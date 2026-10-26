"""Image-to-video: the one job that does not fit on this machine.

Kage renders on a 4 GB card. Video diffusion wants 8-12 GB, so a panel that should
actually move is sent to a GPU elsewhere and the resulting clip is used as that shot's
source instead of the still.

Same shape as ``providers.py``: a list of candidates, tried in order, with the order
decided by ``routing`` from what each has actually done. A candidate that is not
configured is skipped, never failed.

The default candidate is a Hugging Face **ZeroGPU** Space, because it is the only free
GPU this project found that is explicitly permitted to be called as an API. Colab was
the obvious alternative and is not usable: its FAQ bans "connecting to remote proxies"
on every managed runtime, bans "bypassing the notebook UI to interact mainly through a
web UI" on free ones, and names web-UI content generation as a common cause of free-tier
termination. A pipeline that gets the account banned is not a pipeline.

Animating is slow and quota-limited wherever it runs, so this is deliberately *not* used
for every shot. Kage animates the few shots worth animating and leaves depth parallax to
carry the rest - which is also how motion comics are actually made.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR, env

log = logging.getLogger("kage.video")

CLIP_CACHE = Path(env("CLIP_CACHE", str(DATA_DIR / "clips")))

#: How much movement to ask for. SVD's motion bucket runs 1-255; measured on line art,
#: past about 120 the drawing starts to smear rather than move.
DEFAULT_MOTION = int(env("ANIMATE_MOTION", "90"))


@dataclass(frozen=True)
class VideoCandidate:
    """One endpoint that can turn a panel into a clip."""

    space: str                 # a Hugging Face Space id
    api_name: str = "/animate"
    licence: str = "Stability Community (non-commercial / under revenue threshold)"

    @property
    def label(self) -> str:
        return f"space:{self.space}"


def chain() -> tuple[VideoCandidate, ...]:
    raw = str(env("ANIMATE_SPACES", "")).strip()
    if raw:
        return tuple(VideoCandidate(s.strip()) for s in raw.split(",") if s.strip())
    return (VideoCandidate("viperlurk/kage-animator"),)


class VideoUnavailable(RuntimeError):
    """No endpoint could animate the panel. The shot keeps its still."""


def is_configured() -> bool:
    """A token is needed even for a public Space: ZeroGPU meters per account."""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


def _space_url(space: str) -> str:
    return f"https://{space.replace('/', '-').lower()}.hf.space"


def _call_space(space: str, api_name: str, image_path: Path, payload: list,
                timeout: int = 600) -> str:
    """Call a Gradio Space over its REST API and return the produced file's URL.

    Deliberately plain ``requests`` rather than ``gradio_client``: that package pulls
    huggingface-hub 2.x, and transformers pins <1.0, so installing it silently broke
    CLIP, depth estimation and narration. Three HTTP calls are not worth a dependency
    that takes the local pipeline down with it.
    """
    import requests

    base = _space_url(space)
    headers = {}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # 1. upload the panel; Gradio answers with a server-side path
    with open(image_path, "rb") as handle:
        up = requests.post(f"{base}/gradio_api/upload", headers=headers,
                           files={"files": (Path(image_path).name, handle, "image/png")},
                           timeout=180)
    up.raise_for_status()
    uploaded = up.json()[0]

    file_ref = {"path": uploaded, "meta": {"_type": "gradio.FileData"}}

    # 2. start the job
    start = requests.post(f"{base}/gradio_api/call{api_name}", headers=headers,
                          json={"data": [file_ref, *payload]}, timeout=120)
    start.raise_for_status()
    event_id = start.json()["event_id"]

    # 3. stream until it completes; ZeroGPU queues, so this can wait a while
    stream = requests.get(f"{base}/gradio_api/call{api_name}/{event_id}",
                          headers=headers, stream=True, timeout=timeout)
    stream.raise_for_status()
    event = None
    for raw in stream.iter_lines(decode_unicode=True):
        if raw is None or raw == "":
            continue
        if raw.startswith("event:"):
            event = raw.split(":", 1)[1].strip()
        elif raw.startswith("data:"):
            body = raw.split(":", 1)[1].strip()
            if event == "error":
                # ZeroGPU rejects a call with no message once the account's free daily
                # allowance is gone. Saying so beats "space error: null", which sends
                # the reader looking for a bug in their own panel.
                detail = body.strip()
                if detail in ("", "null", "None"):
                    raise RuntimeError(
                        "the Space refused the job with no message - usually the free "
                        "ZeroGPU daily quota. It resets; the still is used meanwhile."
                    )
                raise RuntimeError(f"space error: {detail[:200]}")
            if event == "complete":
                import json as _json

                result = _json.loads(body)
                item = result[0] if isinstance(result, list) and result else result
                return _file_url(item)
    raise RuntimeError("space produced no result")


def _file_url(item) -> str:
    """Dig the file URL out of a Gradio result.

    A video component answers ``{"video": {"url": ...}, "subtitles": null}`` rather than
    a bare file dict, so the URL sits one level down. Walk instead of assuming a shape -
    Gradio has changed this between versions.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("url", "path"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        for value in item.values():
            found = _file_url(value)
            if found:
                return found
    if isinstance(item, list):
        for value in item:
            found = _file_url(value)
            if found:
                return found
    return ""


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None


def clip_path_for(image_path: Path, motion: int, seed: int) -> Path:
    """Where a panel's clip is cached. Keyed on the settings that produced it."""
    stem = Path(image_path).stem
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    return CLIP_CACHE / f"{stem}_m{motion}_s{seed}.mp4"


def animate_panel(image_path: Path, *, motion: int | None = None, seed: int = 0,
                  steps: int = 20, fps: int = 8,
                  project_id: int | None = None) -> Path:
    """Animate one panel and return a local mp4. Cached by panel and settings.

    Raises VideoUnavailable rather than returning a still, so the caller decides whether
    a missing clip is worth failing a render over. It never is: the still already works.
    """
    from . import routing

    motion = DEFAULT_MOTION if motion is None else int(motion)
    cached = clip_path_for(image_path, motion, seed)
    if cached.exists() and cached.stat().st_size > 1024:
        return cached

    if not is_configured():
        raise VideoUnavailable(
            "Animation needs HF_TOKEN set - ZeroGPU is metered per account."
        )

    candidates = chain()
    attempts: list[str] = []
    for candidate in routing.order([c.label for c in candidates], "Animate"):
        target = next(c for c in candidates if c.label == candidate)
        started = time.perf_counter()
        try:
            import requests

            url = _call_space(target.space, target.api_name, Path(image_path),
                              [motion, steps, seed, fps])
            if not url:
                raise RuntimeError("endpoint returned no video")
            if not url.startswith("http"):
                url = f"{_space_url(target.space)}/gradio_api/file={url}"
            got = requests.get(url, timeout=600)
            got.raise_for_status()
            if len(got.content) < 1024:
                raise RuntimeError("endpoint returned an empty video")
            cached.write_bytes(got.content)
        except Exception as exc:
            took = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("Animate via %s failed in %.1fs: %s", target.label, took, reason)
            attempts.append(f"{target.label}: {reason}")
            routing.record(target.label, "Animate", ok=False, latency=took, error=reason)
            routing.usage(project_id, "Animate", "space", target.space, ok=False,
                          latency=took, images=1, error=reason)
            continue

        took = time.perf_counter() - started
        log.info("Animated %s via %s in %.1fs -> %s",
                 Path(image_path).name, target.label, took, cached.name)
        routing.record(target.label, "Animate", ok=True, latency=took)
        routing.usage(project_id, "Animate", "space", target.space, ok=True,
                      latency=took, images=1)
        return cached

    raise VideoUnavailable("; ".join(attempts) or "no animation endpoint configured")


def capability() -> dict:
    """What animation can do right now, for the health endpoint and the UI."""
    candidates = chain()
    return {
        "ready": is_configured(),
        "token": is_configured(),
        "spaces": [c.space for c in candidates],
        "licence": candidates[0].licence if candidates else "",
        "cache": str(CLIP_CACHE),
    }
