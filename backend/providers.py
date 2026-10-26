"""Hosted inference providers, behind one OpenAI-compatible client.

OpenRouter, NVIDIA NIM and Google Gemini all expose the same ``/chat/completions``
shape, so Kage needs one client and a list of candidates rather than an integration per
vendor. A candidate whose key is not configured is skipped, not failed - the chain simply
uses whatever the operator has actually set up.

Why hosted first: the target machine has 4 GB of VRAM and 15 GB of RAM shared with the
app's own backend, frontend and browser. A 1.5B writer alone takes 3.1 GB of VRAM and 2.9
GB of disk. Moving the language work off the machine keeps the studio responsive, and
every provider below has a free tier that comfortably covers one film:
Kage sends **two requests per project**, not two per shot.

Nothing here falls back to written-by-hand text. If every provider fails and local models
are not enabled, narration fails and says why.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger("kage.providers")

TIMEOUT = int(os.environ.get("KAGE_PROVIDER_TIMEOUT", "120"))


@dataclass(frozen=True)
class Provider:
    """One hosted endpoint. ``key_env`` names the environment variable holding its key."""

    name: str
    base_url: str
    key_env: tuple[str, ...]
    signup: str
    free_note: str

    def key(self) -> str:
        for name in self.key_env:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    @property
    def configured(self) -> bool:
        return bool(self.key())


#: Ordered by how much a free account gets for nothing. All verified October 2026.
PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        signup="https://aistudio.google.com/apikey",
        free_note="Free tier, no card. Flash models, ~1500 requests/day, ~10/min.",
    ),
    "openrouter": Provider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        key_env=("OPENROUTER_API_KEY",),
        signup="https://openrouter.ai/keys",
        free_note="Free ':free' models, 20 requests/min and 50/day on a free account.",
    ),
    "nvidia": Provider(
        name="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
        key_env=("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
        signup="https://build.nvidia.com",
        free_note="Free developer tier, no card, ~40 requests/min. Prototyping only "
                  "under NVIDIA's terms - check before shipping commercially.",
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One (provider, model) pair that could do a job."""

    provider: str
    model: str
    vision: bool = False

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def _chain(var: str, default: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    """Let a deployment replace a chain with ``provider:model,provider:model``."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    known = {c.label: c for c in default}
    out: list[Candidate] = []
    for item in raw.split(","):
        label = item.strip()
        if not label or ":" not in label:
            continue
        if label in known:
            out.append(known[label])
        else:
            provider, _, model = label.partition(":")
            out.append(Candidate(provider, model, vision=any(
                k in model.lower() for k in ("vision", "vl", "flash", "gemini", "gpt-4o")
            )))
    return tuple(out) or default


#: Models that can look at an image and say what is drawn.
#: Order is measured, not guessed (October 2026, against a real manga page):
#: flash-lite answered best and fastest; the "latest" alias was 503/timeout under load;
#: 3.5-flash returns no `content` for plain text; OpenRouter's free tier answered 429.
VISION_CHAIN = _chain("KAGE_VISION_CHAIN", (
    Candidate("gemini", "gemini-2.5-flash-lite", vision=True),
    Candidate("gemini", "gemini-2.5-flash", vision=True),
    Candidate("gemini", "gemini-3.5-flash", vision=True),
    Candidate("gemini", "gemini-flash-latest", vision=True),
    Candidate("openrouter", "google/gemma-4-31b-it:free", vision=True),
    Candidate("openrouter", "nvidia/nemotron-3.5-super-vl:free", vision=True),
))

#: Models that turn observations into narration.
TEXT_CHAIN = _chain("KAGE_TEXT_CHAIN", (
    Candidate("gemini", "gemini-2.5-flash-lite"),
    Candidate("gemini", "gemini-2.5-flash"),
    Candidate("gemini", "gemini-flash-latest"),
    Candidate("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    Candidate("openrouter", "google/gemma-4-31b-it:free"),
))


class ProviderError(RuntimeError):
    pass


@dataclass
class Attempt:
    label: str
    outcome: str


@dataclass
class ChainReport:
    """What a chain tried, so the UI can say which model actually wrote the narration."""

    job: str
    used: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def describe(self) -> str:
        if self.used:
            skipped = [a for a in self.attempts if a.outcome != "ok"]
            detail = f"{self.job}: {self.used} in {self.latency_seconds:.1f}s"
            if self.prompt_tokens or self.completion_tokens:
                detail += (f", {self.prompt_tokens}+{self.completion_tokens} tokens")
            return detail + (f" (after {len(skipped)} unavailable)" if skipped else "")
        return f"{self.job}: nothing available - " + "; ".join(
            f"{a.label}: {a.outcome}" for a in self.attempts
        ) or f"{self.job}: no providers configured"


def configured_providers() -> list[Provider]:
    return [p for p in PROVIDERS.values() if p.configured]


def image_part(path: Path, max_edge: int = 768) -> dict:
    """An image as an OpenAI-style content part, downscaled so requests stay small."""
    from io import BytesIO

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buffer = BytesIO()
        im.save(buffer, "JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


@dataclass
class Reply:
    """A completion plus what it cost, so usage can be shown rather than guessed."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __str__(self) -> str:        # callers that only want the words
        return self.text


def complete(candidate: Candidate, messages: list[dict], *, max_tokens: int = 900,
             temperature: float = 0.0) -> Reply:
    """One chat completion. Raises ProviderError with a readable reason."""
    provider = PROVIDERS.get(candidate.provider)
    if provider is None:
        raise ProviderError(f"unknown provider '{candidate.provider}'")
    key = provider.key()
    if not key:
        raise ProviderError(f"no API key set ({' or '.join(provider.key_env)})")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if candidate.provider == "openrouter":
        # OpenRouter asks callers to identify themselves; it also improves rate limits.
        headers["HTTP-Referer"] = "https://github.com/kage-studio"
        headers["X-Title"] = "Kage"

    try:
        response = requests.post(
            f"{provider.base_url}/chat/completions", headers=headers,
            json={"model": candidate.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:180].replace("\n", " ")
        raise ProviderError(f"HTTP {response.status_code}: {detail}")

    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ProviderError(f"unreadable response: {exc}") from exc
    if not str(text or "").strip():
        raise ProviderError("empty response")
    usage = body.get("usage") or {}
    return Reply(
        text=str(text),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def run_chain(job: str, chain: tuple[Candidate, ...], build_messages, parse, *,
              project_id: int | None = None,
              images: int = 0) -> tuple[object, ChainReport]:
    """Try each candidate until one answers and its answer parses.

    A model that replies with something unusable is as failed as one that refuses, so
    parsing happens inside the loop and a parse failure moves to the next candidate.

    The order comes from ``routing``, which remembers what each endpoint has actually
    done. Every attempt is fed back, so the chain gets better at choosing over time.
    """
    from . import routing

    report = ChainReport(job=job)
    usable = [c for c in chain
              if (PROVIDERS.get(c.provider) and PROVIDERS[c.provider].configured)]
    for candidate in chain:
        if candidate not in usable:
            report.attempts.append(Attempt(candidate.label, "no key configured"))

    by_label = {c.label: c for c in usable}
    for label in routing.order([c.label for c in usable], job):
        candidate = by_label[label]
        started = time.perf_counter()
        try:
            reply = complete(candidate, build_messages(candidate))
            result = parse(reply.text)
        except Exception as exc:
            took = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"[:180]
            log.warning("%s via %s failed in %.1fs: %s", job, label, took, reason)
            report.attempts.append(Attempt(label, reason))
            routing.record(label, job, ok=False, latency=took, error=reason)
            routing.usage(project_id, job, candidate.provider, candidate.model,
                          ok=False, latency=took, images=images, error=reason)
            continue
        took = time.perf_counter() - started
        report.attempts.append(Attempt(label, "ok"))
        report.used = label
        report.latency_seconds = took
        report.prompt_tokens = reply.prompt_tokens
        report.completion_tokens = reply.completion_tokens
        routing.record(label, job, ok=True, latency=took)
        routing.usage(project_id, job, candidate.provider, candidate.model, ok=True,
                      latency=took, prompt_tokens=reply.prompt_tokens,
                      completion_tokens=reply.completion_tokens, images=images)
        return result, report
    raise ProviderError(report.describe())


def parse_json_list(raw: str) -> list:
    """Pull a JSON array out of a model reply, fenced or not."""
    text = str(raw).strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in reply")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, list) or not value:
        raise ValueError("JSON array was empty")
    return value
