"""Actual animation: moving the artwork itself, not just the camera over it.

Everything else in Kage slides a crop rectangle around a still image. However well that
is framed, nothing in the picture ever moves, and a viewer reads it as a slideshow with
good manners. This module makes pixels move.

Two techniques, both of which run on a 4 GB card and neither of which needs a diffusion
model:

**Parallax.** A depth model estimates how far away each pixel is, and the frame is
resampled so near things travel further than far things. That is the cue the eye actually
uses for depth, and it turns a flat panel into a space the camera moves through. Measured
on real manga panels: Depth Anything V2 Small loads in about a second, takes 0.3s a panel
and holds 108 MB of VRAM.

**Drift.** A slow, low-frequency displacement field applied on top, so hair, cloth, smoke
and water keep moving while the shot holds. This is what stops a held frame reading as a
freeze. It needs no model at all - it is two sine waves and a noise field.

Both are per-pixel warps through ``cv2.remap`` at output resolution. Measured at
1280x720, that costs **6.33 ms a frame** with a uint8 source and 13.66 ms with float32 -
not the millisecond this docstring claimed before anyone timed it. The cost is the
displacement maps and the dtype, not ``remap`` itself, which is 1.57 ms. Depth is
computed once per panel at analysis time and cached beside it, so rendering never waits
for a model.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR, env

log = logging.getLogger("kage.animation")

MODEL_CACHE = Path(env("MODEL_CACHE", str(DATA_DIR / "models")))

#: Small on purpose: this has to coexist with the renderer on one 4 GB card.
DEPTH_MODELS = (
    "depth-anything/Depth-Anything-V2-Small-hf",   # Apache-2.0, ~100 MB
    "Intel/dpt-swinv2-tiny-256",                   # MIT, fallback
)

#: Longest edge the depth model sees. Depth is low-frequency; full resolution is waste.
DEPTH_EDGE = 518

_LOCK = threading.Lock()
_PIPE: dict = {"pipe": None, "model": None, "failed": None}


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _pipeline():
    """Load the first depth model that works, once per process."""
    if _PIPE["pipe"] is not None:
        return _PIPE["pipe"]
    if _PIPE["failed"]:
        raise RuntimeError(_PIPE["failed"])

    with _LOCK:
        if _PIPE["pipe"] is not None:
            return _PIPE["pipe"]
        import torch
        from transformers import pipeline

        errors = []
        for model_id in DEPTH_MODELS:
            try:
                pipe = pipeline(
                    "depth-estimation", model=model_id,
                    device=0 if torch.cuda.is_available() else -1,
                    model_kwargs={"cache_dir": str(MODEL_CACHE)},
                )
            except Exception as exc:
                errors.append(f"{model_id}: {type(exc).__name__}: {exc}"[:160])
                continue
            _PIPE.update({"pipe": pipe, "model": model_id})
            log.info("Depth model ready: %s", model_id)
            return pipe
        _PIPE["failed"] = "; ".join(errors) or "no depth model available"
        raise RuntimeError(_PIPE["failed"])


def release() -> None:
    with _LOCK:
        _PIPE.update({"pipe": None, "model": None})
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def depth_path_for(image_path: Path) -> Path:
    """Where the cached depth for a panel lives. Beside it, so deleting one cleans both."""
    return Path(image_path).with_suffix(".depth.npy")


def depth_for(image_path: Path, refresh: bool = False) -> np.ndarray | None:
    """Normalised depth for one panel, 0 far .. 1 near. Cached beside the image.

    Returns None rather than raising: a shot without depth still renders, it just does
    not get parallax.
    """
    image_path = Path(image_path)
    cache = depth_path_for(image_path)
    if cache.exists() and not refresh:
        try:
            return np.load(cache)
        except Exception:
            pass

    try:
        from PIL import Image

        with Image.open(image_path) as raw:
            im = raw.convert("RGB")
            scale = DEPTH_EDGE / max(im.size)
            if scale < 1.0:
                im = im.resize(
                    (max(32, round(im.width * scale)), max(32, round(im.height * scale))),
                    Image.LANCZOS,
                )
            result = _pipeline()(im)
        depth = np.asarray(result["depth"], dtype=np.float32)
    except Exception as exc:
        log.warning("No depth for %s: %s", image_path.name, exc)
        return None

    spread = float(np.ptp(depth))
    if spread < 1e-6:
        return None
    depth = (depth - float(depth.min())) / spread
    # A little blur: depth edges are approximate, and a hard seam warps into a tear.
    depth = cv2.GaussianBlur(depth, (0, 0), 2.0)
    try:
        np.save(cache, depth.astype(np.float32))
    except OSError:
        pass
    return depth.astype(np.float32)


def precompute(image_paths: list[Path]) -> int:
    """Warm the depth cache for a whole shot list. Called at analysis time."""
    done = 0
    for path in image_paths:
        if depth_for(path) is not None:
            done += 1
    release()
    return done


class Warp:
    """Per-frame displacement for one shot: parallax from depth, plus drift.

    Built once per shot. ``maps_at`` returns the remap grids for a moment in the shot,
    which ``cv2.remap`` applies to the already-cropped frame.
    """

    def __init__(self, width: int, height: int, depth: np.ndarray | None, *,
                 parallax: float = 0.0, drift: float = 0.0, seed: int = 0) -> None:
        self.width, self.height = width, height
        self.parallax = float(parallax)
        self.drift = float(drift)
        self.seed = seed

        base_x, base_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        self._base_x, self._base_y = base_x, base_y

        # Anchored at the far plane, not the middle: the background holds still and the
        # foreground slides across it. Centring on the median instead makes near and far
        # swing in opposite directions, which is a stereo wobble, not a camera move.
        if depth is not None and parallax > 0.0:
            field = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
            floor = float(np.percentile(field, 10))
            field = np.clip(field - floor, 0.0, None)
            peak = float(field.max())
            self._depth = (field / peak).astype(np.float32) if peak > 1e-6 else None
        else:
            self._depth = None

        # A smooth random field for the drift: upscaling a tiny noise grid gives
        # low-frequency motion that looks like cloth and smoke rather than static.
        if drift > 0.0:
            rng = np.random.default_rng(seed or 1)
            small = rng.standard_normal((5, 5)).astype(np.float32)
            self._drift_x = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
            small2 = rng.standard_normal((5, 5)).astype(np.float32)
            self._drift_y = cv2.resize(small2, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            self._drift_x = self._drift_y = None

        #: Scratch for maps_at, so a 20-minute render does not allocate two
        #: full-resolution float32 arrays 28,800 times.
        self._map_x = np.empty((height, width), dtype=np.float32)
        self._map_y = np.empty((height, width), dtype=np.float32)

    @property
    def active(self) -> bool:
        return self._depth is not None or self._drift_x is not None

    def maps_at(self, t: float, travel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Remap grids for moment ``t`` (0..1). ``travel`` is the camera's own direction.

        Only the scalars change from frame to frame - the depth and drift fields are
        fixed for the whole shot - so each map is just
        ``base + depth * a + drift * b``. Written as accumulating adds onto a pair of
        zero arrays that was six full-resolution float32 passes and two allocations per
        frame; as two cv2 calls into reused buffers it is one or two. At 1280x720 that
        was the single largest remaining cost in the renderer after the grade.

        The returned arrays are scratch buffers reused on every call, so use them
        before calling this again. ``apply`` hands them straight to ``cv2.remap``,
        which is the only caller.
        """
        depth, drift_x = self._depth, self._drift_x

        if depth is not None:
            # Parallax is driven by where the camera has got to, so the two agree: near
            # pixels lag behind the move, which is exactly what depth looks like.
            amount = self.parallax * self.width
            swing = (t - 0.5) * 2.0
            ax = travel[0] * amount + swing * amount * 0.35
            ay = travel[1] * amount * 0.6

        if drift_x is not None:
            # Two incommensurate frequencies, so the loop never visibly repeats.
            amount = self.drift * self.width
            phase = t * 2.0 * math.pi
            bx = amount * math.sin(phase * 0.7 + self.seed)
            by = amount * math.sin(phase * 1.1 + self.seed * 0.5)

        if depth is not None and drift_x is not None:
            # depth * a + drift * b, then + base.
            cv2.addWeighted(depth, ax, drift_x, bx, 0.0, dst=self._map_x)
            cv2.add(self._map_x, self._base_x, dst=self._map_x)
            cv2.addWeighted(depth, ay, self._drift_y, by, 0.0, dst=self._map_y)
            cv2.add(self._map_y, self._base_y, dst=self._map_y)
        elif depth is not None:
            cv2.scaleAdd(depth, ax, self._base_x, dst=self._map_x)
            cv2.scaleAdd(depth, ay, self._base_y, dst=self._map_y)
        else:
            cv2.scaleAdd(drift_x, bx, self._base_x, dst=self._map_x)
            cv2.scaleAdd(self._drift_y, by, self._base_y, dst=self._map_y)

        return self._map_x, self._map_y

    def apply(self, frame: np.ndarray, t: float,
              travel: tuple[float, float]) -> np.ndarray:
        if not self.active:
            return frame
        map_x, map_y = self.maps_at(t, travel)
        return cv2.remap(
            frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
