"""Cinematic motion from static manga artwork.

Every shot is a time-based camera move over the source image: the crop rectangle is
interpolated with eased curves and resampled once per frame with a sub-pixel affine
warp, so motion is smooth rather than a series of integer jumps. Effects are layered on
top of the warped frame. Frames are yielded one at a time (CPU only, no GPU, no
full-resolution frame buffer held in memory).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FPS
from .styles import GradeSpec

# Zoom travel per camera move, scaled by intensity.
INTENSITY_MOTION = {"subtle": 0.55, "balanced": 1.0, "dramatic": 1.55}

BASE_ZOOM_TRAVEL = 0.40   # fraction of frame gained/lost across a zoom
BASE_PAN_TRAVEL = 0.38    # fraction of the extra crop width traversed by a pan
PAN_REACH = 0.92          # how much of the available slack a pan is allowed to use
MIN_CROP_FRACTION = 0.62  # tightest crop, as a share of the page's own dimension
MAX_PAN_RATE = 0.26       # most of a frame a pan may travel per second
WHITE_POINT = 246.0       # where the source's paper is put before grading
INK_FENCE_SHARE = 0.45    # a move may not go where the frame holds less ink than this
SAFE_ZOOM = 1.08          # slight default over-scan so pans always have room


# ------------------------------------------------------------------- easings


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_quad(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


@dataclass
class CameraPath:
    """Normalised start/end crop geometry for one shot."""

    zoom_start: float
    zoom_end: float
    cx_start: float
    cx_end: float
    cy_start: float
    cy_end: float
    easing: str = "smooth"

    def at(self, t: float) -> tuple[float, float, float]:
        if self.easing == "out":
            e = ease_out_cubic(t)
        elif self.easing == "inout":
            e = ease_in_out_quad(t)
        else:
            e = smoothstep(t)
        return (
            self.zoom_start + (self.zoom_end - self.zoom_start) * e,
            self.cx_start + (self.cx_end - self.cx_start) * e,
            self.cy_start + (self.cy_end - self.cy_start) * e,
        )


def build_camera_path(motion: str, intensity: str = "balanced") -> CameraPath:
    """Map a camera-motion name onto a start/end crop. cx/cy are in [-1, 1]."""
    k = INTENSITY_MOTION.get(intensity, 1.0)
    zoom = BASE_ZOOM_TRAVEL * k
    pan = min(1.0, BASE_PAN_TRAVEL * k * 4.0)

    if motion == "slow_zoom_in":
        return CameraPath(SAFE_ZOOM, SAFE_ZOOM + zoom, 0.06 * k, -0.10 * k, 0.14 * k, -0.18 * k, "smooth")
    if motion == "slow_zoom_out":
        return CameraPath(SAFE_ZOOM + zoom, SAFE_ZOOM, -0.10 * k, 0.06 * k, -0.18 * k, 0.14 * k, "smooth")
    if motion == "push_in":
        return CameraPath(SAFE_ZOOM, SAFE_ZOOM + zoom * 1.9, 0.0, 0.08 * k, 0.10 * k, -0.12 * k, "out")
    if motion == "pan_left":
        return CameraPath(SAFE_ZOOM + zoom * 0.5, SAFE_ZOOM + zoom * 0.5, pan, -pan, 0.0, 0.0, "inout")
    if motion == "pan_right":
        return CameraPath(SAFE_ZOOM + zoom * 0.5, SAFE_ZOOM + zoom * 0.5, -pan, pan, 0.0, 0.0, "inout")
    if motion == "pan_up":
        return CameraPath(SAFE_ZOOM + zoom * 0.5, SAFE_ZOOM + zoom * 0.5, 0.0, 0.0, pan, -pan, "inout")
    if motion == "pan_down":
        return CameraPath(SAFE_ZOOM + zoom * 0.5, SAFE_ZOOM + zoom * 0.5, 0.0, 0.0, -pan, pan, "inout")
    if motion == "ken_burns":
        return CameraPath(
            SAFE_ZOOM, SAFE_ZOOM + zoom * 1.3, -pan * 0.7, pan * 0.5, pan * 0.5, -pan * 0.6, "smooth"
        )
    if motion == "rapid_pan":
        return CameraPath(
            SAFE_ZOOM + zoom, SAFE_ZOOM + zoom * 0.6, -pan * 1.0, pan * 1.0, 0.0, 0.0, "out"
        )
    # static_drift and anything unrecognised. "Static" means no cut-like move, not a
    # frozen frame - at 0.35 zoom and 0.05 drift it measured as a held still.
    return CameraPath(
        SAFE_ZOOM, SAFE_ZOOM + zoom * 0.8, -0.24 * k, 0.24 * k, 0.15 * k, -0.15 * k, "smooth"
    )


FAST_MOTIONS = {"rapid_pan", "push_in"}


# --------------------------------------------------------------- shot renderer


@dataclass
class ShotRenderer:
    """Renders the frames of a single shot."""

    image_path: Path
    out_width: int
    out_height: int
    camera_motion: str = "slow_zoom_in"
    effects: list[str] = field(default_factory=list)
    intensity: str = "balanced"
    fps: int = FPS
    seed: int = 0
    grade: GradeSpec | None = None
    framing: float = 1.0   # >1 crops in toward the subject instead of showing the page
    #: Depth-driven parallax and idle drift. These are what actually animate the artwork
    #: rather than only moving a crop around it; see backend/animation.py.
    parallax: float = 0.0
    drift: float = 0.0

    def __post_init__(self) -> None:
        self._source = self._load_source()
        self._path = build_camera_path(self.camera_motion, self.intensity)
        self._framing = self._effective_framing()
        self._anchor, self._centre_limits = self._analyse_ink()
        self._rng = np.random.default_rng(self.seed or 1)
        self._grade = self.grade or GradeSpec()
        self._saturation = self._source_saturation()
        self._grade_lut = self._make_grade_lut()
        self._vignette = self._make_vignette() if self._grade.vignette > 0.01 else None
        self._speed_lines = self._make_speed_lines() if "speed_lines" in self.effects else None
        self._particles = self._make_particles() if "particles" in self.effects else None
        self._rain = self._make_rain() if "rain" in self.effects else None
        grain = self._grade.grain + (0.18 if "film_grain" in self.effects else 0.0)
        self._grain_amount = min(1.0, grain)
        self._grain_bank = self._make_grain_bank() if self._grain_amount > 0.01 else None
        self._warp = self._build_warp()

    def _build_warp(self):
        """Depth parallax + drift for this shot, or a no-op when neither is wanted."""
        from .animation import Warp, depth_for

        if self.parallax <= 0.0 and self.drift <= 0.0:
            return Warp(self.out_width, self.out_height, None, seed=self.seed)
        depth = depth_for(self.image_path) if self.parallax > 0.0 else None
        return Warp(
            self.out_width, self.out_height, depth,
            parallax=self.parallax, drift=self.drift, seed=self.seed,
        )

    # ---- source preparation

    def _source_saturation(self) -> float:
        """Effective saturation: a colour floor only applies to a colourful source.

        Measured as the mean chroma of the page. Ink on paper sits near zero; a tinted
        woodblock print or a colour cover sits well above it.
        """
        g = self._grade
        if g.colour_floor <= g.saturation:
            return g.saturation
        sample = self._source[::7, ::7]
        chroma = float(
            (sample.max(axis=2).astype(np.float32) - sample.min(axis=2)).mean()
        )
        # ~6/255 is the chroma of a slightly warm scan of black-and-white art.
        return g.colour_floor if chroma > 9.0 else g.saturation

    @staticmethod
    def _normalise_levels(arr: np.ndarray) -> np.ndarray:
        """Stretch the source onto a full black-to-white range before grading.

        A photographed page is lit, not printed: its paper lands around 190 and its ink
        around 70, so a style that darkens further leaves everything in the middle. This
        puts the paper back at paper-white and the ink back at black, equally across all
        three channels so hue is untouched.
        """
        luma = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lo, hi = np.percentile(luma[::5, ::5], (1.0, 99.0))
        if hi - lo < 40.0:
            return arr  # too flat to be a lit page; leave it alone
        gain = WHITE_POINT / float(hi - lo)
        if 0.96 < gain < 1.06:
            return arr
        gain = float(np.clip(gain, 0.75, 2.2))
        # Pivot on the white point, not the black one. Clamping the gain while still
        # subtracting the full black offset pulls the paper *down*: a faded page whose
        # range is 135..203 wants a gain of 3.6, gets 1.9, and its paper lands at 137
        # instead of 196 - darker than it started. Anchoring the top means a clamped
        # gain simply lifts the blacks less, and paper is always paper.
        out = (arr.astype(np.float32) - float(hi)) * gain + WHITE_POINT
        return np.clip(out, 0, 255).astype(np.uint8)

    def _load_source(self) -> np.ndarray:
        with Image.open(self.image_path) as raw:
            img = raw.convert("RGB")
            arr = np.asarray(img)

        # Resample once so the per-frame warp is a mild scale change: this keeps motion
        # crisp without aliasing and without resampling a huge scan every frame. Tighter
        # framing crops into less of the page, so it needs more source pixels to stay sharp.
        target_long = int(max(self.out_width, self.out_height) * 1.7 * max(1.0, self.framing))
        h, w = arr.shape[:2]
        longest = max(h, w)
        if longest > target_long * 1.05:
            scale = target_long / longest
            arr = cv2.resize(
                arr, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA
            )
        elif longest < target_long * 0.7:
            scale = (target_long * 0.7) / longest
            arr = cv2.resize(
                arr, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_CUBIC
            )
        if (self.grade or GradeSpec()).normalise_levels:
            arr = self._normalise_levels(arr)
        return np.ascontiguousarray(arr)

    def frame_count(self, duration_seconds: float) -> int:
        return max(2, int(round(duration_seconds * self.fps)))

    # ---- geometry

    def _base_rect(self) -> tuple[float, float]:
        """Largest rect inside the source matching the output aspect ratio."""
        src_h, src_w = self._source.shape[:2]
        target_aspect = self.out_width / self.out_height
        src_aspect = src_w / src_h
        if src_aspect >= target_aspect:
            return src_h * target_aspect, float(src_h)
        return float(src_w), src_w / target_aspect

    def _effective_framing(self) -> float:
        """Clamp the style's framing to something this source can actually carry.

        ``framing`` is a style choice expressed relative to a whole sheet. Sheets are
        split into single pages at import, so applying the same number to a half-size
        source crops twice as tight and the shot lands on a fragment of a sleeve. This
        caps it so the tightest point of the move still shows a usable share of the page.
        """
        if self.framing <= 1.01:
            return 1.0
        src_h, src_w = self._source.shape[:2]
        base_w, base_h = self._base_rect()
        max_zoom = max(self._path.zoom_start, self._path.zoom_end, 1.0)
        # Constrain whichever dimension the output aspect does not already crop.
        if base_w >= src_w * 0.999:
            limit = base_w / (max_zoom * MIN_CROP_FRACTION * src_w)
        else:
            limit = base_h / (max_zoom * MIN_CROP_FRACTION * src_h)
        return float(max(1.0, min(self.framing, limit)))

    def _analyse_ink(self) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
        """Where the drawing is: the best window to frame on, and how far a move may roam.

        A page is mostly paper, so framing on the page centre shows nothing in
        particular and a pan with real travel slides off into the margin. The anchor is
        the densest window of ink at the output aspect; the limits are the range of crop
        centres that still hold a drawing, which ``_crop_rect`` uses as a fence.
        """
        src_h, src_w = self._source.shape[:2]
        centre = (src_w / 2.0, src_h / 2.0)
        whole = (0.0, float(src_w), 0.0, float(src_h))
        if self._framing <= 1.01:
            return centre, whole

        scale = 220.0 / max(src_w, src_h)
        small_w, small_h = max(16, int(src_w * scale)), max(16, int(src_h * scale))
        gray = cv2.cvtColor(
            cv2.resize(self._source, (small_w, small_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY,
        )
        # Relative to this page's own paper, not a fixed level: a pale print whose
        # lightest linework sits at 200 has ink, and a fixed threshold calls it blank -
        # which leaves the camera no reason not to park on a sky.
        level = float(np.median(gray))
        floor_level = float(np.percentile(gray, 2))
        ink = (gray < level - max(12.0, (level - floor_level) * 0.30)).astype(np.float32)

        # Weight by local detail. A panel flooded with black is entirely "ink" and holds
        # no picture; scoring by ink mass alone sent the camera to sit on it. Edge energy
        # asks "is anything drawn here", which is the actual question.
        edges = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
        detail = cv2.GaussianBlur(np.abs(edges), (0, 0), 1.5)
        peak = float(detail.max())
        if peak > 1e-6:
            ink = ink * (0.25 + 0.75 * (detail / peak))
        if ink.sum() < 8:
            return centre, whole

        base_w, base_h = self._base_rect()
        win_w = max(4, int(base_w / self._framing * scale))
        win_h = max(4, int(base_h / self._framing * scale))
        win_w, win_h = min(win_w, small_w), min(win_h, small_h)

        integral = cv2.integral(ink)
        best, best_xy = -1.0, (0, 0)
        step = max(1, min(win_w, win_h) // 8)
        for y in range(0, small_h - win_h + 1, step):
            for x in range(0, small_w - win_w + 1, step):
                total = (
                    integral[y + win_h, x + win_w] - integral[y, x + win_w]
                    - integral[y + win_h, x] + integral[y, x]
                )
                if total > best:
                    best, best_xy = total, (x, y)

        bx, by = best_xy
        cx = (bx + win_w / 2.0) / scale
        cy = (by + win_h / 2.0) / scale
        anchor = (float(np.clip(cx, 0, src_w)), float(np.clip(cy, 0, src_h)))

        # How far the camera may travel and still have a drawing in frame. Not the outer
        # extent of the ink: a page with a picture top and bottom has ink at both ends
        # and nothing in the middle, and a move fenced only by the extent parks in the
        # gap. This walks out from the best window along each axis and stops where the
        # ink in frame falls below a share of what the anchor sees.
        def window(x: int, y: int) -> float:
            return float(
                integral[y + win_h, x + win_w] - integral[y, x + win_w]
                - integral[y + win_h, x] + integral[y, x]
            )

        floor = best * INK_FENCE_SHARE
        x_lo, x_hi = self._walk(lambda x: window(x, by), bx, small_w - win_w, step, floor)
        y_lo, y_hi = self._walk(lambda y: window(bx, y), by, small_h - win_h, step, floor)
        limits = (
            (x_lo + win_w / 2.0) / scale, (x_hi + win_w / 2.0) / scale,
            (y_lo + win_h / 2.0) / scale, (y_hi + win_h / 2.0) / scale,
        )
        return anchor, limits

    @staticmethod
    def _walk(score, start: int, limit: int, step: int, floor: float) -> tuple[int, int]:
        """Widest run around ``start`` whose score stays at or above ``floor``."""
        lo = start
        while lo - step >= 0 and score(lo - step) >= floor:
            lo -= step
        hi = start
        while hi + step <= limit and score(hi + step) >= floor:
            hi += step
        return lo, hi

    def _travel_scale(self, duration_seconds: float) -> tuple[float, float]:
        """Scale the pan so it reads as a camera move, not a whip.

        The path asks for a fraction of the *available slack*, and a 16:9 crop of a tall
        page has an enormous amount of it - taking all of it in three seconds is a swipe.
        What makes a pan feel like a pan is its rate, so cap travel at a share of the
        frame per second and leave short, gentle moves untouched.
        """
        if duration_seconds <= 0.01:
            return (1.0, 1.0)
        src_h, src_w = self._source.shape[:2]
        base_w, base_h = self._base_rect()
        zoom = max(self._path.zoom_start, self._path.zoom_end, 1.0)
        rect_w = min(base_w / self._framing / zoom, float(src_w))
        rect_h = min(base_h / self._framing / zoom, float(src_h))

        out = []
        for span, rect, src in (
            (abs(self._path.cx_end - self._path.cx_start), rect_w, src_w),
            (abs(self._path.cy_end - self._path.cy_start), rect_h, src_h),
        ):
            slack = max(0.0, src - rect) / 2.0
            wanted = span * slack * PAN_REACH
            allowed = MAX_PAN_RATE * duration_seconds * rect
            out.append(1.0 if wanted <= allowed or wanted <= 1.0 else allowed / wanted)
        return (out[0], out[1])

    @staticmethod
    def _fence(centre: float, low: float, high: float) -> float:
        """Hold the crop centre inside the range the drawing supports."""
        if high <= low:
            return (low + high) / 2.0
        return float(np.clip(centre, low, high))

    def _crop_rect(self, t: float, shake: tuple[float, float]) -> tuple[float, float, float, float]:
        src_h, src_w = self._source.shape[:2]
        base_w, base_h = self._base_rect()
        base_w /= self._framing
        base_h /= self._framing
        zoom, cx, cy = self._path.at(t)

        rect_w = min(base_w / zoom, float(src_w))
        rect_h = min(base_h / zoom, float(src_h))

        slack_x = max(0.0, src_w - rect_w) / 2.0
        slack_y = max(0.0, src_h - rect_h) / 2.0
        anchor_x, anchor_y = self._anchor
        tx, ty = getattr(self, "_travel", (1.0, 1.0))
        centre_x = anchor_x + cx * slack_x * PAN_REACH * tx + shake[0]
        centre_y = anchor_y + cy * slack_y * PAN_REACH * ty + shake[1]

        # Fence the move inside the drawing. Travel is worth having only while there is
        # something in frame; without this the pan reaches the end of its slack, which is
        # the edge of the paper, and holds on an empty margin.
        x_lo, x_hi, y_lo, y_hi = self._centre_limits
        centre_x = self._fence(centre_x, x_lo, x_hi)
        centre_y = self._fence(centre_y, y_lo, y_hi)

        x = min(max(centre_x - rect_w / 2.0, 0.0), max(0.0, src_w - rect_w))
        y = min(max(centre_y - rect_h / 2.0, 0.0), max(0.0, src_h - rect_h))
        return x, y, rect_w, rect_h

    def _shake_offset(self, t: float, frame_index: int) -> tuple[float, float]:
        if "screen_shake" not in self.effects:
            return (0.0, 0.0)
        k = INTENSITY_MOTION.get(self.intensity, 1.0)
        amp = 4.6 * k * max(0.0, 1.0 - t * 0.9)  # strongest at the start, settles down
        return (
            amp * math.sin(frame_index * 2.7 + self.seed),
            amp * math.sin(frame_index * 3.9 + self.seed * 1.7),
        )

    # ---- effect assets

    def _make_grade_lut(self) -> np.ndarray:
        """256x1x3 lookup folding contrast, lift and tint into one cv2.LUT pass."""
        g = self._grade
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        x = x * g.exposure                        # overall exposure
        x = (x - 0.5) * g.contrast + 0.5          # contrast around mid grey
        x = x * (1.0 - g.lift) + g.lift           # matte lift of the blacks
        channels = [np.clip(x * t, 0.0, 1.0) * 255.0 for t in g.tint]
        return np.stack(channels, axis=-1).reshape(256, 1, 3).astype(np.uint8)

    def _make_vignette(self) -> np.ndarray:
        h, w = self.out_height, self.out_width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - w / 2.0) / (w / 2.0)
        ny = (yy - h / 2.0) / (h / 2.0)
        radius = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
        strength = self._grade.vignette
        if self.intensity == "dramatic":
            strength = min(1.0, strength * 1.12)
        mask = 1.0 - strength * np.clip((radius - 0.42) / 0.58, 0.0, 1.0) ** 1.6
        return mask[:, :, None].astype(np.float32)

    def _make_speed_lines(self) -> np.ndarray:
        """Radial white streaks, rendered larger than frame so it can be zoomed."""
        scale = 1.45
        h, w = int(self.out_height * scale), int(self.out_width * scale)
        layer = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        rng = np.random.default_rng(self.seed + 11)
        for _ in range(96):
            angle = rng.uniform(0, 2 * math.pi)
            inner = rng.uniform(0.26, 0.58) * min(w, h) / 2
            outer = min(w, h) * rng.uniform(0.75, 1.15)
            p1 = (int(cx + math.cos(angle) * inner), int(cy + math.sin(angle) * inner))
            p2 = (int(cx + math.cos(angle) * outer), int(cy + math.sin(angle) * outer))
            cv2.line(layer, p1, p2, int(rng.integers(120, 255)), int(rng.integers(1, 4)), cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.1)
        return layer

    def _make_particles(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 23)
        n = 44
        return np.stack(
            [
                rng.uniform(0, 1, n),            # x
                rng.uniform(0, 1, n),            # y
                rng.uniform(0.012, 0.05, n),     # drift speed
                rng.uniform(1.0, 3.2, n),        # radius px
                rng.uniform(0, 2 * math.pi, n),  # sway phase
                rng.uniform(0.25, 0.8, n),       # opacity
            ],
            axis=1,
        ).astype(np.float32)

    def _make_rain(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 37)
        n = 220
        return np.stack(
            [
                rng.uniform(-0.1, 1.1, n),
                rng.uniform(0, 1, n),
                rng.uniform(1.1, 2.2, n),      # fall speed (screens/sec)
                rng.uniform(14, 34, n),        # streak length px
                rng.uniform(0.18, 0.45, n),    # opacity
            ],
            axis=1,
        ).astype(np.float32)

    def _make_grain_bank(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed + 53)
        sigma = 2.0 + 9.0 * self._grain_amount
        return [
            rng.normal(0.0, sigma, (self.out_height, self.out_width, 1)).astype(np.float32)
            for _ in range(8)
        ]

    # ---- effect application

    def _apply_grade(self, frame: np.ndarray) -> np.ndarray:
        """Always-on style grade, then any legacy per-shot colour effects."""
        g = self._grade

        # Contrast + lift + tint in a single LUT pass on uint8.
        graded = cv2.LUT(np.clip(frame, 0, 255).astype(np.uint8), self._grade_lut)
        frame = graded.astype(np.float32)

        saturation = self._saturation
        if abs(saturation - 1.0) > 0.01:
            gray3 = cv2.cvtColor(cv2.cvtColor(graded, cv2.COLOR_RGB2GRAY),
                                 cv2.COLOR_GRAY2RGB).astype(np.float32)
            frame = cv2.addWeighted(frame, saturation, gray3, 1.0 - saturation, 0.0)

        if g.bloom > 0.01:
            # Glow only the highlights, computed at quarter resolution. A soft bloom loses
            # nothing visible at that scale and costs about a sixteenth of a full-size blur.
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (max(4, w // 4), max(4, h // 4)),
                               interpolation=cv2.INTER_AREA)
            highlights = np.clip(small - 165.0, 0.0, None)
            blur = cv2.GaussianBlur(highlights, (0, 0), 3.4)
            frame = frame + cv2.resize(blur, (w, h),
                                       interpolation=cv2.INTER_LINEAR) * (g.bloom * 1.6)

        # Legacy named effects, still available from the shot planner.
        if "desaturate" in self.effects:
            gray = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8),
                                cv2.COLOR_RGB2GRAY).astype(np.float32)
            frame = frame * 0.35 + gray[:, :, None] * 0.65
        if "warm_glow" in self.effects:
            frame = frame * np.array([1.06, 1.0, 0.92], dtype=np.float32)
            blur = cv2.GaussianBlur(frame, (0, 0), 9)
            frame = np.clip(frame * 0.86 + blur * 0.18, 0, 255)
        if "cold_tint" in self.effects:
            frame = frame * np.array([0.88, 0.96, 1.10], dtype=np.float32)
        return frame

    def _apply_motion_blur(self, frame: np.ndarray, t: float) -> np.ndarray:
        if "motion_blur" not in self.effects:
            return frame
        # Blur along the dominant travel direction, strongest mid-move.
        dx = self._path.cx_end - self._path.cx_start
        dy = self._path.cy_end - self._path.cy_start
        strength = (1.0 - abs(t - 0.4) / 0.6) if t < 1.0 else 0.0
        strength = max(0.0, strength)
        size = int(3 + 10 * strength * INTENSITY_MOTION.get(self.intensity, 1.0))
        if size < 3:
            return frame
        size += (size + 1) % 2  # force odd
        kernel = np.zeros((size, size), dtype=np.float32)
        if abs(dx) >= abs(dy):
            kernel[size // 2, :] = 1.0
        else:
            kernel[:, size // 2] = 1.0
        kernel /= kernel.sum()
        blurred = cv2.filter2D(frame, -1, kernel)
        return frame * (1.0 - 0.55 * strength) + blurred * (0.55 * strength)

    def _apply_speed_lines(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self._speed_lines is None:
            return frame
        layer = self._speed_lines
        lh, lw = layer.shape
        # Zoom the streak layer outward over time so the lines appear to rush past.
        z = 1.0 + 0.22 * t
        cw, ch = int(self.out_width / z), int(self.out_height / z)
        cw, ch = min(cw, lw), min(ch, lh)
        x0, y0 = (lw - cw) // 2, (lh - ch) // 2
        crop = layer[y0 : y0 + ch, x0 : x0 + cw]
        resized = cv2.resize(crop, (self.out_width, self.out_height), interpolation=cv2.INTER_LINEAR)
        swell = math.sin(math.pi * min(1.0, t * 1.35)) ** 1.5
        alpha = (resized.astype(np.float32) / 255.0) * (0.10 + 0.26 * swell)
        alpha = np.clip(alpha, 0.0, 0.46)[:, :, None]
        return frame * (1.0 - alpha) + 255.0 * alpha

    def _apply_particles(self, frame: np.ndarray, t: float, duration: float) -> np.ndarray:
        if self._particles is None:
            return frame
        overlay = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        elapsed = t * duration
        for px, py, speed, radius, phase, opacity in self._particles:
            y = (py - speed * elapsed * 0.25) % 1.0
            x = (px + math.sin(phase + elapsed * 0.8) * 0.012) % 1.0
            cv2.circle(
                overlay,
                (int(x * self.out_width), int(y * self.out_height)),
                max(1, int(radius)),
                float(opacity),
                -1,
                cv2.LINE_AA,
            )
        overlay = cv2.GaussianBlur(overlay, (0, 0), 1.8)
        alpha = np.clip(overlay, 0.0, 0.42)[:, :, None]
        return frame * (1.0 - alpha) + 242.0 * alpha

    def _apply_rain(self, frame: np.ndarray, t: float, duration: float) -> np.ndarray:
        if self._rain is None:
            return frame
        overlay = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        elapsed = t * duration
        for px, py, speed, length, opacity in self._rain:
            y = (py + speed * elapsed) % 1.15
            x = (px + 0.04 * y) % 1.0
            x0 = int(x * self.out_width)
            y0 = int(y * self.out_height) - int(length)
            cv2.line(overlay, (x0 - 3, y0), (int(x0), int(y0 + length)), float(opacity), 1, cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (0, 0), 0.8)
        alpha = np.clip(overlay, 0.0, 0.32)[:, :, None]
        return frame * (1.0 - alpha) + 226.0 * alpha

    def _apply_impact_flash(self, frame: np.ndarray, t: float, duration: float) -> np.ndarray:
        if "impact_flash" not in self.effects:
            return frame
        flash_len = min(0.22, duration * 0.3)
        elapsed = t * duration
        if elapsed > flash_len:
            return frame
        k = (1.0 - elapsed / flash_len) ** 1.6
        amount = 0.52 * k * (1.15 if self.intensity == "dramatic" else 1.0)
        return frame * (1.0 - amount) + 255.0 * amount

    def _apply_grain(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if not self._grain_bank:
            return frame
        return frame + self._grain_bank[frame_index % len(self._grain_bank)]

    # ---- main loop

    def iter_frames(self, duration_seconds: float):
        """Yield uint8 RGB frames of shape (out_height, out_width, 3)."""
        total = self.frame_count(duration_seconds)
        src_h, src_w = self._source.shape[:2]
        self._travel = self._travel_scale(duration_seconds)
        # Which way the camera is going, so parallax leans the same way it does.
        travel = (
            float(np.clip(self._path.cx_end - self._path.cx_start, -1.0, 1.0)),
            float(np.clip(self._path.cy_end - self._path.cy_start, -1.0, 1.0)),
        )

        for i in range(total):
            t = i / float(total - 1) if total > 1 else 0.0
            shake = self._shake_offset(t, i)
            x, y, rect_w, rect_h = self._crop_rect(t, shake)

            sx = self.out_width / rect_w
            sy = self.out_height / rect_h
            matrix = np.array([[sx, 0.0, -x * sx], [0.0, sy, -y * sy]], dtype=np.float32)
            interp = cv2.INTER_AREA if sx < 0.95 else cv2.INTER_LINEAR
            warped = cv2.warpAffine(
                self._source,
                matrix,
                (self.out_width, self.out_height),
                flags=interp,
                borderMode=cv2.BORDER_REPLICATE,
            )

            frame = warped.astype(np.float32)
            if self._warp.active:
                frame = self._warp.apply(frame, t, travel)
            frame = self._apply_motion_blur(frame, t)
            frame = self._apply_grade(frame)
            frame = self._apply_speed_lines(frame, t)
            frame = self._apply_particles(frame, t, duration_seconds)
            frame = self._apply_rain(frame, t, duration_seconds)
            if self._vignette is not None:
                frame = frame * self._vignette
            frame = self._apply_impact_flash(frame, t, duration_seconds)
            frame = self._apply_grain(frame, i)

            yield np.clip(frame, 0, 255).astype(np.uint8)

        del src_h, src_w  # geometry already consumed; kept explicit for clarity


# ------------------------------------------------------------------ compositing


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear cross-dissolve: alpha=0 -> a, alpha=1 -> b."""
    alpha = min(1.0, max(0.0, alpha))
    return np.clip(
        a.astype(np.float32) * (1.0 - alpha) + b.astype(np.float32) * alpha, 0, 255
    ).astype(np.uint8)


def fade_to_black(frame: np.ndarray, amount: float) -> np.ndarray:
    """amount=0 -> untouched, amount=1 -> black."""
    k = 1.0 - min(1.0, max(0.0, amount))
    return np.clip(frame.astype(np.float32) * k, 0, 255).astype(np.uint8)


def flash_white(frame: np.ndarray, amount: float) -> np.ndarray:
    a = min(1.0, max(0.0, amount))
    return np.clip(frame.astype(np.float32) * (1 - a) + 255.0 * a, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------- captions

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap(text: str, font, draw, max_width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def draw_subtitle(frame: np.ndarray, text: str, opacity: float = 1.0) -> np.ndarray:
    """Burn a styled caption into the frame (keeps it crisp and player-independent)."""
    if not text or opacity <= 0.01:
        return frame

    h, w = frame.shape[:2]
    img = Image.fromarray(frame)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = load_font(max(16, int(h * 0.042)))
    margin = int(w * 0.07)
    lines = _wrap(text, font, draw, w - 2 * margin)
    if not lines:
        return frame

    line_h = int(font.size * 1.34)
    block_h = line_h * len(lines)
    top = h - int(h * 0.085) - block_h
    alpha = int(255 * min(1.0, max(0.0, opacity)))

    pad = int(font.size * 0.5)
    widest = max(draw.textlength(line, font=font) for line in lines)
    box = [
        int(w / 2 - widest / 2) - pad,
        top - pad,
        int(w / 2 + widest / 2) + pad,
        top + block_h + pad // 2,
    ]
    draw.rounded_rectangle(box, radius=pad, fill=(8, 8, 14, int(alpha * 0.62)))

    for n, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        x, y = (w - tw) / 2, top + n * line_h
        draw.text((x, y), line, font=font, fill=(255, 255, 255, alpha),
                  stroke_width=max(2, int(font.size * 0.07)), stroke_fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(img)


def make_title_card(width: int, height: int, title: str, subtitle: str = "") -> np.ndarray:
    """Dark gradient card used for the opening beat."""
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    radial = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    base = np.clip(0.16 - radial * 0.18, 0, 1)
    frame = np.stack(
        [base * 90 + 10, base * 60 + 8, base * 150 + 18], axis=-1
    ).astype(np.float32)
    arr = np.clip(frame, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    font = load_font(max(22, int(height * 0.072)))
    small = load_font(max(13, int(height * 0.032)))
    tw = draw.textlength(title, font=font)
    draw.text(((width - tw) / 2, height * 0.42), title, font=font, fill=(236, 233, 255))
    if subtitle:
        sw = draw.textlength(subtitle, font=small)
        draw.text(((width - sw) / 2, height * 0.42 + font.size * 1.5), subtitle,
                  font=small, fill=(150, 140, 200))
    return np.asarray(img)
