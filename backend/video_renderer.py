"""Shot-plan -> MP4.

The timeline is computed up front (frame counts and transition overlaps), so audio is
built to exactly the video's length. Frames are streamed to FFmpeg over stdin as raw
RGB, which keeps peak memory to a handful of frames instead of a full-resolution
sequence. FFmpeg failures are captured with their stderr tail and reported as
diagnostics rather than silently swallowed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_engine import ShotAudio, build_soundtrack
from .config import ASPECT_PRESETS, FPS, SAMPLE_RATE, ffprobe_path, require_ffmpeg
from .styles import resolve_style
from .visual_engine import (
    ShotRenderer,
    blend,
    draw_subtitle,
    fade_to_black,
    flash_white,
)

DISSOLVE_TRANSITIONS = {"fade", "crossfade", "flash", "whip"}
OPENING_FADE_SECONDS = 0.5
CLOSING_FADE_SECONDS = 0.9
SUBTITLE_FADE_SECONDS = 0.28

ProgressCallback = Callable[[float, str], None]


log = logging.getLogger("kage.render")


@dataclass
class ShotSpec:
    sequence_number: int
    image_path: Path
    duration_seconds: float
    camera_motion: str = "slow_zoom_in"
    transition: str = "fade"
    visual_effects: list[str] = field(default_factory=list)
    sound_cues: list[str] = field(default_factory=list)
    music_intensity: str = "medium"
    subtitle: str = ""
    narration: str = ""   # spoken over this shot, written at analysis time
    #: An animated clip of this panel, generated off-machine at analysis time. When set,
    #: the shot plays these frames under the same camera move instead of a still.
    clip_path: Path | None = None


@dataclass
class RenderSpec:
    shots: list[ShotSpec]
    aspect_ratio: str = "16:9"
    audio_mood: str = "orchestral"
    intensity: str = "balanced"
    style: str = "kage_noir"
    fps: int = FPS
    title: str = "Kage"
    burn_subtitles: bool = False
    target_duration: float | None = None  # final length after transition overlaps
    output_path: Path = Path("output.mp4")
    audio_path: Path = Path("soundtrack.wav")
    subtitle_path: Path | None = None

    @property
    def dimensions(self) -> tuple[int, int]:
        return ASPECT_PRESETS.get(self.aspect_ratio, ASPECT_PRESETS["16:9"])


@dataclass
class RenderResult:
    output_path: Path
    duration_seconds: float
    frame_count: int
    width: int
    height: int
    has_audio: bool
    subtitle_path: Path | None
    diagnostics: dict


class RenderError(RuntimeError):
    """Raised when the render cannot produce a valid MP4."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# ------------------------------------------------------------------- timeline


@dataclass
class TimelineEntry:
    shot: ShotSpec
    frames: int
    overlap_in: int       # frames shared with the previous shot
    start_frame: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.frames


def _lay_out(spec: RenderSpec, counts: list[int]) -> tuple[list[TimelineEntry], int]:
    fps = spec.fps
    max_overlap = max(1, int(round(0.55 * fps)))

    entries: list[TimelineEntry] = []
    cursor = 0
    for i, shot in enumerate(spec.shots):
        if i == 0:
            overlap = 0
        elif shot.transition in DISSOLVE_TRANSITIONS:
            overlap = max(0, min(max_overlap, counts[i] // 3, counts[i - 1] // 3))
        else:
            overlap = 0
        start = 0 if i == 0 else cursor - overlap
        entries.append(TimelineEntry(shot, counts[i], overlap, start))
        cursor = start + counts[i]

    return entries, cursor


def build_timeline(spec: RenderSpec) -> tuple[list[TimelineEntry], int]:
    """Resolve frame counts and transition overlaps; returns (entries, total_frames).

    Dissolves consume frames from both neighbours, so a plan that sums to the requested
    runtime would land short. When ``target_duration`` is set the shot lengths are scaled
    up to absorb the overlaps, and the final shot takes the rounding remainder.
    """
    if not spec.shots:
        raise RenderError("The shot plan is empty - run analysis before rendering.")

    counts = [max(2, int(round(s.duration_seconds * spec.fps))) for s in spec.shots]
    entries, total = _lay_out(spec, counts)

    if spec.target_duration and total > 1:
        target_frames = max(2, int(round(spec.target_duration * spec.fps)))
        for _ in range(3):
            if abs(total - target_frames) <= 1:
                break
            scale = target_frames / float(total)
            counts = [max(2, int(round(c * scale))) for c in counts]
            entries, total = _lay_out(spec, counts)

        drift = target_frames - total
        if drift and counts[-1] + drift >= 2:
            counts[-1] += drift
            entries, total = _lay_out(spec, counts)

    # Keep the shots' reported durations consistent with the frames actually rendered.
    for entry in entries:
        entry.shot.duration_seconds = entry.frames / float(spec.fps)

    return entries, total


def build_subtitle_track(entries: list[TimelineEntry], fps: int,
                         total_frames: int) -> list[tuple[str, float]]:
    """Per-output-frame (text, opacity), so captions fade in and out cleanly."""
    track: list[tuple[str, float]] = [("", 0.0)] * total_frames
    fade_frames = max(1, int(SUBTITLE_FADE_SECONDS * fps))

    for entry in entries:
        text = (entry.shot.subtitle or "").strip()
        if not text:
            continue
        start, end = entry.start_frame, min(total_frames, entry.end_frame)
        span = end - start
        if span <= 0:
            continue
        local_fade = min(fade_frames, max(1, span // 3))
        for n in range(start, end):
            local = n - start
            if local < local_fade:
                opacity = (local + 1) / local_fade
            elif local >= span - local_fade:
                opacity = max(0.0, (span - local) / local_fade)
            else:
                opacity = 1.0
            track[n] = (text, round(opacity, 3))
    return track


def write_srt(entries: list[TimelineEntry], fps: int, path: Path) -> Path | None:
    """Write a sidecar .srt. Returns None when no shot carries text."""
    cues = [(e.start_frame / fps, e.end_frame / fps, e.shot.subtitle.strip())
            for e in entries if (e.shot.subtitle or "").strip()]
    if not cues:
        return None

    def stamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, (start, end, text) in enumerate(cues, start=1):
        lines.append(f"{n}\n{stamp(start)} --> {stamp(end)}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# -------------------------------------------------------------- frame streamer


def _overlap_blend(prev_frame: np.ndarray, new_frame: np.ndarray, transition: str,
                   progress: float) -> np.ndarray:
    """Combine two live frames according to the transition type."""
    if transition == "fade":
        # Dip through black: out to black, then up from black.
        if progress < 0.5:
            return fade_to_black(prev_frame, progress * 2.0)
        return fade_to_black(new_frame, (1.0 - progress) * 2.0)
    if transition == "flash":
        if progress < 0.5:
            return flash_white(prev_frame, progress * 2.0)
        return flash_white(new_frame, (1.0 - progress) * 2.0)
    if transition == "whip":
        mixed = blend(prev_frame, new_frame, progress)
        smear = max(3, int(21 * (1.0 - abs(progress - 0.5) * 2.0)))
        smear += (smear + 1) % 2
        kernel = np.zeros((smear, smear), dtype=np.float32)
        kernel[smear // 2, :] = 1.0 / smear
        import cv2

        return np.clip(cv2.filter2D(mixed.astype(np.float32), -1, kernel), 0, 255).astype(np.uint8)
    return blend(prev_frame, new_frame, progress)


def iter_output_frames(spec: RenderSpec, entries: list[TimelineEntry], total_frames: int,
                       progress_cb: ProgressCallback | None = None,
                       progress_span: tuple[float, float] = (10.0, 85.0)):
    """Yield every final frame, applying transitions, captions and global fades."""
    width, height = spec.dimensions
    subtitle_track = (
        build_subtitle_track(entries, spec.fps, total_frames)
        if spec.burn_subtitles
        else [("", 0.0)] * total_frames
    )

    fade_in_frames = max(1, int(OPENING_FADE_SECONDS * spec.fps))
    fade_out_frames = max(1, int(CLOSING_FADE_SECONDS * spec.fps))
    lo, hi = progress_span
    emitted = 0
    tail: list[np.ndarray] = []  # held-back frames of the previous shot

    def finish(frame: np.ndarray, index: int) -> np.ndarray:
        if index < fade_in_frames:
            frame = fade_to_black(frame, 1.0 - (index + 1) / fade_in_frames)
        remaining = total_frames - index
        if remaining <= fade_out_frames:
            frame = fade_to_black(frame, 1.0 - remaining / fade_out_frames)
        text, opacity = subtitle_track[index] if index < len(subtitle_track) else ("", 0.0)
        if text:
            frame = draw_subtitle(frame, text, opacity)
        return frame

    _preset = resolve_style(spec.style)
    for i, entry in enumerate(entries):
        renderer = ShotRenderer(
            image_path=Path(entry.shot.image_path),
            clip_path=entry.shot.clip_path,
            out_width=width,
            out_height=height,
            camera_motion=entry.shot.camera_motion,
            effects=list(entry.shot.visual_effects),
            intensity=spec.intensity,
            fps=spec.fps,
            seed=entry.shot.sequence_number * 17 + 3,
            grade=_preset.grade,
            framing=_preset.framing,
            parallax=_preset.parallax,
            drift=_preset.drift,
        )
        frames = renderer.iter_frames(entry.shot.duration_seconds)

        overlap = entry.overlap_in if i > 0 else 0
        next_overlap = entries[i + 1].overlap_in if i + 1 < len(entries) else 0

        buffered: list[np.ndarray] = []
        for local_index, frame in enumerate(frames):
            if local_index < overlap and tail:
                prev = tail[local_index] if local_index < len(tail) else tail[-1]
                progress = (local_index + 1) / (overlap + 1)
                frame = _overlap_blend(prev, frame, entry.shot.transition, progress)

            buffered.append(frame)

            # Hold back the tail only once we know the rest has been emitted.
            while len(buffered) > next_overlap:
                out = buffered.pop(0)
                if emitted >= total_frames:
                    break
                yield finish(out, emitted)
                emitted += 1
                if progress_cb and emitted % 12 == 0:
                    progress_cb(lo + (hi - lo) * emitted / max(1, total_frames),
                                f"Rendering shot {i + 1} of {len(entries)}")

        tail = buffered

    for out in tail:
        if emitted >= total_frames:
            break
        yield finish(out, emitted)
        emitted += 1

    # Pad with black if rounding left us a frame short of the planned length.
    while emitted < total_frames:
        yield np.zeros((height, width, 3), dtype=np.uint8)
        emitted += 1


# ---------------------------------------------------------------- ffmpeg stage


def _ffprobe_streams(path: Path) -> dict:
    probe = ffprobe_path()
    if not probe:
        return {}
    try:
        result = subprocess.run(
            [probe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
             str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(result.stdout) if result.stdout else {}
    except Exception:
        return {}


def render(spec: RenderSpec, progress_cb: ProgressCallback | None = None) -> RenderResult:
    """Render the full sequence to an MP4 with a valid video and audio stream."""
    ffmpeg = require_ffmpeg()
    width, height = spec.dimensions

    def report(pct: float, stage: str) -> None:
        if progress_cb:
            progress_cb(min(99.0, max(0.0, pct)), stage)

    report(2.0, "Building timeline")
    entries, total_frames = build_timeline(spec)
    if total_frames < 2:
        raise RenderError("The planned sequence is too short to render.")
    duration = total_frames / float(spec.fps)

    for entry in entries:
        if not Path(entry.shot.image_path).exists():
            raise RenderError(f"Source artwork is missing: {entry.shot.image_path}")

    # --- audio first, so its exact length is known before encoding
    report(5.0, "Composing soundtrack")
    shot_audio = [
        ShotAudio(
            start_seconds=e.start_frame / spec.fps,
            duration_seconds=e.frames / spec.fps,
            cues=list(e.shot.sound_cues),
            music_intensity=e.shot.music_intensity,
        )
        for e in entries
    ]
    spec.audio_path.parent.mkdir(parents=True, exist_ok=True)

    # The script was written at analysis time; only the speaking happens here, and it is
    # a 145 MB model on the CPU, so the GPU stays free for the frames.
    narration_track = None
    if any((e.shot.narration or "").strip() for e in entries):
        try:
            from .narration import Line, release, speak

            lines = [
                Line(sequence_number=n,
                     text=(e.shot.narration or "").strip(),
                     start_seconds=e.start_frame / spec.fps,
                     duration_seconds=e.frames / spec.fps,
                     word_budget=0)
                for n, e in enumerate(entries, start=1)
                if (e.shot.narration or "").strip()
            ]
            narration_track, voice_report = speak(lines, SAMPLE_RATE)
            log.info("Voiceover %s", voice_report.describe())
            release()
        except Exception as exc:
            # A missing voice must not cost the user their film.
            log.warning("Voiceover unavailable, rendering without it: %s", exc)
            narration_track = None

    try:
        build_soundtrack(
            shot_audio,
            spec.audio_mood,
            duration,
            spec.audio_path,
            sr=SAMPLE_RATE,
            narration=narration_track,
            seed=abs(hash(spec.title)) % 9973 or 7,
        )
    except Exception as exc:
        raise RenderError(f"Soundtrack generation failed: {exc}") from exc

    if not spec.audio_path.exists() or spec.audio_path.stat().st_size < 1024:
        raise RenderError("Soundtrack generation produced no audio data.")

    subtitle_file = None
    if spec.subtitle_path is not None:
        subtitle_file = write_srt(entries, spec.fps, Path(spec.subtitle_path))

    # --- encode
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = spec.output_path.with_suffix(".partial.mp4")
    if temp_output.exists():
        temp_output.unlink()

    video_codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-profile:v", "high"]
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(spec.fps), "-i", "-",
        "-i", str(spec.audio_path),
        *video_codec,
        "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE), "-ac", "2",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", "-movflags", "+faststart",
        str(temp_output),
    ]

    report(9.0, "Starting encoder")
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    written = 0
    broken_pipe = False
    try:
        assert process.stdin is not None
        for frame in iter_output_frames(spec, entries, total_frames, progress_cb):
            try:
                process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                broken_pipe = True
                break
            written += 1
    finally:
        # communicate() flushes and closes stdin itself, which is also how FFmpeg is
        # told the frames have ended. Closing it here first made that flush raise
        # ValueError on POSIX, where it is guarded against a broken pipe but not an
        # already-closed one. Windows takes a threaded path and never hit it, so this
        # only surfaced once the suite ran on Linux.
        stdout, stderr = process.communicate(timeout=600)

    stderr_text = (stderr or b"").decode("utf-8", "replace").strip()
    diagnostics = {
        "ffmpeg_returncode": process.returncode,
        "frames_written": written,
        "frames_planned": total_frames,
        "ffmpeg_stderr_tail": stderr_text[-1800:],
        "command": " ".join(command[:14]) + " ... " + str(temp_output),
    }
    del stdout

    if process.returncode != 0 or broken_pipe or not temp_output.exists():
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)
        raise RenderError(
            f"FFmpeg failed (exit {process.returncode}). {stderr_text[-400:] or 'No stderr output.'}",
            diagnostics,
        )

    report(90.0, "Verifying output")
    probe = _ffprobe_streams(temp_output)
    streams = probe.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    probed_duration = float(probe.get("format", {}).get("duration") or 0.0)
    diagnostics.update(
        {
            "probe_duration_seconds": round(probed_duration, 3),
            "probe_has_video": has_video,
            "probe_has_audio": has_audio,
            "probe_streams": [
                {
                    "type": s.get("codec_type"),
                    "codec": s.get("codec_name"),
                    "width": s.get("width"),
                    "height": s.get("height"),
                    "channels": s.get("channels"),
                    "sample_rate": s.get("sample_rate"),
                }
                for s in streams
            ],
        }
    )

    if probe and not has_video:
        temp_output.unlink(missing_ok=True)
        raise RenderError("The encoded file has no video stream.", diagnostics)
    if temp_output.stat().st_size < 4096:
        temp_output.unlink(missing_ok=True)
        raise RenderError("The encoded file is suspiciously small; treating as a failure.",
                          diagnostics)

    spec.output_path.unlink(missing_ok=True)
    shutil.move(str(temp_output), str(spec.output_path))
    report(97.0, "Finalising")

    return RenderResult(
        output_path=spec.output_path,
        duration_seconds=probed_duration or duration,
        frame_count=written,
        width=width,
        height=height,
        has_audio=has_audio if probe else True,
        subtitle_path=subtitle_file,
        diagnostics=diagnostics,
    )


def cleanup_temp(directory: Path, keep: tuple[str, ...] = ()) -> None:
    """Remove intermediate files from a project's temp folder, never source art."""
    directory = Path(directory)
    if not directory.exists() or directory.name != "temp":
        return
    for item in directory.iterdir():
        if item.name in keep:
            continue
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except OSError:
            continue
