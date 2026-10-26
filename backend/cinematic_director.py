"""Turns panels + user preferences into a validated shot plan.

Two paths:
  * ``plan_locally`` - deterministic, always available, style-aware.
  * ``plan_with_hosted_model`` - optional Hugging Face router call; its JSON is validated
    against the Pydantic schema and never executed. Any failure falls back to the local
    planner so the user can always render.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass

from .schemas import PlannedShot, ShotPlan
from .styles import (
    INTENSITY_ACCENT_SHIFT,
    INTENSITY_FACTORS,
    StylePreset,
    resolve_style,
)

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.environ.get("HF_DIRECTOR_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_TIMEOUT = float(os.environ.get("HF_TIMEOUT_SECONDS", "25"))


@dataclass
class PanelRef:
    """A renderable source image handed to the director."""

    panel_id: int
    path: str
    width: int = 0
    height: int = 0
    text: str = ""
    subject: str = ""        # what the vision model saw: face, duo, crowd, scenery...
    energy: float = 0.5      # 0 = still, 1 = action
    vision_score: float = 0.0

    @property
    def is_wide(self) -> bool:
        return self.height > 0 and (self.width / self.height) >= 1.35

    @property
    def is_tall(self) -> bool:
        return self.width > 0 and (self.height / self.width) >= 1.35


# --------------------------------------------------------------- local planner


def plan_locally(
    panels: list[PanelRef],
    style: str,
    target_duration: int,
    intensity: str = "balanced",
    include_subtitles: bool = False,
) -> ShotPlan:
    """Deterministic shot plan in one style's grammar, totalling ``target_duration``.

    Restraint is enforced here, not left to taste: the style's grade carries the look,
    and named effects are accents spent on roughly one shot in ``accent_every``. No shot
    ever exceeds ``preset.max_effects``.
    """
    if not panels:
        raise ValueError("At least one panel is required to plan a sequence.")

    preset = resolve_style(style)
    factor = INTENSITY_FACTORS.get(intensity, 1.0)
    low, high = preset.shot_length[0] * factor, preset.shot_length[1] * factor

    # Reuse panels cyclically when there are few of them, with a different move each pass.
    avg = (low + high) / 2.0
    shot_count = max(2, min(round(target_duration / avg), max(2, len(panels) * 3), 24))

    rng = random.Random(f"{preset.key}|{intensity}|{target_duration}|{len(panels)}")
    accent_every = max(2, preset.accent_every + INTENSITY_ACCENT_SHIFT.get(intensity, 0))
    shots: list[PlannedShot] = []

    for i in range(shot_count):
        panel = panels[i % len(panels)]
        pass_number = i // len(panels)

        camera = _pick_camera(preset.camera, panel, i, pass_number, rng,
                              previous=shots[-1].camera_motion if shots else None)
        transition = "cut" if i == 0 else preset.transitions[i % len(preset.transitions)]
        effects = _accent_effects(preset, i, accent_every, camera)
        cues = _cues_for(preset, i, shot_count)
        music = preset.music_curve[
            min(i * len(preset.music_curve) // shot_count, len(preset.music_curve) - 1)
        ]

        # Opening and closing shots breathe; the middle is cut tighter. A high-energy
        # panel is held for less time, so action does not sit still on screen.
        weight = 1.18 if i in (0, shot_count - 1) else 1.0
        weight *= 1.0 - 0.22 * (panel.energy - 0.5)
        duration = round(min(high, max(low, avg * weight)), 2)

        shots.append(
            PlannedShot(
                panel_id=panel.panel_id,
                duration_seconds=duration,
                camera_motion=camera,
                transition=transition,
                visual_effects=effects,
                sound_cues=cues,
                music_intensity=music,
                subtitle=_subtitle_for(panel, include_subtitles),
            )
        )

    shots = fit_plan_to_duration(shots, target_duration)
    spent = sum(1 for s in shots if s.visual_effects)
    return ShotPlan(
        shots=shots,
        source="local_planner",
        notes=(f"{preset.name}: {len(shots)} shots, intensity {intensity}, "
               f"accents on {spent}/{len(shots)} shots."),
    )


def _accent_effects(preset: StylePreset, index: int, accent_every: int,
                    camera: str) -> list[str]:
    """Spend an accent only when the style's rhythm calls for one."""
    effects: list[str] = []
    if preset.accent_effects:
        if index == 0 and preset.accent_on_first:
            effects.append(preset.accent_effects[0])
        elif index > 0 and index % accent_every == 0:
            slot = (index // accent_every - 1) % len(preset.accent_effects)
            effects.append(preset.accent_effects[slot])

    # A fast pan needs blur to read as motion rather than a jump.
    if camera == "rapid_pan" and len(effects) < preset.max_effects:
        effects.append("motion_blur")

    return effects[: preset.max_effects]


def _cues_for(preset: StylePreset, index: int, shot_count: int) -> list[str]:
    """Sound cues are sparse too; the bed anchors the open and the close."""
    cues = list(preset.cues[index % len(preset.cues)])
    if index in (0, shot_count - 1) and preset.bed not in cues:
        cues.append(preset.bed)
    return cues[:2]


def _pick_camera(options: tuple[str, ...], panel: PanelRef, index: int, pass_number: int,
                 rng: random.Random, previous: str | None = None) -> str:
    """Match the move to the panel shape, vary on repeat visits, never repeat back to back.

    Pages from one source often share a shape, so shape-matching alone would pick the same
    move for every shot. Cutting from a move to itself reads as a glitch, not a cut.
    """
    opts = list(options)

    # When the vision model identified a subject, let that drive the move: a face wants a
    # push in, a crowd wants a drift across it. Shape is only a fallback.
    if panel.subject:
        from .vision import SUBJECT_CAMERA_HINT

        hinted = [c for c in opts if c in SUBJECT_CAMERA_HINT.get(panel.subject, ())]
        if previous:
            hinted = [c for c in hinted if c != previous]
        if hinted:
            return hinted[(index + pass_number) % len(hinted)]
        # Nothing left in the hint that is not a repeat. Pages from one book share a
        # subject, so honouring the hint here is what made every other shot the same
        # slow zoom; shape wins instead, and the hint waits for the next shot.

    if panel.is_wide:
        preferred = [c for c in opts if c in ("pan_left", "pan_right", "ken_burns", "rapid_pan")]
    elif panel.is_tall:
        preferred = [c for c in opts if c in ("pan_up", "pan_down", "slow_zoom_in", "push_in")]
    else:
        preferred = [c for c in opts
                     if "zoom" in c or c in ("push_in", "ken_burns", "static_drift")]
    pool = preferred or opts
    if previous:
        fresh = [c for c in pool if c != previous] or [c for c in opts if c != previous]
        if fresh:
            pool = fresh
    return pool[(index + pass_number * 2 + rng.randrange(0, 2)) % len(pool)]


def _subtitle_for(panel: PanelRef, include_subtitles: bool) -> str:
    """Only ever surface text that OCR actually read from the page - never invented."""
    if not include_subtitles or not panel.text:
        return ""
    text = " ".join(panel.text.split())
    return text[:120]


def fit_plan_to_duration(shots: list[PlannedShot], target_duration: int) -> list[PlannedShot]:
    """Scale shot durations so the sequence totals ``target_duration`` seconds."""
    total = sum(s.duration_seconds for s in shots)
    if total <= 0:
        return shots
    scale = target_duration / total
    for shot in shots:
        shot.duration_seconds = round(max(0.6, min(20.0, shot.duration_seconds * scale)), 2)

    # Absorb rounding drift into the last shot.
    drift = round(target_duration - sum(s.duration_seconds for s in shots), 2)
    if abs(drift) >= 0.01:
        last = shots[-1]
        last.duration_seconds = round(max(0.6, min(20.0, last.duration_seconds + drift)), 2)
    return shots


# -------------------------------------------------------------- hosted planner

SYSTEM_PROMPT = (
    "You are a cinematic director adapting manga panels into a short motion comic. "
    "Reply with JSON only - no prose, no markdown fences."
)


def _build_user_prompt(panels: list[PanelRef], style: str, target_duration: int,
                       intensity: str, audio_mood: str, include_subtitles: bool) -> str:
    from .schemas import CAMERA_MOTIONS, SOUND_CUES, TRANSITIONS, VISUAL_EFFECTS

    preset = resolve_style(style)
    panel_lines = []
    for p in panels:
        shape = "wide" if p.is_wide else "tall" if p.is_tall else "square-ish"
        line = f'- panel_id {p.panel_id}: {p.width}x{p.height} ({shape})'
        if include_subtitles and p.text:
            line += f'; text read from page: "{p.text[:90]}"'
        panel_lines.append(line)

    return (
        f"Panels available (use these panel_id values only):\n" + "\n".join(panel_lines) + "\n\n"
        f"Visual style: {preset.name} - {preset.tagline}\n{preset.description}\n"
        f"Intensity: {intensity}\nAudio mood: {audio_mood}\n"
        f"Hold each shot {preset.shot_length[0]:.1f}-{preset.shot_length[1]:.1f} seconds.\n"
        f"Camera moves to prefer: {', '.join(preset.camera)}\n"
        f"Transitions to prefer: {', '.join(preset.transitions)}\n"
        f"EFFECT BUDGET: at most {preset.max_effects} effect(s) on any shot, and most "
        "shots must have none at all. The style's colour grade already carries the look; "
        "effects are rare accents for a single beat, never decoration.\n"
        f"Total runtime must be about {target_duration} seconds.\n\n"
        "Return this exact JSON shape:\n"
        '{"shots": [{"panel_id": 0, "duration_seconds": 3.0, "camera_motion": "...", '
        '"transition": "...", "visual_effects": ["..."], "sound_cues": ["..."], '
        '"music_intensity": "low|medium|high", "subtitle": ""}]}\n\n'
        f"camera_motion must be one of: {', '.join(CAMERA_MOTIONS)}\n"
        f"transition must be one of: {', '.join(TRANSITIONS)}\n"
        f"visual_effects items must be from: {', '.join(VISUAL_EFFECTS)}\n"
        f"sound_cues items must be from: {', '.join(SOUND_CUES)}\n"
        + (
            "Only put text in 'subtitle' if it was quoted above as read from the page; "
            "never invent dialogue.\n"
            if include_subtitles
            else "Leave every 'subtitle' as an empty string.\n"
        )
        + "Use 4 to 14 shots. Vary camera moves between consecutive shots."
    )


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerating code fences."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object in model response.")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unterminated JSON object in model response.")


def plan_with_hosted_model(
    panels: list[PanelRef],
    style: str,
    target_duration: int,
    intensity: str = "balanced",
    audio_mood: str = "orchestral",
    include_subtitles: bool = False,
) -> ShotPlan:
    """Ask a hosted LLM for a plan. Raises on any problem; callers fall back locally."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("No HF_TOKEN configured for the hosted director.")

    import requests

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    panels, style, target_duration, intensity, audio_mood, include_subtitles
                ),
            },
        ],
        "temperature": 0.7,
        "max_tokens": 1800,
    }
    response = requests.post(
        HF_ENDPOINT,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=HF_TIMEOUT,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    data = _extract_json(content)
    raw_shots = data.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError("Model response contained no shots.")

    valid_ids = {p.panel_id for p in panels}
    by_id = {p.panel_id: p for p in panels}
    shots: list[PlannedShot] = []
    for item in raw_shots[:24]:
        if not isinstance(item, dict):
            continue
        try:
            shot = PlannedShot.model_validate(item)
        except Exception:
            continue
        if shot.panel_id not in valid_ids:
            continue
        # Never let the model put words on screen that OCR did not read.
        if shot.subtitle:
            source_text = by_id[shot.panel_id].text if include_subtitles else ""
            if not source_text or shot.subtitle.lower() not in source_text.lower():
                shot.subtitle = _subtitle_for(by_id[shot.panel_id], include_subtitles)
        shots.append(shot)

    if len(shots) < 2:
        raise ValueError("Model response had fewer than two usable shots.")

    shots = fit_plan_to_duration(shots, target_duration)
    return ShotPlan(
        shots=shots,
        source=f"hosted:{HF_MODEL}",
        notes=f"Hosted director plan validated into {len(shots)} shots.",
    )


def build_plan(
    panels: list[PanelRef],
    style: str,
    target_duration: int,
    intensity: str = "balanced",
    audio_mood: str = "orchestral",
    include_subtitles: bool = False,
    use_hosted: bool = False,
) -> tuple[ShotPlan, str | None]:
    """Return (plan, warning). Falls back to the local planner on any hosted failure."""
    if use_hosted:
        try:
            return (
                plan_with_hosted_model(
                    panels, style, target_duration, intensity, audio_mood, include_subtitles
                ),
                None,
            )
        except Exception as exc:
            warning = f"Hosted director unavailable ({type(exc).__name__}: {exc}). Used the local planner."
            return (
                plan_locally(panels, style, target_duration, intensity, include_subtitles),
                warning[:400],
            )
    return plan_locally(panels, style, target_duration, intensity, include_subtitles), None
