"""Pydantic request / response schemas, including the validated shot-plan schema."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Genre = Literal["action", "horror", "romance", "thriller", "emotional_drama"]  # legacy
Style = Literal["kage_noir", "shonen_kinetic", "shojo_bloom", "seinen_cold", "retro_cel"]
Intensity = Literal["subtle", "balanced", "dramatic"]
AspectRatio = Literal["16:9", "9:16", "1:1"]
AudioMood = Literal["orchestral", "ambient", "dark", "energetic", "suspenseful"]
MusicIntensity = Literal["low", "medium", "high"]

CAMERA_MOTIONS = (
    "slow_zoom_in",
    "slow_zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
    "ken_burns",
    "rapid_pan",
    "push_in",
    "static_drift",
)
TRANSITIONS = ("cut", "fade", "crossfade", "flash", "whip")
VISUAL_EFFECTS = (
    "vignette",
    "speed_lines",
    "screen_shake",
    "impact_flash",
    "motion_blur",
    "particles",
    "film_grain",
    "rain",
    "desaturate",
    "warm_glow",
    "cold_tint",
)
SOUND_CUES = (
    "whoosh",
    "impact",
    "footsteps",
    "wind",
    "rain",
    "ambient_drone",
    "suspense_riser",
    "explosion",
    "heartbeat",
    "sword_clash",
)

CameraMotion = Literal[CAMERA_MOTIONS]  # type: ignore[valid-type]
Transition = Literal[TRANSITIONS]  # type: ignore[valid-type]


# ------------------------------------------------------------------ shot plan


class PlannedShot(BaseModel):
    """One shot as produced by the director (local or hosted model)."""

    model_config = ConfigDict(extra="ignore")

    panel_id: int = Field(ge=0)
    duration_seconds: float = Field(gt=0.2, le=20.0)
    camera_motion: str = "slow_zoom_in"
    transition: str = "fade"
    visual_effects: list[str] = Field(default_factory=list)
    sound_cues: list[str] = Field(default_factory=list)
    music_intensity: MusicIntensity = "medium"
    subtitle: str = ""

    @field_validator("camera_motion")
    @classmethod
    def _camera(cls, v: str) -> str:
        v = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in CAMERA_MOTIONS else "slow_zoom_in"

    @field_validator("transition")
    @classmethod
    def _transition(cls, v: str) -> str:
        v = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in TRANSITIONS else "fade"

    @field_validator("visual_effects", mode="before")
    @classmethod
    def _effects(cls, v) -> list[str]:
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for item in v:
            key = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            if key in VISUAL_EFFECTS and key not in out:
                out.append(key)
        return out[:4]

    @field_validator("sound_cues", mode="before")
    @classmethod
    def _cues(cls, v) -> list[str]:
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for item in v:
            key = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            if key in SOUND_CUES and key not in out:
                out.append(key)
        return out[:3]

    @field_validator("subtitle", mode="before")
    @classmethod
    def _subtitle(cls, v) -> str:
        if v is None:
            return ""
        return str(v).strip()[:160]


class ShotPlan(BaseModel):
    shots: list[PlannedShot] = Field(min_length=1)
    source: str = "local_planner"
    notes: str = ""


# --------------------------------------------------------------- API payloads


class ProjectCreate(BaseModel):
    title: str = Field(default="Untitled Project", max_length=200)
    style: Style = "kage_noir"
    target_duration: Literal[15, 30, 60] = 30
    aspect_ratio: AspectRatio = "16:9"
    audio_mood: AudioMood | None = None   # None -> use the style's own mood
    intensity: Intensity = "balanced"
    subtitles_enabled: bool = False
    narration_enabled: bool = False

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = v.strip()
        return v or "Untitled Project"


class ProjectConfigUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    style: Style | None = None
    target_duration: Literal[15, 30, 60] | None = None
    aspect_ratio: AspectRatio | None = None
    audio_mood: AudioMood | None = None
    intensity: Intensity | None = None
    subtitles_enabled: bool | None = None
    narration_enabled: bool | None = None


class AnalyzeRequest(BaseModel):
    included_pages: list[int] | None = None
    detect_panels: bool = False
    use_hosted_director: bool = False
    extract_subtitles: bool = False
    use_vision: bool = True        # let the model choose which pages are worth filming
    write_narration: bool = False  # have a model watch the pages and write a voiceover
    animate_shots: int = 0         # how many hero shots to send for image-to-video


class ShotEdit(BaseModel):
    sequence_number: int
    duration_seconds: float | None = Field(default=None, gt=0.2, le=20.0)
    camera_motion: str | None = None
    transition_type: str | None = None
    visual_effects: list[str] | None = None
    audio_cues: list[str] | None = None
    subtitle_text: str | None = None
    narration_text: str | None = None


class ProviderKeyUpdate(BaseModel):
    """A user's own API key. Send an empty string to clear it."""

    api_key: str = ""


class ShotsUpdate(BaseModel):
    shots: list[ShotEdit]


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    page_index: int
    kind: str
    image_path: str
    thumb_path: str | None
    width: int
    height: int
    included: int
    ocr_text: str | None = None


class ShotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sequence_number: int
    panel_id: int
    source_panel_path: str
    duration_seconds: float
    camera_motion: str
    transition_type: str
    music_intensity: str
    subtitle_text: str | None
    narration_text: str | None = None
    clip_path: str | None = None
    visual_effects: list[str]
    audio_cues: list[str]


class RenderJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    stage: str | None
    progress_percent: float
    error_message: str | None
    output_video_path: str | None
    subtitle_path: str | None
    started_at: datetime | None
    completed_at: datetime | None


class ProjectSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    style: str
    genre: str = "action"
    target_duration: int
    aspect_ratio: str
    audio_mood: str
    intensity: str
    status: str
    page_count: int
    panel_count: int
    subtitles_enabled: int
    narration_enabled: int = 0
    planner_source: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectSummary):
    source_filename: str | None = None
    pages: list[PageOut] = Field(default_factory=list)
    shots: list[ShotOut] = Field(default_factory=list)
    render_job: RenderJobOut | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    capabilities: dict
    render_ready: bool
    notes: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    project_id: int
    page_count: int
    pages: list[PageOut]
    warnings: list[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    project_id: int
    project_status: str
    job: RenderJobOut | None = None
