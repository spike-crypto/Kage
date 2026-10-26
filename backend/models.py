"""SQLAlchemy ORM models."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    source_filename: Mapped[str | None] = mapped_column(String(400))
    source_file_path: Mapped[str | None] = mapped_column(Text)
    genre: Mapped[str] = mapped_column(String(40), default="action")  # legacy, kept for old rows
    style: Mapped[str] = mapped_column(String(40), default="kage_noir")
    target_duration: Mapped[int] = mapped_column(Integer, default=30)
    aspect_ratio: Mapped[str] = mapped_column(String(10), default="16:9")
    audio_mood: Mapped[str] = mapped_column(String(30), default="orchestral")
    intensity: Mapped[str] = mapped_column(String(20), default="balanced")
    subtitles_enabled: Mapped[int] = mapped_column(Integer, default=0)
    narration_enabled: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="created")
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    panel_count: Mapped[int] = mapped_column(Integer, default=0)
    planner_source: Mapped[str | None] = mapped_column(String(40))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    shots: Mapped[list["Shot"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="Shot.sequence_number",
    )
    render_jobs: Mapped[list["RenderJob"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="RenderJob.id"
    )
    pages: Mapped[list["Page"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="Page.page_index",
    )

    @property
    def latest_job(self) -> "RenderJob | None":
        return self.render_jobs[-1] if self.render_jobs else None


class Page(Base):
    """An imported manga page (or a panel cropped from one)."""

    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    page_index: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(10), default="page")  # page | panel
    image_path: Mapped[str] = mapped_column(Text)
    thumb_path: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    parent_page_index: Mapped[int | None] = mapped_column(Integer)
    ocr_text: Mapped[str | None] = mapped_column(Text)
    included: Mapped[int] = mapped_column(Integer, default=1)

    project: Mapped[Project] = relationship(back_populates="pages")


class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    sequence_number: Mapped[int] = mapped_column(Integer)
    panel_id: Mapped[int] = mapped_column(Integer, default=0)
    source_panel_path: Mapped[str] = mapped_column(Text)
    duration_seconds: Mapped[float] = mapped_column(Float, default=3.0)
    camera_motion: Mapped[str] = mapped_column(String(40), default="slow_zoom_in")
    transition_type: Mapped[str] = mapped_column(String(30), default="fade")
    visual_effects_json: Mapped[str] = mapped_column(Text, default="[]")
    audio_cues_json: Mapped[str] = mapped_column(Text, default="[]")
    music_intensity: Mapped[str] = mapped_column(String(20), default="medium")
    subtitle_text: Mapped[str | None] = mapped_column(Text)
    #: Written at analysis time and stored, so rendering never waits on a model.
    narration_text: Mapped[str | None] = mapped_column(Text)
    #: An animated clip of this panel, generated off-machine at analysis time.
    clip_path: Mapped[str | None] = mapped_column(Text)

    project: Mapped[Project] = relationship(back_populates="shots")

    @property
    def visual_effects(self) -> list[str]:
        try:
            value = json.loads(self.visual_effects_json or "[]")
            return [str(v) for v in value] if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []

    @property
    def audio_cues(self) -> list[str]:
        try:
            value = json.loads(self.audio_cues_json or "[]")
            return [str(v) for v in value] if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []


class RenderJob(Base):
    __tablename__ = "render_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(30), default="queued")
    stage: Mapped[str | None] = mapped_column(String(120))
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text)
    output_video_path: Mapped[str | None] = mapped_column(Text)
    subtitle_path: Mapped[str | None] = mapped_column(Text)
    diagnostics: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    project: Mapped[Project] = relationship(back_populates="render_jobs")


class ProviderStat(Base):
    """What each (provider, model) has actually done for us.

    The router reads this to decide what to try first. It is deliberately a running
    summary rather than a log: the question it answers is "is this endpoint fast and
    working *lately*", not "what happened in March".
    """

    __tablename__ = "provider_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    job: Mapped[str] = mapped_column(String(30), default="")
    ok_count: Mapped[int] = mapped_column(Integer, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    #: Exponentially weighted mean latency in seconds; recent calls dominate.
    latency_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    #: Backoff: a repeatedly failing endpoint is not retried until this passes.
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime)


class UsageEvent(Base):
    """One hosted request, so usage is visible in the app rather than guessed at."""

    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    job: Mapped[str] = mapped_column(String(30), default="")
    provider: Mapped[str] = mapped_column(String(40), default="")
    model: Mapped[str] = mapped_column(String(120), default="")
    ok: Mapped[int] = mapped_column(Integer, default=1)
    latency_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    images: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
