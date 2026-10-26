"""FastAPI application: projects, upload, analysis, rendering, status, video delivery."""

from __future__ import annotations

import json
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import __version__
from .cinematic_director import PanelRef, build_plan
from .config import (
    MAX_PAGES,
    MAX_UPLOAD_BYTES,
    PROJECTS_DIR,
    probe_capabilities,
    project_dir,
)
from .database import get_session, init_db, session_scope
from .logging_config import ERROR_LOG, MAIN_LOG, RING, get_logger, read_log_file, setup_logging
from .manga_processor import (
    MangaImportError,
    classify_upload,
    detect_panels,
    extract_pdf_pages,
    extract_text,
    import_image_file,
    sanitize_filename,
)
from .models import Page, Project, RenderJob, Shot, UsageEvent, utcnow
from .styles import resolve_style
from .schemas import (
    AnalyzeRequest,
    HealthResponse,
    PageOut,
    ProjectConfigUpdate,
    ProjectCreate,
    ProjectDetail,
    ProjectSummary,
    RenderJobOut,
    ShotOut,
    ShotsUpdate,
    StatusResponse,
    UploadResponse,
    ProviderKeyUpdate,
)
from .video_renderer import RenderError, RenderSpec, ShotSpec, cleanup_temp, render

log = get_logger("api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    init_db()
    caps = probe_capabilities()
    log.info(
        "Backend starting - ffmpeg=%s pdf=%s opencv=%s ocr=%s hosted_llm=%s",
        caps.ffmpeg, caps.pymupdf, caps.opencv, caps.ocr, caps.hosted_llm,
    )
    for note in ("FFmpeg missing" if not caps.ffmpeg else None,
                 "PyMuPDF missing" if not caps.pymupdf else None):
        if note:
            log.warning("Capability gap: %s", note)
    yield
    log.info("Backend shutting down")


app = FastAPI(
    title="Kage",
    version=__version__,
    description="From Ink to Imagination - manga pages into cinematic anime sequences.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only one render runs at a time: encoding is CPU-bound and this is a local studio.
_RENDER_LOCK = threading.Lock()
_ACTIVE_RENDERS: set[int] = set()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the full traceback, return a readable message - never a bare 500."""
    log.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal error: {type(exc).__name__}: {exc}. "
                      "See data/logs/errors.log or GET /api/logs for details."
        },
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Request failed: %s %s", request.method, request.url.path)
        raise
    if response.status_code >= 400:
        log.warning("%s %s -> %s", request.method, request.url.path, response.status_code)
    return response


# ------------------------------------------------------------------- helpers


def _get_project(session: Session, project_id: int) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} does not exist.")
    return project


def _safe_project_path(project_id: int, raw_path: str | None) -> Path | None:
    """Only ever serve files that live inside the project's own directory."""
    if not raw_path:
        return None
    candidate = Path(raw_path).resolve()
    root = project_dir(project_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.exists() else None


def _shot_out(shot: Shot) -> ShotOut:
    return ShotOut(
        id=shot.id,
        sequence_number=shot.sequence_number,
        panel_id=shot.panel_id,
        source_panel_path=shot.source_panel_path,
        duration_seconds=shot.duration_seconds,
        camera_motion=shot.camera_motion,
        transition_type=shot.transition_type,
        music_intensity=shot.music_intensity,
        subtitle_text=shot.subtitle_text,
        narration_text=shot.narration_text,
        visual_effects=shot.visual_effects,
        audio_cues=shot.audio_cues,
    )


def _project_detail(project: Project) -> ProjectDetail:
    job = project.render_jobs[-1] if project.render_jobs else None
    return ProjectDetail(
        **{
            **ProjectSummary.model_validate(project).model_dump(),
            "source_filename": project.source_filename,
            "pages": [PageOut.model_validate(p) for p in project.pages],
            "shots": [_shot_out(s) for s in project.shots],
            "render_job": RenderJobOut.model_validate(job) if job else None,
        }
    )


# -------------------------------------------------------------------- health


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    caps = probe_capabilities()
    notes: list[str] = []
    if not caps.ffmpeg:
        notes.append("FFmpeg not found - rendering is disabled until it is installed.")
    if not caps.pymupdf:
        notes.append("PyMuPDF missing - PDF import unavailable, images still work.")
    if not caps.ocr:
        notes.append("Tesseract OCR not detected - subtitle extraction is skipped.")
    if not caps.vision:
        notes.append(
            "Vision model not installed - pages are used in order. "
            "Install with: uv pip install torch transformers"
        )
    from .narration import capability as narration_capability

    voiceover = narration_capability()
    if not voiceover["ready"]:
        missing = ", ".join(p["env"] for p in voiceover["needs_key"])
        notes.append(
            "Voiceover needs a free inference key - set one of: " + missing
            + ". Narration is skipped until then."
        )

    capabilities = caps.as_dict()
    capabilities["narration"] = voiceover

    return HealthResponse(
        status="ok" if caps.ffmpeg else "degraded",
        version=__version__,
        capabilities=capabilities,
        render_ready=caps.ffmpeg and caps.opencv,
        notes=notes,
    )


# ------------------------------------------------------------------ projects


@app.post("/api/projects", response_model=ProjectDetail, status_code=201)
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)) -> ProjectDetail:
    preset = resolve_style(payload.style)
    project = Project(
        title=payload.title,
        style=preset.key,
        genre=payload.style,
        target_duration=payload.target_duration,
        aspect_ratio=payload.aspect_ratio,
        audio_mood=payload.audio_mood or preset.audio_mood,
        intensity=payload.intensity,
        subtitles_enabled=int(payload.subtitles_enabled),
        narration_enabled=int(payload.narration_enabled),
        status="created",
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    project_dir(project.id)
    return _project_detail(project)


@app.get("/api/projects", response_model=list[ProjectSummary])
def list_projects(limit: int = 25, session: Session = Depends(get_session)) -> list[ProjectSummary]:
    limit = max(1, min(100, limit))
    rows = session.scalars(
        select(Project).order_by(Project.created_at.desc(), Project.id.desc()).limit(limit)
    ).all()
    return [ProjectSummary.model_validate(p) for p in rows]


@app.get("/api/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: int, session: Session = Depends(get_session)) -> ProjectDetail:
    return _project_detail(_get_project(session, project_id))


@app.patch("/api/projects/{project_id}", response_model=ProjectDetail)
def update_project(project_id: int, payload: ProjectConfigUpdate,
                   session: Session = Depends(get_session)) -> ProjectDetail:
    project = _get_project(session, project_id)
    data = payload.model_dump(exclude_none=True)
    if "subtitles_enabled" in data:
        data["subtitles_enabled"] = int(data["subtitles_enabled"])
    if "narration_enabled" in data:
        data["narration_enabled"] = int(data["narration_enabled"])
    for key, value in data.items():
        setattr(project, key, value)
    project.updated_at = utcnow()
    session.commit()
    session.refresh(project)
    return _project_detail(project)


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: int, session: Session = Depends(get_session)) -> None:
    project = _get_project(session, project_id)
    if project.id in _ACTIVE_RENDERS:
        raise HTTPException(status_code=409, detail="That project is currently rendering.")
    session.delete(project)
    session.commit()


# -------------------------------------------------------------------- upload


@app.post("/api/projects/{project_id}/upload", response_model=UploadResponse)
async def upload_pages(
    project_id: int,
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
) -> UploadResponse:
    project = _get_project(session, project_id)
    if not files:
        raise HTTPException(status_code=400, detail="No files were received.")

    root = project_dir(project.id)
    warnings: list[str] = []
    imported: list[Page] = []

    # Clear any previous import so re-uploading replaces rather than appends.
    for existing in list(project.pages):
        session.delete(existing)
    for shot in list(project.shots):
        session.delete(shot)
    session.flush()

    next_index = 0
    total_bytes = 0

    for upload in files:
        safe_name = sanitize_filename(upload.filename or "upload")
        try:
            kind = classify_upload(safe_name)
        except MangaImportError as exc:
            warnings.append(str(exc))
            continue

        content = await upload.read()
        total_bytes += len(content)
        if len(content) == 0:
            warnings.append(f"'{safe_name}' is empty and was skipped.")
            continue
        if total_bytes > MAX_UPLOAD_BYTES:
            warnings.append(
                f"Upload limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB reached; "
                f"'{safe_name}' and anything after it was skipped."
            )
            break
        if next_index >= MAX_PAGES:
            warnings.append(f"Page limit of {MAX_PAGES} reached; remaining files were skipped.")
            break

        source_path = root / "source" / safe_name
        source_path.write_bytes(content)
        if project.source_filename is None:
            project.source_filename = safe_name
            project.source_file_path = str(source_path)

        try:
            if kind == "pdf":
                pages = extract_pdf_pages(source_path, project.id, start_index=next_index)
            else:
                pages = import_image_file(source_path, project.id, next_index)
        except MangaImportError as exc:
            warnings.append(f"{safe_name}: {exc}")
            continue
        except Exception as exc:  # unexpected decode failure
            warnings.append(f"{safe_name}: could not be processed ({type(exc).__name__}).")
            continue

        for page in pages:
            row = Page(
                project_id=project.id,
                page_index=page.page_index,
                kind="page",
                image_path=str(page.image_path),
                thumb_path=str(page.thumb_path),
                width=page.width,
                height=page.height,
                included=1,
            )
            session.add(row)
            imported.append(row)
            next_index = page.page_index + 1

    for warning in warnings:
        log.warning("Upload to project %s: %s", project.id, warning)

    if not imported:
        log.error("Upload to project %s imported nothing", project.id)
        project.status = "import_failed"
        project.error_message = "; ".join(warnings)[:800] or "No pages could be imported."
        session.commit()
        raise HTTPException(
            status_code=400,
            detail=project.error_message,
        )

    project.page_count = len(imported)
    project.panel_count = 0
    project.status = "imported"
    project.error_message = None
    project.updated_at = utcnow()
    session.commit()
    for row in imported:
        session.refresh(row)

    return UploadResponse(
        project_id=project.id,
        page_count=len(imported),
        pages=[PageOut.model_validate(p) for p in imported],
        warnings=warnings,
    )


# ------------------------------------------------------------------- analyse


@app.post("/api/projects/{project_id}/analyze", response_model=ProjectDetail)
def analyze_project(project_id: int, payload: AnalyzeRequest,
                    session: Session = Depends(get_session)) -> ProjectDetail:
    project = _get_project(session, project_id)
    if not project.pages:
        raise HTTPException(status_code=400, detail="Upload manga pages before analysing.")

    selected = payload.included_pages
    pages = [p for p in project.pages if p.kind == "page"]
    if selected is not None:
        wanted = set(selected)
        for page in pages:
            page.included = int(page.page_index in wanted)
        chosen = [p for p in pages if p.included]
        if not chosen:
            raise HTTPException(status_code=400, detail="Select at least one page to include.")
    else:
        for page in pages:
            page.included = 1
        chosen = pages

    warnings: list[str] = []

    # Optional panel segmentation. Pages that do not segment confidently stay whole.
    for existing in [p for p in project.pages if p.kind == "panel"]:
        session.delete(existing)
    session.flush()

    renderables: list[Page] = []
    if payload.detect_panels:
        segmented_any = False
        for page in chosen:
            try:
                panels = detect_panels(Path(page.image_path), project.id, page.page_index)
            except Exception:
                panels = []
            if panels:
                segmented_any = True
                for panel in panels:
                    row = Page(
                        project_id=project.id,
                        page_index=1000 + page.page_index * 10 + panel.page_index,
                        kind="panel",
                        image_path=str(panel.image_path),
                        thumb_path=str(panel.thumb_path),
                        width=panel.width,
                        height=panel.height,
                        parent_page_index=page.page_index,
                        included=1,
                    )
                    session.add(row)
                    renderables.append(row)
            else:
                renderables.append(page)
        if segmented_any:
            # Never mix granularity. A whole page holds four to six panels, gutters and
            # speech bubbles at once and reads as a contact sheet; cutting between that
            # and a single panel makes the film keep changing scale. Once any page has
            # segmented, only panels are filmed.
            panels_only = [r for r in renderables if r.kind == "panel"]
            dropped = len(renderables) - len(panels_only)
            # Only when there are enough panels to carry the film. Segmentation that
            # succeeded on two pages out of nineteen must not throw the other seventeen
            # away - that films the same two panels on a loop, which is worse than mixed
            # granularity. Needs roughly a panel per shot, plus slack to choose from.
            wanted_shots = max(4, int(project.target_duration // 4))
            if len(panels_only) >= max(6, wanted_shots):
                renderables = panels_only
                if dropped:
                    warnings.append(
                        f"Filming {len(panels_only)} panels; {dropped} page(s) that did "
                        "not segment were left out so every shot is one panel."
                    )
        else:
            warnings.append(
                "Panel detection did not find a confident layout on any page; "
                "using whole pages instead."
            )
    else:
        renderables = list(chosen)

    session.flush()

    # Optional OCR for subtitles.
    include_subtitles = bool(payload.extract_subtitles and project.subtitles_enabled)
    if include_subtitles:
        found_any = False
        for item in renderables:
            text = extract_text(Path(item.image_path))
            item.ocr_text = text or None
            found_any = found_any or bool(text)
        if not found_any:
            warnings.append("No readable text was extracted, so shots have no subtitles.")

    # --- the one place a model actually looks at the artwork ---------------
    insight_by_index: dict[int, object] = {}
    if payload.use_vision and renderables:
        try:
            from .vision import analyse_pages, select_pages

            insights = analyse_pages(
                [Path(item.image_path) for item in renderables],
                [item.page_index for item in renderables],
            )
            insight_by_index = {i.page_index: i for i in insights}

            skipped = [i for i in insights if i.kind != "artwork"]
            # Roughly how many distinct pages a sequence of this length can use well.
            wanted = max(3, min(len(insights), project.target_duration // 3))
            keep = {i.page_index for i in select_pages(insights, wanted)}

            filtered = [r for r in renderables if r.page_index in keep]
            if len(filtered) >= 2:
                dropped = len(renderables) - len(filtered)
                renderables = filtered
                detail = ", ".join(
                    f"p{i.page_index + 1}:{i.kind}" for i in skipped[:6]
                )
                if dropped:
                    warnings.append(
                        f"Vision model kept {len(filtered)} of {len(filtered) + dropped} "
                        f"pages, skipping covers and text pages"
                        + (f" ({detail})" if detail else "") + "."
                    )
                log.info(
                    "Project %s vision: kept %d/%d pages (%s)",
                    project.id, len(filtered), len(filtered) + dropped,
                    "; ".join(i.describe() for i in insights[:4]),
                )
        except Exception as exc:
            warnings.append(
                f"Vision model unavailable ({type(exc).__name__}); used pages in order."
            )
            log.warning("Project %s vision unavailable: %s", project.id, exc)

    panel_refs = [
        PanelRef(
            panel_id=item.page_index,
            path=item.image_path,
            width=item.width,
            height=item.height,
            text=item.ocr_text or "",
            subject=getattr(insight_by_index.get(item.page_index), "subject", ""),
            energy=getattr(insight_by_index.get(item.page_index), "energy", 0.5),
            vision_score=getattr(insight_by_index.get(item.page_index), "score", 0.0),
        )
        for item in renderables
    ]
    if not panel_refs:
        raise HTTPException(status_code=400, detail="No renderable pages or panels were found.")

    plan, hosted_warning = build_plan(
        panel_refs,
        style=project.style,
        target_duration=project.target_duration,
        intensity=project.intensity,
        audio_mood=project.audio_mood,
        include_subtitles=include_subtitles,
        use_hosted=bool(payload.use_hosted_director),
    )
    if hosted_warning:
        warnings.append(hosted_warning)
        log.warning("Project %s: %s", project.id, hosted_warning)
    log.info(
        "Project %s analysed - %d renderable item(s), %d shots via %s",
        project.id, len(panel_refs), len(plan.shots), plan.source,
    )

    path_by_id = {ref.panel_id: ref.path for ref in panel_refs}
    for shot in list(project.shots):
        session.delete(shot)
    session.flush()

    created: dict[int, Shot] = {}
    for n, planned in enumerate(plan.shots, start=1):
        row = Shot(
            project_id=project.id,
            sequence_number=n,
            panel_id=planned.panel_id,
            source_panel_path=path_by_id.get(planned.panel_id, panel_refs[0].path),
            duration_seconds=planned.duration_seconds,
            camera_motion=planned.camera_motion,
            transition_type=planned.transition,
            visual_effects_json=json.dumps(planned.visual_effects),
            audio_cues_json=json.dumps(planned.sound_cues),
            music_intensity=planned.music_intensity,
            subtitle_text=planned.subtitle or None,
        )
        session.add(row)
        # Keep the rows we just made: project.shots is the old collection until the
        # session is refreshed, so looking narration up through it silently finds
        # nothing and every line is dropped.
        created[n] = row

    session.flush()

    # --- the voiceover -----------------------------------------------------
    # Narration is written here, not at render time: a model reading every page is far
    # slower than generating frames, and the render budget is under a minute.
    if payload.write_narration:
        from . import narration as narration_engine

        try:
            lines, reports = narration_engine.build_script(
                [Path(path_by_id.get(p.panel_id, panel_refs[0].path))
                 for p in plan.shots],
                [{"sequence_number": n, "duration_seconds": planned.duration_seconds}
                 for n, planned in enumerate(plan.shots, start=1)],
                float(project.target_duration),
                project_id=project.id,
            )
            written = 0
            for line in lines:
                row = created.get(line.sequence_number)
                if row is None:
                    continue
                row.narration_text = line.text or None
                written += 1 if line.text else 0
            session.flush()
            warnings.extend(r.describe() for r in reports)
            if written != len(created):
                warnings.append(
                    f"Narration covered {written} of {len(created)} shots."
                )
            log.info("Project %s narration written for %d/%d shots",
                     project.id, written, len(created))
        except Exception as exc:
            warnings.append(f"Narration skipped: {exc}")
            log.warning("Project %s narration unavailable: %s", project.id, exc)

    # Depth for parallax, computed now and cached beside each panel, so the renderer
    # never waits on a model. This is what lets the artwork itself move.
    try:
        from . import animation

        warmed = animation.precompute(
            [Path(row.source_panel_path) for row in created.values()]
        )
        log.info("Project %s depth cached for %d/%d shots",
                 project.id, warmed, len(created))
    except Exception as exc:
        log.warning("Project %s depth unavailable, shots will not parallax: %s",
                    project.id, exc)

    project.panel_count = len([r for r in renderables if r.kind == "panel"])
    project.planner_source = plan.source
    project.status = "planned"
    project.error_message = "; ".join(warnings)[:800] if warnings else None
    project.updated_at = utcnow()
    session.commit()
    session.refresh(project)
    return _project_detail(project)


@app.patch("/api/projects/{project_id}/shots", response_model=ProjectDetail)
def update_shots(project_id: int, payload: ShotsUpdate,
                 session: Session = Depends(get_session)) -> ProjectDetail:
    from .schemas import CAMERA_MOTIONS, SOUND_CUES, TRANSITIONS, VISUAL_EFFECTS

    project = _get_project(session, project_id)
    by_seq = {s.sequence_number: s for s in project.shots}
    if not by_seq:
        raise HTTPException(status_code=400, detail="This project has no shot plan yet.")

    for edit in payload.shots:
        shot = by_seq.get(edit.sequence_number)
        if shot is None:
            continue
        if edit.duration_seconds is not None:
            shot.duration_seconds = round(float(edit.duration_seconds), 2)
        if edit.camera_motion and edit.camera_motion in CAMERA_MOTIONS:
            shot.camera_motion = edit.camera_motion
        if edit.transition_type and edit.transition_type in TRANSITIONS:
            shot.transition_type = edit.transition_type
        if edit.visual_effects is not None:
            shot.visual_effects_json = json.dumps(
                [e for e in edit.visual_effects if e in VISUAL_EFFECTS][:4]
            )
        if edit.audio_cues is not None:
            shot.audio_cues_json = json.dumps(
                [c for c in edit.audio_cues if c in SOUND_CUES][:3]
            )
        if edit.narration_text is not None:
            shot.narration_text = edit.narration_text.strip() or None
        if edit.subtitle_text is not None:
            shot.subtitle_text = edit.subtitle_text.strip()[:160] or None

    project.updated_at = utcnow()
    session.commit()
    session.refresh(project)
    return _project_detail(project)


# -------------------------------------------------------------------- render


def _run_render(project_id: int) -> None:
    """Background worker. Owns its own DB session and never leaves a job dangling."""
    try:
        with session_scope() as session:
            project = session.get(Project, project_id)
            if project is None:
                return
            job = project.render_jobs[-1]
            shots = list(project.shots)
            root = project_dir(project_id)
            spec = RenderSpec(
                shots=[
                    ShotSpec(
                        sequence_number=s.sequence_number,
                        image_path=Path(s.source_panel_path),
                        duration_seconds=s.duration_seconds,
                        camera_motion=s.camera_motion,
                        transition=s.transition_type,
                        visual_effects=s.visual_effects,
                        sound_cues=s.audio_cues,
                        music_intensity=s.music_intensity,
                        subtitle=s.subtitle_text or "",
                        narration=s.narration_text or "",
                    )
                    for s in shots
                ],
                aspect_ratio=project.aspect_ratio,
                audio_mood=project.audio_mood,
                intensity=project.intensity,
                style=project.style,
                title=project.title,
                burn_subtitles=bool(project.subtitles_enabled),
                target_duration=float(project.target_duration),
                output_path=root / "output" / "kage.mp4",
                audio_path=root / "temp" / "soundtrack.wav",
                subtitle_path=root / "output" / "kage.srt",
            )
            job_id = job.id
            job.status = "running"
            job.stage = "Starting"
            job.started_at = utcnow()
            job.progress_percent = 1.0
            project.status = "rendering"
            project.error_message = None

        last_reported = {"pct": -5.0}

        def progress(pct: float, stage: str) -> None:
            # Throttle DB writes so progress polling stays cheap.
            if pct - last_reported["pct"] < 1.0 and pct < 99:
                return
            last_reported["pct"] = pct
            try:
                with session_scope() as s2:
                    j = s2.get(RenderJob, job_id)
                    if j:
                        j.progress_percent = round(pct, 1)
                        j.stage = stage[:120]
            except Exception:
                pass

        log.info(
            "Render %s starting - %d shots, %s, mood=%s, subtitles=%s",
            project_id, len(spec.shots), spec.aspect_ratio, spec.audio_mood,
            spec.burn_subtitles,
        )
        result = render(spec, progress)
        log.info(
            "Render %s completed - %.2fs, %d frames, audio=%s -> %s",
            project_id, result.duration_seconds, result.frame_count, result.has_audio,
            result.output_path,
        )

        with session_scope() as session:
            job = session.get(RenderJob, job_id)
            project = session.get(Project, project_id)
            if job:
                job.status = "completed"
                job.stage = "Done"
                job.progress_percent = 100.0
                job.output_video_path = str(result.output_path)
                job.subtitle_path = str(result.subtitle_path) if result.subtitle_path else None
                job.completed_at = utcnow()
                job.error_message = None
                job.diagnostics = json.dumps(result.diagnostics)[:4000]
            if project:
                project.status = "completed"
                project.updated_at = utcnow()

        cleanup_temp(project_dir(project_id) / "temp")

    except Exception as exc:
        detail = str(exc)
        diagnostics = getattr(exc, "diagnostics", None)
        tb = traceback.format_exc()[-2000:]
        log.exception("Render %s failed: %s", project_id, detail)
        if diagnostics:
            log.error("Render %s diagnostics: %s", project_id,
                      json.dumps(diagnostics, default=str)[:1500])
        try:
            with session_scope() as session:
                project = session.get(Project, project_id)
                if project and project.render_jobs:
                    job = project.render_jobs[-1]
                    job.status = "failed"
                    job.stage = "Failed"
                    job.error_message = detail[:800]
                    job.completed_at = utcnow()
                    job.diagnostics = json.dumps(
                        {"error": detail, "diagnostics": diagnostics, "traceback": tb}
                    )[:4000]
                if project:
                    # Metadata stays intact; only the status reflects the failure.
                    project.status = "render_failed"
                    project.error_message = detail[:800]
                    project.updated_at = utcnow()
        except Exception:
            pass
    finally:
        _ACTIVE_RENDERS.discard(project_id)
        if _RENDER_LOCK.locked():
            try:
                _RENDER_LOCK.release()
            except RuntimeError:
                pass


@app.post("/api/projects/{project_id}/render", response_model=RenderJobOut, status_code=202)
def start_render(project_id: int, session: Session = Depends(get_session)) -> RenderJobOut:
    project = _get_project(session, project_id)
    caps = probe_capabilities()
    if not caps.ffmpeg:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg is not available. Install it or place ffmpeg.exe in tools/bin.",
        )
    if not project.shots:
        raise HTTPException(status_code=400, detail="Run analysis to build a shot plan first.")
    if project_id in _ACTIVE_RENDERS:
        raise HTTPException(status_code=409, detail="This project is already rendering.")
    if not _RENDER_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another render is in progress. Wait for it to finish and try again.",
        )

    job = RenderJob(project_id=project.id, status="queued", stage="Queued", progress_percent=0.0)
    session.add(job)
    project.status = "queued"
    session.commit()
    session.refresh(job)

    _ACTIVE_RENDERS.add(project_id)
    thread = threading.Thread(target=_run_render, args=(project_id,), daemon=True,
                              name=f"render-{project_id}")
    thread.start()
    return RenderJobOut.model_validate(job)


@app.get("/api/projects/{project_id}/status", response_model=StatusResponse)
def render_status(project_id: int, session: Session = Depends(get_session)) -> StatusResponse:
    project = _get_project(session, project_id)
    job = project.render_jobs[-1] if project.render_jobs else None
    return StatusResponse(
        project_id=project.id,
        project_status=project.status,
        job=RenderJobOut.model_validate(job) if job else None,
    )


@app.get("/api/projects/{project_id}/video")
def get_video(project_id: int, session: Session = Depends(get_session)) -> FileResponse:
    project = _get_project(session, project_id)
    job = project.render_jobs[-1] if project.render_jobs else None
    if job is None or job.status != "completed":
        raise HTTPException(status_code=409, detail="No finished render is available yet.")
    path = _safe_project_path(project_id, job.output_video_path)
    if path is None:
        raise HTTPException(status_code=404, detail="The rendered file is missing from disk.")
    return FileResponse(path, media_type="video/mp4", filename=f"kage_{project_id}.mp4")


@app.get("/api/projects/{project_id}/subtitles")
def get_subtitles(project_id: int, session: Session = Depends(get_session)) -> FileResponse:
    project = _get_project(session, project_id)
    job = project.render_jobs[-1] if project.render_jobs else None
    path = _safe_project_path(project_id, job.subtitle_path if job else None)
    if path is None:
        raise HTTPException(status_code=404, detail="This render has no subtitle file.")
    return FileResponse(path, media_type="text/plain", filename=f"kage_{project_id}.srt")


@app.get("/api/projects/{project_id}/pages/{page_index}/thumb")
def get_thumb(project_id: int, page_index: int,
              session: Session = Depends(get_session)) -> FileResponse:
    project = _get_project(session, project_id)
    for page in project.pages:
        if page.page_index == page_index:
            path = _safe_project_path(project_id, page.thumb_path or page.image_path)
            if path is None:
                break
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No such page in this project.")


# ------------------------------------------------- providers, usage and keys


@app.get("/api/providers")
def list_providers(project_id: int | None = None) -> dict:
    """Which providers are set up, how each endpoint is behaving, and what it has cost.

    Keys are never returned - only whether one is present, and how it was supplied.
    """
    from . import routing
    from .narration import MAX_IMAGES_PER_REQUEST
    from .providers import PROVIDERS, TEXT_CHAIN, VISION_CHAIN

    labels = [c.label for c in (*VISION_CHAIN, *TEXT_CHAIN)]
    configured = {name: p.configured for name, p in PROVIDERS.items()}

    with session_scope() as session:
        query = select(UsageEvent)
        if project_id is not None:
            query = query.where(UsageEvent.project_id == project_id)
        events = session.scalars(query.order_by(UsageEvent.id.desc()).limit(500)).all()
        rows = [
            {
                "job": e.job, "provider": e.provider, "model": e.model,
                "ok": bool(e.ok), "latency_seconds": round(e.latency_seconds, 2),
                "prompt_tokens": e.prompt_tokens,
                "completion_tokens": e.completion_tokens,
                "images": e.images, "error": e.error,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "project_id": e.project_id,
            }
            for e in events
        ]

    return {
        "providers": [
            {
                "key": name,
                "name": p.name,
                "configured": configured[name],
                "env": p.key_env[0],
                "signup": p.signup,
                "free": p.free_note,
                "source": _key_source(p),
            }
            for name, p in PROVIDERS.items()
        ],
        "order": {
            "vision": routing.order([c.label for c in VISION_CHAIN
                                     if configured.get(c.provider)], "Observe"),
            "text": routing.order([c.label for c in TEXT_CHAIN
                                   if configured.get(c.provider)], "Write"),
        },
        "health": routing.health(labels),
        "usage": rows,
        "totals": _usage_totals(rows),
        # What a run will cost before you start it: one request per batch of pages,
        # plus one to write the whole script.
        "requests_per_film": {
            "images_per_request": MAX_IMAGES_PER_REQUEST,
            "formula": "ceil(shots / images_per_request) + 1",
        },
    }


def _key_source(provider) -> str:
    """Where a key came from, so the UI can say whether it is the user's own."""
    import os

    for name in provider.key_env:
        if os.environ.get(name, "").strip():
            return "user" if name in _USER_SUPPLIED else "env"
    return ""


def _usage_totals(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    return {
        "requests": len(rows),
        "failed": len(rows) - len(ok),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "images": sum(r["images"] for r in rows),
        "median_latency": round(
            sorted(r["latency_seconds"] for r in ok)[len(ok) // 2], 2
        ) if ok else None,
    }


#: Keys the user typed into the app this session, as opposed to ones from .env.
_USER_SUPPLIED: set[str] = set()


@app.put("/api/providers/{provider_key}/key")
def set_provider_key(provider_key: str, payload: ProviderKeyUpdate) -> dict:
    """Store a user's own API key.

    Written to .env so it survives a restart, and applied immediately so the next call
    uses it. The key is never logged and never returned.
    """
    import os

    from . import routing
    from .providers import PROVIDERS

    provider = PROVIDERS.get(provider_key)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider_key}'.")

    name = provider.key_env[0]
    value = (payload.api_key or "").strip()
    if value and len(value) < 8:
        raise HTTPException(status_code=400, detail="That does not look like an API key.")

    if value:
        os.environ[name] = value
        _USER_SUPPLIED.add(name)
    else:
        os.environ.pop(name, None)
        _USER_SUPPLIED.discard(name)
    _write_env_key(name, value)

    # A new key may well fix an endpoint that was resting, so forget what we learned.
    routing.reset()
    log.info("Provider key for %s %s", provider.name, "set" if value else "cleared")
    return {"provider": provider_key, "configured": bool(value)}


def _write_env_key(name: str, value: str) -> None:
    """Upsert one KEY=value line in .env, leaving everything else untouched."""
    from .config import ENV_FILE

    try:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    except OSError:
        lines = []
    out, replaced = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{name}=") or stripped.startswith(f"# {name}="):
            if not replaced:
                out.append(f"{name}={value}" if value else f"# {name}=")
                replaced = True
            continue
        out.append(line)
    if not replaced and value:
        out.append(f"{name}={value}")
    try:
        ENV_FILE.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    except OSError as exc:
        log.warning("Could not persist %s to .env: %s", name, exc)


@app.post("/api/providers/reset")
def reset_routing() -> dict:
    """Forget measured provider behaviour and start choosing again from scratch."""
    from . import routing

    routing.reset()
    return {"status": "reset"}


@app.get("/api/logs")
def get_logs(limit: int = 100, level: str = "DEBUG", source: str = "buffer") -> dict:
    """Recent log records, newest first.

    ``source`` is 'buffer' (in-memory, structured), 'main' or 'errors' (raw file tails).
    """
    limit = max(1, min(1000, limit))
    if source == "main":
        return {"source": str(MAIN_LOG), "lines": read_log_file(MAIN_LOG, limit)[::-1]}
    if source == "errors":
        return {"source": str(ERROR_LOG), "lines": read_log_file(ERROR_LOG, limit)[::-1]}

    records = RING.tail(limit=limit, min_level=level)
    return {
        "source": "memory",
        "count": len(records),
        "error_count": len([r for r in RING.records if r["level"] in ("ERROR", "CRITICAL")]),
        "warning_count": len([r for r in RING.records if r["level"] == "WARNING"]),
        "log_files": {"main": str(MAIN_LOG), "errors": str(ERROR_LOG)},
        "records": records,
    }


@app.get("/api/stats")
def stats(session: Session = Depends(get_session)) -> dict:
    projects = session.scalars(select(Project)).all()
    completed = [p for p in projects if p.status == "completed"]
    return {
        "projects_total": len(projects),
        "projects_completed": len(completed),
        "projects_failed": len([p for p in projects if p.status == "render_failed"]),
        "storage_root": str(PROJECTS_DIR),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }
