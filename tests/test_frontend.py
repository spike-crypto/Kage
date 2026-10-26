"""Streamlit frontend tests.

These run the real ``app.py`` in-process with ``AppTest`` and a stubbed HTTP layer, so
they need no running backend. They exist mainly to click things: rendering a page without
interacting with it does not catch the session-state errors that only fire on a callback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

APP = str(ROOT / "app.py")

PAGES = [
    "app_pages/home.py",
    "app_pages/import_pages.py",
    "app_pages/style.py",
    "app_pages/shots.py",
    "app_pages/render.py",
    "app_pages/projects.py",
    "app_pages/diagnostics.py",
]

PROJECT = {
    "id": 1, "title": "Stub Project", "style": "kage_noir", "genre": "action",
    "target_duration": 30,
    "aspect_ratio": "16:9", "audio_mood": "orchestral", "intensity": "balanced",
    "status": "planned", "page_count": 2, "panel_count": 0, "subtitles_enabled": 0,
    "planner_source": "local_planner", "error_message": None,
    "created_at": "2026-10-23T10:00:00", "updated_at": "2026-10-23T10:05:00",
    "source_filename": "stub.pdf",
    "pages": [
        {"id": 1, "page_index": 0, "kind": "page", "image_path": "", "thumb_path": None,
         "width": 900, "height": 1300, "included": 1, "ocr_text": None},
        {"id": 2, "page_index": 1, "kind": "page", "image_path": "", "thumb_path": None,
         "width": 900, "height": 1300, "included": 1, "ocr_text": None},
    ],
    "shots": [
        {"id": 1, "sequence_number": 1, "panel_id": 0, "source_panel_path": "",
         "duration_seconds": 3.0, "camera_motion": "slow_zoom_in", "transition_type": "cut",
         "music_intensity": "medium", "subtitle_text": None,
         "visual_effects": ["vignette"], "audio_cues": ["whoosh"]},
        {"id": 2, "sequence_number": 2, "panel_id": 1, "source_panel_path": "",
         "duration_seconds": 3.0, "camera_motion": "pan_left", "transition_type": "crossfade",
         "music_intensity": "high", "subtitle_text": None,
         "visual_effects": ["speed_lines", "impact_flash"],
         "audio_cues": ["impact", "whoosh"]},
    ],
    "render_job": None,
}

#: A project that has got no further than being created, for the gating tests.
EMPTY_PROJECT = {**PROJECT, "id": 2, "status": "created", "page_count": 0,
                 "pages": [], "shots": []}

HEALTH = {
    "status": "ok", "version": "0.1.0", "render_ready": True, "notes": [],
    "capabilities": {
        "ffmpeg": True, "ffmpeg_version": "stub", "ffprobe": True, "pdf_import": True,
        "opencv": True, "ocr_subtitles": False, "hosted_llm_director": False,
        "vision_model": True, "vision_device": "CPU",
        "video_encoders": ["libx264", "aac"],
    },
}

ROUTES = {
    "/api/health": HEALTH,
    "/api/stats": {"projects_total": 1, "projects_completed": 0, "projects_failed": 0,
                   "storage_root": "data/projects", "server_time": "2026-10-23T10:00:00Z"},
    "/api/logs": {"source": "memory", "count": 1, "error_count": 0, "warning_count": 1,
                  "log_files": {"main": "main.log", "errors": "errors.log"},
                  "records": [{"time": "2026-10-23T10:00:00+00:00", "level": "WARNING",
                               "logger": "kage.api", "line": 10,
                               "message": "stub warning"}]},
}

#: Swapped by the gating test so /api/projects/{id} answers with an empty project.
ACTIVE: dict = PROJECT


class StubResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}

    @property
    def content(self) -> bytes:
        return json.dumps(self._payload).encode()

    @property
    def text(self) -> str:
        return json.dumps(self._payload)

    def json(self):
        return self._payload


def _stub_request(method: str, url: str, **_kwargs) -> StubResponse:
    path = url.split("127.0.0.1:8000", 1)[-1].split("?", 1)[0]
    if path in ROUTES:
        return StubResponse(ROUTES[path])
    if path == "/api/projects":
        return StubResponse([PROJECT]) if method.upper() == "GET" else StubResponse(PROJECT, 201)
    if path.endswith("/status"):
        return StubResponse({"project_id": 1, "project_status": "planned", "job": None})
    if path.startswith("/api/projects/"):
        return StubResponse(ACTIVE)
    return StubResponse({"detail": f"unstubbed route {path}"}, 404)


@pytest.fixture(autouse=True)
def stub_http(monkeypatch):
    global ACTIVE
    ACTIVE = PROJECT
    monkeypatch.setattr(requests, "request", _stub_request)


def run_page(page: str, project_id: int | None = 1) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["project_id"] = project_id
    at.run()
    if page != "app_pages/home.py":
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(page):
    at = run_page(page)
    assert not at.exception, f"{page} raised: {[str(e.value) for e in at.exception]}"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_an_active_project(page):
    at = run_page(page, project_id=None)
    assert not at.exception, f"{page} (no project) raised: {at.exception}"


def test_every_button_on_every_page_is_clickable():
    """Click each button on each page and assert none of them raise."""
    failures: list[str] = []
    for page in PAGES:
        at = run_page(page)
        for index in range(len(at.button)):
            fresh = run_page(page)
            if index >= len(fresh.button):
                continue
            button = fresh.button[index]
            label, key = button.label, button.key
            if getattr(button, "disabled", False):
                continue  # a browser user cannot click these either
            try:
                result = button.click().run()
            except Exception as exc:  # a click should never blow up the script
                failures.append(f"{page}/{key or label}: {type(exc).__name__}: {exc}")
                continue
            if result.exception:
                failures.append(
                    f"{page}/{key or label}: {[str(e.value) for e in result.exception]}"
                )
    assert not failures, "clicking these buttons raised:\n" + "\n".join(failures)


def test_saving_shots_with_several_effects_round_trips():
    """A shot carrying more than one effect and cue must save without error."""
    at = run_page("app_pages/shots.py")
    save = [b for b in at.button if b.label == "Save changes"]
    assert save, "the shot table should offer a save button"
    after = save[0].click().run()
    assert not after.exception, [str(e.value) for e in after.exception]


# --- the workflow the navigation is built around ----------------------------


def test_creating_a_project_moves_to_the_first_step():
    at = run_page("app_pages/home.py", project_id=None)
    create = [b for b in at.button if "Create project" in (b.label or "")]
    assert create, "home should offer a create button"
    after = create[0].click().run()
    assert not after.exception
    assert after.session_state["project_id"] == PROJECT["id"]


def test_opening_a_project_selects_it():
    at = run_page("app_pages/home.py", project_id=None)
    opens = [b for b in at.button if b.key and b.key.startswith("open_")]
    assert opens, "home should offer an Open button for the stub project"
    after = opens[0].click().run()
    assert not after.exception
    assert after.session_state["project_id"] == PROJECT["id"]


@pytest.mark.parametrize("page", ["app_pages/style.py", "app_pages/shots.py",
                                  "app_pages/render.py"])
def test_steps_are_gated_until_their_input_exists(page, monkeypatch):
    """A step with nothing to work on must say so, not render an empty screen."""
    global ACTIVE
    ACTIVE = EMPTY_PROJECT
    at = run_page(page, project_id=EMPTY_PROJECT["id"])
    assert not at.exception
    blocked = [i.value for i in at.info]
    assert blocked, f"{page} should explain why it cannot run yet"
    # And it must offer the way back to the step that unblocks it.
    assert at.get("page_link"), f"{page} should link to the step that unblocks it"
