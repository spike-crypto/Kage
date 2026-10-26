"""Shared front-end plumbing: the API client, vocabulary, and the workflow model.

Pages import from here. Nothing in this module draws a whole screen; it holds the pieces
every screen needs, so a page file reads as a description of that one screen.

Design note: the studio is a four-step pipeline, not seven unrelated screens. ``STEPS``
below is the single source of truth for that order, what each step needs before it can
run, and what to say when it cannot. Every page gate and the progress strip are derived
from it, so the flow can only be described one way.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import requests
import streamlit as st

from backend.config import env, load_dotenv
from backend.styles import STYLE_PRESETS

load_dotenv()

API_BASE = env("API", "http://127.0.0.1:8000")
ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample_manga"


# ---------------------------------------------------------------- vocabulary

STYLES = {k: p.name for k, p in STYLE_PRESETS.items()}
STYLE_TAGLINES = {k: p.tagline for k, p in STYLE_PRESETS.items()}
STYLE_DESCRIPTIONS = {k: p.description for k, p in STYLE_PRESETS.items()}

MOODS = {
    "orchestral": "Orchestral",
    "ambient": "Ambient",
    "dark": "Dark",
    "energetic": "Energetic",
    "suspenseful": "Suspenseful",
}
INTENSITIES = {"subtle": "Subtle", "balanced": "Balanced", "dramatic": "Dramatic"}
INTENSITY_HELP = {
    "subtle": "Gentler camera travel and fewer accents.",
    "balanced": "The default. Movement you notice without it pulling focus.",
    "dramatic": "Longer travel, tighter framing, accents spent more freely.",
}
ASPECTS = {
    "16:9": "Widescreen",
    "9:16": "Vertical",
    "1:1": "Square",
}
ASPECT_HELP = {"16:9": "YouTube, players", "9:16": "Reels, Shorts", "1:1": "Feed posts"}
DURATIONS = (15, 30, 60)

CAMERA_MOTIONS = [
    "slow_zoom_in", "slow_zoom_out", "pan_left", "pan_right", "pan_up", "pan_down",
    "ken_burns", "rapid_pan", "push_in", "static_drift",
]
TRANSITIONS = ["cut", "fade", "crossfade", "flash", "whip"]
VISUAL_EFFECTS = [
    "vignette", "speed_lines", "screen_shake", "impact_flash", "motion_blur",
    "particles", "film_grain", "rain", "desaturate", "warm_glow", "cold_tint",
]
SOUND_CUES = [
    "whoosh", "impact", "footsteps", "wind", "rain", "ambient_drone",
    "suspense_riser", "explosion", "heartbeat", "sword_clash",
]

#: label, colour and icon for each project status, so a status reads the same everywhere.
STATUS_BADGE = {
    "completed": ("Rendered", "green", ":material/check_circle:"),
    "rendering": ("Rendering", "blue", ":material/autorenew:"),
    "queued": ("Queued", "blue", ":material/schedule:"),
    "planned": ("Plan ready", "violet", ":material/movie_filter:"),
    "imported": ("Pages in", "orange", ":material/image:"),
    "created": ("Empty", "gray", ":material/add:"),
    "render_failed": ("Render failed", "red", ":material/error:"),
    "import_failed": ("Import failed", "red", ":material/error:"),
}


def humanise(value: str) -> str:
    """snake_case to something a person reads: 'slow_zoom_in' -> 'Slow zoom in'."""
    text = str(value or "").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def style_name(project: dict) -> str:
    """Projects created before the rename only carry a legacy genre."""
    key = project.get("style") or project.get("genre")
    return STYLES.get(key, humanise(str(key or "unknown")))


# ------------------------------------------------------------------ API layer


class ApiError(RuntimeError):
    pass


def api(method: str, path: str, *, timeout: int = 180, **kwargs):
    url = f"{API_BASE}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Cannot reach the backend at {API_BASE}. Start it with `python main.py`."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The backend took too long to respond.") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text[:300]
        if isinstance(detail, list):
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise ApiError(str(detail or f"Request failed ({response.status_code})."))

    if response.status_code == 204 or not response.content:
        return None
    return response.json()


@st.cache_data(ttl=20, show_spinner=False)
def fetch_health() -> dict | None:
    try:
        return api("GET", "/api/health", timeout=15)
    except ApiError:
        return None


def current_project() -> dict | None:
    """The open project, or None. Clears the selection if it has gone away."""
    pid = st.session_state.get("project_id")
    if not pid:
        return None
    try:
        return api("GET", f"/api/projects/{pid}")
    except ApiError as exc:
        st.session_state.project_id = None
        st.warning(str(exc))
        return None


def existing_file(raw: str | None) -> Path | None:
    """Return the path only when it names a real file.

    ``Path("").exists()`` is True - an empty string resolves to the current directory -
    so a blank path from the API would otherwise sail past a plain exists() check and
    crash st.image/st.video.
    """
    if not raw or not str(raw).strip():
        return None
    try:
        candidate = Path(str(raw))
        return candidate if candidate.is_file() else None
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=600, show_spinner=False, max_entries=256)
def thumb_uri(path: str, mtime: float, width: int = 150) -> str | None:
    """A small data URI for a page image, for use in table cells.

    ``st.column_config.ImageColumn`` needs a URL, and these files are local. Keyed on
    mtime so a re-import invalidates it rather than showing the previous artwork.
    """
    del mtime  # part of the cache key only
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((width, width * 3), Image.LANCZOS)
            buffer = io.BytesIO()
            im.save(buffer, "JPEG", quality=72)
    except Exception:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def page_thumb(raw: str | None, width: int = 150) -> str | None:
    file = existing_file(raw)
    if file is None:
        return None
    try:
        return thumb_uri(str(file), file.stat().st_mtime, width)
    except OSError:
        return None


# -------------------------------------------------------------- the workflow


@dataclass(frozen=True)
class Step:
    """One stage of the pipeline, and the condition that unlocks it."""

    key: str
    page: str          # path of the page file, for st.page_link / st.switch_page
    title: str
    icon: str
    needs: str         # what must already be true; "" for the first step
    blocked_hint: str  # shown when it is not

    def ready(self, project: dict | None) -> bool:
        if project is None:
            return self.key == "import"
        if self.needs == "pages":
            return bool([p for p in project.get("pages", []) if p["kind"] == "page"])
        if self.needs == "shots":
            return bool(project.get("shots"))
        return True


STEPS: tuple[Step, ...] = (
    Step("import", "app_pages/import_pages.py", "Add pages", ":material/upload_file:",
         "", "Open or create a project first."),
    Step("style", "app_pages/style.py", "Choose a look", ":material/palette:",
         "pages", "Add some pages before choosing a look."),
    Step("shots", "app_pages/shots.py", "Review shots", ":material/view_timeline:",
         "shots", "Choose a look and build the shot plan first."),
    Step("render", "app_pages/render.py", "Render", ":material/movie:",
         "shots", "Build a shot plan before rendering."),
)

STEP_BY_KEY = {s.key: s for s in STEPS}


def step_state(step: Step, project: dict | None, current: str) -> str:
    """One of 'done', 'current', 'ready', 'locked'."""
    if step.key == current:
        return "current"
    if not step.ready(project):
        return "locked"
    # A step counts as done once the step after it has what it produces.
    order = [s.key for s in STEPS]
    following = order[order.index(step.key) + 1:]
    if any(STEP_BY_KEY[k].ready(project) for k in following):
        return "done"
    return "ready"


_STATE_LOOK = {
    "done": (":material/check_circle:", "green", "Done"),
    "current": (":material/radio_button_checked:", "violet", "You are here"),
    "ready": (":material/circle:", "gray", "Next"),
    "locked": (":material/lock:", "gray", "Locked"),
}


def progress_strip(project: dict | None, current: str) -> None:
    """The four steps, their state, and a link to each one that is open.

    This is the main piece of wayfinding in the app: the pipeline is linear, and without
    it the sidebar is just seven names with no hint that three of them need the others to
    have run first.
    """
    columns = st.columns(len(STEPS), gap="small")
    for column, step in zip(columns, STEPS):
        state = step_state(step, project, current)
        icon, colour, label = _STATE_LOOK[state]
        with column, st.container(border=True, height="stretch"):
            st.badge(label, icon=icon, color=colour)
            st.page_link(
                step.page, label=step.title, icon=step.icon,
                disabled=state == "locked", width="stretch",
                help=step.blocked_hint if state == "locked" else None,
            )


def project_bar(project: dict | None) -> None:
    """One line of context: which project is open and what it is set to."""
    if project is None:
        st.caption("No project open.")
        return
    label, colour, icon = STATUS_BADGE.get(
        project["status"], (humanise(project["status"]), "gray", ":material/info:")
    )
    pages = len([p for p in project.get("pages", []) if p["kind"] == "page"])
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(label, icon=icon, color=colour)
        st.caption(
            f"**{project['title']}**  ·  {style_name(project)}  ·  "
            f"{project['target_duration']}s  ·  {project['aspect_ratio']}  ·  "
            f"{pages} page{'s' if pages != 1 else ''}"
        )


def require(step_key: str, project: dict | None) -> bool:
    """Gate a step page. Returns True when it may run.

    When it may not, this says why and offers the one link that fixes it, rather than
    leaving the reader on a page that quietly does nothing.
    """
    step = STEP_BY_KEY[step_key]
    if project is None:
        st.info(
            "Open a project to continue, or start a new one.",
            icon=":material/folder_open:",
        )
        st.page_link("app_pages/home.py", label="Go to home", icon=":material/home:")
        return False
    if step.ready(project):
        return True

    order = [s.key for s in STEPS]
    earlier = [s for s in STEPS[: order.index(step_key)] if s.ready(project)]
    target = earlier[-1] if earlier else STEPS[0]
    st.info(step.blocked_hint, icon=":material/lock:")
    st.page_link(target.page, label=f"Go to {target.title.lower()}", icon=target.icon)
    return False


def open_project(project_id: int, *, page: str | None = None) -> None:
    """Make a project active and jump to the right step for its state."""
    st.session_state.project_id = project_id
    st.session_state.included_pages = None
    if page is None:
        try:
            project = api("GET", f"/api/projects/{project_id}")
        except ApiError:
            project = None
        page = next(
            (s.page for s in reversed(STEPS) if s.ready(project)), STEPS[0].page
        )
    st.switch_page(page)
