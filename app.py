"""Kage - From Ink to Imagination. Streamlit studio front end.

Entry point only: page config, navigation, the sidebar, and the state every page shares.
Each screen lives in ``app_pages/``; the pieces they share live in ``frontend/studio.py``.

Talks to the FastAPI backend over HTTP; it holds no rendering logic of its own.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Kage - From Ink to Imagination",
    page_icon=":material/movie_filter:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.studio import (  # noqa: E402  - must follow set_page_config
    API_BASE,
    STEPS,
    fetch_health,
)

# Shared, per-user state. One place, before any page runs.
st.session_state.setdefault("project_id", None)
st.session_state.setdefault("included_pages", None)

PAGES = {
    "": [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Make a film": [
        st.Page(step.page, title=step.title, icon=step.icon) for step in STEPS
    ],
    "Studio": [
        st.Page("app_pages/projects.py", title="Projects", icon=":material/folder:"),
        st.Page("app_pages/settings.py", title="Providers", icon=":material/key:"),
        st.Page("app_pages/diagnostics.py", title="Diagnostics",
                icon=":material/monitor_heart:"),
    ],
}


def sidebar(health: dict | None) -> None:
    """Brand, backend health and the active project. Navigation is drawn by st.navigation."""
    with st.sidebar:
        st.markdown("#### Kage")
        st.caption("From ink to imagination")

        if health is None:
            st.badge("Backend offline", icon=":material/cloud_off:", color="red")
            st.caption(f"Expected at {API_BASE}. Run `python main.py`.")
            return

        caps = health["capabilities"]
        ready = bool(health.get("render_ready"))
        st.badge(
            "Ready to render" if ready else "Not ready",
            icon=":material/check_circle:" if ready else ":material/warning:",
            color="green" if ready else "orange",
        )
        with st.expander("What is installed", icon=":material/host:"):
            for name, ok, note in (
                ("FFmpeg", caps["ffmpeg"], "encodes the video"),
                ("PDF import", caps["pdf_import"], "reads PDF sources"),
                ("OpenCV", caps["opencv"], "camera moves and effects"),
                ("Vision model", caps.get("vision_model", False), "picks the best pages"),
                ("OCR", caps["ocr_subtitles"], "subtitles from page text"),
            ):
                st.badge(
                    name,
                    icon=":material/check:" if ok else ":material/remove:",
                    color="green" if ok else "gray",
                )
                st.caption(note if ok else f"{note} - not installed, skipped")
            if caps.get("vision_model"):
                st.caption(f"Vision runs on {caps.get('vision_device', 'CPU')}")

        for note in health.get("notes", []):
            st.caption(note)

        pid = st.session_state.get("project_id")
        if pid:
            if st.button("Close project", icon=":material/close:", width="stretch"):
                st.session_state.project_id = None
                st.session_state.included_pages = None
                st.rerun()


page = st.navigation(PAGES, position="sidebar")
sidebar(fetch_health())
page.run()
