"""Home: what the studio has made, and the one button that starts something new."""

from __future__ import annotations

import streamlit as st

from frontend.studio import (
    DURATIONS,
    STATUS_BADGE,
    STYLE_DESCRIPTIONS,
    STYLE_TAGLINES,
    STYLES,
    ApiError,
    api,
    fetch_health,
    humanise,
    open_project,
    style_name,
)

st.title("From ink to imagination", icon=":material/movie_filter:")
st.caption(
    "Upload manga artwork and Kage frames it like film - choosing pages, moving the "
    "camera, cutting the sequence and scoring it - then renders an MP4."
)

try:
    projects = api("GET", "/api/projects?limit=25")
    stats = api("GET", "/api/stats")
except ApiError as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.caption("Start the backend in another terminal:  `python main.py`")
    st.stop()

health = fetch_health()
ready = bool((health or {}).get("render_ready"))

a, b, c = st.columns(3)
a.metric("Projects", stats["projects_total"], border=True, icon=":material/folder:")
b.metric("Rendered", stats["projects_completed"], border=True, icon=":material/movie:")
c.metric(
    "Engine", "Ready" if ready else "Degraded", border=True,
    icon=":material/check_circle:" if ready else ":material/warning:",
    help="Whether FFmpeg and the imaging libraries are available.",
)

st.space("small")
new, recent = st.columns([1, 1.3], gap="large")

with new:
    st.subheader("Start a new film", divider=False)
    title = st.text_input("Project name", value="Untitled sequence")
    style = st.selectbox(
        "Look", list(STYLES), format_func=STYLES.get,
        help="Sets the grade, camera vocabulary, pacing and score. Changeable later.",
    )
    with st.container(border=True):
        st.markdown(f"**{STYLE_TAGLINES[style]}**")
        st.caption(STYLE_DESCRIPTIONS[style])
    duration = st.segmented_control(
        "Length", DURATIONS, default=30, format_func=lambda s: f"{s}s",
        selection_mode="single", width="stretch",
    )
    if st.button("Create project", type="primary", icon=":material/add:", width="stretch"):
        try:
            created = api("POST", "/api/projects", json={
                "title": title or "Untitled sequence",
                "style": style,
                "target_duration": duration or 30,
            })
        except ApiError as exc:
            st.error(str(exc))
        else:
            open_project(created["id"], page="app_pages/import_pages.py")

with recent:
    st.subheader("Pick up where you left off", divider=False)
    if not projects:
        with st.container(border=True):
            st.markdown("**Nothing here yet**")
            st.caption(
                "Create a project on the left, add a few manga pages, and render your "
                "first sequence."
            )
    for project in projects[:6]:
        label, colour, icon = STATUS_BADGE.get(
            project["status"], (humanise(project["status"]), "gray", ":material/info:")
        )
        with st.container(border=True):
            text, action = st.columns([3, 1], vertical_alignment="center")
            with text:
                st.markdown(f"**{project['title']}**")
                st.caption(
                    f"{style_name(project)} · {project['target_duration']}s · "
                    f"{project['aspect_ratio']} · {project['page_count']} pages · "
                    f"created {project['created_at'][:10]}"
                )
                st.badge(label, icon=icon, color=colour)
            if action.button(
                "Open", key=f"open_{project['id']}", width="stretch",
                icon=":material/arrow_forward:",
            ):
                open_project(project["id"])

st.space("medium")
with st.expander("How Kage works", icon=":material/help:"):
    steps = [
        ("Add pages", "Drop in a PDF or page images. Kage trims the scan down to the "
                      "printed page and splits two-page spreads."),
        ("Choose a look", "A style sets the grade, the camera moves, the pacing and the "
                          "score. A local vision model then picks which pages are worth "
                          "filming and skips covers and pages of text."),
        ("Review shots", "Every shot is listed with its length, camera move, transition "
                         "and effects. Change any of them."),
        ("Render", "Frames are generated one at a time, scored, and muxed into an MP4 "
                   "you can preview and download."),
    ]
    for n, (name, body) in enumerate(steps, start=1):
        st.markdown(f"**{n}. {name}** — {body}")
    st.caption(
        "Kage frames and moves your artwork like film. It does not generate new drawings "
        "or animate characters within a panel."
    )
