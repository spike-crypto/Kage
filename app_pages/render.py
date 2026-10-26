"""Step 4: render, watch it happen, then preview and download."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.studio import (
    MOODS,
    ApiError,
    api,
    current_project,
    existing_file,
    progress_strip,
    project_bar,
    require,
)

project = current_project()
st.title("Render", icon=":material/movie:")
progress_strip(project, "render")
project_bar(project)

if not require("render", project):
    st.stop()

try:
    state = api("GET", f"/api/projects/{project['id']}/status")
except ApiError as exc:
    st.error(str(exc))
    st.stop()

job = state.get("job")
status = (job or {}).get("status")
busy = status in ("queued", "running")


def start_render() -> None:
    try:
        api("POST", f"/api/projects/{project['id']}/render")
    except ApiError as exc:
        st.error(str(exc))
    else:
        st.rerun()


st.space("small")

if busy:
    # A fragment reruns on its own timer, so the progress bar updates without the rest of
    # the page - and the whole script - being re-executed every two seconds.
    @st.fragment(run_every=2)
    def progress() -> None:
        try:
            live = api("GET", f"/api/projects/{project['id']}/status")
        except ApiError as exc:
            st.error(str(exc))
            return
        current = live.get("job") or {}
        percent = float(current.get("progress_percent") or 0.0)
        if current.get("status") in ("queued", "running"):
            st.progress(
                max(0.0, min(1.0, percent / 100.0)),
                text=f"{current.get('stage') or 'Working'} — {percent:.0f}%",
            )
        else:
            st.rerun(scope="app")

    progress()
    st.caption(
        f"Rendering {len(project['shots'])} shots. This runs in a background worker, so "
        "you can leave the page and come back."
    )
    st.stop()

if status == "failed":
    st.error(job.get("error_message") or "The render failed.", icon=":material/error:")
    with st.expander("What usually causes this", icon=":material/help:"):
        st.caption(
            "FFmpeg missing from tools/bin or PATH, or a source image that was moved or "
            "deleted after import. The Diagnostics page carries the backend log."
        )
        st.code(job.get("error_message") or "No details recorded.")
    if st.button("Try again", type="primary", icon=":material/refresh:"):
        start_render()
    st.stop()

if status == "completed":
    video = existing_file(job.get("output_video_path"))
    if video is None:
        st.error("The render finished, but the file is no longer on disk.",
                 icon=":material/error:")
        if st.button("Render again", type="primary", icon=":material/refresh:"):
            start_render()
        st.stop()

    st.success(
        f"{project['target_duration']}s · {project['aspect_ratio']} · "
        f"{MOODS.get(project['audio_mood'], project['audio_mood'])} score",
        icon=":material/check_circle:",
    )
    data = video.read_bytes()
    st.video(data, alt=f"The rendered motion comic for {project['title']}.")

    with st.container(horizontal=True):
        st.download_button(
            "Download MP4", data, file_name=f"kage_{project['id']}.mp4",
            mime="video/mp4", type="primary", icon=":material/download:",
        )
        srt = existing_file(job.get("subtitle_path"))
        if srt:
            st.download_button(
                "Download subtitles", Path(srt).read_text(encoding="utf-8"),
                file_name=f"kage_{project['id']}.srt", mime="text/plain",
                icon=":material/subtitles:",
            )
        if st.button("Render again", icon=":material/refresh:"):
            start_render()
    st.caption(f"Saved at {job['output_video_path']}")
    st.stop()

# Nothing rendered yet.
with st.container(border=True):
    st.markdown("**Ready to render**")
    st.caption(
        f"{len(project['shots'])} shots · {project['target_duration']}s · "
        f"{project['aspect_ratio']}. Frames are generated one at a time and muxed with "
        "the score into an MP4."
    )
    if st.button("Start render", type="primary", icon=":material/play_arrow:"):
        start_render()
