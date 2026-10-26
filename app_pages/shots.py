"""Step 3: the whole sequence as one editable table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.studio import (
    CAMERA_MOTIONS,
    MOODS,
    SOUND_CUES,
    TRANSITIONS,
    VISUAL_EFFECTS,
    ApiError,
    api,
    current_project,
    humanise,
    page_thumb,
    progress_strip,
    project_bar,
    require,
)

project = current_project()
st.title("Review shots", icon=":material/view_timeline:")
progress_strip(project, "shots")
project_bar(project)

if not require("shots", project):
    st.stop()

shots = project["shots"]
planned = sum(s["duration_seconds"] for s in shots)

# What the director decided, and why - the vision model's page choices land here.
if project.get("error_message"):
    for note in str(project["error_message"]).split("; "):
        st.info(note, icon=":material/auto_awesome:")

a, b, c, d = st.columns(4)
a.metric("Shots", len(shots), border=True, icon=":material/view_timeline:")
b.metric("Planned", f"{planned:.1f}s", border=True, icon=":material/timer:")
c.metric("Target", f"{project['target_duration']}s", border=True, icon=":material/flag:")
d.metric(
    "Score", MOODS.get(project["audio_mood"], project["audio_mood"]),
    border=True, icon=":material/music_note:",
)
st.caption(
    "Dissolves overlap the shots either side of them, so the encoder stretches the plan "
    f"slightly to land on exactly {project['target_duration']}s."
)

st.space("small")
st.subheader("The sequence", divider=False)
st.caption(
    "Edit any cell. Each row is one shot, in order - the thumbnail is the page it is "
    "framed on."
)

table = pd.DataFrame([
    {
        "Shot": shot["sequence_number"],
        "Page": page_thumb(shot.get("source_panel_path")),
        "Seconds": float(shot["duration_seconds"]),
        "Camera": shot["camera_motion"],
        "Cut in": shot["transition_type"],
        "Effects": list(shot["visual_effects"]),
        "Sound": list(shot["audio_cues"]),
        "Moving": bool(shot.get("clip_path")),
        "Narration": shot.get("narration_text") or "",
        "Subtitle": shot["subtitle_text"] or "",
    }
    for shot in shots
])

edited = st.data_editor(
    table,
    key="shot_table",
    hide_index=True,
    width="stretch",
    row_height=72,
    num_rows="fixed",
    disabled=["Shot", "Page", "Moving"],
    alt="The shot plan: one row per shot, with its length, camera move and effects.",
    column_config={
        "Shot": st.column_config.NumberColumn(width="small", help="Order in the film."),
        "Page": st.column_config.ImageColumn("Page", width="small"),
        "Seconds": st.column_config.NumberColumn(
            min_value=0.6, max_value=12.0, step=0.1, format="%.1f s", width="small",
            help="How long this shot holds.",
        ),
        "Camera": st.column_config.SelectboxColumn(
            options=CAMERA_MOTIONS, format_func=humanise, required=True,
            help="How the camera moves across the page.",
        ),
        "Cut in": st.column_config.SelectboxColumn(
            options=TRANSITIONS, format_func=humanise, required=True, width="small",
            help="How this shot arrives from the one before it.",
        ),
        "Effects": st.column_config.MultiselectColumn(
            options=VISUAL_EFFECTS, format_func=humanise,
            help="Accents. The style's grade is always on and is not listed here.",
        ),
        "Sound": st.column_config.MultiselectColumn(
            options=SOUND_CUES, format_func=humanise,
            help="One-off sounds placed on this shot.",
        ),
        "Moving": st.column_config.CheckboxColumn(
            width="small",
            help="Generated image-to-video. Unticked shots use depth parallax.",
        ),
        "Narration": st.column_config.TextColumn(
            width="large",
            help="Spoken over this shot. Written by a model from what it saw on the "
                 "page, trimmed to what fits the shot's length. Edit freely.",
        ),
        "Subtitle": st.column_config.TextColumn(
            help="Only text read from the page by OCR; nothing is invented.",
        ),
    },
)


def items(cell) -> list[str]:
    """A multiselect cell as a plain list of strings.

    Cleared cells can arrive as None or NaN rather than an empty list, and NaN is not
    iterable, so normalise here instead of at each call site.
    """
    if cell is None:
        return []
    try:
        return [str(v) for v in cell]
    except TypeError:
        return []


def collect() -> list[dict]:
    """The table as the API expects it, capped at the limits the backend enforces."""
    return [
        {
            "sequence_number": int(row["Shot"]),
            "duration_seconds": float(row["Seconds"]),
            "camera_motion": row["Camera"],
            "transition_type": row["Cut in"],
            "visual_effects": items(row["Effects"])[:4],
            "audio_cues": items(row["Sound"])[:3],
            "narration_text": str(row["Narration"] or ""),
            "subtitle_text": str(row["Subtitle"] or ""),
        }
        for _, row in edited.iterrows()
    ]


st.space("small")
edited_total = float(edited["Seconds"].sum())
if abs(edited_total - planned) > 0.05:
    st.caption(f"Edited plan is {edited_total:.1f}s before transition overlap.")

with st.container(horizontal=True):
    if st.button("Save changes", icon=":material/save:"):
        try:
            api("PATCH", f"/api/projects/{project['id']}/shots", json={"shots": collect()})
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.toast("Shot plan saved.", icon=":material/check_circle:")
            st.rerun()
    if st.button("Save and render", type="primary", icon=":material/movie:"):
        try:
            api("PATCH", f"/api/projects/{project['id']}/shots", json={"shots": collect()})
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.switch_page("app_pages/render.py")
