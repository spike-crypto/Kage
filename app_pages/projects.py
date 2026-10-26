"""The library: everything the studio has made, with one row per project."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.studio import (
    STATUS_BADGE,
    ApiError,
    api,
    humanise,
    open_project,
    style_name,
)

st.title("Projects", icon=":material/folder:")
st.caption("Every project in the studio. Select a row to open or delete it.")

try:
    projects = api("GET", "/api/projects?limit=100")
except ApiError as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()

if not projects:
    with st.container(border=True):
        st.markdown("**Nothing here yet**")
        st.caption("Projects and their rendered output will be listed here.")
    st.page_link("app_pages/home.py", label="Start one", icon=":material/home:")
    st.stop()

table = pd.DataFrame([
    {
        "#": project["id"],
        "Name": project["title"],
        "Status": STATUS_BADGE.get(project["status"], (humanise(project["status"]),))[0],
        "Look": style_name(project),
        "Length": f"{project['target_duration']}s",
        "Shape": project["aspect_ratio"],
        "Pages": project["page_count"],
        "Updated": project["updated_at"][:16].replace("T", " "),
    }
    for project in projects
])

selected = st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row",
    key="project_table",
    alt="All projects, with their status, settings and last update.",
    column_config={
        "#": st.column_config.NumberColumn(width="small"),
        "Name": st.column_config.TextColumn(width="medium"),
        "Pages": st.column_config.NumberColumn(width="small"),
    },
)

rows = selected.selection.rows if selected and selected.selection else []
if not rows:
    st.caption("Select a row to act on it.")
    st.stop()

project = projects[rows[0]]
st.space("small")
with st.container(border=True):
    st.markdown(f"**{project['title']}**")
    st.caption(
        f"#{project['id']} · created {project['created_at'][:16].replace('T', ' ')} · "
        f"{project['page_count']} pages · {project['panel_count']} panels"
    )
    if project.get("error_message"):
        st.caption(project["error_message"])
    with st.container(horizontal=True):
        if st.button("Open", type="primary", icon=":material/arrow_forward:"):
            open_project(project["id"])
        confirm = st.checkbox("I want to delete this", key=f"del_ok_{project['id']}")
        if st.button("Delete", icon=":material/delete:", disabled=not confirm):
            try:
                api("DELETE", f"/api/projects/{project['id']}")
            except ApiError as exc:
                st.error(str(exc))
            else:
                if st.session_state.get("project_id") == project["id"]:
                    st.session_state.project_id = None
                    st.session_state.included_pages = None
                st.toast("Project deleted.", icon=":material/delete:")
                st.rerun()
