"""Step 1: bring artwork in and choose which pages are in the running."""

from __future__ import annotations

import streamlit as st

from frontend.studio import (
    SAMPLE_DIR,
    ApiError,
    api,
    current_project,
    existing_file,
    progress_strip,
    project_bar,
    require,
)

project = current_project()
st.title("Add pages", icon=":material/upload_file:")
progress_strip(project, "import")
project_bar(project)

if not require("import", project):
    st.stop()

st.space("small")
st.subheader("Bring in artwork", divider=False)

uploaded = st.file_uploader(
    "PDF or page images",
    type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    help="Up to 80 MB and 80 pages in one go. Uploading REPLACES everything already "
         "imported, so send a book in a single selection rather than in batches.",
)
st.caption(
    "Scans of a book are cleaned up on the way in: the camera background and margins are "
    "trimmed off, and a two-page spread becomes two pages. Select every page at once - a "
    "second upload replaces the first rather than adding to it."
)

sample_files = sorted(SAMPLE_DIR.glob("sample_page_*.png")) if SAMPLE_DIR.exists() else []
with st.container(horizontal=True):
    do_upload = st.button(
        "Upload and extract", type="primary", icon=":material/upload:",
        disabled=not uploaded,
    )
    do_sample = (
        st.button(f"Use {len(sample_files)} sample pages", icon=":material/science:")
        if sample_files else False
    )

if do_upload or do_sample:
    if do_upload:
        payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                   for f in uploaded]
        message = "Reading pages, trimming scans and splitting spreads..."
    else:
        payload = [("files", (p.name, p.read_bytes(), "image/png")) for p in sample_files]
        message = "Importing sample artwork..."
    with st.spinner(message):
        try:
            result = api("POST", f"/api/projects/{project['id']}/upload",
                         files=payload, timeout=600)
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state.included_pages = None
            for warning in result.get("warnings", []):
                st.warning(warning, icon=":material/warning:")
            st.toast(f"Imported {result['page_count']} page(s).",
                     icon=":material/check_circle:")
            st.rerun()

pages = [p for p in project.get("pages", []) if p["kind"] == "page"]
if not pages:
    st.space("small")
    with st.container(border=True):
        st.markdown("**No pages yet**")
        st.caption(
            "Upload a PDF or images above. Generated sample artwork is there for a quick "
            "test run - no third-party manga ships with Kage."
        )
    st.stop()

# --- choose what makes the cut ---------------------------------------------

st.space("medium")
chosen = st.session_state.get("included_pages")
if chosen is None:
    chosen = {p["page_index"] for p in pages if p["included"]}

head, tools = st.columns([2, 1], vertical_alignment="bottom")
with head:
    st.subheader(f"{len(pages)} pages imported", divider=False)
    st.caption(
        f"{len(chosen)} selected. Untick anything you do not want filmed - though the "
        "vision model does this for you in the next step."
    )
with tools, st.container(horizontal=True, horizontal_alignment="right"):
    if st.button("Select all", icon=":material/select_all:"):
        st.session_state.included_pages = {p["page_index"] for p in pages}
        st.rerun()
    if st.button("Clear", icon=":material/deselect:"):
        st.session_state.included_pages = set()
        st.rerun()

selection: set[int] = set()
columns = st.columns(6, gap="small")
for n, page in enumerate(pages):
    with columns[n % 6]:
        thumb = existing_file(page.get("thumb_path") or page.get("image_path"))
        if thumb:
            st.image(str(thumb), width="stretch",
                     alt=f"Page {page['page_index'] + 1} of the imported artwork")
        keep = st.checkbox(
            f"Page {page['page_index'] + 1}",
            value=page["page_index"] in chosen,
            key=f"pg_{project['id']}_{page['page_index']}",
        )
        st.caption(f"{page['width']}×{page['height']}")
        if keep:
            selection.add(page["page_index"])

st.session_state.included_pages = selection

st.space("medium")
if st.button(
    "Continue to the look", type="primary", icon=":material/arrow_forward:",
    disabled=not selection,
    help=None if selection else "Select at least one page.",
):
    st.switch_page("app_pages/style.py")
