"""Step 2: the creative decisions, then build the shot plan."""

from __future__ import annotations

import streamlit as st

from frontend.studio import (
    ASPECT_HELP,
    ASPECTS,
    DURATIONS,
    INTENSITIES,
    INTENSITY_HELP,
    MOODS,
    STYLE_DESCRIPTIONS,
    STYLE_TAGLINES,
    STYLES,
    ApiError,
    api,
    current_project,
    fetch_health,
    progress_strip,
    project_bar,
    require,
)

project = current_project()
st.title("Choose a look", icon=":material/palette:")
progress_strip(project, "style")
project_bar(project)

if not require("style", project):
    st.stop()

# Widgets are live rather than wrapped in a form: picking a style should show what that
# style is before you commit to it, and nothing here costs an API call until you save.

st.space("small")
st.subheader("Style", divider=False)
current_style = project.get("style") or project.get("genre") or "kage_noir"
if current_style not in STYLES:
    current_style = "kage_noir"
style = st.segmented_control(
    "Style", list(STYLES), format_func=STYLES.get, default=current_style,
    selection_mode="single", label_visibility="collapsed", width="stretch",
) or current_style
with st.container(border=True):
    st.markdown(f"**{STYLE_TAGLINES[style]}**")
    st.caption(STYLE_DESCRIPTIONS[style])

st.space("small")
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Shape and length", divider=False)
    title = st.text_input("Project name", value=project["title"])
    duration = st.segmented_control(
        "Length", DURATIONS, default=project["target_duration"],
        format_func=lambda s: f"{s} seconds", selection_mode="single", width="stretch",
    ) or project["target_duration"]
    aspect = st.segmented_control(
        "Shape", list(ASPECTS),
        default=project["aspect_ratio"] if project["aspect_ratio"] in ASPECTS else "16:9",
        format_func=lambda a: f"{a} {ASPECTS[a]}", selection_mode="single", width="stretch",
    ) or project["aspect_ratio"]
    st.caption(ASPECT_HELP.get(aspect, ""))

with right:
    st.subheader("Feel", divider=False)
    intensity = st.segmented_control(
        "Intensity", list(INTENSITIES), format_func=INTENSITIES.get,
        default=project["intensity"] if project["intensity"] in INTENSITIES else "balanced",
        selection_mode="single", width="stretch",
    ) or project["intensity"]
    st.caption(INTENSITY_HELP.get(intensity, ""))
    mood = st.selectbox(
        "Score", list(MOODS), format_func=MOODS.get,
        index=list(MOODS).index(project["audio_mood"])
        if project["audio_mood"] in MOODS else 0,
        help="Music bed and ambience. Each style already suggests one.",
    )
    use_vision = st.toggle(
        "Let the vision model pick the pages", value=True,
        help="A local CLIP model reads every page, ranks them by how much artwork they "
             "hold, and skips covers, blank pages and pages of pure text. Without it, "
             "pages are filmed in reading order.",
    )

st.space("small")
st.subheader("Voiceover", divider=False)
voice = (fetch_health() or {}).get("capabilities", {}).get("narration", {})
if voice.get("ready"):
    write_narration = st.toggle(
        "Write and speak a narration", value=bool(project.get("narration_enabled")),
        help="A vision model reads every page, a language model writes one line per shot "
             "inside a word budget set by that shot's length, and a local voice speaks it "
             "over the score.",
    )
    pages_per_request = voice.get("images_per_request", 12)
    shots = len(project.get("shots") or []) or 8
    st.caption(
        f"Writing runs on {', '.join(voice['providers'])}; the voice is "
        f"{voice['voice']} ({voice['voice_licence']}), on this machine. "
        f"This costs about {max(1, -(-shots // pages_per_request)) + 1} hosted requests "
        "and adds roughly 18s to the render."
    )
    st.page_link("app_pages/settings.py", label="Providers and usage",
                 icon=":material/key:")
else:
    write_narration = False
    st.caption("Voiceover needs a free inference key - any one provider is enough.")
    st.page_link("app_pages/settings.py", label="Add a provider key",
                 icon=":material/key:")

st.space("small")
st.subheader("Animation", divider=False)
st.caption(
    "Every shot gets depth parallax, which moves the artwork locally. A few shots can "
    "also be sent to a GPU Space for true image-to-video - generated motion rather than "
    "a warp. It is slow and quota-limited, so it is for the moments that matter."
)
animate_shots = st.slider(
    "Hero shots to animate", 0, 6, 0,
    help="0 keeps every shot on parallax. Each animated shot is one call to the "
         "ZeroGPU Space and takes about a minute. A shot whose clip fails keeps its "
         "still, so this never costs you the render.",
)
if animate_shots:
    st.caption(
        f"About {animate_shots} minute(s) added at analysis time. Free ZeroGPU has a "
        "small daily allowance; when it runs out the remaining shots stay on parallax."
    )

with st.expander("Advanced", icon=":material/tune:"):
    subtitles = st.checkbox(
        "Burn in subtitles read from the page", value=bool(project["subtitles_enabled"]),
        help="Needs Tesseract OCR installed. Only text actually read from the page is "
             "used; nothing is invented.",
    )
    detect_panels = st.checkbox(
        "Try to split pages into panels", value=False,
        help="Best effort. Only confident, mostly-rectangular layouts are accepted; "
             "anything else stays a whole page.",
    )
    hosted = st.checkbox(
        "Use the hosted AI director when available", value=False,
        help="Needs a Hugging Face token with inference credit. Falls back to the local "
             "planner automatically.",
    )

st.space("medium")
if st.button("Save and build the shot plan", type="primary", icon=":material/auto_awesome:",
             width="stretch"):
    try:
        api("PATCH", f"/api/projects/{project['id']}", json={
            "title": title, "style": style, "target_duration": duration,
            "aspect_ratio": aspect, "audio_mood": mood, "intensity": intensity,
            "subtitles_enabled": subtitles,
            "narration_enabled": write_narration,
        })
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

    included = sorted(
        st.session_state.get("included_pages")
        or [p["page_index"] for p in project["pages"] if p["kind"] == "page"]
    )
    jobs = ["Directing the sequence"]
    if use_vision:
        jobs.insert(0, "Reading every page")
    if write_narration:
        jobs.append("writing the narration")
    if animate_shots:
        jobs.append(f"animating {animate_shots} shot(s)")
    spinner = ", ".join(jobs) + "..."
    with st.spinner(spinner):
        try:
            updated = api("POST", f"/api/projects/{project['id']}/analyze", json={
                "included_pages": included,
                "detect_panels": detect_panels,
                "use_hosted_director": hosted,
                "extract_subtitles": subtitles,
                "use_vision": use_vision,
                "write_narration": write_narration,
                "animate_shots": int(animate_shots),
            }, timeout=900)
        except ApiError as exc:
            st.error(str(exc))
            st.stop()

    st.toast(f"{len(updated['shots'])} shots planned.", icon=":material/check_circle:")
    if updated.get("error_message"):
        st.warning(updated["error_message"], icon=":material/warning:")
    st.switch_page("app_pages/shots.py")
