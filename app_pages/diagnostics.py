"""The backend log, so a failure is visible here rather than only in a file."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.studio import ApiError, api

SOURCES = {"buffer": "Live", "errors": "Errors only", "main": "Full log file"}

st.title("Diagnostics", icon=":material/monitor_heart:")
st.caption("What the backend has been doing, newest last.")

controls, refresh = st.columns([3, 1], vertical_alignment="bottom")
with controls, st.container(horizontal=True, vertical_alignment="bottom"):
    source = st.segmented_control(
        "Source", list(SOURCES), format_func=SOURCES.get, default="buffer",
        selection_mode="single",
    ) or "buffer"
    level = st.selectbox("From level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    limit = st.number_input("Records", min_value=20, max_value=500, value=120, step=20)
with refresh, st.container(horizontal=True, horizontal_alignment="right"):
    if st.button("Refresh", icon=":material/refresh:"):
        st.rerun()

try:
    payload = api("GET", f"/api/logs?limit={int(limit)}&level={level}&source={source}")
except ApiError as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()

if source in ("errors", "main"):
    lines = payload.get("lines", [])
    st.caption(f"Tailing {payload.get('source')}")
    if not lines:
        st.success("That file is empty - nothing has been written to it yet.",
                   icon=":material/check_circle:")
    else:
        st.code("\n".join(lines), language="log")
    st.stop()

errors = payload.get("error_count", 0)
warnings = payload.get("warning_count", 0)
a, b, c = st.columns(3)
a.metric("Errors", errors, border=True,
         icon=":material/error:" if errors else ":material/check_circle:")
b.metric("Warnings", warnings, border=True,
         icon=":material/warning:" if warnings else ":material/check_circle:")
c.metric("Shown", payload.get("count", 0), border=True, icon=":material/list:")

records = payload.get("records", [])
if not records:
    with st.container(border=True):
        st.markdown("**Nothing logged yet**")
        st.caption("Run an import or a render and anything that goes wrong appears here.")
    st.stop()

st.space("small")
st.dataframe(
    pd.DataFrame([
        {
            "Time": record["time"][11:19],
            "Level": record["level"],
            "Where": f"{record['logger']}:{record['line']}",
            "Message": record["message"][:600],
        }
        for record in records
    ]),
    hide_index=True,
    width="stretch",
    height=420,
    alt="Backend log records, with their level, source and message.",
    column_config={
        "Time": st.column_config.TextColumn(width="small"),
        "Level": st.column_config.TextColumn(width="small"),
        "Where": st.column_config.TextColumn(width="small"),
        "Message": st.column_config.TextColumn(width="large"),
    },
)

tracebacks = [r for r in records if r.get("traceback")]
if tracebacks:
    st.space("small")
    st.subheader("Tracebacks", divider=False)
    for record in tracebacks:
        with st.expander(f"{record['time'][11:19]} · {record['message'][:80]}",
                         icon=":material/bug_report:"):
            st.code(record["traceback"], language="python")

files = payload.get("log_files", {})
st.caption(f"Files: {files.get('main', '?')}  ·  {files.get('errors', '?')}")
