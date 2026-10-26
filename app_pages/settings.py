"""Providers: bring your own key, see which endpoint Kage is choosing, and what it costs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.studio import ApiError, api, fetch_health

st.title("Providers", icon=":material/key:")
st.caption(
    "Kage writes narration on a hosted model. Any one key below is enough - a provider "
    "with no key is skipped, never failed."
)

try:
    info = api("GET", "/api/providers")
except ApiError as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()

# --- your keys --------------------------------------------------------------

st.subheader("Your API keys", divider=False)
st.caption(
    "Keys are written to the project's `.env`, which is git-ignored, and are never sent "
    "anywhere except the provider they belong to. Kage never displays a key back to you."
)

for provider in info["providers"]:
    with st.container(border=True):
        head, state = st.columns([3, 1], vertical_alignment="center")
        with head:
            st.markdown(f"**{provider['name']}**")
            st.caption(provider["free"])
        with state:
            if provider["configured"]:
                st.badge(
                    "Your key" if provider["source"] == "user" else "Configured",
                    icon=":material/check_circle:", color="green",
                )
            else:
                st.badge("Not set", icon=":material/remove:", color="gray")

        entry, actions = st.columns([3, 1], vertical_alignment="bottom")
        typed = entry.text_input(
            f"{provider['name']} API key", type="password",
            placeholder="Paste a key to set or replace it",
            key=f"key_{provider['key']}", label_visibility="collapsed",
        )
        with actions, st.container(horizontal=True):
            if st.button("Save", key=f"save_{provider['key']}", disabled=not typed,
                         icon=":material/save:"):
                try:
                    api("PUT", f"/api/providers/{provider['key']}/key",
                        json={"api_key": typed})
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    fetch_health.clear()
                    st.toast(f"{provider['name']} key saved.",
                             icon=":material/check_circle:")
                    st.rerun()
            if provider["configured"] and st.button(
                "Clear", key=f"clear_{provider['key']}", icon=":material/delete:"
            ):
                try:
                    api("PUT", f"/api/providers/{provider['key']}/key",
                        json={"api_key": ""})
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    fetch_health.clear()
                    st.rerun()
        st.caption(f"Environment variable `{provider['env']}` · [get a key]({provider['signup']})")

# --- what Kage is choosing, and why -----------------------------------------

st.space("medium")
st.subheader("How Kage is choosing", divider=False)
st.caption(
    "The order is measured, not fixed. Every call records whether it worked and how long "
    "it took; a repeatedly failing endpoint is rested with a growing backoff, and "
    "reliability outranks speed."
)

order = info.get("order", {})
a, b = st.columns(2)
for column, (job, label) in zip((a, b), (("vision", "Reading pages"),
                                         ("text", "Writing narration"))):
    with column, st.container(border=True):
        st.markdown(f"**{label}**")
        picks = order.get(job) or []
        if not picks:
            st.caption("No provider configured yet.")
        for n, pick in enumerate(picks[:5], start=1):
            st.caption(f"{n}. {pick}" + ("  ← tried first" if n == 1 else ""))

health = info.get("health") or []
if health:
    st.dataframe(
        pd.DataFrame([
            {
                "Endpoint": h["label"],
                "Job": h["job"],
                "Calls": h["calls"],
                "Success": h["success"],
                "Latency": h["latency_seconds"],
                "Resting": h["resting"],
                "Last error": (h["last_error"] or "")[:90],
            }
            for h in health
        ]),
        hide_index=True, width="stretch",
        alt="Measured behaviour of each hosted endpoint.",
        column_config={
            "Success": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent", width="small"),
            "Latency": st.column_config.NumberColumn(format="%.1f s", width="small"),
            "Resting": st.column_config.CheckboxColumn(
                width="small", help="Backing off after repeated failures."),
        },
    )
    if st.button("Forget what was measured", icon=":material/restart_alt:",
                 help="Start choosing from scratch - useful after fixing a key."):
        try:
            api("POST", "/api/providers/reset")
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.toast("Routing reset.", icon=":material/restart_alt:")
            st.rerun()
else:
    st.caption("Nothing measured yet. Write a narration and this fills in.")

# --- usage -------------------------------------------------------------------

st.space("medium")
st.subheader("Usage", divider=False)
totals = info.get("totals") or {}
per_film = info.get("requests_per_film") or {}
st.caption(
    f"Kage sends one request per {per_film.get('images_per_request', 12)} pages it reads, "
    "plus one to write the whole script - two for a typical film, not two per shot."
)

m = st.columns(4)
m[0].metric("Requests", totals.get("requests", 0), border=True, icon=":material/send:")
m[1].metric("Failed", totals.get("failed", 0), border=True,
            icon=":material/error:" if totals.get("failed") else ":material/check_circle:")
m[2].metric("Tokens",
            f"{totals.get('prompt_tokens', 0) + totals.get('completion_tokens', 0):,}",
            border=True, icon=":material/toll:")
m[3].metric("Median latency",
            f"{totals['median_latency']:.1f}s" if totals.get("median_latency") else "—",
            border=True, icon=":material/timer:")

rows = info.get("usage") or []
if rows:
    st.dataframe(
        pd.DataFrame([
            {
                "When": (r["created_at"] or "")[:19].replace("T", " "),
                "Project": r["project_id"],
                "Job": r["job"],
                "Model": f"{r['provider']}:{r['model']}",
                "OK": r["ok"],
                "Seconds": r["latency_seconds"],
                "Images": r["images"],
                "Tokens": r["prompt_tokens"] + r["completion_tokens"],
                "Error": (r["error"] or "")[:70],
            }
            for r in rows[:200]
        ]),
        hide_index=True, width="stretch", height=320,
        alt="Every hosted request Kage has made, newest first.",
        column_config={
            "OK": st.column_config.CheckboxColumn(width="small"),
            "Seconds": st.column_config.NumberColumn(format="%.1f", width="small"),
        },
    )
else:
    st.caption("No hosted requests yet.")
