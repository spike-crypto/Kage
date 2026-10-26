"""What Kage has spent, against what each free tier allows."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.studio import ApiError, api

st.title("Usage", icon=":material/monitoring:")
st.caption(
    "Kage's own tally of what it sent. If you use the same key elsewhere the real figure "
    "is higher - this is a smoke alarm, not an invoice."
)

days = st.segmented_control(
    "Window", [7, 14, 30], default=14, format_func=lambda d: f"{d} days",
    selection_mode="single",
) or 14

try:
    data = api("GET", f"/api/usage?days={days}")
except ApiError as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()

budgets = data.get("budgets") or []
tracked = [b for b in budgets if b["requests"] or (b["allowance"] or 0) > 0]

st.space("small")
for budget in tracked:
    with st.container(border=True):
        head, figure = st.columns([3, 1], vertical_alignment="center")
        with head:
            st.markdown(f"**{budget['name']}**")
            st.caption(
                f"per {budget['window']} · {budget['requests']} request"
                f"{'s' if budget['requests'] != 1 else ''}"
                + (f", {budget['failed']} failed" if budget["failed"] else "")
                + (f" · {budget['tokens']:,} tokens" if budget["tokens"] else "")
                + (f" · {budget['seconds']:.0f}s GPU" if budget["seconds"] else "")
            )
        with figure:
            if budget["unit"] == "usd":
                st.metric("Spent", f"${budget['spent']:.2f}",
                          delta=f"${budget['remaining']:.2f} left"
                          if budget["remaining"] is not None else None,
                          delta_color="off", label_visibility="collapsed")
            else:
                st.metric("Requests", f"{int(budget['spent'])}",
                          label_visibility="collapsed")

        fraction = budget.get("used_fraction")
        if fraction is not None:
            limit = (f"${budget['allowance']:.0f}" if budget["unit"] == "usd"
                     else f"{int(budget['allowance'])}")
            st.progress(
                min(1.0, max(0.0, fraction)),
                text=f"{fraction * 100:.0f}% of {limit} per {budget['window']}",
            )
            if fraction >= 0.8:
                st.warning(
                    f"{budget['name']} is at {fraction * 100:.0f}% of its allowance.",
                    icon=":material/warning:",
                )
        elif budget["confidence"] == "unknown":
            st.caption("No published allowance to measure against - usage only.")

        with st.expander("Where this limit comes from", icon=":material/info:"):
            st.caption(f"**{budget['confidence'].title()}.** {budget['source']}")

if not tracked:
    with st.container(border=True):
        st.markdown("**Nothing spent yet**")
        st.caption("Write a narration or animate a shot and this fills in.")

# --- trend -------------------------------------------------------------------

history = data.get("history") or []
if history:
    st.space("medium")
    st.subheader("By day", divider=False)
    frame = pd.DataFrame(history)
    chart = frame.pivot_table(index="day", columns="provider", values="requests",
                              aggfunc="sum", fill_value=0).sort_index()
    st.bar_chart(chart, height=240,
                 alt="Requests per day, stacked by provider.")

    spend = frame[frame["usd"] > 0]
    if not spend.empty:
        st.caption(f"Paid compute: ${spend['usd'].sum():.2f} over {days} days.")

    st.dataframe(
        frame.rename(columns={"day": "Day", "provider": "Provider",
                              "requests": "Requests", "failed": "Failed",
                              "usd": "USD", "seconds": "Seconds"}),
        hide_index=True, width="stretch", height=260,
        alt="Requests, failures and spend per day and provider.",
        column_config={
            "USD": st.column_config.NumberColumn(format="$%.4f", width="small"),
            "Seconds": st.column_config.NumberColumn(format="%.0f s", width="small"),
        },
    )

with st.expander("Modal GPU rates", icon=":material/payments:"):
    rates = (data.get("rates") or {}).get("modal_usd_per_second") or {}
    st.caption("From modal.com/pricing. Cost is recorded per call, not recomputed later.")
    st.dataframe(
        pd.DataFrame(
            [{"GPU": k.upper(), "$/second": v, "$/hour": round(v * 3600, 2)}
             for k, v in rates.items()]
        ),
        hide_index=True, width="stretch",
        alt="Modal per-second GPU rates.",
        column_config={"$/second": st.column_config.NumberColumn(format="$%.6f")},
    )
