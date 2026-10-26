"""What each free tier allows, and how much of it has been spent.

Kage calls several metered services. None of them tell you how much is left in a way the
app can read, so this keeps its own count from ``usage_events`` and compares it against
the published allowance.

Two rules about the numbers below:

1. **Where an allowance came from is recorded.** Modal's rates and credit are from their
   own pricing page. The per-day request limits are from vendor docs or community
   reporting and change often, so each carries its ``source``. A number nobody can trace
   is a number nobody should trust.
2. **This is Kage's own tally, not the vendor's.** It counts what this app sent. If you
   use the same key elsewhere, the real figure is higher. It is a smoke alarm, not an
   invoice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from .database import session_scope
from .models import UsageEvent


@dataclass(frozen=True)
class Budget:
    """One metered service's free allowance."""

    key: str
    name: str
    #: "requests" or "usd"
    unit: str
    allowance: float
    #: "day" or "month"
    window: str
    source: str
    confidence: str = "published"   # or "reported" when it comes from the community


BUDGETS: dict[str, Budget] = {
    "gemini": Budget(
        key="gemini", name="Google Gemini", unit="requests", allowance=1500,
        window="day", confidence="reported",
        source="Community reporting, Oct 2026. Google publishes no fixed table; the "
               "live figure is in the AI Studio rate-limit console.",
    ),
    "openrouter": Budget(
        key="openrouter", name="OpenRouter", unit="requests", allowance=50,
        window="day", confidence="published",
        source="OpenRouter help centre: 50/day on a free account, 1000/day once $10 "
               "has ever been bought.",
    ),
    "nvidia": Budget(
        key="nvidia", name="NVIDIA NIM", unit="requests", allowance=40 * 60,
        window="day", confidence="reported",
        source="~40 requests/minute is the commonly reported default; NVIDIA publishes "
               "no daily cap. Treated as generous rather than tracked precisely.",
    ),
    "modal": Budget(
        key="modal", name="Modal", unit="usd", allowance=30.0,
        window="month", confidence="published",
        source="modal.com/pricing: Starter includes $30/month of credits.",
    ),
    "space": Budget(
        key="space", name="HF ZeroGPU", unit="requests", allowance=0,
        window="day", confidence="unknown",
        source="Hugging Face publishes no ZeroGPU figure for free accounts. Measured "
               "here: one 60s call succeeded, the next was refused with no message. "
               "Allowance left at 0 so the UI reports usage without a false limit.",
    ),
}

#: Modal GPU rates, dollars per second, from modal.com/pricing (October 2026).
MODAL_GPU_RATES: dict[str, float] = {
    "l4": 0.000222,
    "a10": 0.000306,
    "a100-40gb": 0.000583,
    "a100-80gb": 0.000694,
    "h100": 0.001097,
}


def modal_cost(gpu: str, seconds: float) -> float:
    """What a Modal call cost, from the GPU it ran on and how long it took."""
    rate = MODAL_GPU_RATES.get(str(gpu).strip().lower())
    if rate is None:
        return 0.0
    return round(max(0.0, float(seconds)) * rate, 6)


def _window_start(window: str, now: datetime) -> datetime:
    if window == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def report(now: datetime | None = None) -> list[dict]:
    """Spend against each allowance, for the UI.

    Counts failures too: a refused call still consumed a request against most limits,
    and a run of them is exactly what the reader needs to see.
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    out: list[dict] = []

    with session_scope() as session:
        for budget in BUDGETS.values():
            since = _window_start(budget.window, now)
            rows = session.scalars(
                select(UsageEvent).where(
                    UsageEvent.provider == budget.key,
                    UsageEvent.created_at >= since,
                )
            ).all()

            requests = len(rows)
            failed = sum(1 for r in rows if not r.ok)
            spent = (
                round(sum(r.cost_usd or 0.0 for r in rows), 4)
                if budget.unit == "usd" else float(requests)
            )
            allowance = budget.allowance
            used_fraction = (spent / allowance) if allowance > 0 else None

            out.append({
                "key": budget.key,
                "name": budget.name,
                "unit": budget.unit,
                "window": budget.window,
                "since": since.isoformat(),
                "allowance": allowance,
                "spent": spent,
                "requests": requests,
                "failed": failed,
                "remaining": round(allowance - spent, 4) if allowance > 0 else None,
                "used_fraction": round(used_fraction, 4) if used_fraction is not None else None,
                "seconds": round(sum(r.latency_seconds or 0.0 for r in rows), 1),
                "tokens": sum((r.prompt_tokens or 0) + (r.completion_tokens or 0)
                              for r in rows),
                "confidence": budget.confidence,
                "source": budget.source,
            })
    return out


def history(days: int = 14) -> list[dict]:
    """Requests and spend per day, so a trend is visible rather than only a total."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    since = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    buckets: dict[tuple[str, str], dict] = {}

    with session_scope() as session:
        rows = session.scalars(
            select(UsageEvent).where(UsageEvent.created_at >= since)
        ).all()
        for row in rows:
            day = (row.created_at or now).strftime("%Y-%m-%d")
            key = (day, row.provider or "?")
            slot = buckets.setdefault(key, {"day": day, "provider": row.provider or "?",
                                            "requests": 0, "failed": 0, "usd": 0.0,
                                            "seconds": 0.0})
            slot["requests"] += 1
            slot["failed"] += 0 if row.ok else 1
            slot["usd"] += row.cost_usd or 0.0
            slot["seconds"] += row.latency_seconds or 0.0

    return sorted(
        ({**v, "usd": round(v["usd"], 4), "seconds": round(v["seconds"], 1)}
         for v in buckets.values()),
        key=lambda d: (d["day"], d["provider"]),
        reverse=True,
    )
