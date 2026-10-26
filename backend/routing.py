"""Which hosted model to try first, decided from what each one has actually done.

A fixed chain goes stale the moment a provider retires a model or gets busy. Measured on
this machine in October 2026: Gemini Flash answered in about 3 seconds, OpenRouter's free
tier returned 429, and NVIDIA's free endpoints returned 410 for most models and took 76
seconds for the one that worked. Any order written by hand encodes that one afternoon.

So the order is computed instead. Every call records whether it worked and how long it
took; the next call sorts the chain by that record. Three rules, and no more than three,
because a router nobody can predict is worse than a slow one:

1. **A failing endpoint is rested.** Consecutive failures push a cooldown out
   exponentially, so a dead model is skipped instead of being retried every time.
2. **Reliability outranks speed.** A fast endpoint that fails half the time is worse than
   a slower one that always answers.
3. **Unknowns get one chance.** Something never tried sorts just below the proven ones,
   so a new model gets used without displacing a working one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select

from .database import session_scope
from .models import ProviderStat, UsageEvent, utcnow

log = logging.getLogger("kage.routing")

#: How much a new measurement moves the running latency. Recent calls should dominate
#: without a single slow response rewriting history.
LATENCY_WEIGHT = 0.4
#: Backoff after repeated failures: 1 min, 4 min, 9 min ... capped.
COOLDOWN_BASE_SECONDS = 60
COOLDOWN_MAX_SECONDS = 30 * 60
#: Latency we assume for something never tried - optimistic, but not free.
UNKNOWN_LATENCY = 6.0


def _cooldown_for(consecutive_failures: int) -> timedelta:
    seconds = min(COOLDOWN_MAX_SECONDS, COOLDOWN_BASE_SECONDS * consecutive_failures ** 2)
    return timedelta(seconds=seconds)


def record(label: str, job: str, ok: bool, latency: float,
           error: str | None = None) -> None:
    """Fold one call into the running summary for that endpoint."""
    now = utcnow()
    try:
        with session_scope() as session:
            stat = session.scalar(select(ProviderStat).where(ProviderStat.label == label))
            if stat is None:
                # Column defaults are applied on INSERT, so a pending object still has
                # None for every counter. Spell them out rather than add to None.
                stat = ProviderStat(
                    label=label, job=job, ok_count=0, fail_count=0,
                    latency_seconds=0.0, consecutive_failures=0,
                )
                session.add(stat)
            stat.job = job or stat.job
            stat.last_used_at = now
            if ok:
                stat.ok_count = (stat.ok_count or 0) + 1
                stat.consecutive_failures = 0
                stat.cooldown_until = None
                stat.last_error = None
                previous = stat.latency_seconds or latency
                stat.latency_seconds = (
                    previous * (1 - LATENCY_WEIGHT) + latency * LATENCY_WEIGHT
                )
            else:
                stat.fail_count = (stat.fail_count or 0) + 1
                stat.consecutive_failures = (stat.consecutive_failures or 0) + 1
                stat.last_error = (error or "")[:400] or None
                stat.cooldown_until = now + _cooldown_for(stat.consecutive_failures)
    except Exception as exc:  # routing memory must never break a render
        log.debug("Could not record routing stat for %s: %s", label, exc)


def usage(project_id: int | None, job: str, provider: str, model: str, ok: bool,
          latency: float, prompt_tokens: int = 0, completion_tokens: int = 0,
          images: int = 0, error: str | None = None) -> None:
    """Log one hosted request so the app can show what it has spent."""
    try:
        with session_scope() as session:
            session.add(UsageEvent(
                project_id=project_id, job=job, provider=provider, model=model,
                ok=int(bool(ok)), latency_seconds=float(latency),
                prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens),
                images=int(images), error=(error or "")[:400] or None,
            ))
    except Exception as exc:
        log.debug("Could not record usage: %s", exc)


@dataclass(frozen=True)
class Snapshot:
    """A detached copy of one endpoint's record, safe to read after the session closes."""

    ok_count: int = 0
    fail_count: int = 0
    latency_seconds: float = 0.0
    cooldown_until: datetime | None = None


def _load(labels: list[str]) -> dict[str, Snapshot]:
    if not labels:
        return {}
    try:
        with session_scope() as session:
            rows = session.scalars(
                select(ProviderStat).where(ProviderStat.label.in_(labels))
            ).all()
            return {
                r.label: Snapshot(
                    ok_count=r.ok_count or 0,
                    fail_count=r.fail_count or 0,
                    latency_seconds=r.latency_seconds or 0.0,
                    cooldown_until=r.cooldown_until,
                )
                for r in rows
            }
    except Exception as exc:
        log.debug("Could not read routing stats: %s", exc)
        return {}


def score(stat: Snapshot | None, now: datetime) -> tuple[int, float]:
    """Sort key: lower is tried earlier. Returns (tier, cost).

    The tier keeps the three rules legible - resting endpoints can never jump ahead of
    working ones no matter how fast they once were.
    """
    if stat is None:
        return (1, UNKNOWN_LATENCY)                       # never tried
    resting = stat.cooldown_until is not None and stat.cooldown_until > now
    attempts = stat.ok_count + stat.fail_count
    if attempts == 0:
        return (1, UNKNOWN_LATENCY)
    success = stat.ok_count / attempts
    latency = stat.latency_seconds or UNKNOWN_LATENCY
    if resting:
        return (3, latency)                               # rested: last resort
    if success < 0.5:
        return (2, latency)                               # unreliable
    # Reliability outranks speed: a 50% endpoint is penalised more than a slow one.
    return (0, latency / max(0.05, success))


def order(labels: list[str], job: str = "") -> list[str]:
    """Reorder a chain by what each endpoint has actually done. Stable for ties."""
    del job
    now = utcnow()
    stats = _load(labels)
    ranked = sorted(
        enumerate(labels),
        key=lambda pair: (*score(stats.get(pair[1]), now), pair[0]),
    )
    return [label for _, label in ranked]


def health(labels: list[str] | None = None) -> list[dict]:
    """A readable view of the routing table, for the UI."""
    now = utcnow()
    try:
        with session_scope() as session:
            query = select(ProviderStat)
            if labels:
                query = query.where(ProviderStat.label.in_(labels))
            rows = session.scalars(query).all()
            out = []
            for r in rows:
                attempts = r.ok_count + r.fail_count
                out.append({
                    "label": r.label,
                    "job": r.job,
                    "calls": attempts,
                    "success": round(r.ok_count / attempts, 3) if attempts else None,
                    "latency_seconds": round(r.latency_seconds, 2),
                    "resting": bool(r.cooldown_until and r.cooldown_until > now),
                    "last_error": r.last_error,
                    "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
                })
            return sorted(out, key=lambda d: (-(d["calls"] or 0), d["label"]))
    except Exception as exc:
        log.debug("Could not read routing health: %s", exc)
        return []


def reset(labels: list[str] | None = None) -> None:
    """Forget what was learned, for ``labels`` or for everything.

    Pass labels to forget only those: a caller that is not the user asking to start over
    - a test, say - must not wipe real measurements it did not make.
    """
    try:
        with session_scope() as session:
            query = select(ProviderStat)
            if labels is not None:
                query = query.where(ProviderStat.label.in_(labels))
            for stat in session.scalars(query).all():
                session.delete(stat)
    except Exception as exc:
        log.debug("Could not reset routing stats: %s", exc)
