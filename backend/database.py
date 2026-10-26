"""SQLite engine / session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .config import DB_PATH, adopt_legacy_database

adopt_legacy_database()

ENGINE = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
    future=True,
)


@event.listens_for(ENGINE, "connect")
def _set_sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - driver hook
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they do not exist yet, then apply lightweight migrations."""
    from .models import Base  # local import to avoid a cycle

    Base.metadata.create_all(ENGINE)
    _migrate()


def _migrate() -> None:
    """Additive, idempotent schema updates.

    SQLAlchemy's create_all never alters an existing table, so a column added after a
    database already exists has to be introduced by hand. Each step checks first, so this
    is safe to run on every start.
    """
    from sqlalchemy import text

    from .styles import LEGACY_GENRE_TO_STYLE

    with ENGINE.begin() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(projects)"))}
        if "projects" not in {
            row[0] for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
        }:
            return

        if "style" not in columns:
            conn.execute(text(
                "ALTER TABLE projects ADD COLUMN style VARCHAR(40) DEFAULT 'kage_noir'"
            ))
            # Give pre-existing projects the style closest to their old genre.
            for genre, style in LEGACY_GENRE_TO_STYLE.items():
                conn.execute(
                    text("UPDATE projects SET style = :style WHERE genre = :genre"),
                    {"style": style, "genre": genre},
                )
            conn.execute(text(
                "UPDATE projects SET style = 'kage_noir' WHERE style IS NULL OR style = ''"
            ))

        if "narration_enabled" not in columns:
            conn.execute(text(
                "ALTER TABLE projects ADD COLUMN narration_enabled INTEGER DEFAULT 0"
            ))

        shot_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(shots)"))}
        if shot_columns and "narration_text" not in shot_columns:
            conn.execute(text("ALTER TABLE shots ADD COLUMN narration_text TEXT"))


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Used by background render threads."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
