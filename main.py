"""Entry point for the Kage FastAPI backend.

Run with:  python main.py          (or)  uvicorn backend.api:app --reload
"""

from __future__ import annotations

import os

import uvicorn

from backend.audio_engine import ensure_sfx_library
from backend.config import env, probe_capabilities
from backend.database import init_db
from backend.logging_config import ERROR_LOG, MAIN_LOG, setup_logging


def bootstrap() -> None:
    setup_logging()
    init_db()
    caps = probe_capabilities()
    print("Kage - From Ink to Imagination")
    print(f"  FFmpeg        : {caps.ffmpeg_version or 'NOT FOUND'}")
    print(f"  PDF import    : {'yes' if caps.pymupdf else 'no (install pymupdf)'}")
    print(f"  OpenCV        : {'yes' if caps.opencv else 'no'}")
    print(f"  OCR subtitles : {'yes' if caps.ocr else 'no (optional)'}")
    print(f"  Hosted LLM    : {'yes' if caps.hosted_llm else 'no (local planner)'}")
    library = ensure_sfx_library()
    print(f"  Sound library : {len(library)} cues ready")
    print(f"  Log file      : {MAIN_LOG}")
    print(f"  Error log     : {ERROR_LOG}")


if __name__ == "__main__":
    bootstrap()
    uvicorn.run(
        "backend.api:app",
        host=env("HOST", "127.0.0.1"),
        port=int(env("PORT", "8000")),
        reload=bool(env("RELOAD")),
        log_level="info",
    )
