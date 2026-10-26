"""Pre-flight environment check. Run:  python scripts/check_environment.py"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import ffmpeg_path, ffprobe_path, probe_capabilities  # noqa: E402

REQUIRED = ["fastapi", "uvicorn", "streamlit", "PIL", "numpy", "cv2", "sqlalchemy", "requests"]
OPTIONAL = ["fitz", "pytesseract", "pytest", "httpx"]


def main() -> int:
    print(f"Python  : {sys.version.split()[0]} ({sys.executable})")
    failures: list[str] = []

    print("\nRequired packages")
    for name in REQUIRED:
        try:
            importlib.import_module(name)
            print(f"  [ ok ] {name}")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failures.append(name)

    print("\nOptional packages")
    for name in OPTIONAL:
        try:
            importlib.import_module(name)
            print(f"  [ ok ] {name}")
        except Exception:
            print(f"  [skip] {name} (not installed)")

    print("\nBinaries")
    ff, probe = ffmpeg_path(), ffprobe_path()
    print(f"  ffmpeg  : {ff or 'NOT FOUND'}")
    print(f"  ffprobe : {probe or 'NOT FOUND'}")
    if not ff:
        failures.append("ffmpeg")

    caps = probe_capabilities()
    print("\nCapabilities")
    for key, value in caps.as_dict().items():
        print(f"  {key:22s}: {value}")

    if failures:
        print(f"\nFAILED: missing {', '.join(failures)}")
        print("Install Python deps with:  pip install -r requirements.txt")
        print("FFmpeg: put ffmpeg.exe and ffprobe.exe in tools/bin/ or on PATH.")
        return 1

    print("\nAll required components are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
