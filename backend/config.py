"""Central configuration: paths, binaries, render constants, capability probing."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


#: The one .env the app reads and writes. Named here so the key-settings endpoint and
#: the loader can never disagree about which file that is.
ENV_FILE = BASE_DIR / ".env"


def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Read KEY=VALUE pairs from .env into the environment.

    Kept dependency-free on purpose. Real environment variables win by default, so an
    explicit `set HF_TOKEN=...` still overrides the file.
    """
    path = path or ENV_FILE
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return loaded

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


DOTENV_VALUES = load_dotenv()

DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_ASSETS_DIR = ASSETS_DIR / "audio"
SFX_DIR = AUDIO_ASSETS_DIR / "sfx"
EFFECTS_DIR = ASSETS_DIR / "effects"
TOOLS_BIN = BASE_DIR / "tools" / "bin"
DB_PATH = DATA_DIR / "kage.db"
LEGACY_DB_PATH = DATA_DIR / "mangamotion.db"

for _d in (DATA_DIR, PROJECTS_DIR, ASSETS_DIR, AUDIO_ASSETS_DIR, SFX_DIR, EFFECTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- render specs

FPS = 24
SAMPLE_RATE = 44100

ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (900, 900),
}

GENRES = ("action", "horror", "romance", "thriller", "emotional_drama")
INTENSITIES = ("subtle", "balanced", "dramatic")
AUDIO_MOODS = ("orchestral", "ambient", "dark", "energetic", "suspenseful")
DURATIONS = (15, 30, 60)

MAX_UPLOAD_BYTES = 80 * 1024 * 1024
# A sheet holding a spread is split into two pages at import, so this counts roughly
# twice as fast as it used to. 80 keeps a book-length source from being truncated.
MAX_PAGES = 80
MAX_SOURCE_EDGE = 2600  # downscale huge scans so rendering stays responsive
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_PDF_SUFFIXES = {".pdf"}


# -------------------------------------------------------------- binary lookup


def _find_binary(name: str) -> str | None:
    """Prefer the FFmpeg bundled under tools/bin, then fall back to PATH."""
    local = TOOLS_BIN / f"{name}.exe"
    if local.exists():
        return str(local)
    local_posix = TOOLS_BIN / name
    if local_posix.exists():
        return str(local_posix)
    env_override = os.environ.get(f"{name.upper()}_BINARY")
    if env_override and Path(env_override).exists():
        return env_override
    return shutil.which(name)


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    return _find_binary("ffmpeg")


@lru_cache(maxsize=1)
def ffprobe_path() -> str | None:
    return _find_binary("ffprobe")


def require_ffmpeg() -> str:
    path = ffmpeg_path()
    if not path:
        raise RuntimeError(
            "FFmpeg was not found. Install it and put ffmpeg.exe on PATH, place it in "
            "tools/bin/, or set the FFMPEG_BINARY environment variable."
        )
    return path


# ------------------------------------------------------------- capability info


@dataclass
class Capabilities:
    ffmpeg: bool = False
    ffmpeg_version: str | None = None
    ffprobe: bool = False
    pymupdf: bool = False
    opencv: bool = False
    ocr: bool = False
    hosted_llm: bool = False
    vision: bool = False
    vision_device: str = "none"
    encoders: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ffmpeg": self.ffmpeg,
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe": self.ffprobe,
            "pdf_import": self.pymupdf,
            "opencv": self.opencv,
            "ocr_subtitles": self.ocr,
            "hosted_llm_director": self.hosted_llm,
            "vision_model": self.vision,
            "vision_device": self.vision_device,
            "video_encoders": self.encoders,
        }


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def hosted_llm_available() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN"))


@lru_cache(maxsize=1)
def probe_capabilities() -> Capabilities:
    caps = Capabilities()

    ff = ffmpeg_path()
    caps.ffmpeg = bool(ff)
    caps.ffprobe = bool(ffprobe_path())
    if ff:
        try:
            out = subprocess.run(
                [ff, "-version"], capture_output=True, text=True, timeout=20
            )
            caps.ffmpeg_version = out.stdout.splitlines()[0].strip() if out.stdout else None
            enc = subprocess.run(
                [ff, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            for wanted in ("libx264", "aac", "libopenh264", "mpeg4"):
                if wanted in enc.stdout:
                    caps.encoders.append(wanted)
        except Exception:
            pass

    try:
        import fitz  # noqa: F401

        caps.pymupdf = True
    except Exception:
        caps.pymupdf = False

    try:
        import cv2  # noqa: F401

        caps.opencv = True
    except Exception:
        caps.opencv = False

    caps.ocr = ocr_available()
    caps.hosted_llm = hosted_llm_available()
    try:
        from .vision import device_name, is_available

        caps.vision = is_available()
        caps.vision_device = device_name() if caps.vision else "none"
    except Exception:
        caps.vision = False
    return caps


def env(name: str, default: str | None = None) -> str | None:
    """Read KAGE_<NAME>, falling back to the old MANGAMOTION_<NAME> spelling."""
    return os.environ.get(f"KAGE_{name}") or os.environ.get(f"MANGAMOTION_{name}") or default


def adopt_legacy_database() -> None:
    """Carry projects over from the pre-rename database, once."""
    if DB_PATH.exists() or not LEGACY_DB_PATH.exists():
        return
    try:
        shutil.copy2(LEGACY_DB_PATH, DB_PATH)
    except OSError:
        pass


def project_dir(project_id: int) -> Path:
    """Per-project working directory with isolated sub-folders."""
    root = PROJECTS_DIR / f"project_{int(project_id):05d}"
    for sub in ("source", "pages", "panels", "thumbs", "render", "output", "temp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
