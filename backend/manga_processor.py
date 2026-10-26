"""Manga import: PDF page extraction, image validation, thumbnails, panel segmentation.

Everything here is deterministic image processing. Panel segmentation is best-effort and
is never required for a successful render: callers fall back to whole pages.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import (
    ALLOWED_IMAGE_SUFFIXES,
    ALLOWED_PDF_SUFFIXES,
    MAX_PAGES,
    MAX_SOURCE_EDGE,
    ocr_available,
    project_dir,
)

Image.MAX_IMAGE_PIXELS = 300_000_000  # allow big scans, still guards decompression bombs

THUMB_WIDTH = 260
MIN_EDGE = 80


class MangaImportError(ValueError):
    """Raised for user-correctable import problems."""


@dataclass
class ImportedPage:
    page_index: int
    kind: str
    image_path: Path
    thumb_path: Path
    width: int
    height: int
    parent_page_index: int | None = None


# ----------------------------------------------------------------- file names


def sanitize_filename(name: str, fallback: str = "upload") -> str:
    """Strip directories and anything that is not a safe filename character."""
    name = Path(str(name or "")).name
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if not name:
        name = fallback
    stem, dot, suffix = name.rpartition(".")
    if dot:
        name = f"{stem[:80]}.{suffix[:10]}"
    else:
        name = name[:80]
    return name


def classify_upload(filename: str) -> str:
    suffix = Path(sanitize_filename(filename)).suffix.lower()
    if suffix in ALLOWED_PDF_SUFFIXES:
        return "pdf"
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        return "image"
    raise MangaImportError(
        f"Unsupported file type '{suffix or filename}'. Upload a PDF or "
        f"{', '.join(sorted(s.lstrip('.') for s in ALLOWED_IMAGE_SUFFIXES))} images."
    )


# ------------------------------------------------------------- normalisation


def _normalise(img: Image.Image) -> Image.Image:
    """Apply EXIF rotation, flatten to RGB, bound the long edge."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        canvas.alpha_composite(img)
        img = canvas.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > MAX_SOURCE_EDGE:
        scale = MAX_SOURCE_EDGE / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


def _save_page(img: Image.Image, pages_dir: Path, thumbs_dir: Path, index: int,
               prefix: str = "page") -> tuple[Path, Path, int, int]:
    image_path = pages_dir / f"{prefix}_{index:03d}.png"
    img.save(image_path, "PNG", optimize=False)

    thumb = img.copy()
    ratio = THUMB_WIDTH / thumb.width
    thumb = thumb.resize(
        (THUMB_WIDTH, max(1, round(thumb.height * ratio))), Image.LANCZOS
    )
    thumb_path = thumbs_dir / f"{prefix}_{index:03d}.jpg"
    thumb.save(thumb_path, "JPEG", quality=84)
    return image_path, thumb_path, img.width, img.height


# -------------------------------------------------------- sheet preparation

# A photographed book carries things that are not artwork: the camera's dark
# background, the edge of the paper block, and wide blank margins. Filming those is what
# makes a shot read as a photo of a book instead of a frame of film, so they come off
# once, at import, rather than being fought by the camera on every shot.

PAPER_MIN_AREA = 0.25     # a detected sheet smaller than this is not trusted
TRIM_MIN_KEEP = 0.25      # never crop an edge down past this fraction of the sheet
TRIM_MAX_EDGE = 0.18      # margin pass may take at most this much off any one edge
TRIM_MARGIN = 0.012       # breathing room left around the ink
SPREAD_ASPECT = 1.18      # wider than this and the sheet holds two facing pages
_PROFILE_EDGE = 480       # long edge used for the profile pass


def _span(profile: np.ndarray, threshold: float, min_run: int) -> tuple[int, int] | None:
    """First and last index covered by runs of >= ``min_run`` samples above ``threshold``.

    Everything between the first and last run is kept, so the shadow down the gutter of a
    spread does not split the sheet in two - but a dark band at an *edge* is excluded,
    which is how the gutter comes off each half after the split.
    """
    hits = profile > threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(hits):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(hits) - start >= min_run:
        runs.append((start, len(hits) - 1))
    if not runs:
        return None
    return runs[0][0], runs[-1][1]


def trim_sheet(img: Image.Image) -> Image.Image:
    """Crop a sheet down to the printed artwork.

    Two passes: find the paper (everything bright enough not to be the surround), then
    find the ink inside it. Both are guarded - on any doubt the original comes back
    untouched, because a bad crop is far worse than an untrimmed page.
    """
    w, h = img.size
    if min(w, h) < MIN_EDGE * 2:
        return img

    scale = min(1.0, _PROFILE_EDGE / max(w, h))
    small = img.convert("L")
    if scale < 1.0:
        small = small.resize((max(8, round(w * scale)), max(8, round(h * scale))), Image.BILINEAR)
    gray = np.asarray(small, dtype=np.float32)
    sh, sw = gray.shape

    dark, bright = float(np.percentile(gray, 4)), float(np.percentile(gray, 96))
    if bright - dark < 25.0:
        return img  # flat image - nothing to separate paper from

    # Pass 1: the sheet.
    paper = gray > (dark + bright) / 2.0
    cols = _span(paper.mean(axis=0), 0.45, max(3, sw // 40))
    rows = _span(paper.mean(axis=1), 0.45, max(3, sh // 40))
    if cols is None or rows is None:
        return img
    x0, x1 = cols
    y0, y1 = rows
    if (x1 - x0 + 1) * (y1 - y0 + 1) < sw * sh * PAPER_MIN_AREA:
        return img

    # Pass 2: the ink inside the sheet.
    # The ink threshold is relative to this sheet's own contrast: a faintly printed or
    # sun-faded page still has ink, it is just closer to the paper than 45 levels.
    block = gray[y0 : y1 + 1, x0 : x1 + 1]
    bh, bw = block.shape
    level = float(np.median(block))
    floor = float(np.percentile(block, 2))
    ink = block < level - max(16.0, (level - floor) * 0.30)
    icols = _span(ink.mean(axis=0), 0.004, max(2, bw // 80))
    irows = _span(ink.mean(axis=1), 0.004, max(2, bh // 80))
    if icols and irows:
        ix0, ix1 = icols
        iy0, iy1 = irows
        wide_enough = (ix1 - ix0 + 1) >= bw * TRIM_MIN_KEEP
        tall_enough = (iy1 - iy0 + 1) >= bh * TRIM_MIN_KEEP
        if wide_enough and tall_enough:
            mx, my = int(bw * TRIM_MARGIN), int(bh * TRIM_MARGIN)
            # Margins are a few percent of a sheet. Capping how much this pass may take
            # means a page whose ink reads weakly loses its margin, never its artwork.
            cap_x, cap_y = int(bw * TRIM_MAX_EDGE), int(bh * TRIM_MAX_EDGE)
            x1 = x0 + min(bw - 1, max(ix1 + mx, bw - 1 - cap_x))
            y1 = y0 + min(bh - 1, max(iy1 + my, bh - 1 - cap_y))
            x0 = x0 + max(0, min(ix0 - mx, cap_x))
            y0 = y0 + max(0, min(iy0 - my, cap_y))

    box = (
        int(x0 / scale), int(y0 / scale),
        min(w, int(round((x1 + 1) / scale))), min(h, int(round((y1 + 1) / scale))),
    )
    if box[2] - box[0] < MIN_EDGE or box[3] - box[1] < MIN_EDGE:
        return img
    if box == (0, 0, w, h):
        return img
    return img.crop(box)


def split_spread(img: Image.Image) -> list[Image.Image]:
    """Split a sheet holding two facing pages into single pages.

    A camera move across a spread shows half of one drawing, the gutter, then half of
    another; no single composition ever owns the frame. Right half first, matching the
    right-to-left reading order ``detect_panels`` already uses for panels within a page.
    """
    w, h = img.size
    if w < h * SPREAD_ASPECT:
        return [img]
    half = w // 2
    return [img.crop((half, 0, w, h)), img.crop((0, 0, half, h))]


def prepare_sheet(img: Image.Image) -> list[Image.Image]:
    """Turn one imported sheet into the page images Kage will actually film."""
    sheet = trim_sheet(img)
    halves = split_spread(sheet)
    if len(halves) == 1:
        return [sheet]
    # Each half is trimmed again: this is what removes the gutter shadow left on the
    # inner edge, and the margins that pass 1 could not reach while the sheet was whole.
    return [trim_sheet(half) for half in halves]


# ------------------------------------------------------------------- importers


def extract_pdf_pages(pdf_path: Path, project_id: int, start_index: int = 0,
                      dpi: int = 160, prepare: bool = True) -> list[ImportedPage]:
    """Rasterise PDF pages to PNG. Requires PyMuPDF.

    One PDF page can yield two Kage pages when it holds a spread, so the returned
    indices are assigned here rather than tracking the PDF's own numbering.
    """
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MangaImportError(
            "PDF import needs PyMuPDF. Install it with: pip install pymupdf"
        ) from exc

    root = project_dir(project_id)
    pages_dir, thumbs_dir = root / "pages", root / "thumbs"
    results: list[ImportedPage] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise MangaImportError(f"Could not open the PDF: {exc}") from exc

    with doc:
        if doc.page_count == 0:
            raise MangaImportError("That PDF has no pages.")
        if doc.needs_pass:
            raise MangaImportError("That PDF is password protected.")

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        idx = start_index

        for n in range(doc.page_count):
            if idx >= MAX_PAGES:
                break
            page = doc.load_page(n)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = _normalise(img)
            sheets = prepare_sheet(img) if prepare else [img]
            for sheet in sheets:
                if idx >= MAX_PAGES:
                    break
                image_path, thumb_path, w, h = _save_page(sheet, pages_dir, thumbs_dir, idx)
                results.append(ImportedPage(idx, "page", image_path, thumb_path, w, h))
                idx += 1

    if not results:
        raise MangaImportError("No pages could be extracted from that PDF.")
    return results


def import_image_file(src_path: Path, project_id: int, page_index: int,
                      prepare: bool = True) -> list[ImportedPage]:
    """Import one image file. Returns two pages when the image is a spread."""
    root = project_dir(project_id)
    try:
        with Image.open(src_path) as raw:
            raw.load()
            img = _normalise(raw)
    except MangaImportError:
        raise
    except Exception as exc:
        raise MangaImportError(f"'{src_path.name}' is not a readable image: {exc}") from exc

    if min(img.size) < MIN_EDGE:
        raise MangaImportError(
            f"'{src_path.name}' is only {img.width}x{img.height}px; "
            f"pages must be at least {MIN_EDGE}px on the short edge."
        )

    out: list[ImportedPage] = []
    for n, sheet in enumerate(prepare_sheet(img) if prepare else [img]):
        if page_index + n >= MAX_PAGES:
            break
        image_path, thumb_path, w, h = _save_page(
            sheet, root / "pages", root / "thumbs", page_index + n
        )
        out.append(ImportedPage(page_index + n, "page", image_path, thumb_path, w, h))
    return out


# -------------------------------------------------------- panel segmentation


def detect_panels(page_path: Path, project_id: int, page_index: int,
                  max_panels: int = 6) -> list[ImportedPage]:
    """Best-effort panel segmentation.

    Manga layouts are irregular (bleeds, overlaps, diagonal gutters) so this only
    returns panels when it finds a confident, mostly-rectangular grid. On any doubt it
    returns an empty list and the caller keeps the whole page.
    """
    try:
        import cv2
    except ImportError:
        return []

    try:
        with Image.open(page_path) as raw:
            page = _normalise(raw)
    except Exception:
        return []

    arr = np.asarray(page)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    page_h, page_w = gray.shape
    page_area = float(page_h * page_w)

    # Gutters are near-white; ink is dark. Binarise so panel interiors become blobs.
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < page_area * 0.04 or area > page_area * 0.92:
            continue
        if w < page_w * 0.12 or h < page_h * 0.08:
            continue
        aspect = w / float(h)
        if aspect > 6.0 or aspect < 0.16:
            continue
        # The bounding box should actually be filled by content, not a thin frame.
        if cv2.contourArea(contour) < area * 0.55:
            continue
        boxes.append((x, y, w, h))

    if len(boxes) < 2:
        return []

    # Drop boxes nested inside a larger one.
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        inside = any(
            x >= kx - 4 and y >= ky - 4 and x + w <= kx + kw + 4 and y + h <= ky + kh + 4
            for kx, ky, kw, kh in kept
        )
        if not inside:
            kept.append(box)

    if not 2 <= len(kept) <= max_panels:
        return []
    if sum(w * h for _, _, w, h in kept) < page_area * 0.30:
        return []

    # Manga reading order: top-to-bottom rows, right-to-left inside a row.
    row_tolerance = page_h * 0.08
    kept.sort(key=lambda b: b[1])
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in kept:
        if rows and abs(box[1] - rows[-1][0][1]) <= row_tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    ordered: list[tuple[int, int, int, int]] = []
    for row in rows:
        row.sort(key=lambda b: b[0], reverse=True)
        ordered.extend(row)

    root = project_dir(project_id)
    panels_dir, thumbs_dir = root / "panels", root / "thumbs"
    out: list[ImportedPage] = []
    for n, (x, y, w, h) in enumerate(ordered):
        pad = 2
        crop = page.crop(
            (
                max(0, x - pad),
                max(0, y - pad),
                min(page_w, x + w + pad),
                min(page_h, y + h + pad),
            )
        )
        if min(crop.size) < MIN_EDGE:
            continue
        prefix = f"panel_p{page_index:03d}"
        image_path, thumb_path, cw, ch = _save_page(crop, panels_dir, thumbs_dir, n, prefix)
        out.append(
            ImportedPage(n, "panel", image_path, thumb_path, cw, ch, parent_page_index=page_index)
        )

    return out if len(out) >= 2 else []


# ------------------------------------------------------------------------ OCR


def extract_text(image_path: Path) -> str:
    """Return OCR text for a page, or '' when OCR is unavailable or finds nothing."""
    if not ocr_available():
        return ""
    try:
        import pytesseract

        with Image.open(image_path) as raw:
            img = _normalise(raw)
        text = pytesseract.image_to_string(img)
    except Exception:
        return ""

    cleaned = " ".join(part.strip() for part in text.split()) if text else ""
    cleaned = re.sub(r"[^\w\s,.!?'\"-]", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Short fragments from OCR noise are not worth showing as a subtitle.
    return cleaned[:160] if len(cleaned) >= 8 else ""
