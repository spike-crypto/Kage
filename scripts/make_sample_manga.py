"""Generate original sample 'manga' pages for testing.

The artwork is drawn procedurally here, so the repository ships no third-party or
commercial manga. Run:  python scripts/make_sample_manga.py
Output: data/sample_manga/*.png plus a combined sample_manga.pdf
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import DATA_DIR  # noqa: E402
from backend.visual_engine import load_font  # noqa: E402

OUT_DIR = DATA_DIR / "sample_manga"
PAGE_W, PAGE_H = 1100, 1550

INK = (18, 18, 22)
PAPER = (247, 245, 240)


def screentone(draw: ImageDraw.ImageDraw, box, spacing: int = 7, radius: int = 2,
               shade=(140, 140, 150)) -> None:
    x0, y0, x1, y1 = box
    for y in range(int(y0), int(y1), spacing):
        offset = spacing // 2 if (y // spacing) % 2 else 0
        for x in range(int(x0) + offset, int(x1), spacing):
            draw.ellipse([x, y, x + radius, y + radius], fill=shade)


def speed_lines(draw: ImageDraw.ImageDraw, box, count: int = 70, rng=None) -> None:
    rng = rng or random.Random(1)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    reach = math.hypot(x1 - x0, y1 - y0)
    for _ in range(count):
        angle = rng.uniform(0, 2 * math.pi)
        inner = rng.uniform(0.22, 0.45) * reach / 2
        outer = rng.uniform(0.75, 1.0) * reach / 2
        draw.line(
            [
                cx + math.cos(angle) * inner, cy + math.sin(angle) * inner,
                cx + math.cos(angle) * outer, cy + math.sin(angle) * outer,
            ],
            fill=INK,
            width=rng.choice((1, 1, 2)),
        )


def silhouette(draw: ImageDraw.ImageDraw, cx: float, base_y: float, height: float,
               facing: int = 1, colour=INK) -> None:
    """A simple standing figure, enough to read as a character in a panel."""
    unit = height / 8.0
    head_r = unit * 0.62
    head_y = base_y - height + head_r
    draw.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r], fill=colour)
    torso_top = head_y + head_r * 1.1
    torso_bottom = base_y - unit * 3.0
    draw.polygon(
        [
            (cx - unit * 0.95, torso_top + unit * 0.4),
            (cx + unit * 0.95, torso_top + unit * 0.4),
            (cx + unit * 0.7, torso_bottom),
            (cx - unit * 0.7, torso_bottom),
        ],
        fill=colour,
    )
    # arms
    draw.line([cx - unit * 0.8, torso_top + unit * 0.7,
               cx - unit * 2.1 * facing, torso_top + unit * 2.0], fill=colour, width=int(unit * 0.5))
    draw.line([cx + unit * 0.8, torso_top + unit * 0.7,
               cx + unit * 1.9 * facing, torso_top + unit * 0.2], fill=colour, width=int(unit * 0.5))
    # legs
    draw.line([cx - unit * 0.35, torso_bottom, cx - unit * 1.0, base_y], fill=colour,
              width=int(unit * 0.6))
    draw.line([cx + unit * 0.35, torso_bottom, cx + unit * 1.1, base_y], fill=colour,
              width=int(unit * 0.6))


def speech_bubble(draw: ImageDraw.ImageDraw, box, text: str, tail_down: bool = True) -> None:
    x0, y0, x1, y1 = box
    draw.ellipse([x0, y0, x1, y1], fill=PAPER, outline=INK, width=4)
    cx = (x0 + x1) / 2
    if tail_down:
        draw.polygon([(cx - 14, y1 - 10), (cx + 16, y1 - 12), (cx - 4, y1 + 34)],
                     fill=PAPER, outline=INK)
        draw.line([(cx - 13, y1 - 6), (cx - 4, y1 + 32)], fill=PAPER, width=5)
        draw.line([(cx + 15, y1 - 9), (cx - 3, y1 + 31)], fill=PAPER, width=5)

    font = load_font(28)
    words, lines, current = text.split(), [], ""
    max_w = (x1 - x0) * 0.74
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    total_h = len(lines) * 34
    ty = (y0 + y1) / 2 - total_h / 2
    for line in lines:
        tw = draw.textlength(line, font=font)
        draw.text((cx - tw / 2, ty), line, font=font, fill=INK)
        ty += 34


def frame(draw: ImageDraw.ImageDraw, box, width: int = 6) -> None:
    draw.rectangle(box, fill=PAPER, outline=INK, width=width)


def page_one() -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
    d = ImageDraw.Draw(img)
    rng = random.Random(11)
    m = 48

    top = [m, m, PAGE_W - m, 560]
    frame(d, top)
    screentone(d, [m + 8, m + 8, PAGE_W - m - 8, 330], spacing=9, radius=3, shade=(196, 198, 210))
    d.polygon([(m + 8, 430), (330, 250), (640, 400), (900, 230), (PAGE_W - m - 8, 350),
               (PAGE_W - m - 8, 552), (m + 8, 552)], fill=(60, 62, 76))
    for x in range(m + 40, PAGE_W - m - 60, 90):
        h = rng.randint(120, 260)
        d.rectangle([x, 552 - h, x + 52, 552], fill=(30, 31, 40))
    speech_bubble(d, [120, 90, 520, 250], "The city never sleeps. Tonight, neither do we.")

    mid = [m, 600, 560, 1030]
    frame(d, mid)
    screentone(d, [m + 8, 608, 552, 1022], spacing=8, radius=2, shade=(205, 206, 216))
    silhouette(d, 300, 1010, 360, facing=1)
    speech_bubble(d, [90, 630, 500, 760], "Three blocks. Go!")

    mid2 = [600, 600, PAGE_W - m, 1030]
    frame(d, mid2)
    speed_lines(d, [608, 608, PAGE_W - m - 8, 1022], 90, rng)
    silhouette(d, 870, 980, 300, facing=-1, colour=(24, 24, 30))

    bottom = [m, 1070, PAGE_W - m, PAGE_H - m]
    frame(d, bottom)
    screentone(d, [m + 8, 1078, PAGE_W - m - 8, PAGE_H - m - 8], spacing=6, radius=2,
               shade=(168, 170, 186))
    d.ellipse([360, 1150, 740, 1430], fill=(28, 28, 36))
    d.ellipse([430, 1215, 560, 1300], fill=PAPER)
    d.ellipse([480, 1245, 530, 1288], fill=INK)
    d.ellipse([600, 1215, 700, 1295], fill=PAPER)
    d.ellipse([635, 1245, 678, 1285], fill=INK)
    return img


def page_two() -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
    d = ImageDraw.Draw(img)
    rng = random.Random(23)
    m = 48

    left = [m, m, 520, 760]
    frame(d, left)
    screentone(d, [m + 8, m + 8, 512, 752], spacing=10, radius=3, shade=(200, 190, 210))
    silhouette(d, 280, 740, 420, facing=1)
    speech_bubble(d, [90, 80, 470, 230], "You should not have followed me here.")

    right = [540, m, PAGE_W - m, 760]
    frame(d, right)
    speed_lines(d, [548, m + 8, PAGE_W - m - 8, 752], 120, rng)
    d.polygon([(700, 300), (900, 180), (980, 420), (820, 520)], fill=INK)
    d.polygon([(730, 330), (880, 240), (930, 400), (820, 470)], fill=PAPER)

    wide = [m, 790, PAGE_W - m, 1180]
    frame(d, wide)
    screentone(d, [m + 8, 798, PAGE_W - m - 8, 1172], spacing=5, radius=2, shade=(150, 152, 170))
    silhouette(d, 260, 1160, 330, facing=-1)
    silhouette(d, 840, 1160, 330, facing=1)
    for i in range(16):
        x = 400 + i * 20
        d.line([(x, 980 - i * 6), (x + 60, 1000 + i * 4)], fill=INK, width=3)

    bottom = [m, 1220, PAGE_W - m, PAGE_H - m]
    frame(d, bottom)
    speed_lines(d, [m + 8, 1228, PAGE_W - m - 8, PAGE_H - m - 8], 150, rng)
    font = load_font(96)
    text = "CLASH!"
    tw = d.textlength(text, font=font)
    d.text(((PAGE_W - tw) / 2 + 4, 1330), text, font=font, fill=(120, 120, 130))
    d.text(((PAGE_W - tw) / 2, 1326), text, font=font, fill=INK)
    return img


def page_three() -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
    d = ImageDraw.Draw(img)
    m = 48

    hero = [m, m, PAGE_W - m, 980]
    frame(d, hero)
    for i in range(70):
        shade = int(70 + i * 1.9)
        d.rectangle([m + 8, m + 8 + i * 13, PAGE_W - m - 8, m + 20 + i * 13],
                    fill=(shade, shade, min(255, shade + 24)))
    silhouette(d, PAGE_W / 2, 940, 620, facing=1, colour=(16, 16, 22))
    d.ellipse([PAGE_W / 2 - 240, 180, PAGE_W / 2 + 240, 640], outline=(250, 250, 255), width=8)
    speech_bubble(d, [140, 120, 560, 300], "This ends tonight.")

    low_l = [m, 1020, 540, PAGE_H - m]
    frame(d, low_l)
    screentone(d, [m + 8, 1028, 532, PAGE_H - m - 8], spacing=7, radius=2, shade=(186, 188, 200))
    d.ellipse([160, 1150, 420, 1400], fill=(26, 26, 34))
    d.ellipse([215, 1215, 300, 1280], fill=PAPER)
    d.ellipse([300, 1215, 380, 1280], fill=PAPER)

    low_r = [560, 1020, PAGE_W - m, PAGE_H - m]
    frame(d, low_r)
    screentone(d, [568, 1028, PAGE_W - m - 8, PAGE_H - m - 8], spacing=12, radius=4,
               shade=(210, 206, 220))
    speech_bubble(d, [600, 1080, 1010, 1260], "Then let us finish it.")
    silhouette(d, 820, 1480, 240, facing=-1)
    return img


def page_four() -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
    d = ImageDraw.Draw(img)
    rng = random.Random(37)
    m = 48

    top = [m, m, PAGE_W - m, 700]
    frame(d, top)
    speed_lines(d, [m + 8, m + 8, PAGE_W - m - 8, 692], 180, rng)
    d.ellipse([PAGE_W / 2 - 300, 180, PAGE_W / 2 + 300, 600], fill=(252, 252, 255),
              outline=INK, width=6)
    font = load_font(110)
    text = "BOOM"
    tw = d.textlength(text, font=font)
    d.text(((PAGE_W - tw) / 2, 330), text, font=font, fill=INK)

    mid = [m, 740, PAGE_W - m, 1180]
    frame(d, mid)
    screentone(d, [m + 8, 748, PAGE_W - m - 8, 1172], spacing=6, radius=2, shade=(160, 162, 178))
    for i in range(12):
        x = 120 + i * 80
        d.line([(x, 1160), (x + rng.randint(-40, 40), 820 + rng.randint(0, 120))],
               fill=(40, 40, 52), width=rng.randint(4, 10))
    silhouette(d, 560, 1165, 300, facing=1, colour=(22, 22, 30))

    bottom = [m, 1220, PAGE_W - m, PAGE_H - m]
    frame(d, bottom)
    for i in range(24):
        shade = 250 - i * 8
        d.rectangle([m + 8 + i * 4, 1228 + i * 4, PAGE_W - m - 8 - i * 4, PAGE_H - m - 8 - i * 4],
                    outline=(shade, shade, shade), width=3)
    font = load_font(54)
    text = "TO BE CONTINUED"
    tw = d.textlength(text, font=font)
    d.text(((PAGE_W - tw) / 2, 1390), text, font=font, fill=INK)
    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = [page_one(), page_two(), page_three(), page_four()]
    paths = []
    for n, page in enumerate(pages, start=1):
        path = OUT_DIR / f"sample_page_{n:02d}.png"
        page.save(path, "PNG")
        paths.append(path)
        print(f"wrote {path}")

    pdf_path = OUT_DIR / "sample_manga.pdf"
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:], resolution=150.0)
    print(f"wrote {pdf_path}")
    print(f"\n{len(paths)} pages + 1 PDF in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
