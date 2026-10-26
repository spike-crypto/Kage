# Roadmap

Ordered by how much each would change what a viewer sees. Everything here is written
from something measured on real pages, not from a guess about what might help.

## 1. Read the dialogue

**Kage currently throws the story away.** Speech bubbles are visible in almost every
rendered frame — often cropped mid-sentence — and nothing reads them. Narration describes
what the picture *looks like* ("A group of people sit in a room") rather than what is
*happening*, so a 30–60s film has no narrative thread. This is the single biggest reason
output reads as a gallery rather than a story.

**This already works.** The vision model in the existing provider chain transcribes
bubbles accurately. Tested on `panel_p017_000.png` from a Junji Ito volume with
`gemini-2.5-flash`:

> THE NEXT DAY, SHUICHI'S FATHER VISITED MY FATHER AT WORK. MR. GOSHIMA, YOU'RE ALWAYS
> WORKING SO HARD... WELL, YOU KNOW...

That is the panel's actual text, in reading order, including the caption box. No
Tesseract, no new dependency — the same request that already observes the panel can
return its dialogue.

Proposed:

1. `observe()` in `backend/narration.py` returns `{"description": ..., "dialogue": [...]}`
   per panel instead of a bare description.
2. Dialogue is stored on the shot alongside `narration_text`.
3. Used three ways:
   - **Subtitles** — burn the real lines, rather than relying on OCR that is usually not
     installed.
   - **Shot length** — hold a panel as long as its dialogue takes to read (~3.5 words a
     second) instead of a fixed style pace. This is most of what makes a sequence feel
     directed rather than metronomic.
   - **Narration** — speak the dialogue or the caption box, rather than describing the
     picture.
4. Keep the existing rule: only text actually read from the page, never invented.

Cost stays at roughly two hosted requests per film. Panels with no dialogue must return
an empty list, not a guess.

## 2. Better panel segmentation

`detect_panels` only accepts confident, mostly-rectangular grids. Measured on a Junji Ito
volume: 61 panels from 42 pages, but several pages returned a single blob covering 96% of
the sheet because the panels bleed into each other. Pages that fail fall back to whole
pages, and a whole page holds four to six panels at once — unreadable in a 16:9 frame.

This is the highest-impact change available after dialogue.

## 3. Reading-order detection

Panels within a page are ordered right-to-left, which is correct for manga and wrong for
western comics. Nothing detects which a book is. Spread splitting makes the same
assumption.

## 4. Character motion

Parallax and drift move the artwork, but no character ever moves within a panel: no
in-betweens, no redrawn cels. Real character animation needs image-to-video diffusion at
8–12 GB of VRAM, which Kage does not assume. Checked October 2026: Cloudflare Workers AI
has no video model (73 models, text-to-image and TTS only), and Hugging Face Inference
Providers bill through one router that returns 402 without credit.

If a free or cheap image-to-video endpoint appears, it belongs in the existing
`backend/providers.py` chain, not as a special case.

## 5. Hosted voice

The narrator is a local 139 MB model (`facebook/mms-tts-eng`, CC-BY-NC — non-commercial).
It adds roughly 18s to a render and the licence blocks commercial use. A hosted option, or
a permissively-licensed local one such as Kokoro-82M (Apache-2.0), would fix both.

## Known limits, stated plainly

- Narration can only be as right as the captioner. On a Hokusai volume it read a garden
  of stone lanterns as "a man on a horse". Narration is editable in the shot table for
  exactly this reason.
- Depth is approximate on dense line art. Parallax is subtle by design; pushed harder it
  tears at depth edges.
- Panel detection failing is normal on irregular layouts, and the fallback — whole pages —
  is visibly worse. See item 2.
