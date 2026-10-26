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

## 4. More animation quota

Image-to-video now works, via a ZeroGPU Space (`spaces/kage-animator/`, deployed at
[viperlurk/kage-animator](https://huggingface.co/spaces/viperlurk/kage-animator)). A
panel goes up, a 25-frame clip comes back, and the shot plays those frames under the same
camera move, grade and effects as any other.

Measured on a Junji Ito street panel: SVD generated a camera push down the alley with the
buildings moving in correct perspective and the line-art style intact — frame-to-frame
motion 26.9, first-to-last difference 49.0. Generated motion, not a warp.

The limit is quota, not capability. A free ZeroGPU account refuses further calls once its
small daily allowance is gone — instantly, with no message — so Kage animates a handful
of hero shots and leaves depth parallax to carry the rest. Options, in order of appeal:

- Reduce the reservation further. ZeroGPU charges the `@spaces.GPU(duration=...)`
  reservation, not the time used; 75s is already down from 150s but a 10-step run may fit
  in 45s.
- A faster model. LTX-Video is built for speed and would cost less quota per clip, at the
  cost of needing a text prompt — which the dialogue work in item 1 could supply.
- HF PRO, which raises the ZeroGPU allowance.

## 5. Character motion

Even with image-to-video, no character is drawn walking. SVD moves the *image*; it does
not know a figure has limbs. It produces camera motion and parallax through a scene, not
in-betweens or redrawn cels. That gap is inherent to image-conditioned video models, and
closing it needs a different class of model entirely.

Routes ruled out, October 2026. Cloudflare Workers AI has no video model at all: 73
models, text-to-image and TTS only. Hugging Face Inference Providers bill through one
router that returns 402 without credit. Google Colab is not usable either, however well a
tunnel would work: its own FAQ bans connecting to remote proxies on every managed
runtime, bans bypassing the notebook UI on free ones, and names web-UI content generation
as a common cause of free-tier termination.

If a better image-to-video endpoint appears, it belongs in the existing chain in
backend/video_providers.py, not as a special case.

## 6. Hosted voice

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
