# Kage

> **From Ink to Imagination**

Turn manga pages into cinematic motion comics — panel detection, camera moves with depth
parallax, visual effects, a synthesised score with timed sound effects, optional AI
narration, and an MP4 you can preview and download.

```
PDF / images  ->  trim + split  ->  pages & panels  ->  shot plan  ->  frames + effects  ->  score  ->  MP4
```

**What it needs.** The core pipeline — import, framing, camera, grade, score, render — is
deterministic and runs on CPU with no model and no network. Three things go beyond that,
and each is optional:

| | Runs on | Needed for | Without it |
|---|---|---|---|
| CLIP + Depth Anything V2 Small | your GPU (CPU fallback), **1.3 GB downloaded** | choosing pages, depth parallax | pages used in reading order, no parallax |
| A hosted vision/text model | **free tier of Gemini, OpenRouter or NVIDIA** | narration | no voiceover |
| Image-to-video | a GPU elsewhere — Modal or an HF ZeroGPU Space | animated hero shots | parallax carries every shot |

So: it renders offline on a CPU, and it is better with a GPU, ~1.3 GB of weights and one
free API key. Earlier versions of this file claimed "no GPU, no paid API, no model weights"
— that was true once and stopped being true as the above was added.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | 3.12 recommended (this build was verified on 3.12.12) |
| FFmpeg + ffprobe | Needed for encoding and verification |
| ~500 MB free disk | Per-project pages, frames and output |
| +1.4 GB if you use the vision models | Measured: CLIP 1.2 GB, Depth Anything V2 Small 99 MB, the voice 145 MB |

Optional, and each degrades gracefully:

- **`torch` + `transformers`** — page selection (CLIP) and depth parallax. Uses a GPU when
  one is present and falls back to CPU. Without them, pages are filmed in reading order
  and shots get no parallax.
- **One free inference key** — `GEMINI_API_KEY`, `OPENROUTER_API_KEY` or `NVIDIA_API_KEY`
  for narration. A provider with no key is skipped, never failed. Without any, the film
  is silent of voice but still scored.
- **`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`**, or a Hugging Face ZeroGPU Space — animated
  hero shots. Without either, depth parallax carries every shot.
- **Tesseract OCR** + `pytesseract` — subtitle extraction from speech bubbles. Without it,
  subtitles are skipped.
- **`HF_TOKEN`** — downloading model weights, and calling a ZeroGPU Space. Not needed for
  the hosted narration providers above.

### FFmpeg

The app looks for FFmpeg in this order: `tools/bin/` → `FFMPEG_BINARY` env var → `PATH`.

**Windows** — drop `ffmpeg.exe` and `ffprobe.exe` into `tools/bin/`, or:

```powershell
winget install Gyan.FFmpeg
```

**macOS** — `brew install ffmpeg`  **Linux** — `sudo apt install ffmpeg`

---

## Installation

```powershell
# Windows / PowerShell
cd D:\Projects\Kage
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Verify everything the app needs is present:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

It prints each dependency, the resolved FFmpeg path, and the capability matrix, and exits
non-zero if something required is missing.

### Configuration

```powershell
copy .env.example .env
```

All settings have working defaults, so `.env` is optional. Tokens are read from the
environment only — nothing is hardcoded, and `.env` is git-ignored.

| Variable | Default | Purpose |
|---|---|---|
| `KAGE_HOST` / `KAGE_PORT` | `127.0.0.1` / `8000` | Backend bind address |
| `KAGE_API` | `http://127.0.0.1:8000` | Where the frontend looks for the API |
| `HF_TOKEN` | *unset* | Enables the hosted AI director |
| `HF_DIRECTOR_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Hosted director model |
| `FFMPEG_BINARY` / `FFPROBE_BINARY` | *unset* | Explicit binary paths |

---

## Running

### One command (Windows)

```powershell
.\run_local.ps1
```

It runs the environment check, then starts both services in separate processes.

### Manual (two terminals)

```powershell
# Terminal 1 - backend
.\.venv\Scripts\python.exe main.py

# Terminal 2 - frontend
.\.venv\Scripts\streamlit.exe run app.py
```

| Service | URL |
|---|---|
| Streamlit frontend | <http://localhost:8501> |
| FastAPI backend | <http://127.0.0.1:8000> |
| API documentation | <http://127.0.0.1:8000/docs> |

### Sample artwork

The repository ships no third-party manga. Generate original test pages instead:

```powershell
.\.venv\Scripts\python.exe scripts\make_sample_manga.py
```

This writes four procedurally drawn pages plus a combined PDF to `data/sample_manga/`.
The **Manga Import** screen has a one-click button to load them.

---

## Using the studio

The studio is a four-step pipeline, and the app is built around that: every step page
carries a progress strip showing which steps are done, which one you are on, and which are
still locked, so the order is visible rather than something you have to learn.

1. **Add pages** — drop in a PDF or page images (up to 80 MB / 80 pages). Sheets are
   cleaned up and split on the way in (see *Page preparation*), so a 33-sheet book arrives
   as ~65 pages. Untick anything you do not want filmed.
2. **Choose a look** — pick a style, length, shape and score. The style sets the grade,
   camera vocabulary and pacing all at once. Saving builds the shot plan.
3. **Review shots** — the whole sequence is one editable table: length, camera move,
   transition, effects, sound cues and subtitle, with a thumbnail of the page each shot is
   framed on.
4. **Render** — live progress, then play and download the MP4.

**Projects** lists everything the studio has made; **Diagnostics** carries the backend log.

A step you cannot run yet says why and links to the step that unblocks it, instead of
rendering an empty screen.

---

## Architecture

```
app.py                    Streamlit entry point: page config, navigation, sidebar
main.py                   Backend entry point
app_pages/                One file per screen (home, the four steps, projects, diagnostics)
frontend/studio.py        API client, vocabulary, and the workflow model shared by pages
.streamlit/config.toml    Theme: colours, fonts, radii, light and dark
backend/
  api.py                  FastAPI routes, background render worker
  config.py               Paths, binary discovery, capability probing
  database.py             SQLite engine and sessions
  models.py               Project / Page / Shot / RenderJob tables
  schemas.py              Pydantic request, response and shot-plan schemas
  manga_processor.py      PDF extraction, sheet trim + spread split, panel segmentation
  cinematic_director.py   Local planner + optional hosted LLM planner
  visual_engine.py        Camera paths, effects, captions - frame generation
  audio_engine.py         Synthesised music, SFX library, multi-track mixing
  video_renderer.py       Timeline, transitions, FFmpeg encode and verification
assets/audio/sfx/         Generated sound-effect cache (created on first run)
data/projects/            Per-project source, pages, panels, render output
tests/test_pipeline.py    Backend unit + integration tests
tests/test_frontend.py    Streamlit pages, driven headlessly with AppTest
```

### Anime styles

A style is not a filter. It bundles the whole visual grammar: allowed camera moves, shot
length, transition set, colour grade and how sparingly accents are spent.

| Style | Look |
|---|---|
| **Kage Noir** | Near-monochrome, crushed blacks, slow push-ins. Modern dark. |
| **Shonen Kinetic** | Bright and punchy, hard cuts, speed lines saved for the hit. |
| **Shojo Bloom** | Warm highlights, soft bloom, long dissolves. |
| **Seinen Cold** | Desaturated blue, locked-off framing, minimal motion. |
| **Retro Cel** | Heavy grain and amber cast, like a 90s OVA on tape. |

**Restraint is enforced in code, not left to taste.** The always-on grade (exposure,
contrast, saturation, tint, vignette, grain, bloom) carries the look. Named effects are
*accents*, spent on roughly one shot in three to five, and a shot can never exceed its
style's `max_effects`. Tests assert both rules. Consecutive shots also never reuse the same
camera move, because cutting from a move to itself reads as a glitch.

### Page preparation

A photographed book is not artwork. It carries the camera's dark background, the edge of
the paper block, wide blank margins and - on a spread - two separate drawings either side
of a gutter. Filming that is what makes a shot read as a photo of a book rather than a
frame of film, so it is dealt with once, at import, instead of being fought by the camera
on every shot:

- **Trim.** Two guarded passes find the sheet (everything bright enough to be paper) and
  then the ink inside it. The margin pass may take at most 18% off any one edge, so a
  faintly printed page loses its margin and never its artwork. On any doubt the page is
  left untouched - a bad crop is far worse than an untrimmed page.
- **Split.** A sheet wider than 1.18:1 holds two facing pages and becomes two pages, right
  half first, matching the right-to-left reading order used for panels. Each half is then
  trimmed again, which is what removes the gutter shadow left on its inner edge.

Because one sheet can become two pages, `MAX_PAGES` counts roughly twice as fast as it
used to; it is 80 so that a book-length source is not silently truncated.

### Subject framing

A page is still mostly paper, so centring a shot on it shows nothing in particular - that
is what makes a slideshow. Kage finds the densest region of drawing using an integral-image
search at the output aspect ratio, and frames the shot there. It does not need to know
*what* is drawn, only *where* - so this works with or without the vision model.

The same ink map fences the camera: a move may not travel to where the frame would hold
less than 45% of the ink the anchor sees. Without that fence a pan reaches the end of its
slack, which is the edge of the paper, and holds on an empty margin - or parks in the gap
between two pictures stacked on one page.

Each style sets its own `framing` tightness, from Kage Noir at 1.95 (intimate) to Seinen
Cold at 1.35 (observational). That number is expressed relative to a whole sheet, so it is
clamped per source: since pages are split at import, applying it unclamped to a half-size
page crops twice as tight and the shot lands on a fragment of a sleeve.

### Animation

Kage moves the artwork, not only a camera over it. Two techniques, both of which run on a
4 GB card and neither of which needs a diffusion model:

- **Parallax.** A depth model (Depth Anything V2 Small, Apache-2.0, ~100 MB) estimates how
  far away each pixel is, and the frame is resampled so near things travel further than
  far ones. Depth is anchored at the *far* plane, so the background holds still and the
  foreground slides across it - centring on the median instead produces a stereo wobble
  rather than a camera move. Measured: 0.3s per panel, 108 MB of VRAM.
- **Drift.** A slow low-frequency displacement field on top, so hair, cloth and smoke keep
  moving while a shot holds. No model at all - two sine waves and a noise field.

Both are `cv2.remap` passes at output resolution, about a millisecond a frame. Depth is
computed once per panel at analysis time and cached beside it, so rendering never waits on
a model. Each style sets its own amounts, from Shonen Kinetic at 0.034 to Seinen Cold at
0.014.

**What this is not.** No frame is drawn, no character is animated, no in-betweens exist.
Kage is a motion comic engine: it makes your artwork move, it does not generate new
artwork. Image-to-video diffusion needs 8-12 GB of VRAM, which this does not assume.

### Voiceover

Optional narration, written from what a model actually sees on the pages and spoken over
the score. Three jobs, each a chain of candidates tried in order:

| Job | Runs on | Default chain |
|---|---|---|
| **See** - what is drawn on each page | hosted | Gemini Flash → OpenRouter free vision → NVIDIA NIM |
| **Write** - one line per shot, inside a word budget | hosted | Gemini Flash → OpenRouter free → NVIDIA NIM |
| **Speak** - render the script to audio | local, CPU | `facebook/mms-tts-eng` (139 MB) → SpeechT5 |

Seeing and writing are hosted on purpose. A local 1.5B writer was built and measured
first: it wanted 3.1 GB of VRAM on a 4 GB card and 2.9 GB of disk, and it made the studio
fight its own renderer for the GPU. Hosted, the whole film costs **two requests** - one
batch of pages to look at, one call to write every line - which sits inside every free
tier below. Speaking stays local because free hosted TTS generally wants a card, and
139 MB on the CPU costs nothing the renderer needs.

Set **any one** of these in `.env`; a provider with no key is skipped, not failed:

| Provider | Free tier | Key |
|---|---|---|
| Google AI Studio | no card, ~1500 req/day | `GEMINI_API_KEY` — <https://aistudio.google.com/apikey> |
| OpenRouter | `:free` models, 20/min, 50/day | `OPENROUTER_API_KEY` — <https://openrouter.ai/keys> |
| NVIDIA NIM | no card, ~40/min, prototyping only per their terms | `NVIDIA_API_KEY` — <https://build.nvidia.com> |

Override a chain with `KAGE_VISION_CHAIN` / `KAGE_TEXT_CHAIN`, as
`provider:model,provider:model`.

**Narration never invents.** Measured on real pages: asking a model for "cinematic
narration in the Kage Noir style" turned the observation *"a group of people in a room"*
into *"the lights flicker, casting shadows across faces"* - every image fabricated. The
prompt now frames the job as describing a picture to someone who cannot see it, decoding
is greedy, and every noun must come from the observation. Narration is written at analysis
time, stored on the shot, and fully editable in the shot table - so when a captioner is
wrong, you can fix it rather than argue with it.

Narration adds roughly 18s to a render (load the voice model, then about a second a line).

### Colour and levels

A style's grade is a choice about mood, not a licence to throw away a colour original. Two
rules keep it honest:

- **Levels are normalised before grading.** A photographed page is lit, not printed: its
  paper sits around 190 and a style that darkens further leaves everything in the middle.
  Kage puts the paper back at paper-white first, equally across all three channels so hue
  is untouched.
- **A colour source keeps some colour.** Each grade carries a `colour_floor` that applies
  only when the page actually has chroma. Kage Noir still reads as near-monochrome on
  black-and-white art, but a hand-tinted woodblock print no longer comes out a photocopy.

### How the motion is made

Each shot interpolates a crop rectangle across the source image with eased curves and
resamples it once per frame with a sub-pixel affine warp, so motion is smooth rather than a
series of integer jumps. Frames stream to FFmpeg over stdin as raw RGB — peak memory stays
at a handful of frames regardless of clip length.

Available camera moves: `slow_zoom_in`, `slow_zoom_out`, `pan_left`, `pan_right`, `pan_up`,
`pan_down`, `ken_burns`, `rapid_pan`, `push_in`, `static_drift`.

Effects: vignette, speed lines, screen shake, impact flash, directional motion blur,
particles, film grain, rain, desaturate, warm glow, cold tint.

Transitions: cut, fade (dip to black), crossfade, flash, whip. Dissolves consume frames
from both neighbouring shots, so the timeline scales the plan up to land on exactly the
requested runtime.

### How the sound is made

No audio assets are bundled or assumed. Music is generated per mood from a chord
progression with pad, bass, melody and percussion voices, through a convolution reverb
built from a decaying-noise impulse response. Ten sound effects are synthesised, cached as
WAVs under `assets/audio/sfx/`, and verified on disk before use.

Four logical tracks are mixed — music, ambience, shot effects, narration — with the music
following a per-shot intensity curve and ducking under loud effects. The master bus is
soft-limited and normalised to about −1 dBFS, so the output never clips.

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status and capability matrix |
| `GET` | `/api/projects` | Recent projects |
| `POST` | `/api/projects` | Create a project |
| `GET` | `/api/projects/{id}` | Project detail, pages, shots, render job |
| `PATCH` | `/api/projects/{id}` | Update configuration |
| `DELETE` | `/api/projects/{id}` | Delete a project |
| `POST` | `/api/projects/{id}/upload` | Upload a PDF or images |
| `POST` | `/api/projects/{id}/analyze` | Extract panels and build the shot plan |
| `PATCH` | `/api/projects/{id}/shots` | Edit the shot plan |
| `POST` | `/api/projects/{id}/render` | Start a render (202, background worker) |
| `GET` | `/api/projects/{id}/status` | Progress, stage and errors |
| `GET` | `/api/projects/{id}/video` | Download the MP4 |
| `GET` | `/api/projects/{id}/subtitles` | Download the `.srt` sidecar |
| `GET` | `/api/logs` | Recent log records (`level`, `limit`, `source`) |

Rendering runs in a background thread with a single-render lock, so the API event loop is
never blocked. Job status lives in SQLite, so the frontend recovers state after a reload.

---

## Logging and diagnostics

Logging is configured centrally in `backend/logging_config.py` and writes to three sinks:

| Sink | Contents |
|---|---|
| `data/logs/kage.log` | INFO and above, rotating 2 MB x5 |
| `data/logs/errors.log` | WARNING and above only - look here first |
| In-memory ring buffer (500) | Served by `GET /api/logs` and the Diagnostics screen |

Captured automatically: unhandled exceptions with full tracebacks (the API returns a
readable message instead of a bare 500), every 4xx/5xx response, render start/finish/failure
including the FFmpeg exit code and stderr tail, upload rejections, and planner fallbacks.

Read it from the **Diagnostics** screen in the sidebar - filter by level, expand tracebacks,
or tail the raw files - or over HTTP:

```powershell
curl "http://127.0.0.1:8000/api/logs?level=WARNING&limit=50"
curl "http://127.0.0.1:8000/api/logs?source=errors&limit=100"
```

Set `KAGE_LOG_LEVEL=DEBUG` for verbose output.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

39 tests cover input sanitisation, planner determinism and schema coercion, camera motion,
aspect ratios, captions, every audio mood and sound effect, mix levels, timeline overlap
maths, a full render verified with `ffprobe`, and the complete API workflow from upload
through download. `tests/test_frontend.py` runs the real Streamlit app with a stubbed HTTP
layer and clicks every enabled button on every screen - rendering a page without interacting
with it does not catch session-state errors that only fire on a widget callback. Render tests
skip automatically when FFmpeg is absent.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar shows **Backend offline** | Start it: `python main.py`. Check `KAGE_API`. |
| `FFmpeg is not available` on render | Put `ffmpeg.exe`/`ffprobe.exe` in `tools/bin/`, or set `FFMPEG_BINARY`. Re-run `scripts\check_environment.py`. |
| PDF upload rejected | Needs PyMuPDF (`pip install pymupdf`), and the PDF must not be password protected. |
| Panel detection found nothing | Expected on irregular layouts — the app falls back to whole pages and still renders. |
| Subtitles are empty | OCR is optional. Install Tesseract and `pytesseract`, then re-run analysis. |
| `429 Another render is in progress` | Only one render runs at a time. Wait for it to finish. |
| Hosted director always falls back | Check **Diagnostics**. A 402 means the HF account has no inference credits; 401 means a bad or expired token. |
| Frontend slow to start | Streamlit's first boot can exceed 60s. The launcher waits 90s and opens the browser regardless. |
| Render failed | Open **Diagnostics** on the Render screen — the FFmpeg exit code and stderr tail are stored on the job. |
| Video is very large | CRF is 20 for quality. Raise it in `video_renderer.py` for smaller files. |

---

## Known limitations

- **This is a motion comic, not generated animation.** Your artwork is preserved and
  moved; no character moves within a panel. Image-to-video adds generated camera motion
  through a scene, not in-betweens or redrawn cels.
- **Panel detection is best-effort.** It only accepts confident, mostly-rectangular
  layouts. Bleeds, overlaps and diagonal gutters fall back to whole pages. Measured on a
  Junji Ito volume: 61 panels from 42 pages, with several pages returning one blob
  covering 96% of the sheet. This is the biggest open problem.
- **The score is synthesised, not composed.** Procedural audio, mixed carefully, but not
  a licensed soundtrack.
- **Narration is only as right as the captioner.** It describes what a vision model
  reported seeing, and that model is sometimes wrong — a Hokusai garden came back as "a
  man on a horse". Editable in the shot table for this reason.
- **The voice is non-commercial.** `facebook/mms-tts-eng` is CC-BY-NC-4.0 and adds about
  18s to a render. Fine for portfolio; a blocker for commercial use.
- **OCR is still unexercised.** No Tesseract binary has been present in any build so far,
  so subtitles fall back cleanly but have never actually run.
- **Animation is rationed.** A free ZeroGPU account allows very few calls a day, and
  Modal requires a payment method for any GPU. Shots that cannot be animated keep their
  still and the render completes.
- **One render at a time**, single machine, no queue.

## What is and is not AI

Kage began deterministic and has grown five model-driven steps. Each is optional and each
degrades to something that still produces a film, so it is worth being exact about which
is which.

### Models, and where they run

| Model | Job | Size | Runs on |
|---|---|---|---|
| CLIP ViT-B/32 | ranks pages by artwork, so covers and text pages are skipped | 1.2 GB | your GPU, CPU fallback |
| Depth Anything V2 Small | depth map for parallax | 99 MB | your GPU, CPU fallback |
| MMS-TTS eng | speaks the narration | 145 MB | your CPU |
| A hosted vision model | reads each panel and describes it | — | Gemini / OpenRouter / NVIDIA free tier |
| A hosted text model | writes one narration line per shot | — | the same chain |
| Stable Video Diffusion | animates a hero shot | 4.5 GB | Modal, or an HF ZeroGPU Space |

Weights download from the Hugging Face hub, which is free and unrelated to its paid
Inference API. The hosted steps use **two requests per film**, not two per shot, which
sits inside every free tier Kage supports.

### What stays deterministic

Style presets, shot selection, panel segmentation, the camera paths, framing, the colour
grade, every visual effect, the whole score and the render itself. No model touches any of
them. That is why a render is reproducible and fast.

### What the models are not allowed to do

**Invent.** Narration is written only from what a vision model reported seeing, with a
prompt that forbids adding weather, lighting, emotion or action, and greedy decoding so it
does not drift. This was measured: asked for "cinematic narration" the same model turned
*"a group of people in a room"* into *"the lights flicker, casting shadows across faces"* —
every image fabricated. Subtitles are only ever text OCR actually read from the page.

Narration is still only as right as the captioner, which is why it is editable in the shot
table. On a Hokusai volume one page of stone lanterns came back as "a man on a horse".

### What none of it does

**No frame is drawn.** Parallax and drift move your artwork; image-to-video generates
camera motion through a scene. Neither animates a character — there are no in-betweens and
no redrawn cels. Kage is a motion comic engine: it makes your artwork move, it does not
make new artwork.

### Why the video model runs elsewhere

Stable Video Diffusion needs more VRAM than the 4 GB card this was built on, so it runs
on a GPU elsewhere and the clip comes back — see `modal_app/`. Everything else is small
enough to stay local: CLIP, depth and the voice are 1.4 GB together.

Even with it, Kage produces a **motion comic** — your artwork, framed and moved like film
— not generated character motion. That distinction is deliberate.

## Licensing

Do not upload manga you are not authorised to adapt. The sample pages in
`data/sample_manga/` are generated by `scripts/make_sample_manga.py` and contain no
third-party work. All audio is synthesised at runtime — no commercial music is bundled or
used.
