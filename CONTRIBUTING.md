# Contributing to Kage

Thanks for looking. Kage turns manga artwork into cinematic motion comics, locally.

**Everyone is welcome here.** This was built for people who love manga and anime and want
to see these stories move, and it stays open to anyone who wants to help — no invitation,
no experience bar, no gatekeeping. You do not have to be a Python developer to be useful:
knowing how a scene should be cut, or noticing that a shot lands wrong, is worth as much
as code. Open an issue, send a pull request, or just say what you would change.

Everything here is measured and written down, so you can see why a decision was made
before you change it. If you disagree with one, bring a measurement and change it — that
is the whole bar.

## Ground rules

Read [ENGINEERING.md](ENGINEERING.md) first. Every rule in it exists because it was broken
once here and cost something real, and several are enforced by tests.

**Never commit artwork you do not own.** Kage ships no third-party manga, and a public
repo is publication. `.gitignore` already excludes `*.cbz`, `*.cbr`, `*.rar` and common
scan folder names — keep your source material outside the repo directory entirely, not
just untracked. Use `python scripts/make_sample_manga.py` for test pages.

**Never commit secrets.** `.env` is git-ignored. Copy `.env.example` and fill it in.

**Claims in this repo are measured, not asserted.** Several comments cite numbers — depth
model VRAM, provider latency, how often a camera move left the frame blank. If you change
the behaviour behind one, re-measure and update the number. A stale measurement in a
comment is worse than none, because the next person trusts it.

## Setup

```bash
uv venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
# GPU users, install torch first from the index matching your CUDA:
#   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cp .env.example .env          # add one free inference key for narration
./run_local.ps1               # backend on :8000, frontend on :8501
```

FFmpeg is required. Drop `ffmpeg.exe` / `ffprobe.exe` in `tools/bin/`, or have them on
PATH. `python scripts/check_environment.py` reports what is missing.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

`tests/test_pipeline.py` covers the backend end to end, including a real render.
`tests/test_frontend.py` drives the Streamlit app headlessly with `AppTest` — it clicks
every button on every page, because rendering a page without interacting with it does not
catch session-state bugs.

Please add a test with a behaviour change. Several tests in here exist because a bug
shipped silently once; a regression test naming the symptom is the most useful thing you
can leave behind.

## Where things live

| Path | What |
|---|---|
| `app.py`, `app_pages/` | Streamlit front end, one file per screen |
| `frontend/studio.py` | API client and the workflow model the pages share |
| `backend/manga_processor.py` | Import: scan trimming, spread splitting, panel segmentation |
| `backend/vision.py` | Local CLIP: which pages are worth filming |
| `backend/cinematic_director.py` | Shot plan: length, camera, transitions, cues |
| `backend/visual_engine.py` | Camera paths, framing, grade, effects, frame generation |
| `backend/animation.py` | Depth parallax and drift — moving the artwork itself |
| `backend/narration.py` | Voiceover: observe, write, speak |
| `backend/providers.py`, `backend/routing.py` | Hosted models, and which to try first |
| `backend/audio_engine.py` | Synthesised score, SFX, mixing |
| `backend/video_renderer.py` | Timeline, transitions, FFmpeg encode |

## Good first issues

- **Panel segmentation** only accepts confident rectangular grids. Bleeds, overlaps and
  diagonal gutters fall back to whole pages. Better segmentation is the single highest
  impact change available.
- **Reading order** within a page assumes right-to-left. Western comics need the opposite,
  and nothing detects which a book is.
- **OCR** is wired but unexercised — no Tesseract in most setups. Speech bubbles are
  visible in frame and currently unused.
- **Hosted TTS.** The voice is a local 139 MB model; a hosted option would free it up.

## Style

Match the surrounding code. Comments explain *why*, especially where a choice looks odd —
most odd-looking choices here are the result of something measured. Keep them.
