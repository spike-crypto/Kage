# The anime engine

How Kage goes from moving a camera over artwork to performing it, and why the route is
limited animation rather than frame generation.

This is the architecture document. The research it rests on — papers, models, licences,
and every measurement with its date — is in [KNOWLEDGE.md](KNOWLEDGE.md).

---

## The problem

Kage renders a **motion comic**: a camera moves over a panel, effects play on top, and
nothing inside the frame ever moves. That is a finished, solid thing, and it is a small
thing. The goal is an anime engine that runs on an ordinary laptop and costs nothing.

The obvious route is to generate every frame with a video diffusion model. A 20-minute
episode is 28,800 frames. Stable Video Diffusion on a rented L4 produces 25 frames per
call in roughly a minute. That is about 19 hours of GPU time per episode, on hardware
most users do not have, at a price this project does not have. It is not a near miss; it
is off by three orders of magnitude.

## The thesis

**Anime is already a cost-optimised medium. Copy how it solved this instead of
brute-forcing it.**

TV anime draws about **8 unique drawings per second**, not 24 — "animating on threes",
each drawing held for three frames. It holds cels and moves the camera over them. During
dialogue, only the mouth and eyes move. Limited animation was invented in the 1960s for
exactly our problem: convincing animation on a budget that cannot afford every frame.

So the engine does not generate frames. It **rigs the artwork as a puppet and performs
it**, and the performance is data.

---

## Frame budget

Everything downstream depends on how long one frame takes, so that was settled first,
by measurement rather than estimate. The starting pipeline was float32 end to end.

Measured at 1280×720 with the `shonen_kinetic` grade on this machine (Ryzen 5 5600H,
CPU only — no GPU is involved in any of this):

| preset | start | after uint8 core | after full conversion |
|---|---|---|---|
| camera only | 17.8 fps | 29.7 fps | **97.9 fps** |
| + parallax + drift | 13.7 fps | 23.6 fps | **53.8 fps** |
| + speed lines + grain | 10.2 fps | 11.8 fps | **49.2 fps** |

A 20-minute episode on the heavy preset: **35.1 minutes of frame work down to 9.8**.

The work was not clever. It was noticing that a float32 promotion is *sticky* — every
effect downstream of one takes its own float path — so a single effect promoting the
frame made the whole chain expensive:

- The four overlay effects each blended toward a flat colour in float32. Speed lines
  alone cost 23.4 ms/frame, 36% of the total, and dragged the grade, vignette and grain
  onto float with it.
- `_float_effects` named bloom, but `_apply_grade` already promotes and converts back
  internally. So the loop cast to float32 (2.6 ms) and the grade cast straight back
  (4.1 ms): **6.7 ms a frame to accomplish nothing**, on every shot of a preset carrying
  `bloom=0.06`.
- `maps_at` rebuilt two full-resolution float32 arrays from zeros every frame, when each
  remap grid is just `base + depth * a + drift * b` with only the scalars changing.

None of it changed the picture. Each rewrite was diffed against the float version by
rendering the same shot both ways: worst case **1/255** on pixels, and 6×10⁻⁵ px on remap
coordinates. Three tests pin that equivalence.

### What one frame costs now

No single operation dominates any more; the time is spread across cv2 primitives, which
is the shape you want before adding work. Representative costs at 1280×720:

| operation | ms | tier |
|---|---|---|
| `warpAffine` (camera crop) | 1.28 | optical |
| `remap` (parallax + drift) | 1.41 | cel |
| LUT grade | 0.92 | cel |
| saturation | 1.50 | cel |
| vignette | 1.04 | optical |
| grain | 1.62 | optical |
| impact flash | 0.90 | optical |
| uint8 composite, full frame | 2.97 | — |
| uint8 composite, character bbox only | **0.92** | — |
| float32 composite, full frame | 23.00 | — |

The last three rows are the rule for everything that follows: **never composite a full
frame in float, and prefer a region to a frame.** A character occupying a sixth of the
frame costs a third of what the full frame costs.

---

## On twos and threes: an honest correction

An earlier version of this plan claimed that holding cels on threes was primarily a **3×
performance win** and should be done immediately. Having measured it, that is wrong, and
it is worth saying why.

A held cel is only free if the held thing is reused. In this pipeline nearly all the work
happens *after* the camera crop, in screen space — and the camera moves every frame, so
every frame's crop is different and nothing downstream can be reused. The genuinely
holdable work today (`remap`, LUT, saturation) is about **3.8 ms of a 20 ms frame**.
Holding it on threes saves roughly 12%, not 200%.

So on-threes is adopted for two real reasons, neither of which is today's frame budget:

1. **It is what makes it read as anime.** Smooth 24 fps interpolation reads as motion
   comic or as AI video. The hold is the aesthetic.
2. **It becomes a genuine 3× later.** Once Phase 4's per-region deformation lands, the
   cel tier is the expensive tier, and holding it is the saving that makes the rig
   affordable.

The pipeline therefore needs a **cel tier** (deformation, grade — holdable on 2s/3s) and
an **optical tier** (camera, vignette, grain, flash, subtitles — every frame), and the
split has to exist *before* the rig, because retrofitting it afterwards means redoing the
rig. `build_timeline` itself needs no change: a held cel is the same buffer yielded three
times.

---

## Hardware tiers

| tier | hardware | what it does |
|---|---|---|
| **0** | any laptop, CPU only | ONNX segmentation, puppet deformation, lip sync. **This is the product.** |
| **1** | any GPU | Talking Head Anime 4 for better face and head motion |
| **2** | rented GPU, hero shots only | the existing diffusion path, a handful of standout cuts |

Tier 0 is not a fallback. It is the thing everyone can run, and the project fails if it
is not good enough on its own.

The GPU problem is already solved for extraction: **`onnxruntime` is in the shipped
371 MB build** (pymupdf pulls it in), and the models needed are ONNX under permissive
licences. No torch, no new dependency, no download at install time. Sizes and licences
are in [KNOWLEDGE.md](KNOWLEDGE.md).

---

## Build order

Phase 1 is done. The rest is designed, not built.

### Phase 1 — frame budget ✅
Convert the pipeline to uint8, composite by region, publish a real fps number. Done; the
numbers are above, and all three presets clear the target of 3 minutes of output in
under 2 minutes of CPU, which none of them did before.

### Phase 2 — layers (`backend/layers.py`)
An RGBA `Layer` and a ROI-aware `LayerStack` composited per frame. Nothing in the
codebase composites two images today — effects blend a mask toward a *scalar colour*, and
there is no fourth channel anywhere (`_load_source` does `.convert("RGB")`).

Add the cel/optical tier split here, not later.

*Done when:* a character PNG renders over a background with correct alpha, at the Phase 1
frame budget.

### Phase 3 — extraction (`backend/characters.py`)
ONNX `anime-seg` → alpha matte per panel, cached beside the panel like depth already is.
Background inpainting behind the character (`cv2.inpaint` to start) — but note that
for hair this turned out to be unnecessary and actively harmful: what sits behind moving
hair is more hair, already on the page, and inpainting it put flat white across a
subject's jaw and erased her chin outline. Reach for the original drawing before
inventing anything. CPU fallback reusing
the ink mask `_analyse_ink()` already computes and currently throws away.

**Kill criterion:** if segmentation cannot clear roughly 0.7 IoU on real manga panels,
stop here and ship Phases 1–2 as a better motion comic. This is the phase most likely to
fail, and it should be allowed to fail cheaply.

*Done when:* a real panel yields a clean character layer and a filled background.

### Phase 4 — the rig (`backend/rig.py`)

**This phase was planned around displacing pixels, and that has now been shown not to
work.** Five attempts at warping hair were each rejected on sight; the measurements are
in `KNOWLEDGE.md` §1.6. The short version:

- A warp moves texture inside a fixed outline. The silhouette changed 0.35–0.60% while
  the texture inside changed 51 grey levels, which is what "the photo is wobbling"
  describes.
- Manga line art is binary ink on paper. Strands are 1–2 px wide with a median 8 px gap,
  so most of the hair is high-contrast edge, and displacing it by *any* amount flips edge
  pixels black to white. At half the strand spacing the motion is nearly invisible and
  11.3% of the panel still blows out.

So a rig cannot be a displacement field over line art. **A strand is a line, not a
texture: to move it you redraw it.** The route is to extract the drawing as curves and
re-render them each frame.

`lineart_anime.onnx` (207 MB, MIT, CPU, no new dependency) does the extraction well and
is verified working — see `KNOWLEDGE.md` §2 for the two traps. What is not solved is
tracing that line art into strand curves: a hand-rolled skeleton tracer keeps 53.9% of
the skeleton in 23 px fragments, because lines cross constantly and cutting at every
crossing loses the drawing. **That tracer is the blocking problem for this phase.**

Displacement still has uses where the thing being moved is a *region* rather than line
work — a whole limb or head shifting, where the lines travel together and their spacing
relative to each other does not change.

**Not ARAP.** It needs scipy, which is not in the build and is not worth adding for this.

**Mouths cannot be warped into existence.** `cv2.remap` rearranges pixels that already
exist; an open mouth needs teeth, tongue and a dark cavity that are not in a closed-mouth
drawing. Warping a closed mouth open produces a smeared skin-coloured hole. Blinks work
the other way round — closing an eye hides information, which a warp can do — so blinks
are warped and **mouths are composited from drawn viseme sprites**.

*Done when:* a still character blinks, breathes and sways at under 2 ms/frame.

### Phase 5 — performance track (`backend/performance.py`)
A per-frame event timeline generalising `build_subtitle_track`, carrying mouth shapes,
blinks, gestures, camera and FX. Animation becomes data rather than code.

*Done when:* the same character performs two different scripts from data alone.

### Phase 6 — lip sync
Six visemes (A/I/U/E/O/closed) is what anime actually uses.

The plan previously said VITS' phoneme timing could be recovered from the model output.
It cannot: `VitsModelOutput` carries waveform, sequence lengths, spectrogram, hidden
states and attentions, the attentions are text-encoder self-attention rather than
alignment, and the duration tensor is discarded internally. The tokenizer is
character-level with 39 tokens and no phonemes, and the duration predictor is stochastic.

Two routes, in order: **ship energy-envelope timing first** (robust, no model internals,
good enough for six visemes), then add a forward hook on `model.duration_predictor` —
about fifteen lines — for real per-phoneme timing.

Also invert the causality in `cinematic_director.py`: shot length currently comes from
style presets and dialogue is cut to fit. For anime that is backwards — the line sets the
shot.

### Phase 7 — the proof
One manga scene, 2–3 minutes, start to finish. Judged by watching it. That is the only
test that matters.

---

## Known risks

- **Extraction on black-and-white line art is the real risk.** `anime-seg` is trained on
  coloured anime art; manga is screentone and hatching. Hence the kill criterion.
- **Tone crawl — confirmed, and worse than this entry assumed.** It is not only
  screentone and not only sub-pixel. Any displacement of any size over dense line work
  flips edge pixels between ink and paper, because the drawing is binary rather than
  continuous-tone. Measured in `KNOWLEDGE.md` §1.6. This is what rules out warping as a
  way to animate drawn detail, and it is the single most important finding so far.
- **It will look like Live2D, not Ghibli.** Puppet deformation has a ceiling. Accept it
  as the aesthetic or be disappointed.
- **Character consistency across panels.** The same person drawn at different angles
  needs re-rigging per panel. Unsolved.
- **Hands and complex poses warp badly.** Anime hides this with cuts; so will we.
- **"Reads as anime" may be the wrong target.** What separates anime from manga visually
  is colour as much as motion, and this engine does not colourise. **"Animated manga"** is
  the honest description and probably the better product.
- **20 minutes is still ambitious.** The compute works. Having enough distinct rigged
  material to fill 20 minutes without looking repetitive is the harder problem.

## Constraints that bind

- **No `ultralytics`.** AGPL-3.0, and it requires `opencv-python>=4.7`, which collides
  with the repo's `opencv-python-headless`. ONNX and `onnxruntime` only.
- **No scipy.** Not in the build; the lattice approach avoids needing it.
- **New dataclass fields go last.** Tests construct `ShotRenderer` and `ShotSpec`
  positionally.
- **Every phase adds a measurement, not an assertion** — per `ENGINEERING.md`. The claim
  that the warp cost "about a millisecond a frame" survived in a docstring for months and
  was out by a factor of thirteen.
