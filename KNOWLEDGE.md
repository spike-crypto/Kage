# Knowledge base

Everything learned while building Kage that is not obvious from the code: measurements
with their dates, models with their licences and sizes, what the research actually says,
what the free infrastructure actually allows, and the things that turned out to be false.

This exists so nothing has to be rediscovered. Add to it; do not prune it.

The architecture it supports is in [ANIME_ENGINE.md](ANIME_ENGINE.md).

---

## 1. Measurements

All on the development machine unless stated: **Ryzen 5 5600H, 15.4 GB RAM, GTX 1650
4 GB**, Windows 11, Python 3.12, `opencv-python-headless` 5.0.0. All CPU-only.

### 1.1 Frame operations, 1280×720 (2026-10-27)

| operation | ms/frame |
|---|---|
| `warpAffine`, camera crop | 1.28 |
| `remap`, uint8 source | 1.41 |
| `remap`, float32 source | 6.33 |
| LUT grade | 0.92 |
| saturation, two `cvtColor` | 1.50 |
| vignette, `cv2.multiply` uint8 | 1.04 |
| grain, `cv2.add` int16 | 1.62 |
| impact flash, `convertScaleAbs` | 0.90 |
| alpha composite, float32 full frame | 23.00 |
| alpha composite, uint8 `cv2` full frame | 2.97 |
| alpha composite, uint8 character bbox (⅙ area) | **0.92** |
| `astype(float32)` promotion | 2.58 |
| `np.clip` + `astype(uint8)` back | 4.11 |

**The rule:** a float32 promotion costs 6.7 ms round trip and is *sticky* — every effect
after it takes its own float path. One effect promoting makes the whole chain expensive.

**The second rule:** composite by region, never by frame. 0.92 ms vs 2.97 ms vs 23.00 ms
for the same visual result.

### 1.2 End-to-end throughput, 1280×720, `shonen_kinetic` (2026-10-27)

The grade matters: `shonen_kinetic` carries `bloom=0.06`, which used to force every shot
onto the float path regardless of effects. Benchmarks run without a grade are not
comparable and will look about 2× too good.

| preset | start | uint8 core | full conversion |
|---|---|---|---|
| camera only | 17.8 | 29.7 | **97.9** fps |
| + parallax + drift | 13.7 | 23.6 | **53.8** fps |
| + speed lines + grain | 10.2 | 11.8 | **49.2** fps |

20-minute episode (28,800 frames), heavy preset: **35.1 min → 9.8 min** of frame work.

Same-script profile, 72 frames: **4.720 s → 1.304 s, 3.6×.**

Where it went, heavy preset, before → after:

| | before | after |
|---|---|---|
| `_apply_speed_lines` | 23.4 ms | 0.9 ms |
| `_apply_vignette` | 9.4 ms | 1.0 ms |
| `maps_at` | 8.3 ms | 2.3 ms |
| `_apply_grade` | 19.7 ms | ~4 ms |

`_apply_vignette` measured 9.4 ms in situ against 1.04 ms in isolation — because it was
receiving a float32 frame and taking its float path. That gap is the clearest evidence
for the stickiness rule above.

### 1.3 Equivalence checks (2026-10-27)

Every optimisation was diffed against the code it replaced by rendering the same shot
both ways:

| change | worst difference |
|---|---|
| overlay blend, uint8 vs float | 1/255 |
| bloom, uint8 vs float | 1/255, 0% of pixels differ by >1 |
| remap grids | 6×10⁻⁵ px on coordinates, 1/255 on pixels |

Pinned by `test_uint8_overlay_blend_matches_float_arithmetic`,
`test_bloom_stays_in_uint8_and_grade_returns_uint8`,
`test_warp_maps_reuse_buffers_without_changing_them`.

### 1.4 Holdable work, for on-threes (2026-10-27)

Cel tier — deformation, grade, saturation — is **≈3.8 ms of a ≈20 ms frame**. Holding it
on threes saves about 12%, not the 3× an earlier plan assumed. Nearly all the work
happens after the camera crop, in screen space, and the camera moves every frame, so
there is little to reuse. On-threes is an aesthetic decision now and a performance
decision after Phase 4.

### 1.5 Models, local (earlier sessions)

| model | load | per panel | VRAM |
|---|---|---|---|
| Depth Anything V2 Small | ~1 s | 0.3 s | 108 MB |

### 1.6 Animating hair by warping — why it cannot work (2026-10-27)

Five attempts, on two panels, each rejected on sight by the person who asked for it.
The measurements below are why, and they generalise past hair to any dense line work.

**A warp cannot change a silhouette.** Displacing pixels inside a fixed shape moves the
texture and leaves the outline where it was. Measured over a four-second clip, the
hair's outline changed by **0.35–0.60%** of its area while the texture inside it changed
by 51 grey levels. That gap is exactly what "the photo is wobbling" describes. Cutting
the hair out as a layer and bending it took the outline change to **7.45%**.

**Too much of the frame moving reads as the frame moving.** On a chapter-title splash
page where hair covered 76.6% of the picture, **66% of the whole page** was displaced by
more than 2 px, and it read as a wobble no matter what else was fixed. The same code on
an ordinary panel moved **27%**, with the same outline change. Picking the most
hair-dominated page in the volume as the test case — which is what asking CLIP for "the
most hair" gets you — was itself the bug.

**A spatial wave through the hair is not hair.** Optical flow on the output showed the
motion reversing direction **2–3 times** between root and tip, with neighbouring pixels
agreeing 0.993: a rubber sheet with a ripple in it. A lock of hair reverses **zero**
times — all of it goes one way, the tip travels further and arrives later. Replacing
1.6 spatial wave crests with a 0.15-cycle **time lag** took reversals to 0.17 and raised
the outline change at the same time.

**And then the hard limit.** Manga line art is binary ink on paper, not continuous tone.
On this panel strands are 1–2 px wide with a **median 8 px gap**, so a large share of
the hair is high-contrast edge. Displace it at all and edge pixels swing black to white:

| sway | pixels blown out at some point |
|---|---|
| 12.5 px (1.56× strand spacing) | 19.5% of the panel |
| 7.8 px | 16.4% |
| 4.0 px (0.5× spacing) | 11.3% |

There is no amplitude small enough to avoid it and large enough to see. **Warping cannot
animate dense line art.** To move a drawn strand you have to redraw it, because a strand
is a line, not a texture.

**What the backdrop should be.** Step 4 of the plan assumes the region a moving layer
uncovers must be inpainted. For hair that is wrong and expensive: behind moving hair in
a dense mass is *more hair*, and the page already contains it. Inpainting it instead put
flat white across the subject's jaw and erased her chin outline, because beside a face
the only pixels to borrow from are skin. Using the original drawing untouched costs some
silhouette change at the outer edge (7.57% rather than 10.21%) and removes the artifact
completely.

**Line art is drawn in both polarities.** Near the subject's head the hair is dark
strokes on white paper; at the edges of the same drawing it flips to white strokes on
black. Anything keyed on "ink is the dark pixels" works on half an image and fails on
the other half. Gradient-based methods (structure tensors, the line-art model below) are
unaffected.

---

## 2. Models for the anime work

Not yet integrated. Verified for licence, size and format.

| model | job | size | licence |
|---|---|---|---|
| `skytnt/anime-seg` (`isnetis.onnx`) | character matting → alpha | 176 MB | Apache-2.0 |
| `deepghs/anime_person_detection` | locate characters | 103 MB | MIT |
| `deepghs/imgutils-models` (`lineart/lineart_anime.onnx`) | line extraction | **207 MB, run and verified** | MIT |
| `dreMaz/AnimeInstanceSegmentation` | instance masks | — | MIT |

**`onnxruntime` 1.31.0 is already in the shipped build**, pulled in by pymupdf. torch is
not shipped. So the whole extraction stage runs CPU-only with no new dependency.


### `lineart_anime.onnx` — verified working, with two traps (2026-10-27)

Downloaded to `data/models` and run on CPU through the `onnxruntime` already in the
build. On a manga panel it produces clean, continuous, flowing strand lines — far better
than `cv2.adaptiveThreshold`, which breaks strokes wherever local contrast dips. This is
the representation hair animation needs, and it costs no new dependency and no GPU.

Two things will waste an afternoon if you do not know them:

1. **Pad each side to a power of two.** It is a deep U-Net and its skip connections only
   line up if every level halves exactly. Padding to a multiple of 8 fails with
   `Non concat axis dimensions must match: Axis 2 has mismatched dimensions of 8 and 9`;
   padding to 64 moves the same failure one level deeper onto axis 3. Resizing instead
   of padding distorts line directions and should be avoided.
2. **It emits lines LOW and paper HIGH.** Normalise and then invert, or every
   `map > threshold` selects the paper. Thresholded the wrong way round it reported 41%
   of the panel as "line" and traced the gaps between strands, which looks like the
   model failing rather than the caller.

Input shape is `[batch, 3, h, w]` float, output `[batch, 1, h, w]`.

### Skeleton tracing is the open problem, not line extraction

Turning that clean line art into strand curves is where it currently stops. A hand-rolled
Zhang–Suen skeleton, cut at junctions and stitched back by direction continuity, keeps at
best **53.9%** of the skeleton in fragments with a **median length of 23 px**, where real
strands run for hundreds. Lines cross constantly in hair, and cutting at every crossing
then guessing how to rejoin loses most of the drawing.

This is consistent with the literature: AnimeInbet matches line *vertices* with a trained
model rather than tracing, which is a strong hint that naive tracing does not hold up.

### Forbidden dependencies

- **`ultralytics`** — AGPL-3.0, and requires `opencv-python>=4.7`, which collides with
  the repo's `opencv-python-headless` 5.0.0. Use ONNX directly.
- **`scipy`** — not in the build. Rules out ARAP; use a coarse control lattice.

Also absent, checked 2026-10-27: **`cv2.ximgproc`** (so no `thinning`), **`skimage`**,
**`svgwrite`**, **`potrace`**. Thinning has to be written by hand. **`networkx` is
present**, which helps if the tracer becomes a proper graph problem.

---

## 3. Research

- **Talking Head Anime 4** (WACV 2025) — 512×512 at 30 FPS on a consumer GPU. MIT code,
  CC-BY model. The Tier 1 face/head path.
- **AnimeInbet** (ICCV 2023) — inbetweening by matching line vertices rather than
  interpolating pixels. Relevant if frame interpolation is ever wanted.
- **ToonCrafter** — Apache-2.0 per its model card, 10.5 GB, 512×320, 16 frames.
  Tier 2 only.
- **APES** — part segmentation for puppet rigs.
- **ARAP / Igarashi 2005** — as-rigid-as-possible shape manipulation. The textbook
  answer for puppet warping, and ruled out here: needs scipy.

### Anime timing conventions

TV anime is drawn on twos and threes — roughly **8 unique drawings per second**, each
held 2–3 frames. Camera moves are on ones (the cel physically slides under the camera),
which is why the engine needs a cel tier and an optical tier rather than one frame rate.
During dialogue only the mouth and eyes move. Six visemes (A/I/U/E/O/closed) is the
standard mouth set.

### Things that are not what they sound like

- **`netflix/void-model`** — video **object removal**, not animation. Takes video plus a
  mask, erases objects and their shadows and reflections. Apache-2.0, 22.3 GB (32.5 GB
  with the T5-XXL encoder), needs 40 GB+ VRAM. Useful one day for erasing speech bubbles
  from artwork; not a route to animation.
- **PixAI** — anime **image** generation from text prompts. No video, no manga input.
  Not a competitor.

### VITS lip sync — what does not work

`VitsModelOutput` exposes waveform, sequence lengths, spectrogram, hidden states and
attentions. The attentions are **text-encoder self-attention, not alignment**, and the
duration tensor is discarded internally. The tokenizer is **character-level, 39 tokens,
no phonemes**, and the duration predictor is **stochastic**.

So phoneme timing is *not* recoverable from the output, as an earlier plan assumed. Two
routes: an energy envelope (robust, no model internals, adequate for six visemes), or a
forward hook on `model.duration_predictor` (~15 lines) for real timing. Ship the envelope
first.

---

## 4. Infrastructure, verified

Checked against real accounts in October 2026. Most published guidance on this is wrong
or out of date.

| service | finding |
|---|---|
| **Modal** | $30/month Starter credit, but **a payment method is required for any GPU**. L4 and T4 both refused on a card-less account; the gate is account-wide. CPU functions run fine. The pricing page does not mention this; the API does. |
| **HF Spaces** | Gradio/Docker Spaces need a paid plan. Free personal accounts get **2 ZeroGPU Gradio Spaces**. `viperlurk/kage-animator` deployed on `zero-a10g`, SVD working — one 60 s clip succeeded, then quota exhausted (instant `error: null`). |
| **HF Inference API** | Token valid for downloads; hosted inference returns **402, no credits**. |
| **Cloudflare** | Tunnel available (API 200). **R2 not enabled** on the account (`10042: Please enable R2`). Workers AI has **no video model**. |
| **Render.com free** | 512 MB RAM, 0.1 CPU, no persistent disk. Unusable. |
| **Colab free** | FAQ bans "connecting to remote proxies" and names web-UI content generation as a common cause of termination. Not a backend. |
| **GitHub Actions** | Terms prohibit "activity unrelated to the production, testing, deployment, or publication of the software project". Not a render farm. |

**Conclusion:** there is no free GPU that can be relied on as infrastructure. This is the
single strongest argument for the CPU-first design.

---

## 5. Claims that turned out to be false

Kept deliberately. Each cost real time, and the pattern — a plausible number nobody
measured — is the thing to watch for.

| claim | reality |
|---|---|
| `animation.py`: the warp costs "about a millisecond a frame" | 13.66 ms in float32, 6.33 ms in uint8. Out by 13×. Lived in a docstring for months. |
| Holding cels on threes is a 3× performance win | ≈12% in the current pipeline. The holdable work is 3.8 ms of a 20 ms frame. |
| VITS alignment is recoverable from the model output | It is not. Duration tensor discarded internally; attentions are the wrong thing. |
| Modal needs no payment method | It does, for every GPU. |
| README: "no GPU, no paid API, no model weights" | False on all three counts at the time. Rewritten. |
| The shot table had a numpy-array bug | It did not. Cells are Python lists. The diagnosis was wrong. |
| Mouths can be warped open | They cannot. `cv2.remap` rearranges existing pixels; an open mouth needs teeth and a cavity that a closed-mouth drawing does not contain. Blinks work (hiding information); mouths need drawn sprites. |
| Masked drift would animate hair | It does not. It moves texture inside a fixed outline, which reads as a wobbling photograph. Four rounds of fixing masks, directions, fills and amplitudes changed nothing, because the model of the motion was wrong. |
| There was an edge leak around the matte | There was not. The audit defined background as "matte below 40", so eroding the matte moved the measuring region into the hair and the number could not improve. Measured properly the original scored 19:1 and the "fix" 15:1 — the fix was worse than the thing it fixed. |
| "Line work stays sharp, 96% retained" means the lines survive | It does not. That metric measures edge *energy*, which stays high while the edges move to the wrong places every frame. It passed on every version the user rejected. |
| The white gash was the fill being poor | It was the fill existing at all. Behind moving hair is more hair, already on the page. |
| The line-art model was unsuitable | It is excellent. The caller thresholded its output the wrong way round and traced the paper. |

Two "regression tests" once passed with their fix reverted, and three others gave false
passes because stale dev services held ports 8000 and 8501. Render and test on isolated
ports (`KAGE_PORT`, `KAGE_UI_PORT`).

---

## 6. Bugs worth remembering

Mechanisms that will recur, not a changelog.

- **ctypes truncates handles.** The launcher's orphan protection never worked: without
  `restype`/`argtypes`, ctypes assumes C `int` and truncates a 64-bit `HANDLE`, so
  `AssignProcessToJobObject` failed with error 6 while the code reported success.
  Declare signatures for every Win32 call.
- **`communicate()` flushes stdin itself.** Closing it first raised
  `ValueError: flush of closed file` — on POSIX only, so CI caught it and Windows never
  would have.
- **SQLAlchemy `default=0` applies at INSERT**, so pending objects carry `None`. Routing
  counters silently never learned.
- **A flushed-but-uncommitted transaction holds the write lock.** Running models for
  minutes between flush and commit produced "database is locked". Commit before slow
  work. `busy_timeout=30000`, WAL on.
- **A stale ORM collection is not a cache miss, it is silence.** `project.shots` was
  stale, so narration was written nowhere and nothing errored.
- **Build artefacts leak into releases.** v0.1.0 shipped 20 MB of test state and
  `data/`. The build script now strips `data/`, `__pycache__`, `.env`,
  `.cloudflared_token`.
- **A public repo will happily take copyrighted material.** 206 files / 93 MB of manga
  were nearly committed; caught in a pre-push audit. `.gitignore` now covers archives and
  sample manga directories.
- **Never hold a child's stderr in a pipe you only drain at the end.** Writing frames to
  ffmpeg while its stderr pipe fills up deadlocks both processes, and it looks exactly
  like a slow render rather than a hang. Send stderr to a file, or add `-nostats`.
- **libx264 with `yuv420p` needs even dimensions.** A 277-pixel-wide panel makes ffmpeg
  exit immediately, and the first symptom is `OSError: [Errno 22]` on the pipe write,
  which points at the wrong thing entirely.
- **Flood-fill hole filling must start from a known background pixel.** Flooding from
  image (0, 0) is wrong whenever the shape touches that corner — the fill then consumes
  the shape and the inversion returns the whole image. Pad a one-pixel border first.
- **Filling holes in a mask puts enclosed objects back.** Hair rings a head, so the face
  is a hole in it; hole-filling reinstated the entire face as "hair". Subtract the
  protected region again afterwards.
- **A measurement region defined in terms of the thing being changed cannot detect a
  change.** Twice this session a defect was an artifact of how it was measured, and both
  times a control — a still video, the untouched drawing — was what caught it.

---

## 7. Open questions

- Does `anime-seg` clear ~0.7 IoU on black-and-white screentone manga? **This decides
  whether the project proceeds past Phase 3.**
- Does deforming screentone produce unacceptable crawl, and can flattening fix it?
- How is character identity held across panels drawn at different angles?
- Is "reads as anime" the right target, or is "animated manga" both honest and better?
- Can line-art strands be traced through crossings well enough to animate? This is now
  the single blocking question for hair, and the only one left between here and a strand
  that moves without shimmering.
- Does the same tracing problem appear on faces and clothing, or is hair the worst case
  because its lines are densest?
