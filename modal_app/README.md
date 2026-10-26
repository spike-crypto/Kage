# Kage on Modal

GPU work, deployed serverless. Everything that does not need a GPU stays on the local
machine — CLIP, depth estimation and the voice are under 1.5 GB together and already run
on a 4 GB card. Moving them here would spend credits and buy nothing.

## Status

The framework deploys. The image builds, the Volume is created, the function registers.

**GPU execution is blocked until a payment method is on the Modal account.** Verified
October 2026 against this account:

```
CPU function    ✓ ran remotely      {'python': '3.12.10', 'cpus': 17}
L4  function    ✗ "Please add a payment method to use L4 GPU functions."
T4  function    ✗ "Please add a payment method to use T4 GPU functions."
```

The gate is account-wide, not per-GPU. Modal's pricing page states the $30/month Starter
credit but does not mention the card requirement; the API does. Third-party guides
claiming "no card required" are describing CPU use, or are wrong.

Once a card is added the $30/month credit applies and nothing here changes.

## Deploy

```bash
# credentials come from .env; they are already set if Kage is configured
modal deploy -m modal_app.animate
```

Smoke test a real panel:

```bash
modal run -m modal_app.animate --panel data/projects/project_00116/panels/panel_p007_002.png
```

It prints the measured GPU seconds and the dollar cost of that one call.

## What runs here

| Function | Model | GPU | Rate |
|---|---|---|---|
| `Animator.animate` | `stabilityai/stable-video-diffusion-img2vid-xt` | L4 | $0.000222/s |

SVD conditions on the image alone, with no text prompt. Nothing upstream knows what a
panel depicts well enough to describe it, and a wrong prompt fights the artwork.

## Keeping the $30 from leaking

Three things, and they matter more than which model you pick:

1. **Weights live on a Volume** (`kage-weights`), downloaded once rather than per call.
   Without this, every invocation pays to re-fetch gigabytes — the fastest way to burn
   the credit.
2. **Containers scale to zero after 120s.** An idle warm container bills; the Volume
   makes the next cold start cheap enough that paying to idle is the worse trade.
3. **Each call returns the GPU seconds it actually used**, so Kage records real cost
   rather than estimating from wall-clock time, which includes queueing it did not pay
   for. Those land on the Usage page.

## Adding another model

Copy `animate.py`. Keep three things: the shared `BASE_IMAGE` and `WEIGHTS` volume from
`common.py`, a `@modal.enter()` that loads once per container, and a return value
carrying `gpu_seconds` and `gpu` so the cost is recorded honestly.

### netflix/void-model

Worth being explicit, since it gets mistaken for an animation model: **VOID is video
object removal**, not image-to-video. It takes an existing video plus a 4-value mask and
a prompt describing the scene after removal, and erases objects along with their physical
consequences — shadows, reflections, things that would fall. Netflix built it for VFX
cleanup.

- Apache-2.0, 22.3 GB over two passes (32.5 GB with the T5-XXL text encoder)
- Needs **40 GB+ VRAM**, so A100-80GB here at $0.000694/s
- 384×672, up to 197 frames

It cannot animate a panel: it needs video in, and its job is subtraction. If Kage ever
wants speech bubbles erased from artwork it would be the right tool, and it slots in as
another function — but it is not a route to animation.
