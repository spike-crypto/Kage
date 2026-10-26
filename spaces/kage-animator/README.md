---
title: Kage Animator
emoji: 🌀
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: other
short_description: Image-to-video for a single manga panel, for the Kage studio
---

# Kage Animator

Image-to-video for **one manga panel**, on ZeroGPU.

[Kage](https://github.com/spike-crypto/Kage) turns manga pages into cinematic motion
comics and runs on a 4 GB card. Everything it does — page selection, panel segmentation,
framing, camera, depth parallax, grade, score, narration — stays on that machine. Video
diffusion is the one part that does not fit, so it lives here and Kage calls it only for
the shots worth animating.

## Model

[`stabilityai/stable-video-diffusion-img2vid-xt`](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt),
conditioned on the **image alone**. Models needing a text prompt are a poor fit: nothing
in the pipeline knows what a panel depicts well enough to describe it, and a wrong prompt
fights the artwork. SVD is told nothing and moves what it is given.

Panels are letterboxed into SVD's 1024×576 training shape using the panel's own paper
tone, never stretched — warping a face is the first thing a reader notices.

**Licence:** the SVD weights are under the Stability AI Community License, which is free
for non-commercial use and for commercial use below a revenue threshold. Check it before
shipping anything commercial.

## API

```python
from gradio_client import Client, handle_file

client = Client("viperlurk/kage-animator")
clip = client.predict(
    image=handle_file("panel.png"),
    motion=90,      # SVD motion bucket; 60-120 suits line art
    steps=20,
    seed=0,
    fps=8,
    api_name="/animate",
)
```

Returns an mp4 of 25 frames. Motion strength is the main dial: low holds the drawing,
high moves more and distorts more.

## Honest limits

SVD is trained on photographic video. On black-and-white line art it produces plausible
drift and parallax rather than character animation — no in-betweens, no redrawn cels.
It animates the *image*; it does not understand that a figure has limbs.
