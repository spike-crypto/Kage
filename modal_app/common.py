"""Shared Modal pieces: one app, one image, one weight cache.

Everything that needs a GPU runs here. Everything that does not stays on the local
machine, because moving it would cost credits and buy nothing - CLIP, depth estimation
and the voice together are under 1.5 GB and already run on a 4 GB card.

Three things keep the $30/month allowance from leaking, and all three matter more than
the model choice:

1. **Weights live on a Volume.** Downloaded once, not per call. Without this every
   invocation pays to re-fetch gigabytes, which is the fastest way to burn the credit.
2. **Containers scale to zero quickly.** An idle warm container bills. The Volume makes
   the next cold start cheap enough that keeping one alive is the worse trade.
3. **Each call reports the GPU seconds it used.** Kage records the real cost rather than
   estimating from wall-clock time, which includes queueing it did not pay for.
"""

from __future__ import annotations

import modal

APP_NAME = "kage-gpu"

#: One cache for every function. Models are addressed by their Hugging Face repo id, so
#: two functions wanting the same weights share them instead of each paying to download.
WEIGHTS = modal.Volume.from_name("kage-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

#: Pinned deliberately. An unpinned image rebuilds on someone else's release schedule,
#: and a rebuild is both a surprise and a cost.
BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.26.2",
        "imageio[ffmpeg]==2.36.0",
        "pillow==11.0.0",
    )
    .env({"HF_HOME": WEIGHTS_DIR, "HF_HUB_DISABLE_TELEMETRY": "1"})
)

app = modal.App(APP_NAME)


def gpu_seconds(start: float) -> float:
    """Seconds of GPU time this call actually used, for honest cost accounting."""
    import time

    return round(time.perf_counter() - start, 3)
