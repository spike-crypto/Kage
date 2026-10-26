"""Image-to-video: one manga panel in, a short clip out.

Runs Stable Video Diffusion on an L4 — the cheapest GPU Modal offers that holds the
model comfortably, at $0.000222/second. SVD conditions on the image alone, with no text
prompt, which matters because nothing upstream knows what a panel depicts well enough to
describe it and a wrong prompt fights the artwork.

The class keeps weights loaded between calls in the same container, so a burst of shots
in one film pays the load cost once.
"""

from __future__ import annotations

import time

import modal

from .common import BASE_IMAGE, WEIGHTS, WEIGHTS_DIR, app, gpu_seconds

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

#: SVD's training shape. A panel is letterboxed into it rather than stretched - warping
#: a face is the first thing a reader notices.
WIDTH, HEIGHT = 1024, 576


@app.cls(
    image=BASE_IMAGE,
    gpu="L4",
    volumes={WEIGHTS_DIR: WEIGHTS},
    timeout=900,
    # Scale to zero soon after a film finishes. The Volume makes the next cold start
    # cheap, so paying to idle is the worse trade.
    scaledown_window=120,
)
class Animator:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import StableVideoDiffusionPipeline

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16",
            cache_dir=WEIGHTS_DIR,
        )
        self.pipe.to("cuda")
        self.pipe.unet.enable_forward_chunking()
        WEIGHTS.commit()        # persist anything just downloaded, for the next container

    @modal.method()
    def animate(self, image_bytes: bytes, motion: int = 90, steps: int = 20,
                seed: int = 0, fps: int = 8) -> dict:
        """Animate one panel. Returns the mp4 bytes and what the call actually cost."""
        import io

        import torch
        from diffusers.utils import export_to_video
        from PIL import Image

        started = time.perf_counter()
        panel = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        scale = min(WIDTH / panel.width, HEIGHT / panel.height)
        fitted = panel.resize((max(1, int(panel.width * scale)),
                               max(1, int(panel.height * scale))), Image.LANCZOS)
        # Pad in the panel's own paper tone, so the fill does not read as a letterbox
        # bar that the camera then moves across.
        edge = fitted.crop((0, 0, fitted.width, 1)).resize((1, 1)).getpixel((0, 0))
        canvas = Image.new("RGB", (WIDTH, HEIGHT), edge)
        canvas.paste(fitted, ((WIDTH - fitted.width) // 2, (HEIGHT - fitted.height) // 2))

        frames = self.pipe(
            canvas,
            decode_chunk_size=4,
            motion_bucket_id=int(motion),
            noise_aug_strength=0.02,
            num_inference_steps=int(steps),
            generator=torch.manual_seed(int(seed)),
        ).frames[0]

        out = "/tmp/panel.mp4"
        export_to_video(frames, out, fps=int(fps))
        with open(out, "rb") as handle:
            video = handle.read()

        return {
            "video": video,
            "gpu_seconds": gpu_seconds(started),
            "gpu": "l4",
            "model": MODEL_ID,
            "frames": len(frames),
            "fps": int(fps),
        }


@app.local_entrypoint()
def main(panel: str = "", motion: int = 90, steps: int = 20) -> None:
    """Smoke test: `modal run -m modal_app.animate --panel path/to/panel.png`."""
    from pathlib import Path

    if not panel:
        print("pass --panel <image> to try a real panel")
        return
    data = Path(panel).read_bytes()
    result = Animator().animate.remote(data, motion=motion, steps=steps)
    out = Path("modal_clip.mp4")
    out.write_bytes(result["video"])
    rate = 0.000222
    print(f"{result['frames']} frames, {result['gpu_seconds']}s on {result['gpu']} "
          f"= ${result['gpu_seconds'] * rate:.4f} -> {out}")
