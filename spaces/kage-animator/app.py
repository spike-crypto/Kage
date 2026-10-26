"""Kage Animator - image-to-video for a single manga panel, on ZeroGPU.

Kage runs on a 4 GB card, which is not enough for video diffusion. This Space is the
piece that needs a real GPU: it takes one panel and returns a short clip of it moving.
Everything else - page selection, framing, camera, grade, score, narration - stays local.

Why Stable Video Diffusion: it conditions on the *image alone*. Models that need a text
prompt are a poor fit here, because nothing in the pipeline knows what a panel depicts
well enough to describe it, and a wrong prompt actively fights the artwork. SVD is told
nothing and simply moves what it is given.

ZeroGPU gives a time-boxed slice of an A10G per call, so the work is sized to fit inside
one: 25 frames, decoded in small chunks, model offloaded between calls.
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr
import spaces
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

MODEL_ID = os.environ.get("KAGE_VIDEO_MODEL",
                          "stabilityai/stable-video-diffusion-img2vid-xt")

#: SVD was trained at this shape. Feeding it a tall manga panel unchanged produces
#: smearing, so panels are fitted into it rather than squashed.
WIDTH, HEIGHT = 1024, 576

_pipe = None


def _pipeline():
    global _pipe
    if _pipe is None:
        _pipe = StableVideoDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16"
        )
        _pipe.enable_model_cpu_offload()
        _pipe.unet.enable_forward_chunking()
    return _pipe


def fit(image: Image.Image) -> Image.Image:
    """Letterbox a panel into SVD's frame without distorting the drawing.

    A panel is whatever shape the artist drew. Stretching it to 16:9 warps faces, which
    is the one thing a reader notices immediately.
    """
    image = image.convert("RGB")
    scale = min(WIDTH / image.width, HEIGHT / image.height)
    new = image.resize((max(1, int(image.width * scale)),
                        max(1, int(image.height * scale))), Image.LANCZOS)
    # Pad with the panel's own paper tone rather than black, so the fill does not read
    # as a letterbox bar the camera then moves across.
    edge = new.crop((0, 0, new.width, 1)).resize((1, 1)).getpixel((0, 0))
    canvas = Image.new("RGB", (WIDTH, HEIGHT), edge)
    canvas.paste(new, ((WIDTH - new.width) // 2, (HEIGHT - new.height) // 2))
    return canvas


# ZeroGPU charges the *reservation*, not the time used, and a free account's daily
# allowance is small. A 20-step 25-frame run measured ~60s on an A10G slice, so asking
# for 150 burnt more than twice the quota each call for nothing.
@spaces.GPU(duration=75)
def animate(image: Image.Image, motion: int = 90, steps: int = 20,
            seed: int = 0, fps: int = 8) -> str:
    """Animate one panel. Returns a path to an mp4."""
    if image is None:
        raise gr.Error("Give me a panel to animate.")

    generator = torch.manual_seed(int(seed))
    frames = _pipeline()(
        fit(image),
        decode_chunk_size=4,          # keeps VAE decode inside the slice's memory
        motion_bucket_id=int(motion),  # how much movement SVD tries for
        noise_aug_strength=0.02,       # low: stay faithful to the drawing
        num_inference_steps=int(steps),
        generator=generator,
    ).frames[0]

    out = os.path.join(tempfile.mkdtemp(), "panel.mp4")
    export_to_video(frames, out, fps=int(fps))
    return out


with gr.Blocks(title="Kage Animator") as demo:
    gr.Markdown(
        "## Kage Animator\n"
        "Image-to-video for one manga panel. Built for "
        "[Kage](https://github.com/spike-crypto/Kage), which runs everything else "
        "locally and calls this only for the shots worth animating.\n\n"
        "Conditioned on the image alone - no prompt. Motion strength is the main dial: "
        "low holds the drawing, high moves more and distorts more."
    )
    with gr.Row():
        with gr.Column():
            panel = gr.Image(type="pil", label="Panel")
            motion = gr.Slider(1, 255, value=90, step=1, label="Motion strength",
                               info="SVD's motion bucket. 60-120 suits line art.")
            steps = gr.Slider(10, 30, value=20, step=1, label="Steps")
            seed = gr.Number(value=0, precision=0, label="Seed")
            fps = gr.Slider(4, 12, value=8, step=1, label="FPS")
            go = gr.Button("Animate", variant="primary")
        with gr.Column():
            out = gr.Video(label="Clip")

    go.click(animate, [panel, motion, steps, seed, fps], out, api_name="animate")

demo.queue(max_size=8).launch()
