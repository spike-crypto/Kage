"""Local vision model - the part that actually looks at the artwork.

Everything else in Kage is deterministic. This module is the one place a neural network
reads the pages and reports what is on them, so the director can frame a drawing instead
of a page of prose.

Model: OpenAI CLIP ViT-B/32 (~600 MB, MIT licence). It is used zero-shot: each page is
scored against short English descriptions, and the best match wins. CLIP was chosen over a
captioning VLM because it needs no remote code, runs comfortably in 4 GB of VRAM, and the
question being asked - "is this artwork or a wall of text, and how busy is it?" - is
exactly what contrastive image/text scoring is good at.

The weights download once from the Hugging Face hub. That is free and unrelated to the
paid Inference API: nothing here calls a hosted endpoint.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, env

# The hub cache cannot use symlinks on a normal Windows account; the fallback copy works
# fine and the warning is just noise.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

log = logging.getLogger("kage.vision")

MODEL_ID = env("VISION_MODEL", "openai/clip-vit-base-patch32")
MODEL_CACHE = Path(env("MODEL_CACHE", str(DATA_DIR / "models")))

_LOCK = threading.Lock()
_STATE: dict = {"model": None, "processor": None, "device": None, "failed": None}

# --- zero-shot prompt banks -------------------------------------------------
# Each bank is scored independently; the winner within a bank is the label.

KIND_PROMPTS: dict[str, str] = {
    "text": "a page of dense printed text, columns of writing, no pictures",
    "artwork": "a comic book page with drawings of people and scenery",
    "cover": "a plain decorated book cover with a title and no illustration",
    "blank": "a blank empty page, plain paper, nothing on it",
}

SUBJECT_PROMPTS: dict[str, str] = {
    "face": "a close-up of a person's face",
    "figure": "a drawing of a single person standing",
    "duo": "two people facing each other",
    "crowd": "a crowd of many small figures",
    "scenery": "a landscape, building or empty scenery",
    "creature": "a drawing of an animal, bird or monster",
}

ENERGY_PROMPTS: dict[str, str] = {
    "calm": "a calm, still, quiet scene",
    "active": "a dynamic action scene with movement and motion",
}

#: Camera moves that suit each detected subject.
SUBJECT_CAMERA_HINT = {
    "face": ("push_in", "slow_zoom_in"),
    "figure": ("slow_zoom_in", "push_in"),
    "duo": ("pan_left", "pan_right"),
    "crowd": ("ken_burns", "pan_right"),
    "scenery": ("pan_left", "slow_zoom_out", "ken_burns"),
    "creature": ("push_in", "rapid_pan"),
}

#: How tightly to frame each subject, relative to the style's own framing.
SUBJECT_FRAMING_SCALE = {
    "face": 1.25, "figure": 1.12, "duo": 0.95,
    "crowd": 0.82, "scenery": 0.78, "creature": 1.05,
}


@dataclass
class PageInsight:
    """What the model saw on one page."""

    page_index: int
    kind: str = "artwork"
    kind_confidence: float = 0.0
    subject: str = "scenery"
    subject_confidence: float = 0.0
    energy: float = 0.5
    score: float = 0.0          # suitability as a shot, 0..1
    scores: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.kind == "artwork"

    def describe(self) -> str:
        return (f"{self.kind} / {self.subject} "
                f"(energy {self.energy:.2f}, score {self.score:.2f})")


# --------------------------------------------------------------------- model


def is_available() -> bool:
    """True when torch and transformers are importable. Does not load the model."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def _load():
    """Load CLIP once, cached for the process. Returns (model, processor, device)."""
    if _STATE["model"] is not None:
        return _STATE["model"], _STATE["processor"], _STATE["device"]
    if _STATE["failed"]:
        raise RuntimeError(_STATE["failed"])

    with _LOCK:
        if _STATE["model"] is not None:
            return _STATE["model"], _STATE["processor"], _STATE["device"]
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading vision model %s on %s", MODEL_ID, device)

            processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=str(MODEL_CACHE))
            dtype = torch.float16 if device == "cuda" else torch.float32
            # use_safetensors matters for more than tidiness: transformers refuses to
            # torch.load a legacy .bin checkpoint on torch < 2.6 because of CVE-2025-32434.
            # safetensors cannot execute code on load, so it sidesteps the issue entirely.
            common = {
                "cache_dir": str(MODEL_CACHE),
                "use_safetensors": True,
            }
            try:
                # transformers >= 5 renamed torch_dtype to dtype.
                model = CLIPModel.from_pretrained(MODEL_ID, dtype=dtype, **common)
            except TypeError:
                model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=dtype, **common)
            model = model.to(device)
            model.eval()

            _STATE.update({"model": model, "processor": processor, "device": device})
            log.info("Vision model ready on %s", device)
            return model, processor, device
        except Exception as exc:
            _STATE["failed"] = f"{type(exc).__name__}: {exc}"
            log.exception("Vision model failed to load")
            raise RuntimeError(_STATE["failed"]) from exc


def warm_up() -> bool:
    """Load the model ahead of time. Returns False instead of raising."""
    try:
        _load()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ scoring


_TEXT_CACHE: dict[int, object] = {}


def _text_features(prompts: dict[str, str], model, processor, device):
    """Embed a prompt bank once per process - this used to run for every page."""
    key = id(prompts)
    cached = _TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    import torch

    with torch.no_grad():
        inputs = processor(
            text=list(prompts.values()), return_tensors="pt", padding=True
        ).to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    _TEXT_CACHE[key] = feats
    return feats


def _score_bank(image_features, prompts: dict[str, str], model, processor, device):
    """Softmax similarity of one image (or a stack of crops) against a prompt bank.

    With several crops the strongest response per label wins, so a drawing in one half of
    a double-page spread is not averaged away by text in the other half.
    """
    import torch

    labels = list(prompts)
    text_features = _text_features(prompts, model, processor, device)
    with torch.no_grad():
        logits = (image_features @ text_features.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1).float().cpu().numpy()
    return dict(zip(labels, probs.max(axis=0).tolist()))


def _crops(image):
    """Split a double-page spread so each page is seen at full detail.

    CLIP resizes its input to 224px. A wide spread squashed to that size loses the line
    work entirely, which is what made artwork read as 'text'.
    """
    w, h = image.size
    if w > h * 1.25:
        half = w // 2
        return [image.crop((0, 0, half, h)), image.crop((half, 0, w, h))]
    return [image]


def analyse_pages(paths: list[Path], page_indices: list[int] | None = None,
                  batch_size: int = 8) -> list[PageInsight]:
    """Look at every page and report what is on it.

    Raises RuntimeError if the model cannot be loaded - callers decide whether to fall
    back to taking pages in order.
    """
    import torch
    from PIL import Image

    model, processor, device = _load()
    indices = page_indices if page_indices is not None else list(range(len(paths)))
    insights: list[PageInsight] = []

    del batch_size  # each page is scored with its own crops, so batching is per page

    for path, page_index in zip(paths, indices):
        try:
            with Image.open(path) as im:
                crops = _crops(im.convert("RGB"))
        except Exception:
            crops = [Image.new("RGB", (224, 224), (255, 255, 255))]

        with torch.no_grad():
            inputs = processor(images=crops, return_tensors="pt").to(device)
            if device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

        kind = _score_bank(feats, KIND_PROMPTS, model, processor, device)
        subject = _score_bank(feats, SUBJECT_PROMPTS, model, processor, device)
        energy = _score_bank(feats, ENERGY_PROMPTS, model, processor, device)

        # Score by artwork affinity and RANK, rather than trusting an argmax label.
        # On real pages where drawings and writing share the paper, the argmax collapses
        # to "text" almost everywhere, but the artwork probability still orders the pages
        # correctly - which is all that page selection needs.
        score = float(kind["artwork"])
        best_kind = max(kind, key=kind.get)
        if score < 0.01:
            # Vanishingly low artwork affinity is a reliable negative signal.
            best_kind = "text" if kind["text"] >= kind["cover"] else "cover"

        insights.append(
            PageInsight(
                page_index=page_index,
                kind=best_kind,
                kind_confidence=kind[best_kind],
                subject=max(subject, key=subject.get),
                subject_confidence=max(subject.values()),
                energy=energy["active"],
                score=score,
                scores={"kind": kind, "subject": subject, "energy": energy},
            )
        )

    return insights


def select_pages(insights: list[PageInsight], wanted: int,
                 min_score: float = 0.01) -> list[PageInsight]:
    """Pick the best pages to film, keeping reading order.

    Selection is purely by rank. Measured on a real woodblock volume whose pages mix art
    and calligraphy, hard classification scored 5/14, while the top 8 pages by artwork
    affinity were 8/8 correct. So this ranks and takes the top N, and only uses the
    absolute score to drop pages with essentially zero artwork signal.
    """
    if not insights:
        return []

    ranked = sorted(insights, key=lambda i: i.score, reverse=True)
    usable = [i for i in ranked if i.score >= min_score]
    if len(usable) < 2:
        usable = ranked[: max(2, min(wanted, len(ranked)))]

    chosen = usable[:wanted]
    # Restore reading order so the sequence still follows the story.
    return sorted(chosen, key=lambda i: i.page_index)
