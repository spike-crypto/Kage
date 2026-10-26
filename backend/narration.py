"""Voiceover: models look at the pages, write the narration, and speak it.

Three jobs:

1. **See** - a vision model says what is actually drawn on each page.
2. **Write** - a language model turns those observations into one narration line per shot,
   inside a word budget derived from how long that shot holds.
3. **Speak** - a text-to-speech model renders the script to audio.

Seeing and writing run on a *hosted* provider (see ``providers.py``). This machine has
4 GB of VRAM and 15 GB of RAM shared with the studio's own backend, frontend and browser,
and a local 1.5B writer alone wanted 3.1 GB of VRAM and 2.9 GB of disk - measured, then
moved off the box. The whole film costs **two hosted requests**, which sits inside every
free tier Kage supports, so the GPU stays free for the camera work it is actually needed
for.

Speaking stays local, because free hosted TTS generally wants a card. The default voice is
a 145 MB model - small enough not to matter next to the rest of the studio.

There is deliberately no written-by-hand fallback text in this module. If every model
fails, narration fails and says why. Canned narration pretending to have read the artwork
would be worse than none.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_DIR, env
from .providers import (
    PROVIDERS,
    TEXT_CHAIN,
    VISION_CHAIN,
    Attempt,
    ChainReport,
    ProviderError,
    configured_providers,
    image_part,
    parse_json_list,
    run_chain,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

log = logging.getLogger("kage.narration")

MODEL_CACHE = Path(env("MODEL_CACHE", str(DATA_DIR / "models")))

#: Narration is read, not performed - about 2.6 words a second sits naturally under music.
WORDS_PER_SECOND = float(env("NARRATION_WPS", "2.6"))
#: Leave the head and tail of a shot clear so the voice does not collide with a cut.
SHOT_VOICE_MARGIN = 0.35
#: Pages sent to the vision model in one request.
MAX_IMAGES_PER_REQUEST = int(env("NARRATION_MAX_IMAGES", "12"))


class NarrationUnavailable(RuntimeError):
    """Every model in every chain failed. Carries what was tried and why."""


@dataclass
class Line:
    """One shot's narration and the budget it had to fit."""

    sequence_number: int
    text: str
    start_seconds: float
    duration_seconds: float
    word_budget: int


def _budget(duration: float) -> int:
    usable = max(0.0, duration - SHOT_VOICE_MARGIN * 2)
    return max(3, int(usable * WORDS_PER_SECOND))


#: Words a line must never end on. Cutting to a budget mid-clause produced lines like
#: "Carts and logs are pushed and." - grammatical debris that a voice reads aloud.
_DANGLING = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "that", "which", "who", "while",
    "their", "his", "her", "its", "some", "many", "several", "into", "over", "under",
}


def _trim_to_budget(text: str, budget: int) -> str:
    """Fit a line to its budget, ending on a clean boundary.

    This is arithmetic, not a fallback: the model wrote the words, and the shot only has
    room for so many of them. What matters is that what survives is a whole thought -
    a sentence, a clause, or at worst a phrase that does not trail off mid-grammar.
    """
    clean = " ".join(str(text or "").split())
    clean = re.sub(r"^[\"'\s>*\-]+|[\"'\s*]+$", "", clean)
    if not clean:
        return ""
    words = clean.split()
    if len(words) <= budget:
        return clean

    cut = " ".join(words[:budget])
    # Best: a complete sentence already ended inside the budget.
    ends = [cut.rfind(stop) for stop in (". ", "! ", "? ")]
    if max(ends) > 0:
        return cut[: max(ends) + 1].strip()
    # Next best: a clause boundary. A coordinate phrase that lost its other half reads
    # worse than no phrase at all - "a circular object and wooden" - so cut the whole
    # "and ..." away rather than keep its stump.
    boundaries = [cut.rfind(mark) for mark in (", ", " and ", " or ", "; ")]
    boundary = max(b for b in boundaries if b > len(cut) // 2) if any(
        b > len(cut) // 2 for b in boundaries
    ) else -1
    if boundary > 0:
        cut = cut[:boundary]
    # Finally, never leave the sentence hanging on a connective or article.
    kept = cut.rstrip(" ,;:").split()
    while kept and kept[-1].lower().strip(",;:") in _DANGLING:
        kept.pop()
    if not kept:
        return ""
    return " ".join(kept).rstrip(" ,;:") + "."


# ------------------------------------------------------------------- 1. see

OBSERVE_SYSTEM = (
    "You are a visual describer for an accessibility tool. For each image you are given, "
    "write one short literal sentence listing what is drawn: the figures, objects and "
    "setting. Describe only what is visibly there. Do not interpret mood, invent names, "
    "or guess at a story. These are pages of manga or illustrated books."
)


def observe(paths: list[Path], project_id: int | None = None) -> tuple[list[str], ChainReport]:
    """One literal description per page, in one hosted request per batch of pages."""
    observations: list[str] = []
    report = ChainReport(job="Observe")

    for start in range(0, len(paths), MAX_IMAGES_PER_REQUEST):
        batch = paths[start : start + MAX_IMAGES_PER_REQUEST]

        def build(_candidate, batch=batch):
            content: list[dict] = [{
                "type": "text",
                "text": (
                    f"Here are {len(batch)} pages, in order. Reply with a JSON array of "
                    f"exactly {len(batch)} strings - one literal description per page, in "
                    "the same order. No other text."
                ),
            }]
            for path in batch:
                content.append(image_part(path))
            return [{"role": "system", "content": OBSERVE_SYSTEM},
                    {"role": "user", "content": content}]

        def parse(raw, expected=len(batch)):
            items = [" ".join(str(v).split()) for v in parse_json_list(raw)]
            items = [i for i in items if i]
            if len(items) < expected:
                raise ValueError(f"got {len(items)} descriptions, expected {expected}")
            return items[:expected]

        batch_result, batch_report = run_chain(
            "Observe", VISION_CHAIN, build, parse,
            project_id=project_id, images=len(batch),
        )
        observations.extend(batch_result)
        report.used = batch_report.used
        report.attempts.extend(batch_report.attempts)

    return observations, report


# ----------------------------------------------------------------- 2. write

WRITE_SYSTEM = (
    "You are a narrator describing pictures to someone who cannot see them. You are given "
    "a literal description of each shot's picture and a word budget for that shot. Write "
    "one spoken sentence per shot. Every noun you use must come from that shot's "
    "description. Do not add weather, lighting, emotion, action or anything not listed. "
    "Present tense. Do not repeat the previous shot's wording. "
    "Reply with a JSON array of objects: [{\"shot\": 1, \"line\": \"...\"}]."
)


def write_script(observations: list[str], shots: list[dict], total_duration: float,
                 project_id: int | None = None) -> tuple[list[Line], ChainReport]:
    """Turn the observations into one narration line per shot, in a single request."""
    plan: list[tuple[int, dict, float, float, int]] = []
    start = 0.0
    for index, shot in enumerate(shots):
        duration = float(shot["duration_seconds"])
        plan.append((index, shot, start, duration, _budget(duration)))
        start += duration

    brief = [
        {
            "shot": index + 1,
            "seconds": round(duration, 1),
            "max_words": budget,
            "picture": observations[index] if index < len(observations) else "",
        }
        for index, _shot, _begin, duration, budget in plan
    ]

    def build(_candidate):
        return [
            {"role": "system", "content": WRITE_SYSTEM},
            {"role": "user", "content": (
                f"The film is {total_duration:.0f} seconds long and has {len(plan)} shots.\n"
                f"{json.dumps(brief, ensure_ascii=False)}\n"
                f"Reply with exactly {len(plan)} objects."
            )},
        ]

    def parse(raw):
        items = parse_json_list(raw)
        by_shot: dict[int, str] = {}
        for item in items:
            if isinstance(item, dict):
                number = int(item.get("shot") or item.get("sequence") or 0)
                text = str(item.get("line") or item.get("text") or "")
            else:
                number, text = len(by_shot) + 1, str(item)
            if text.strip():
                by_shot[number] = text
        if len(by_shot) < len(plan):
            raise ValueError(f"got {len(by_shot)} lines, expected {len(plan)}")
        return by_shot

    by_shot, report = run_chain("Write", TEXT_CHAIN, build, parse,
                                project_id=project_id)

    lines: list[Line] = []
    for index, shot, begin, duration, budget in plan:
        written = _trim_to_budget(by_shot.get(index + 1, ""), budget)
        lines.append(Line(
            sequence_number=int(shot["sequence_number"]), text=written,
            start_seconds=begin, duration_seconds=duration, word_budget=budget,
        ))
    return lines, report


# ----------------------------------------------------------------- 3. speak

_LOCK = threading.Lock()
_VOICES: dict[str, tuple] = {}


@dataclass(frozen=True)
class Voice:
    model_id: str
    licence: str
    note: str = ""


#: Small on purpose. The first is 145 MB and runs on CPU without touching the GPU the
#: studio needs for its own camera work.
VOICE_MODELS: tuple[Voice, ...] = (
    Voice("facebook/mms-tts-eng", "CC-BY-NC-4.0", "~145 MB, non-commercial"),
    Voice("microsoft/speecht5_tts", "MIT", "~600 MB, needs sentencepiece"),
)


def voice_chain() -> tuple[Voice, ...]:
    raw = str(env("VOICE_MODELS", "")).strip()
    if not raw:
        return VOICE_MODELS
    known = {v.model_id: v for v in VOICE_MODELS}
    out = [known.get(i.strip(), Voice(i.strip(), "unknown", "from env"))
           for i in raw.split(",") if i.strip()]
    return tuple(out) or VOICE_MODELS


def _load_voice(voice: Voice):
    import torch
    from transformers import AutoTokenizer

    # Speech is short and the GPU belongs to the renderer; the CPU is plenty for this.
    device = env("VOICE_DEVICE", "cpu")
    cache = str(MODEL_CACHE)

    if "speecht5" in voice.model_id.lower():
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

        processor = SpeechT5Processor.from_pretrained(voice.model_id, cache_dir=cache)
        model = SpeechT5ForTextToSpeech.from_pretrained(voice.model_id, cache_dir=cache)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",
                                                  cache_dir=cache)
        generator = torch.Generator().manual_seed(7)
        speaker = (torch.randn(1, 512, generator=generator) * 0.08).to(device)
        return ("speecht5", processor, model.to(device).eval(), vocoder.to(device).eval(),
                speaker, device, 16000)

    from transformers import VitsModel

    tokeniser = AutoTokenizer.from_pretrained(voice.model_id, cache_dir=cache)
    model = VitsModel.from_pretrained(voice.model_id, cache_dir=cache)
    return ("vits", tokeniser, model.to(device).eval(), None, None, device,
            int(model.config.sampling_rate))


def speak(lines: list[Line], sample_rate: int) -> tuple[np.ndarray, ChainReport]:
    """Render the script to one mono track, each line placed at its shot.

    Lines are placed, never time-stretched: a line that overruns is allowed to spill a
    little rather than be pitch-shifted into a chipmunk.
    """
    spoken_lines = [line for line in lines if line.text.strip()]
    if not spoken_lines:
        raise NarrationUnavailable("Speak: the script is empty")

    total = max(line.start_seconds + line.duration_seconds for line in spoken_lines)
    length = max(1, int(round((total + 1.0) * sample_rate)))
    report = ChainReport(job="Speak")

    for voice in voice_chain():
        try:
            import torch

            with _LOCK:
                bundle = _VOICES.get(voice.model_id)
                if bundle is None:
                    log.info("Loading voice model %s", voice.model_id)
                    bundle = _load_voice(voice)
                    _VOICES[voice.model_id] = bundle

            kind, device, native_rate = bundle[0], bundle[5], bundle[6]
            track = np.zeros(length, dtype=np.float32)
            spoken = 0
            for line in spoken_lines:
                with torch.no_grad():
                    if kind == "speecht5":
                        _, processor, model, vocoder, speaker, _, _ = bundle
                        inputs = processor(text=line.text, return_tensors="pt").to(device)
                        audio = model.generate_speech(inputs["input_ids"], speaker,
                                                      vocoder=vocoder)
                    else:
                        _, tokeniser, model, _, _, _, _ = bundle
                        inputs = tokeniser(line.text, return_tensors="pt").to(device)
                        audio = model(**inputs).waveform[0]
                clip = audio.detach().float().cpu().numpy()
                if clip.size < 8:
                    continue
                clip = _resample(clip, native_rate, sample_rate)
                peak = float(np.abs(clip).max())
                if peak < 1e-6:
                    continue
                clip = clip / peak * 0.82
                begin = int((line.start_seconds + SHOT_VOICE_MARGIN) * sample_rate)
                end = min(length, begin + clip.size)
                if begin < length:
                    track[begin:end] += clip[: end - begin]
                    spoken += 1
            if not spoken:
                raise RuntimeError("voice model produced no audio")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:180]
            log.warning("Voice model %s unusable: %s", voice.model_id, reason)
            report.attempts.append(Attempt(voice.model_id, reason))
            _VOICES.pop(voice.model_id, None)
            continue
        report.attempts.append(Attempt(voice.model_id, "ok"))
        report.used = f"{voice.model_id} ({voice.licence})"
        return track, report

    raise NarrationUnavailable(report.describe())


def _resample(clip: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample. Speech is band-limited well below Nyquist, so this is enough."""
    if source_rate == target_rate or clip.size < 2:
        return clip.astype(np.float32)
    count = max(2, int(round(clip.size * target_rate / float(source_rate))))
    source = np.linspace(0.0, 1.0, clip.size, dtype=np.float64)
    target = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return np.interp(target, source, clip).astype(np.float32)


def release() -> None:
    """Drop every loaded voice model and free the GPU."""
    with _LOCK:
        _VOICES.clear()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -------------------------------------------------------------- the whole job


def capability() -> dict:
    """What narration can do right now, for the health endpoint and the UI."""
    providers = configured_providers()
    voices = voice_chain()
    try:
        import transformers  # noqa: F401
        voice_ready = True
    except Exception:
        voice_ready = False
    return {
        "ready": bool(providers) and voice_ready,
        "providers": [p.name for p in providers],
        "needs_key": [
            {"name": p.name, "env": p.key_env[0], "signup": p.signup, "free": p.free_note}
            for p in PROVIDERS.values() if not p.configured
        ],
        "images_per_request": MAX_IMAGES_PER_REQUEST,
        "voice": voices[0].model_id if voices else "",
        "voice_licence": voices[0].licence if voices else "",
        "voice_ready": voice_ready,
    }


def build_script(page_paths: list[Path], shots: list[dict], total_duration: float,
                 project_id: int | None = None) -> tuple[list[Line], list[ChainReport]]:
    """See the pages, then write the narration.

    Raises NarrationUnavailable, with what was tried, if no model could do either job.
    """
    if not configured_providers():
        raise NarrationUnavailable(
            "No inference provider is configured. Add a free API key - "
            + "; ".join(f"{p.name}: {p.signup}" for p in PROVIDERS.values())
        )
    try:
        observations, observe_report = observe(page_paths, project_id)
        log.info("Narration %s", observe_report.describe())
        lines, write_report = write_script(observations, shots, total_duration, project_id)
        log.info("Narration %s", write_report.describe())
    except ProviderError as exc:
        raise NarrationUnavailable(str(exc)) from exc
    return lines, [observe_report, write_report]
