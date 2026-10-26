"""Procedural sound design: music, ambience, and a timed sound-effect library.

No external audio assets are required or assumed. Everything is synthesised with numpy
(additive/subtractive synthesis, spectral filtering and a convolution reverb built from a
decaying-noise impulse response), cached to WAV under ``assets/audio/sfx`` and verified on
disk before use. Four logical tracks - music, ambience, shot effects and narration - are
mixed, soft-limited and normalised, then written as a single 16-bit stereo WAV.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, SFX_DIR

# ------------------------------------------------------------------ primitives


def _t(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.arange(max(1, int(round(duration * sr))), dtype=np.float32) / sr


def adsr(n: int, sr: int, attack: float, decay: float, sustain: float,
         release: float, sustain_level: float = 0.7) -> np.ndarray:
    """Sample-accurate ADSR envelope of exactly ``n`` samples."""
    a, d, r = int(attack * sr), int(decay * sr), int(release * sr)
    a, d, r = max(1, a), max(1, d), max(1, r)
    if a + d + r > n:
        scale = n / float(a + d + r)
        a, d, r = max(1, int(a * scale)), max(1, int(d * scale)), max(1, int(r * scale))
    s = max(0, n - a - d - r)
    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, a, dtype=np.float32) ** 0.6,
            np.linspace(1.0, sustain_level, d, dtype=np.float32),
            np.full(s, sustain_level, dtype=np.float32),
            np.linspace(sustain_level, 0.0, r, dtype=np.float32) ** 1.4,
        ]
    )
    del sustain
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def spectral_filter(x: np.ndarray, sr: int = SAMPLE_RATE, low: float | None = None,
                    high: float | None = None, rolloff: float = 0.35) -> np.ndarray:
    """Zero-phase band shaping in the frequency domain with raised-cosine edges."""
    if x.size == 0:
        return x
    spec = np.fft.rfft(x.astype(np.float32))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    mask = np.ones_like(freqs, dtype=np.float32)

    if low is not None and low > 0:
        width = max(1.0, low * rolloff)
        mask *= np.clip((freqs - (low - width)) / (2 * width), 0.0, 1.0)
    if high is not None and high < sr / 2:
        width = max(1.0, high * rolloff)
        mask *= np.clip(((high + width) - freqs) / (2 * width), 0.0, 1.0)

    return np.fft.irfft(spec * mask, n=x.size).astype(np.float32)


def noise(duration: float, sr: int = SAMPLE_RATE, seed: int = 0, pink: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed + 7)
    x = rng.normal(0.0, 0.3, max(1, int(round(duration * sr)))).astype(np.float32)
    if pink:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
        shape = np.ones_like(freqs)
        shape[1:] = 1.0 / np.sqrt(freqs[1:])
        x = np.fft.irfft(spec * shape, n=x.size).astype(np.float32)
        peak = np.max(np.abs(x)) or 1.0
        x = x / peak * 0.3
    return x


def tone(freq: float, duration: float, sr: int = SAMPLE_RATE, wave_type: str = "sine",
         detune_cents: float = 0.0, vibrato_hz: float = 0.0, vibrato_cents: float = 0.0,
         partials: int = 1) -> np.ndarray:
    """Oscillator with optional detune, vibrato and additive partials."""
    t = _t(duration, sr)
    f = freq * (2.0 ** (detune_cents / 1200.0))
    phase_mod = 0.0
    if vibrato_hz > 0 and vibrato_cents > 0:
        depth = (2.0 ** (vibrato_cents / 1200.0) - 1.0) * f
        phase_mod = depth / (2 * math.pi * vibrato_hz) * np.sin(2 * math.pi * vibrato_hz * t)

    out = np.zeros_like(t)
    for k in range(1, max(1, partials) + 1):
        phase = 2 * math.pi * f * k * t + (phase_mod * k if isinstance(phase_mod, np.ndarray) else 0.0)
        if wave_type == "saw":
            comp = 2.0 * ((phase / (2 * math.pi)) % 1.0) - 1.0
        elif wave_type == "square":
            comp = np.sign(np.sin(phase))
        elif wave_type == "triangle":
            comp = 2.0 / math.pi * np.arcsin(np.clip(np.sin(phase), -1, 1))
        else:
            comp = np.sin(phase)
        out += comp.astype(np.float32) / (k ** 1.4)
    return out / max(1.0, np.max(np.abs(out)))


@dataclass
class Reverb:
    """Convolution reverb using a synthesised exponential-decay noise tail."""

    seconds: float = 1.8
    sr: int = SAMPLE_RATE
    seed: int = 101
    damping: float = 4200.0

    def __post_init__(self) -> None:
        n = int(self.seconds * self.sr)
        rng = np.random.default_rng(self.seed)
        ir = rng.normal(0.0, 1.0, n).astype(np.float32)
        ir *= np.exp(-np.linspace(0.0, 6.5, n, dtype=np.float32))
        ir = spectral_filter(ir, self.sr, high=self.damping)
        ir[: int(0.004 * self.sr)] = 0.0  # small pre-delay
        self.ir = ir / (np.sqrt(np.sum(ir ** 2)) or 1.0)

    def apply(self, x: np.ndarray, mix: float = 0.3) -> np.ndarray:
        if x.size == 0 or mix <= 0:
            return x
        n = x.size + self.ir.size - 1
        size = 1 << (n - 1).bit_length()
        wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(self.ir, size), size)[: x.size]
        wet = wet.astype(np.float32)
        peak = np.max(np.abs(wet)) or 1.0
        wet = wet / peak * (np.max(np.abs(x)) or 1.0)
        return (x * (1.0 - mix) + wet * mix).astype(np.float32)


def fade(x: np.ndarray, sr: int = SAMPLE_RATE, fade_in: float = 0.01,
         fade_out: float = 0.05) -> np.ndarray:
    out = x.copy()
    n_in = min(len(out), int(fade_in * sr))
    n_out = min(len(out) - n_in, int(fade_out * sr))
    if n_in > 1:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 1:
        out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return out


def to_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n].copy()
    return np.pad(x, (0, n - len(x)))


def loop_to_length(x: np.ndarray, n: int, crossfade: float = 0.25,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Tile a bed with short crossfades so the seams are inaudible."""
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) >= n:
        return x[:n].copy()

    xf = min(int(crossfade * sr), len(x) // 3)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        chunk = x
        end = min(n, pos + len(chunk))
        seg = chunk[: end - pos]
        if pos > 0 and xf > 1:
            ramp_len = min(xf, len(seg))
            ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
            out[pos : pos + ramp_len] = out[pos : pos + ramp_len] * (1 - ramp) + seg[:ramp_len] * ramp
            out[pos + ramp_len : end] = seg[ramp_len:]
        else:
            out[pos:end] = seg
        pos += max(1, len(chunk) - xf)
    return out


# ------------------------------------------------------------- sound effects

SFX_BUILDERS: dict[str, str] = {
    "whoosh": "a swept band of filtered noise passing the mic",
    "impact": "low-end thump with a transient crack",
    "footsteps": "four damped steps on hard ground",
    "wind": "slow-moving filtered noise bed",
    "rain": "dense rain bed with sparse droplets",
    "ambient_drone": "detuned low drone with slow movement",
    "suspense_riser": "rising noise and pitch sweep into a cut-off",
    "explosion": "broadband blast with a long sub tail",
    "heartbeat": "two-beat muffled pulse",
    "sword_clash": "metallic ring with a bright transient",
}

_REVERB_SHORT = None
_REVERB_LONG = None


def _reverb_short() -> Reverb:
    global _REVERB_SHORT
    if _REVERB_SHORT is None:
        _REVERB_SHORT = Reverb(seconds=0.9, seed=11, damping=5200.0)
    return _REVERB_SHORT


def _reverb_long() -> Reverb:
    global _REVERB_LONG
    if _REVERB_LONG is None:
        _REVERB_LONG = Reverb(seconds=2.6, seed=29, damping=3600.0)
    return _REVERB_LONG


def synth_sfx(name: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Build one mono sound effect. Unknown names return a soft neutral transient."""
    name = (name or "").strip().lower()

    if name == "whoosh":
        dur = 0.85
        t = _t(dur, sr)
        src = noise(dur, sr, seed=3, pink=True)
        # Sweep a band upward then away, with a doppler-ish amplitude arc.
        bands = []
        centres = (320.0, 900.0, 2400.0)
        for i, centre in enumerate(centres):
            band = spectral_filter(src, sr, low=centre * 0.6, high=centre * 1.7)
            weight = np.exp(-((t / dur - (0.2 + i * 0.3)) ** 2) / 0.016)
            bands.append(band * weight)
        out = np.sum(bands, axis=0)
        out *= np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.3
        return fade(_reverb_short().apply(out, 0.22), sr, 0.01, 0.2) * 0.9

    if name == "impact":
        dur = 1.1
        t = _t(dur, sr)
        body = np.sin(2 * math.pi * (95.0 * np.exp(-t * 7.0) + 42.0) * t).astype(np.float32)
        body *= np.exp(-t * 6.0)
        crack = spectral_filter(noise(dur, sr, seed=5), sr, low=1200.0) * np.exp(-t * 38.0)
        sub = np.sin(2 * math.pi * 48.0 * t).astype(np.float32) * np.exp(-t * 4.0)
        out = body * 0.8 + crack * 0.45 + sub * 0.5
        return fade(_reverb_short().apply(out, 0.3), sr, 0.002, 0.25)

    if name == "footsteps":
        dur = 1.6
        out = np.zeros(int(dur * sr), dtype=np.float32)
        for i, offset in enumerate((0.0, 0.42, 0.84, 1.24)):
            step_dur = 0.22
            st = _t(step_dur, sr)
            click = spectral_filter(noise(step_dur, sr, seed=11 + i), sr, low=180.0, high=2800.0)
            click *= np.exp(-st * 26.0)
            thud = np.sin(2 * math.pi * 120.0 * st).astype(np.float32) * np.exp(-st * 22.0)
            step = (click * 0.8 + thud * 0.4) * (0.85 if i % 2 else 1.0)
            start = int(offset * sr)
            end = min(len(out), start + len(step))
            out[start:end] += step[: end - start]
        return fade(_reverb_short().apply(out, 0.28), sr, 0.002, 0.15) * 0.8

    if name == "wind":
        dur = 6.0
        t = _t(dur, sr)
        src = noise(dur, sr, seed=17, pink=True)
        low = spectral_filter(src, sr, low=90.0, high=700.0)
        mid = spectral_filter(src, sr, low=600.0, high=2600.0)
        lfo = 0.5 + 0.5 * np.sin(2 * math.pi * 0.13 * t + 0.6)
        lfo2 = 0.5 + 0.5 * np.sin(2 * math.pi * 0.07 * t)
        out = low * (0.55 + 0.45 * lfo2) + mid * (0.18 + 0.32 * lfo)
        return fade(out, sr, 0.5, 0.5) * 0.85

    if name == "rain":
        dur = 6.0
        src = noise(dur, sr, seed=23, pink=True)
        bed = spectral_filter(src, sr, low=700.0, high=9000.0) * 0.65
        drops = np.zeros(len(bed), dtype=np.float32)
        rng = np.random.default_rng(31)
        for _ in range(120):
            pos = int(rng.uniform(0, dur - 0.1) * sr)
            d = _t(0.05, sr)
            drop = np.sin(2 * math.pi * rng.uniform(1800, 5200) * d).astype(np.float32)
            drop *= np.exp(-d * 90.0) * rng.uniform(0.1, 0.35)
            drops[pos : pos + len(drop)] += drop
        out = bed + drops
        return fade(_reverb_short().apply(out, 0.18), sr, 0.5, 0.5) * 0.8

    if name == "ambient_drone":
        dur = 8.0
        t = _t(dur, sr)
        base = 55.0
        out = np.zeros_like(t)
        for ratio, det, gain in ((1.0, -6, 1.0), (1.0, 7, 0.9), (1.5, -4, 0.45), (2.0, 5, 0.3), (3.0, 0, 0.12)):
            out += tone(base * ratio, dur, sr, "sine", detune_cents=det, vibrato_hz=0.08,
                        vibrato_cents=12) * gain
        swell = 0.6 + 0.4 * np.sin(2 * math.pi * 0.05 * t)
        out = out / (np.max(np.abs(out)) or 1.0) * swell
        return fade(_reverb_long().apply(out, 0.4), sr, 1.2, 1.2) * 0.8

    if name == "suspense_riser":
        dur = 3.0
        t = _t(dur, sr)
        prog = (t / dur) ** 1.5
        src = noise(dur, sr, seed=41, pink=True)
        swept = np.zeros_like(src)
        # Band-split and crossfade upward to imitate a rising resonant sweep.
        centres = np.array([200, 420, 900, 1900, 4000, 8200], dtype=np.float32)
        for i, centre in enumerate(centres):
            band = spectral_filter(src, sr, low=centre * 0.7, high=centre * 1.5)
            target = i / (len(centres) - 1)
            weight = np.exp(-((prog - target) ** 2) / 0.02)
            swept += band * weight
        pitch = 140.0 * (2.0 ** (prog * 2.6))
        sweep_tone = np.sin(2 * math.pi * np.cumsum(pitch) / sr).astype(np.float32)
        out = swept * 0.8 + sweep_tone * 0.3 * prog
        out *= prog ** 1.2
        out[-int(0.05 * sr) :] *= np.linspace(1.0, 0.0, int(0.05 * sr), dtype=np.float32)
        return fade(_reverb_long().apply(out, 0.25), sr, 0.1, 0.05) * 0.9

    if name == "explosion":
        dur = 2.6
        t = _t(dur, sr)
        blast = noise(dur, sr, seed=47)
        blast = spectral_filter(blast, sr, high=6000.0) * np.exp(-t * 3.2)
        sub = np.sin(2 * math.pi * (70.0 * np.exp(-t * 2.2) + 28.0) * t).astype(np.float32)
        sub *= np.exp(-t * 1.9)
        crack = spectral_filter(noise(dur, sr, seed=53), sr, low=2500.0) * np.exp(-t * 45.0)
        out = blast * 0.85 + sub * 0.8 + crack * 0.4
        return fade(_reverb_long().apply(out, 0.35), sr, 0.001, 0.5)

    if name == "heartbeat":
        dur = 1.8
        out = np.zeros(int(dur * sr), dtype=np.float32)
        for offset, gain in ((0.0, 1.0), (0.33, 0.72), (1.0, 1.0), (1.33, 0.72)):
            bt = _t(0.4, sr)
            beat = np.sin(2 * math.pi * 52.0 * bt).astype(np.float32) * np.exp(-bt * 14.0)
            beat += np.sin(2 * math.pi * 78.0 * bt).astype(np.float32) * np.exp(-bt * 22.0) * 0.4
            start = int(offset * sr)
            end = min(len(out), start + len(beat))
            out[start:end] += beat[: end - start] * gain
        return fade(spectral_filter(out, sr, high=420.0), sr, 0.005, 0.2) * 0.9

    if name == "sword_clash":
        dur = 1.4
        t = _t(dur, sr)
        out = np.zeros_like(t)
        for freq, decay, gain in ((1480.0, 5.0, 1.0), (2230.0, 7.0, 0.6), (3570.0, 9.0, 0.35),
                                  (5100.0, 13.0, 0.2)):
            out += np.sin(2 * math.pi * freq * t).astype(np.float32) * np.exp(-t * decay) * gain
        transient = spectral_filter(noise(dur, sr, seed=59), sr, low=1800.0) * np.exp(-t * 60.0)
        out = out * 0.6 + transient * 0.7
        return fade(_reverb_long().apply(out, 0.32), sr, 0.001, 0.3) * 0.85

    # Neutral fallback so an unknown cue is quiet rather than wrong.
    dur = 0.4
    t = _t(dur, sr)
    out = np.sin(2 * math.pi * 220.0 * t).astype(np.float32) * np.exp(-t * 9.0)
    return fade(out, sr, 0.005, 0.2) * 0.3


def ensure_sfx_library(force: bool = False, sr: int = SAMPLE_RATE) -> dict[str, Path]:
    """Render every cue to a WAV once, then verify each file exists before returning it."""
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    library: dict[str, Path] = {}
    for name in SFX_BUILDERS:
        path = SFX_DIR / f"{name}.wav"
        if force or not path.exists() or path.stat().st_size < 1024:
            write_wav(path, synth_sfx(name, sr), sr)
        if path.exists() and path.stat().st_size >= 1024:
            library[name] = path
    return library


def load_sfx(name: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a cue from the cached library, synthesising it if the file is missing."""
    path = SFX_DIR / f"{name}.wav"
    if path.exists() and path.stat().st_size >= 1024:
        try:
            data, file_sr = read_wav(path)
            if file_sr == sr:
                return data
        except Exception:
            pass
    return synth_sfx(name, sr)


# -------------------------------------------------------------------- music

# (scale degrees as semitone offsets, chord progression as root semitones, tempo)
MOOD_SPEC: dict[str, dict] = {
    "orchestral": {
        "root": 55.0,  # G3-ish
        "progression": [0, -3, -7, -5],
        "chords": [[0, 4, 7, 11], [0, 3, 7, 10], [0, 4, 7], [0, 3, 7, 10]],
        "bpm": 82,
        "pad_wave": "saw",
        "partials": 4,
        "reverb": 0.38,
        "percussion": "timpani",
        "melody": True,
    },
    "ambient": {
        "root": 49.0,
        "progression": [0, 5, 0, 7],
        "chords": [[0, 7, 12, 19], [0, 7, 14], [0, 5, 12], [0, 7, 12, 16]],
        "bpm": 56,
        "pad_wave": "triangle",
        "partials": 3,
        "reverb": 0.55,
        "percussion": None,
        "melody": False,
    },
    "dark": {
        "root": 41.0,
        "progression": [0, 1, 0, -2],
        "chords": [[0, 7, 13], [0, 6, 13], [0, 7, 12], [0, 8, 15]],
        "bpm": 64,
        "pad_wave": "saw",
        "partials": 3,
        "reverb": 0.45,
        "percussion": "pulse",
        "melody": False,
    },
    "energetic": {
        "root": 58.0,
        "progression": [0, -4, -2, -5],
        "chords": [[0, 3, 7, 10], [0, 4, 7], [0, 3, 7], [0, 5, 10]],
        "bpm": 136,
        "pad_wave": "square",
        "partials": 2,
        "reverb": 0.2,
        "percussion": "kit",
        "melody": True,
    },
    "suspenseful": {
        "root": 46.0,
        "progression": [0, 0, 1, -1],
        "chords": [[0, 6, 11], [0, 7, 11], [0, 6, 13], [0, 5, 11]],
        "bpm": 72,
        "pad_wave": "saw",
        "partials": 4,
        "reverb": 0.42,
        "percussion": "pulse",
        "melody": True,
    },
}


def _midi_to_hz(semitones_from_a4: float) -> float:
    return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))


def render_music(mood: str, duration: float, sr: int = SAMPLE_RATE,
                 seed: int = 7) -> np.ndarray:
    """Build a looping chord bed (+ bass, optional melody and percussion) of ``duration``."""
    spec = MOOD_SPEC.get(mood, MOOD_SPEC["orchestral"])
    rng = np.random.default_rng(seed)

    beat = 60.0 / spec["bpm"]
    bar = beat * 4
    bars = len(spec["progression"])
    loop_len = bar * bars
    n_loop = int(loop_len * sr)

    pad = np.zeros(n_loop, dtype=np.float32)
    bass = np.zeros(n_loop, dtype=np.float32)
    lead = np.zeros(n_loop, dtype=np.float32)
    perc = np.zeros(n_loop, dtype=np.float32)

    root_offset = spec["root"] - 69.0  # treat spec root as a MIDI note number

    for b, degree in enumerate(spec["progression"]):
        chord = spec["chords"][b % len(spec["chords"])]
        start = int(b * bar * sr)
        chord_dur = bar * 1.08  # slight overlap so chords bleed into each other

        for voice, semis in enumerate(chord):
            freq = _midi_to_hz(root_offset + degree + semis)
            v = tone(
                freq,
                chord_dur,
                sr,
                spec["pad_wave"],
                detune_cents=(-7 + 5 * voice),
                vibrato_hz=4.6,
                vibrato_cents=9,
                partials=spec["partials"],
            )
            v = spectral_filter(v, sr, high=3400.0)
            env = adsr(len(v), sr, attack=bar * 0.28, decay=bar * 0.2, sustain=0.0,
                       release=bar * 0.45, sustain_level=0.72)
            v *= env * (0.82 / max(1, len(chord)))
            end = min(n_loop, start + len(v))
            pad[start:end] += v[: end - start]

        # Bass: root, one octave down, re-articulated on beats 1 and 3.
        for hit in (0.0, 2.0) if spec["bpm"] > 100 else (0.0,):
            bdur = bar * (0.45 if spec["bpm"] > 100 else 0.95)
            bfreq = _midi_to_hz(root_offset + degree - 12)
            bv = tone(bfreq, bdur, sr, "triangle", partials=2)
            bv += tone(bfreq, bdur, sr, "sine") * 0.6
            bv *= adsr(len(bv), sr, 0.02, 0.15, 0.0, bdur * 0.5, 0.6) * 0.75
            bstart = start + int(hit * beat * sr)
            bend = min(n_loop, bstart + len(bv))
            bass[bstart:bend] += bv[: bend - bstart]

        if spec["melody"]:
            steps = 8 if spec["bpm"] > 100 else 4
            note_dur = bar / steps
            for s in range(steps):
                if spec["bpm"] <= 100 and rng.random() < 0.35:
                    continue
                semis = chord[int(rng.integers(0, len(chord)))] + 12 * int(rng.integers(1, 3))
                freq = _midi_to_hz(root_offset + degree + semis)
                nv = tone(freq, note_dur * 1.4, sr, "triangle", vibrato_hz=5.2,
                          vibrato_cents=14, partials=3)
                nv *= adsr(len(nv), sr, 0.01, note_dur * 0.4, 0.0, note_dur * 0.8, 0.45)
                nv *= 0.3
                nstart = start + int(s * note_dur * sr)
                nend = min(n_loop, nstart + len(nv))
                lead[nstart:nend] += nv[: nend - nstart]

        kind = spec["percussion"]
        if kind == "kit":
            for s in range(8):
                pos = start + int(s * (bar / 8) * sr)
                if s % 4 == 0:
                    hit = _kick(sr)
                elif s % 4 == 2:
                    hit = _snare(sr, int(rng.integers(0, 999)))
                else:
                    hit = _hat(sr, int(rng.integers(0, 999))) * 0.5
                end = min(n_loop, pos + len(hit))
                perc[pos:end] += hit[: end - pos]
        elif kind == "timpani":
            for hit_beat in (0.0, 2.5):
                pos = start + int(hit_beat * beat * sr)
                hit = _timpani(_midi_to_hz(root_offset + degree - 24), sr)
                end = min(n_loop, pos + len(hit))
                perc[pos:end] += hit[: end - pos] * 0.7
        elif kind == "pulse":
            pos = start
            hit = _kick(sr, pitch=58.0, decay=5.0) * 0.6
            end = min(n_loop, pos + len(hit))
            perc[pos:end] += hit[: end - pos]

    reverb = Reverb(seconds=2.4 if spec["reverb"] > 0.35 else 1.3, seed=71)
    mixed = pad * 0.62 + bass * 0.5 + lead * 0.38 + perc * 0.42
    mixed = reverb.apply(mixed, spec["reverb"])
    mixed = mixed / (np.max(np.abs(mixed)) or 1.0) * 0.9

    total = int(round(duration * sr))
    return loop_to_length(mixed, total, crossfade=min(1.0, loop_len * 0.12), sr=sr)


def _kick(sr: int = SAMPLE_RATE, pitch: float = 110.0, decay: float = 14.0) -> np.ndarray:
    t = _t(0.42, sr)
    body = np.sin(2 * math.pi * (pitch * np.exp(-t * 18.0) + 40.0) * t).astype(np.float32)
    return (body * np.exp(-t * decay) * 0.95).astype(np.float32)


def _snare(sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    t = _t(0.26, sr)
    body = spectral_filter(noise(0.26, sr, seed=seed), sr, low=320.0, high=7200.0)
    tonal = np.sin(2 * math.pi * 195.0 * t).astype(np.float32) * 0.3
    return ((body + tonal) * np.exp(-t * 20.0) * 0.6).astype(np.float32)


def _hat(sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    t = _t(0.1, sr)
    body = spectral_filter(noise(0.1, sr, seed=seed), sr, low=5500.0)
    return (body * np.exp(-t * 60.0) * 0.5).astype(np.float32)


def _timpani(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = _t(1.3, sr)
    out = np.zeros_like(t)
    for ratio, gain, decay in ((1.0, 1.0, 3.2), (1.5, 0.4, 4.5), (2.0, 0.25, 6.0)):
        out += np.sin(2 * math.pi * freq * ratio * t).astype(np.float32) * gain * np.exp(-t * decay)
    thump = spectral_filter(noise(1.3, sr, seed=83), sr, high=900.0) * np.exp(-t * 22.0)
    return ((out * 0.8 + thump * 0.3) * 0.8).astype(np.float32)


# --------------------------------------------------------------- track mixing

INTENSITY_GAIN = {"low": 0.52, "medium": 0.78, "high": 1.0}

AMBIENCE_CUES = {"wind", "rain", "ambient_drone"}


@dataclass
class ShotAudio:
    """What the mixer needs to know about one shot."""

    start_seconds: float
    duration_seconds: float
    cues: list[str]
    music_intensity: str = "medium"


def _smooth_envelope(points: list[tuple[float, float]], total_samples: int,
                     sr: int = SAMPLE_RATE, smooth_seconds: float = 0.8) -> np.ndarray:
    """Step envelope from (start_seconds, value) pairs, then box-smoothed."""
    env = np.ones(total_samples, dtype=np.float32)
    if not points:
        return env
    for i, (start, value) in enumerate(points):
        s = int(start * sr)
        e = int(points[i + 1][0] * sr) if i + 1 < len(points) else total_samples
        env[max(0, s) : min(total_samples, e)] = value
    width = max(3, int(smooth_seconds * sr))
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(env, kernel, mode="same").astype(np.float32)


def soft_limit(x: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """tanh-style knee so peaks round over instead of clipping."""
    return (np.tanh(x / ceiling) * ceiling).astype(np.float32)


def normalise(x: np.ndarray, target_peak: float = 0.89) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-6:
        return x
    return (x / peak * target_peak).astype(np.float32)


def build_soundtrack(
    shots: list[ShotAudio],
    mood: str,
    total_duration: float,
    out_path: Path,
    sr: int = SAMPLE_RATE,
    narration: np.ndarray | None = None,
    seed: int = 7,
) -> Path:
    """Mix music + ambience + timed effects (+ optional narration) to a 16-bit stereo WAV."""
    total_samples = max(1, int(round(total_duration * sr)))
    ensure_sfx_library(sr=sr)

    # 1. music, ducked by the per-shot intensity curve
    music = to_length(render_music(mood, total_duration + 1.0, sr, seed), total_samples)
    curve_points = [
        (s.start_seconds, INTENSITY_GAIN.get(s.music_intensity, 0.78)) for s in shots
    ]
    music *= _smooth_envelope(curve_points, total_samples, sr, smooth_seconds=1.2)

    # 2. ambience bed: the longest-running ambient cue across the sequence
    ambience = np.zeros(total_samples, dtype=np.float32)
    ambient_names = [c for s in shots for c in s.cues if c in AMBIENCE_CUES]
    if ambient_names:
        chosen = max(set(ambient_names), key=ambient_names.count)
        bed = load_sfx(chosen, sr)
        ambience = loop_to_length(bed, total_samples, crossfade=0.6, sr=sr)

    # 3. one-shot effects placed at shot boundaries
    effects = np.zeros(total_samples, dtype=np.float32)
    for index, shot in enumerate(shots):
        one_shots = [c for c in shot.cues if c not in AMBIENCE_CUES]
        for n, cue in enumerate(one_shots):
            clip = load_sfx(cue, sr)
            # Risers land before the next cut; other cues hit just after the cut.
            if cue == "suspense_riser":
                start = max(0.0, shot.start_seconds + shot.duration_seconds - len(clip) / sr)
            else:
                start = shot.start_seconds + 0.04 + n * 0.22
            s0 = int(start * sr)
            if s0 >= total_samples:
                continue
            clip = fade(clip, sr, 0.004, 0.08)
            e0 = min(total_samples, s0 + len(clip))
            effects[s0:e0] += clip[: e0 - s0] * (0.85 if index else 0.95)

    # 4. optional narration / dialogue track
    narration_track = to_length(narration, total_samples) if narration is not None else None

    # Duck music and ambience slightly where effects are loud.
    effect_energy = np.convolve(
        np.abs(effects), np.ones(int(0.12 * sr), dtype=np.float32) / int(0.12 * sr), mode="same"
    )
    duck = 1.0 - 0.32 * np.clip(effect_energy / 0.35, 0.0, 1.0)

    mix = music * 0.52 * duck + ambience * 0.26 * duck + effects * 0.60
    if narration_track is not None:
        voice_energy = np.convolve(
            np.abs(narration_track),
            np.ones(int(0.2 * sr), dtype=np.float32) / int(0.2 * sr),
            mode="same",
        )
        voice_duck = 1.0 - 0.45 * np.clip(voice_energy / 0.25, 0.0, 1.0)
        mix = mix * voice_duck + narration_track * 0.8

    mix = fade(mix, sr, fade_in=0.6, fade_out=1.2)
    mix = normalise(soft_limit(mix), target_peak=0.89)

    # Gentle stereo widening: a few ms of inter-channel delay on the bed.
    delay = int(0.008 * sr)
    left = mix.copy()
    right = np.concatenate([np.zeros(delay, dtype=np.float32), mix[:-delay]]) if delay else mix.copy()
    stereo = np.stack([left * 0.98 + right * 0.02, right * 0.98 + left * 0.02], axis=1)
    stereo = normalise(soft_limit(stereo), target_peak=0.9)

    write_wav(out_path, stereo, sr)
    return out_path


# ------------------------------------------------------------------ WAV files


def write_wav(path: Path, data: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write float data (mono 1-D or stereo Nx2) as 16-bit PCM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    channels = arr.shape[1]
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV into a mono float array."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr, sr
