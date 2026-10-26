"""Anime style presets - the creative grammar of Kage.

A style is not a filter. It bundles everything that makes a sequence read as one
deliberate look: which camera moves are allowed, how long shots hold, which transitions
cut between them, how the image is graded, and how sparingly accents are spent.

Design rule: restraint. The grade is always on and does the quiet work; named effects are
*accents*, spent on roughly one shot in three or four. A shot may never carry more than
``max_effects``. This is what separates a cinematic look from an effects pile-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GradeSpec:
    """Always-on image treatment. Applied to every frame, never listed as an effect."""

    exposure: float = 1.0         # <1 darkens the whole frame before contrast
    vignette: float = 0.40        # 0..1 corner falloff
    grain: float = 0.15           # 0..1 film grain
    contrast: float = 1.0         # 1.0 = untouched
    saturation: float = 1.0       # 0 = mono, 1 = original
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)   # RGB multipliers
    bloom: float = 0.0            # 0..1 soft highlight glow
    lift: float = 0.0             # raises blacks slightly, for a filmic matte
    #: Floor on saturation when the *source* is genuinely colourful. A near-monochrome
    #: style is a choice about mood, not a licence to throw away a colour original: a
    #: hand-tinted print graded to 0.08 saturation comes out as a photocopy. Styles that
    #: are deliberately colourless keep this at 0.
    colour_floor: float = 0.0
    #: Normalise the source's black and white points before grading, so paper reads as
    #: paper instead of mid-grey. Off for art that is meant to sit dark.
    normalise_levels: bool = True


@dataclass(frozen=True)
class StylePreset:
    key: str
    name: str
    tagline: str
    description: str

    camera: tuple[str, ...]
    transitions: tuple[str, ...]
    shot_length: tuple[float, float]

    grade: GradeSpec

    # Accents are deliberately rare.
    framing: float = 1.6   # >1 crops in toward the drawing instead of showing the page
    #: How far near pixels travel relative to far ones, as a fraction of frame width.
    #: This is the depth illusion; it is what makes a panel a space rather than a picture.
    parallax: float = 0.020
    #: Idle displacement so a held shot keeps breathing instead of freezing.
    drift: float = 0.004
    max_effects: int = 1
    accent_effects: tuple[str, ...] = ()
    accent_every: int = 3          # spend an accent on every Nth shot
    accent_on_first: bool = False

    audio_mood: str = "orchestral"
    bed: str = "wind"
    cues: tuple[tuple[str, ...], ...] = (("wind",),)
    music_curve: tuple[str, ...] = ("low", "medium", "medium")

    def accent_for(self, index: int) -> list[str]:
        """Which accent effects (if any) this shot earns."""
        if not self.accent_effects:
            return []
        if index == 0 and self.accent_on_first:
            return [self.accent_effects[0]]
        if index > 0 and index % self.accent_every == 0:
            return [self.accent_effects[(index // self.accent_every - 1) % len(self.accent_effects)]]
        return []


STYLE_PRESETS: dict[str, StylePreset] = {
    "kage_noir": StylePreset(
        key="kage_noir",
        name="Kage Noir",
        tagline="Modern dark. Ink, shadow, menace.",
        description=(
            "Near-monochrome with crushed blacks and hard contrast. The camera pushes in "
            "slowly and holds. Almost no effects - the darkness does the work."
        ),
        camera=("push_in", "slow_zoom_in", "static_drift", "pan_up", "slow_zoom_out", "pan_down"),
        transitions=("cut", "fade", "cut"),
        shot_length=(2.8, 4.6),
        grade=GradeSpec(
            exposure=0.86, vignette=0.52, grain=0.20, contrast=1.22, saturation=0.08,
            tint=(0.97, 0.99, 1.09), bloom=0.10, lift=0.015, colour_floor=0.34,
        ),
        parallax=0.024, drift=0.005,
        framing=1.95,
        max_effects=1,
        accent_effects=("impact_flash",),
        accent_every=4,
        audio_mood="dark",
        bed="ambient_drone",
        cues=(("ambient_drone",), ("heartbeat",), (), ("impact",), ()),
        music_curve=("low", "low", "medium", "high"),
    ),
    "shonen_kinetic": StylePreset(
        key="shonen_kinetic",
        name="Shonen Kinetic",
        tagline="Fast cuts. Clean energy. Impact.",
        description=(
            "Bright and punchy with hard cuts and quick pans. Speed lines are saved for "
            "the hit, not sprayed across every shot."
        ),
        camera=("rapid_pan", "push_in", "pan_left", "pan_right", "ken_burns", "slow_zoom_in"),
        transitions=("cut", "cut", "whip", "cut"),
        shot_length=(1.5, 2.6),
        grade=GradeSpec(
            vignette=0.28, grain=0.08, contrast=1.16, saturation=1.12,
            tint=(1.02, 1.0, 0.99), bloom=0.06,
        ),
        parallax=0.034, drift=0.006,
        framing=1.75,
        max_effects=2,
        accent_effects=("speed_lines", "screen_shake", "impact_flash"),
        accent_every=3,
        accent_on_first=False,
        audio_mood="energetic",
        bed="wind",
        cues=(("whoosh",), (), ("impact",), (), ("sword_clash",), ()),
        music_curve=("medium", "high", "high"),
    ),
    "shojo_bloom": StylePreset(
        key="shojo_bloom",
        name="Shojo Bloom",
        tagline="Soft light. Long breaths. Feeling.",
        description=(
            "Warm highlights and gentle bloom. The camera drifts rather than moves, and "
            "shots dissolve into one another instead of cutting."
        ),
        camera=("slow_zoom_in", "ken_burns", "static_drift", "slow_zoom_out", "pan_up"),
        transitions=("crossfade", "fade", "crossfade"),
        shot_length=(3.2, 5.0),
        grade=GradeSpec(
            exposure=1.02, vignette=0.34, grain=0.10, contrast=0.96, saturation=1.06,
            tint=(1.07, 1.00, 0.93), bloom=0.30, lift=0.03,
        ),
        parallax=0.018, drift=0.007,
        framing=1.55,
        max_effects=1,
        accent_effects=("particles",),
        accent_every=3,
        accent_on_first=True,
        audio_mood="ambient",
        bed="wind",
        cues=(("wind",), (), ("footsteps",), ()),
        music_curve=("low", "medium", "low"),
    ),
    "seinen_cold": StylePreset(
        key="seinen_cold",
        name="Seinen Cold",
        tagline="Desaturated. Still. Observational.",
        description=(
            "Cold blue cast, muted colour, locked-off framing. Motion is minimal and "
            "deliberate - the restraint is the point."
        ),
        camera=("static_drift", "slow_zoom_in", "pan_left", "slow_zoom_out", "pan_down"),
        transitions=("cut", "crossfade", "cut"),
        shot_length=(3.0, 4.8),
        grade=GradeSpec(
            exposure=0.92, vignette=0.44, grain=0.18, contrast=1.10, saturation=0.40,
            tint=(0.88, 0.97, 1.14), bloom=0.04, lift=0.035, colour_floor=0.55,
        ),
        parallax=0.014, drift=0.003,
        framing=1.35,
        max_effects=1,
        accent_effects=("rain",),
        accent_every=4,
        audio_mood="suspenseful",
        bed="rain",
        cues=(("rain",), (), ("footsteps",), (), ("suspense_riser",)),
        music_curve=("low", "medium", "medium"),
    ),
    "retro_cel": StylePreset(
        key="retro_cel",
        name="Retro Cel",
        tagline="90s OVA. Grain, amber, celluloid.",
        description=(
            "Heavy film grain and a warm amber cast, like a cel-animated OVA transferred "
            "to tape. Ken Burns moves and honest cuts."
        ),
        camera=("ken_burns", "pan_right", "slow_zoom_in", "push_in", "pan_left"),
        transitions=("cut", "fade", "crossfade"),
        shot_length=(2.4, 4.0),
        grade=GradeSpec(
            exposure=0.93, vignette=0.50, grain=0.30, contrast=1.14, saturation=0.86,
            tint=(1.07, 1.00, 0.88), bloom=0.14, lift=0.045, colour_floor=0.86,
        ),
        parallax=0.022, drift=0.006,
        framing=1.65,
        max_effects=1,
        accent_effects=("film_grain",),
        accent_every=5,
        audio_mood="orchestral",
        bed="ambient_drone",
        cues=(("footsteps",), (), ("wind",), (), ("impact",)),
        music_curve=("low", "medium", "medium"),
    ),
}

DEFAULT_STYLE = "kage_noir"

#: Older projects stored a narrative genre. Map those onto the closest visual style.
LEGACY_GENRE_TO_STYLE = {
    "action": "shonen_kinetic",
    "horror": "kage_noir",
    "romance": "shojo_bloom",
    "thriller": "seinen_cold",
    "emotional_drama": "retro_cel",
}

#: Intensity scales motion travel and how freely accents are spent.
INTENSITY_FACTORS = {"subtle": 1.28, "balanced": 1.0, "dramatic": 0.78}
INTENSITY_ACCENT_SHIFT = {"subtle": 1, "balanced": 0, "dramatic": -1}


def resolve_style(value: str | None) -> StylePreset:
    """Accept a style key, a legacy genre name, or None."""
    if value:
        key = str(value).strip().lower()
        if key in STYLE_PRESETS:
            return STYLE_PRESETS[key]
        mapped = LEGACY_GENRE_TO_STYLE.get(key)
        if mapped:
            return STYLE_PRESETS[mapped]
    return STYLE_PRESETS[DEFAULT_STYLE]


def style_choices() -> list[tuple[str, str, str, str]]:
    """(key, name, tagline, description) for building UI selectors."""
    return [(p.key, p.name, p.tagline, p.description) for p in STYLE_PRESETS.values()]
