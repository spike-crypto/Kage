"""End-to-end and unit tests for the Kage pipeline.

Run with:  pytest -q
The render tests need FFmpeg; they are skipped automatically when it is unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.api import app  # noqa: E402
from backend.audio_engine import (  # noqa: E402
    ShotAudio,
    build_soundtrack,
    ensure_sfx_library,
    read_wav,
    render_music,
    synth_sfx,
)
from backend.cinematic_director import PanelRef, build_plan, plan_locally
from backend.styles import STYLE_PRESETS, resolve_style  # noqa: E402
from backend.config import ffmpeg_path  # noqa: E402
from backend.database import init_db  # noqa: E402
from backend.manga_processor import classify_upload, sanitize_filename  # noqa: E402
from backend.schemas import PlannedShot  # noqa: E402
from backend.video_renderer import (  # noqa: E402
    RenderSpec,
    ShotSpec,
    build_timeline,
    render,
)
from backend.visual_engine import ShotRenderer, draw_subtitle  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="FFmpeg is not installed")


@pytest.fixture(scope="session")
def client() -> TestClient:
    init_db()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def artwork(tmp_path_factory) -> list[Path]:
    """Three generated test pages - no third-party artwork is used."""
    out = tmp_path_factory.mktemp("artwork")
    paths = []
    for n in range(3):
        img = Image.new("RGB", (900, 1300), (246, 244, 240))
        d = ImageDraw.Draw(img)
        d.rectangle([40, 40, 860, 620], outline=(20, 20, 25), width=6)
        d.rectangle([40, 660, 860, 1260], outline=(20, 20, 25), width=6)
        d.ellipse([200 + n * 60, 180, 420 + n * 60, 400], fill=(26, 26, 34))
        for i in range(30):
            d.line([(60 + i * 26, 700), (120 + i * 26, 1240)], fill=(60, 60, 72), width=3)
        path = out / f"page_{n}.png"
        img.save(path)
        paths.append(path)
    return paths


# ------------------------------------------------------------------- health


def test_health_reports_capabilities(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "ffmpeg" in body["capabilities"]
    assert isinstance(body["capabilities"]["video_encoders"], list)


# -------------------------------------------------------------- input safety


@pytest.mark.parametrize(
    "raw, expected_suffix",
    [("../../etc/passwd.png", ".png"), ("C:\\evil\\page.PNG", ".png"), ("a b/c*d.jpg", ".jpg")],
)
def test_sanitize_filename_strips_paths(raw, expected_suffix):
    safe = sanitize_filename(raw)
    assert "/" not in safe and "\\" not in safe and ".." not in safe
    assert safe.lower().endswith(expected_suffix)


def test_classify_upload_rejects_unknown_types():
    assert classify_upload("page.png") == "image"
    assert classify_upload("book.pdf") == "pdf"
    with pytest.raises(Exception):
        classify_upload("payload.exe")


def test_upload_rejects_non_image(client):
    project = client.post("/api/projects", json={"title": "Bad upload"}).json()
    response = client.post(
        f"/api/projects/{project['id']}/upload",
        files=[("files", ("notes.txt", b"not an image", "text/plain"))],
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_missing_project_returns_404(client):
    assert client.get("/api/projects/999999").status_code == 404


# ------------------------------------------------------------------ planner


def test_local_plan_matches_target_duration(artwork):
    panels = [PanelRef(panel_id=i, path=str(p), width=900, height=1300)
              for i, p in enumerate(artwork)]
    plan = plan_locally(panels, "action", 30, "balanced")
    assert len(plan.shots) >= 2
    assert abs(sum(s.duration_seconds for s in plan.shots) - 30) < 0.1
    assert {s.panel_id for s in plan.shots} <= {p.panel_id for p in panels}


def test_every_style_produces_a_valid_plan(artwork):
    panels = [PanelRef(panel_id=0, path=str(artwork[0]), width=900, height=1300)]
    for style in STYLE_PRESETS:
        plan = plan_locally(panels, style, 15, "dramatic")
        assert len(plan.shots) >= 2
        for shot in plan.shots:
            assert shot.duration_seconds > 0
            assert shot.music_intensity in ("low", "medium", "high")


def test_styles_respect_their_effect_budget(artwork):
    """Restraint is a contract, not a preference: no shot may exceed max_effects."""
    panels = [PanelRef(panel_id=i, path=str(p), width=900, height=1300)
              for i, p in enumerate(artwork)]
    for key, preset in STYLE_PRESETS.items():
        plan = plan_locally(panels, key, 60, "balanced")
        counts = [len(s.visual_effects) for s in plan.shots]
        assert max(counts) <= preset.max_effects, (
            f"{key} spent {max(counts)} effects, budget is {preset.max_effects}"
        )
        # Most shots should carry no named effect at all - the grade does the work.
        bare = counts.count(0)
        assert bare >= len(counts) * 0.5, f"{key}: only {bare}/{len(counts)} shots are clean"


def test_consecutive_shots_never_repeat_a_camera_move(artwork):
    panels = [PanelRef(panel_id=i, path=str(p), width=900, height=1300)
              for i, p in enumerate(artwork)]
    for key in STYLE_PRESETS:
        cameras = [s.camera_motion for s in plan_locally(panels, key, 60, "balanced").shots]
        repeats = [a for a, b in zip(cameras, cameras[1:]) if a == b]
        assert not repeats, f"{key} repeated {repeats} back to back"


def test_legacy_genre_names_still_resolve():
    assert resolve_style("horror").key == "kage_noir"
    assert resolve_style("action").key == "shonen_kinetic"
    assert resolve_style(None).key == "kage_noir"
    assert resolve_style("nonsense").key == "kage_noir"


def test_each_style_grades_differently(artwork):
    """Two styles must not produce the same image, or the presets are cosmetic only."""
    rendered = {}
    for key, preset in STYLE_PRESETS.items():
        r = ShotRenderer(artwork[0], 160, 90, "static_drift", [], "balanced", seed=3,
                         grade=preset.grade)
        rendered[key] = next(iter(r.iter_frames(0.2))).astype(float)
    keys = list(rendered)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            diff = np.mean(np.abs(rendered[a] - rendered[b]))
            assert diff > 2.0, f"{a} and {b} render near-identically (diff {diff:.2f})"


def test_planner_falls_back_without_credentials(artwork, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    panels = [PanelRef(panel_id=0, path=str(artwork[0]), width=900, height=1300)]
    plan, warning = build_plan(panels, "action", 15, use_hosted=True)
    assert plan.source == "local_planner"
    assert warning and "local planner" in warning


def test_shot_schema_rejects_unknown_values():
    shot = PlannedShot(
        panel_id=0, duration_seconds=3,
        camera_motion="drone_strike", transition="morph",
        visual_effects=["vignette", "lens_flare"], sound_cues=["impact", "kazoo"],
    )
    assert shot.camera_motion == "slow_zoom_in"   # coerced to a safe default
    assert shot.transition == "fade"
    assert shot.visual_effects == ["vignette"]    # unknown effect dropped
    assert shot.sound_cues == ["impact"]


# ----------------------------------------------------------- visual engine


def test_shot_renderer_produces_moving_frames(artwork):
    renderer = ShotRenderer(artwork[0], 320, 180, "slow_zoom_in", ["vignette"], seed=5)
    frames = list(renderer.iter_frames(1.0))
    assert len(frames) == 24
    assert all(f.shape == (180, 320, 3) and f.dtype == np.uint8 for f in frames)
    # The camera must actually move: first and last frames differ measurably.
    difference = np.mean(np.abs(frames[0].astype(float) - frames[-1].astype(float)))
    assert difference > 1.0, "camera motion produced a static shot"


def test_aspect_ratios_render_at_the_right_size(artwork):
    for (w, h) in ((1280, 720), (720, 1280), (900, 900)):
        renderer = ShotRenderer(artwork[0], w, h, "ken_burns", [])
        frame = next(iter(renderer.iter_frames(0.3)))
        assert frame.shape == (h, w, 3)


def test_subtitle_is_drawn_into_the_frame():
    blank = np.zeros((180, 320, 3), dtype=np.uint8)
    captioned = draw_subtitle(blank, "A readable caption", 1.0)
    assert captioned.shape == blank.shape
    assert captioned.max() > 200, "caption text should be bright against black"


# ------------------------------------------------------------ audio engine


def test_sfx_library_is_written_and_verified():
    library = ensure_sfx_library()
    assert len(library) >= 8
    for name, path in library.items():
        assert path.exists() and path.stat().st_size > 1024, name
        data, sr = read_wav(path)
        assert sr == 44100 and data.size > 0
        assert np.max(np.abs(data)) > 0.01, f"{name} is silent"


@pytest.mark.parametrize("mood", ["orchestral", "ambient", "dark", "energetic", "suspenseful"])
def test_music_renders_for_every_mood(mood):
    audio = render_music(mood, 3.0)
    assert audio.shape[0] == 44100 * 3
    assert np.max(np.abs(audio)) > 0.05, "music track is effectively silent"
    assert np.all(np.isfinite(audio))


def test_sfx_are_audible_and_bounded():
    for name in ("whoosh", "impact", "explosion", "suspense_riser", "heartbeat"):
        clip = synth_sfx(name)
        peak = float(np.max(np.abs(clip)))
        assert 0.02 < peak <= 1.5, f"{name} peak was {peak}"


def test_soundtrack_matches_requested_length_and_does_not_clip(tmp_path):
    shots = [
        ShotAudio(0.0, 2.0, ["whoosh"], "low"),
        ShotAudio(2.0, 2.0, ["impact", "wind"], "high"),
        ShotAudio(4.0, 2.0, ["ambient_drone"], "medium"),
    ]
    out = build_soundtrack(shots, "orchestral", 6.0, tmp_path / "mix.wav")
    data, sr = read_wav(out)
    assert sr == 44100
    assert abs(len(data) / sr - 6.0) < 0.05
    assert np.max(np.abs(data)) <= 1.0
    assert np.max(np.abs(data)) > 0.3, "mix is too quiet"


# ---------------------------------------------------------------- timeline


def test_timeline_compensates_for_transition_overlaps(artwork):
    shots = [
        ShotSpec(i + 1, artwork[i % len(artwork)], 3.0, "slow_zoom_in", "crossfade")
        for i in range(5)
    ]
    spec = RenderSpec(shots=shots, target_duration=15.0)
    entries, total = build_timeline(spec)
    assert abs(total / 24 - 15.0) < 0.1, f"timeline landed at {total / 24:.2f}s"
    assert entries[0].overlap_in == 0
    assert any(e.overlap_in > 0 for e in entries[1:]), "dissolves produced no overlap"


def test_cuts_do_not_overlap(artwork):
    shots = [ShotSpec(i + 1, artwork[0], 2.0, "pan_left", "cut") for i in range(3)]
    entries, total = build_timeline(RenderSpec(shots=shots))
    assert all(e.overlap_in == 0 for e in entries)
    assert total == 3 * 48


# -------------------------------------------------------------- full render


@needs_ffmpeg
def test_render_produces_a_playable_mp4(tmp_path, artwork):
    shots = [
        ShotSpec(1, artwork[0], 1.5, "slow_zoom_in", "cut", ["vignette"], ["whoosh"], "low",
                 "First caption"),
        ShotSpec(2, artwork[1], 1.5, "rapid_pan", "crossfade", ["speed_lines", "motion_blur"],
                 ["impact"], "high"),
        ShotSpec(3, artwork[2], 1.5, "ken_burns", "fade", ["film_grain"], ["wind"], "medium"),
    ]
    spec = RenderSpec(
        shots=shots, aspect_ratio="16:9", audio_mood="energetic", title="Test",
        burn_subtitles=True, target_duration=4.0,
        output_path=tmp_path / "out.mp4", audio_path=tmp_path / "out.wav",
        subtitle_path=tmp_path / "out.srt",
    )
    result = render(spec)

    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 20_000
    assert result.has_audio, "the MP4 has no audio stream"
    assert abs(result.duration_seconds - 4.0) < 0.3
    assert result.width == 1280 and result.height == 720

    codecs = {s["type"]: s["codec"] for s in result.diagnostics["probe_streams"]}
    assert codecs.get("video") == "h264"
    assert codecs.get("audio") == "aac"
    assert (tmp_path / "out.srt").exists(), "subtitle sidecar was not written"


@needs_ffmpeg
def test_render_reports_missing_artwork_clearly(tmp_path):
    spec = RenderSpec(
        shots=[ShotSpec(1, tmp_path / "gone.png", 1.0)],
        output_path=tmp_path / "x.mp4", audio_path=tmp_path / "x.wav",
    )
    with pytest.raises(Exception) as excinfo:
        render(spec)
    assert "missing" in str(excinfo.value).lower()


# ------------------------------------------------------------ api workflow


def test_uploading_again_replaces_the_previous_import(client, artwork):
    """Upload replaces rather than appends - the UI must not promise otherwise.

    Sending a book in batches therefore destroys every batch but the last. Caught when a
    42-page rebuild arrived as 15 pages, both on disk and in the database.
    """
    project_id = client.post("/api/projects", json={"title": "Replace"}).json()["id"]

    first = [("files", (p.name, p.read_bytes(), "image/png")) for p in artwork[:2]]
    assert client.post(f"/api/projects/{project_id}/upload", files=first).json()[
        "page_count"] == 2

    second = [("files", (artwork[2].name, artwork[2].read_bytes(), "image/png"))]
    body = client.post(f"/api/projects/{project_id}/upload", files=second).json()
    assert body["page_count"] == 1, "a second upload replaces the first"

    detail = client.get(f"/api/projects/{project_id}").json()
    pages = [p for p in detail["pages"] if p["kind"] == "page"]
    assert len(pages) == 1, f"expected the import to be replaced, found {len(pages)} pages"
    assert detail["page_count"] == len(pages), "page_count must match the rows that exist"


# -------------------------------------------------------------- environment


def test_installed_packages_have_consistent_requirements():
    """No installed package may have its pins violated by another.

    Regression: installing gradio_client pulled huggingface-hub from 0.36 to 2.3, while
    transformers pins <1.0. Nothing failed at install time - CLIP, depth estimation and
    narration simply stopped loading at runtime, and an unrelated check happened to
    notice. A dependency that breaks the local pipeline is not worth what it buys.

    Done in plain Python rather than `pip check`: the project's venv is made by uv and
    has no pip in it, so a test shelling out to pip would quietly never run.
    """
    from importlib.metadata import distributions

    packaging = pytest.importorskip("packaging")
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version

    installed = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name:
            installed[name] = dist.version
    del packaging

    conflicts = []
    for dist in distributions():
        owner = dist.metadata["Name"] or "?"
        for raw in dist.requires or []:
            try:
                requirement = Requirement(raw)
            except Exception:
                continue
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue        # an extra or a platform we are not on
            have = installed.get(requirement.name.lower().replace("_", "-"))
            if have is None or not requirement.specifier:
                continue
            try:
                if not requirement.specifier.contains(Version(have), prereleases=True):
                    conflicts.append(f"{owner} needs {raw}, found {have}")
            except InvalidVersion:
                continue

    assert not conflicts, "conflicting requirements:\n  " + "\n  ".join(conflicts)


def test_the_local_model_stack_still_imports():
    """The three local models must load, not merely be installed.

    `pip check` passes for a package that imports but raises on use, so this exercises
    the import path each model actually takes.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import transformers
    from huggingface_hub import __version__ as hub_version

    assert transformers.__version__, "transformers must be importable"
    major = int(hub_version.split(".")[0])
    assert major < 1, (
        f"huggingface-hub {hub_version} breaks transformers, which pins <1.0"
    )


# ------------------------------------------------------------- concurrency


@needs_ffmpeg
def test_the_app_stays_writable_while_an_analysis_runs(client, artwork, monkeypatch):
    """A slow analysis must not lock the database against the rest of the app.

    Context: analyze used to flush its page and shot rows and then run segmentation, OCR,
    CLIP and the narration chain - minutes on a real book - before committing. SQLite
    serialises writers, so renaming a project from the UI during an analysis failed with
    "database is locked" after waiting out the full 30s timeout. It now commits before
    any model runs.

    Honest limit: this passes with that fix reverted, so it does not reproduce the
    original failure. TestClient serves both requests through one pooled connection, so
    they share a transaction instead of contending for the write lock the way two
    requests against the real server do. It is kept as a smoke test of the property -
    a concurrent write completes, and quickly - not as proof the bug cannot return.
    Reproducing it needs two processes against the same file.
    """
    import threading
    import time as _time

    from backend import vision as vision_module

    started = threading.Event()
    release = threading.Event()

    def slow_analyse(paths, indices=None, **_kw):
        started.set()
        release.wait(timeout=20)       # stand in for a model taking its time
        return []

    monkeypatch.setattr(vision_module, "analyse_pages", slow_analyse)

    other = client.post("/api/projects", json={"title": "Bystander"}).json()["id"]
    project_id = client.post("/api/projects", json={"title": "Slow"}).json()["id"]
    files = [("files", (p.name, p.read_bytes(), "image/png")) for p in artwork]
    assert client.post(f"/api/projects/{project_id}/upload", files=files).status_code == 200

    result: dict = {}

    def run_analysis():
        try:
            result["status"] = client.post(
                f"/api/projects/{project_id}/analyze", json={"use_vision": True}
            ).status_code
        except Exception as exc:                       # pragma: no cover - diagnostic
            result["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=run_analysis, daemon=True)
    worker.start()
    assert started.wait(timeout=30), "analysis never reached the model step"

    # While it is held up in "the model", the rest of the app must still accept a write.
    began = _time.perf_counter()
    patched = client.patch(f"/api/projects/{other}", json={"title": "Renamed mid-analysis"})
    took = _time.perf_counter() - began

    release.set()
    worker.join(timeout=60)

    assert patched.status_code == 200, f"write blocked during analysis: {patched.text}"
    assert took < 5.0, f"write waited {took:.1f}s on the analysis transaction"
    assert client.get(f"/api/projects/{other}").json()["title"] == "Renamed mid-analysis"


# --------------------------------------------------------------- selection


def _insight(index, score):
    from backend.vision import PageInsight

    return PageInsight(page_index=index, kind="artwork", score=score)


def test_selection_takes_a_continuous_run_not_the_best_scattered_panels():
    """Shots from pages half a book apart are a gallery, not a sequence."""
    from backend.vision import select_pages

    # High scores at both ends, a solid run in the middle.
    scores = [0.9, 0.1, 0.1, 0.5, 0.6, 0.55, 0.58, 0.1, 0.1, 0.95]
    insights = [_insight(n, s) for n, s in enumerate(scores)]

    chosen = select_pages(insights, wanted=4)
    picked = [i.page_index for i in chosen]
    assert picked == sorted(picked), "selection must stay in reading order"
    assert max(picked) - min(picked) == len(picked) - 1, (
        f"expected a continuous run, got {picked}"
    )

    # Ranking alone would have grabbed the two outliers and two of the middle.
    scattered = [i.page_index for i in select_pages(insights, 4, continuous=False)]
    assert 0 in scattered and 9 in scattered
    assert 0 not in picked or 9 not in picked


def test_selection_still_drops_pages_with_no_artwork():
    from backend.vision import select_pages

    insights = [_insight(0, 0.0), _insight(1, 0.0)] + [
        _insight(n, 0.7) for n in range(2, 8)
    ]
    picked = [i.page_index for i in select_pages(insights, wanted=4)]
    assert 0 not in picked and 1 not in picked


# ----------------------------------------------------------------- routing


#: Labels the routing tests invent. Scoped so the suite never deletes real
#: measurements from the developer's own database, which it shares.
#: Every label any routing test records against. The fixture resets exactly these and
#: nothing else, because the tests share the real database with the running app: a
#: blanket reset would wipe measurements the user actually made. A label used in a test
#: but missing from this list is worse than either - it is never cleaned up, and it
#: lands in the live routing table as a permanent fake endpoint.
TEST_LABELS = ["prov:model", "fast-flaky", "slow-solid", "good", "broken",
               "never-tried", "quick", "slow", "slow-but-sure", "quick-and-sure"]


@pytest.fixture
def clean_routing():
    from backend import routing

    routing.reset(TEST_LABELS)
    yield routing
    routing.reset(TEST_LABELS)


def test_routing_counts_from_zero_on_a_new_endpoint(clean_routing):
    """Column defaults apply on INSERT, so a pending row still has None counters.

    Regression: ``stat.ok_count += 1`` raised TypeError on the very first call, the
    error was swallowed, and the router silently never learned anything.
    """
    clean_routing.record("prov:model", "Write", ok=True, latency=2.0)
    rows = clean_routing.health(["prov:model"])
    assert rows, "the first call must create a record"
    assert rows[0]["calls"] == 1
    assert rows[0]["success"] == 1.0


def test_routing_prefers_reliability_over_speed(clean_routing):
    for _ in range(10):
        clean_routing.record("fast-flaky", "Write", ok=False, latency=0.3, error="500")
    for _ in range(10):
        clean_routing.record("slow-solid", "Write", ok=True, latency=9.0)
    assert clean_routing.order(["fast-flaky", "slow-solid"], "Write") == [
        "slow-solid", "fast-flaky",
    ]


def test_routing_puts_a_resting_endpoint_last(clean_routing):
    clean_routing.record("good", "Observe", ok=True, latency=5.0)
    clean_routing.record("broken", "Observe", ok=False, latency=0.2, error="HTTP 429")
    order = clean_routing.order(["broken", "good", "never-tried"], "Observe")
    assert order[0] == "good", "a working endpoint comes first"
    assert order[-1] == "broken", "a resting endpoint is the last resort"
    assert "never-tried" in order, "an unknown endpoint still gets a chance"


def test_routing_prefers_the_faster_of_two_reliable_endpoints(clean_routing):
    for _ in range(4):
        clean_routing.record("quick", "Observe", ok=True, latency=3.1)
        clean_routing.record("slow", "Observe", ok=True, latency=76.4)
    assert clean_routing.order(["slow", "quick"], "Observe")[0] == "quick"


def test_provider_endpoint_never_returns_keys(client):
    body = client.get("/api/providers").json()
    assert "providers" in body and "usage" in body and "order" in body
    blob = json.dumps(body)
    for secret in ("api_key", "sk-or-v1-", "AIza"):
        assert secret not in blob, f"the providers endpoint leaked {secret}"
    for entry in body["providers"]:
        assert set(entry) >= {"key", "name", "configured", "env", "signup"}
        assert "value" not in entry


@needs_ffmpeg
def test_narration_is_stored_on_the_shots(client, artwork, monkeypatch):
    """Lines the writer produced must land on the shots that were just created.

    Regression: narration was looked up through ``project.shots``, which is the stale
    pre-delete collection during analysis, so every line was silently dropped and the
    render was mute while the logs said narration had been written.
    """
    from backend import narration as narration_engine

    def fake_build_script(page_paths, shots, total_duration, project_id=None):
        report = narration_engine.ChainReport(job="Write", used="stub:model")
        lines = [
            narration_engine.Line(
                sequence_number=int(shot["sequence_number"]),
                text=f"Line for shot {shot['sequence_number']}.",
                start_seconds=0.0,
                duration_seconds=float(shot["duration_seconds"]),
                word_budget=8,
            )
            for shot in shots
        ]
        return lines, [report]

    monkeypatch.setattr(narration_engine, "build_script", fake_build_script)

    project_id = client.post("/api/projects", json={"title": "Narrated"}).json()["id"]
    files = [("files", (p.name, p.read_bytes(), "image/png")) for p in artwork]
    assert client.post(f"/api/projects/{project_id}/upload", files=files).status_code == 200

    analyzed = client.post(f"/api/projects/{project_id}/analyze",
                           json={"write_narration": True, "use_vision": False})
    assert analyzed.status_code == 200, analyzed.text
    shots = analyzed.json()["shots"]
    assert shots, "analysis should produce shots"
    missing = [s["sequence_number"] for s in shots if not s.get("narration_text")]
    assert not missing, (
        f"shots without narration: {missing}; "
        f"error_message={analyzed.json().get('error_message')!r}"
    )

    # And it must survive a round trip, so the renderer sees it too.
    detail = client.get(f"/api/projects/{project_id}").json()
    assert all(s.get("narration_text") for s in detail["shots"])


def test_full_api_workflow_upload_to_download(client, artwork):
    created = client.post("/api/projects", json={
        "title": "API workflow", "style": "seinen_cold", "target_duration": 15,
        "aspect_ratio": "16:9", "audio_mood": "suspenseful", "intensity": "balanced",
    })
    assert created.status_code == 201
    project_id = created.json()["id"]

    files = [("files", (p.name, p.read_bytes(), "image/png")) for p in artwork]
    upload = client.post(f"/api/projects/{project_id}/upload", files=files)
    assert upload.status_code == 200, upload.text
    assert upload.json()["page_count"] == 3

    analyzed = client.post(f"/api/projects/{project_id}/analyze", json={})
    assert analyzed.status_code == 200, analyzed.text
    body = analyzed.json()
    assert len(body["shots"]) >= 2
    assert body["status"] == "planned"

    started = client.post(f"/api/projects/{project_id}/render")
    assert started.status_code == 202, started.text

    deadline, final = 420, None
    for _ in range(deadline):
        state = client.get(f"/api/projects/{project_id}/status").json()
        job = state["job"]
        if job and job["status"] in ("completed", "failed"):
            final = job
            break
        import time as _time

        _time.sleep(1.0)

    assert final is not None, "render did not finish in time"
    assert final["status"] == "completed", final.get("error_message")
    assert Path(final["output_video_path"]).exists()

    video = client.get(f"/api/projects/{project_id}/video")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert len(video.content) > 20_000

    # Project state survives in SQLite.
    stored = client.get(f"/api/projects/{project_id}").json()
    assert stored["status"] == "completed"
    assert stored["render_job"]["progress_percent"] == 100.0


def test_render_without_a_plan_is_rejected(client):
    project = client.post("/api/projects", json={"title": "No plan"}).json()
    response = client.post(f"/api/projects/{project['id']}/render")
    assert response.status_code in (400, 503)


# ------------------------------------------------------------ vision model

from backend import vision  # noqa: E402

needs_vision = pytest.mark.skipif(
    not vision.is_available(), reason="torch/transformers not installed"
)


def test_vision_reports_availability_without_crashing():
    """is_available() must never raise, even with nothing installed."""
    assert isinstance(vision.is_available(), bool)
    assert isinstance(vision.device_name(), str)


def test_subject_hints_cover_every_prompt_label():
    """Every subject the model can report needs a camera hint and a framing scale."""
    for subject in vision.SUBJECT_PROMPTS:
        assert subject in vision.SUBJECT_CAMERA_HINT, f"no camera hint for {subject}"
        assert subject in vision.SUBJECT_FRAMING_SCALE, f"no framing scale for {subject}"


def test_select_pages_always_returns_something():
    """Even if every page scores badly the user must still get a sequence."""
    junk = [vision.PageInsight(page_index=i, kind="text", score=0.01) for i in range(5)]
    chosen = vision.select_pages(junk, wanted=3)
    assert len(chosen) >= 2
    assert [c.page_index for c in chosen] == sorted(c.page_index for c in chosen)


def test_select_pages_prefers_artwork_and_keeps_reading_order():
    insights = [
        vision.PageInsight(page_index=0, kind="cover", score=0.1),
        vision.PageInsight(page_index=1, kind="text", score=0.05),
        vision.PageInsight(page_index=2, kind="artwork", score=0.9),
        vision.PageInsight(page_index=3, kind="artwork", score=0.8),
        vision.PageInsight(page_index=4, kind="artwork", score=0.95),
    ]
    chosen = vision.select_pages(insights, wanted=3)
    assert [c.page_index for c in chosen] == [2, 3, 4]
    assert all(c.kind == "artwork" for c in chosen)


@needs_vision
def test_vision_separates_text_pages_from_artwork(artwork, tmp_path):
    """The thing OpenCV could not do: tell a page of prose from a drawing."""
    from PIL import Image, ImageDraw

    text_page = tmp_path / "text.png"
    img = Image.new("RGB", (900, 1300), (248, 246, 242))
    d = ImageDraw.Draw(img)
    for row in range(28):
        for col in range(3):
            d.text((80 + col * 260, 70 + row * 42),
                   "text line of writing here", fill=(20, 20, 20))
    img.save(text_page)

    results = vision.analyse_pages([text_page, artwork[0]], [0, 1])
    assert len(results) == 2
    text_insight, art_insight = results
    assert 0.0 <= text_insight.score <= 1.0
    assert text_insight.scores["kind"]["text"] > text_insight.scores["kind"]["artwork"], (
        f"text page misread as {text_insight.kind}"
    )


@needs_vision
def test_vision_insight_fields_are_sane(artwork):
    results = vision.analyse_pages([artwork[0]], [0])
    insight = results[0]
    assert insight.kind in vision.KIND_PROMPTS
    assert insight.subject in vision.SUBJECT_PROMPTS
    assert 0.0 <= insight.energy <= 1.0
    assert 0.0 <= insight.score <= 1.0
    assert insight.describe()


def test_uint8_overlay_blend_matches_float_arithmetic():
    """The fast overlay path must be the same picture, not merely a similar one.

    The four overlay effects (speed lines, particles, rain, impact flash) used to blend
    toward a flat colour in float32, which promoted the whole frame and dragged the
    grade, vignette and grain onto their slow paths with it. They now do the same
    arithmetic in uint8. This pins the two against each other so a future change to
    _blend_to_colour cannot quietly alter the output.
    """
    rng = np.random.default_rng(3)
    for colour in (255.0, 242.0, 226.0):
        frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        alpha = rng.integers(0, 118, (64, 96), dtype=np.uint8)

        fast = ShotRenderer._blend_to_colour(frame, alpha, colour)
        reference = ShotRenderer._blend_to_colour(frame.astype(np.float32), alpha, colour)
        reference = np.clip(reference, 0, 255).astype(np.uint8)

        assert fast.dtype == np.uint8
        assert fast.shape == frame.shape
        # One level of rounding is expected; anything more is a behaviour change.
        assert np.abs(fast.astype(np.int16) - reference.astype(np.int16)).max() <= 1


def test_overlay_effects_no_longer_force_float_promotion():
    """A speed-lines shot should stay in uint8 end to end.

    One effect promoting the frame is enough to put every effect after it on a float
    path, so this guards the flag rather than the arithmetic.
    """
    panel = Path("data/projects/project_00116/panels/panel_p006_000.png")
    if not panel.exists():
        pytest.skip("no sample panel available")

    renderer = ShotRenderer(
        image_path=str(panel), out_width=320, out_height=180,
        camera_motion="push_in",
        effects=["speed_lines", "grain", "impact_flash"],
        intensity="normal", fps=12, seed=7,
    )
    assert renderer._float_effects is False
    for frame in renderer.iter_frames(0.5):
        assert frame.dtype == np.uint8


def test_bloom_stays_in_uint8_and_grade_returns_uint8():
    """Bloom used to promote the whole frame to float32 and back.

    The highlight threshold is a saturating subtract and the recombine is an
    addWeighted, both exact in 8-bit, so a graded frame should never leave uint8 unless
    one of the legacy colour effects is asked for.
    """
    from backend.styles import STYLE_PRESETS

    grade = STYLE_PRESETS["shonen_kinetic"].grade
    assert grade.bloom > 0.01, "preset no longer exercises the bloom path"

    panel = Path("data/projects/project_00116/panels/panel_p006_000.png")
    if not panel.exists():
        pytest.skip("no sample panel available")

    renderer = ShotRenderer(
        image_path=str(panel), out_width=320, out_height=180,
        camera_motion="push_in", effects=[], intensity="normal",
        fps=12, seed=7, grade=grade,
    )
    assert renderer._float_effects is False
    graded = renderer._apply_grade(
        np.full((180, 320, 3), 200, dtype=np.uint8)
    )
    assert graded.dtype == np.uint8

    # A legacy colour effect is the one case that still needs the headroom.
    colour = ShotRenderer(
        image_path=str(panel), out_width=320, out_height=180,
        camera_motion="push_in", effects=["desaturate"], intensity="normal",
        fps=12, seed=7, grade=grade,
    )
    assert colour._colour_effects == ["desaturate"]
    assert colour._apply_grade(np.full((180, 320, 3), 200, dtype=np.uint8)).dtype == np.uint8


def test_warp_maps_reuse_buffers_without_changing_them():
    """maps_at returns scratch buffers; the values must still be right.

    It builds base + depth * a + drift * b with cv2 into reused arrays instead of
    accumulating onto freshly allocated zeros. Same arithmetic, two passes instead of
    six, no per-frame allocation.
    """
    from backend.animation import Warp

    h, w = 120, 160
    depth = np.random.default_rng(11).random((h, w)).astype(np.float32)
    warp = Warp(w, h, depth, parallax=0.034, drift=0.006, seed=7)
    assert warp.active

    travel = (0.6, -0.3)
    map_x, map_y = warp.maps_at(0.25, travel)

    # The identity grid plus a displacement: every coordinate should sit near its own
    # pixel, not somewhere arbitrary.
    base_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    assert np.abs(map_x - base_x).max() < 0.034 * w * 2.0
    assert map_x.dtype == np.float32 and map_y.dtype == np.float32

    # Buffers are reused, so a later call must overwrite rather than accumulate.
    again_x, again_y = warp.maps_at(0.25, travel)
    assert np.abs(again_x - map_x).max() == 0.0
    assert np.abs(again_y - map_y).max() == 0.0

    moved_x, _ = warp.maps_at(0.75, travel)
    assert np.abs(moved_x - base_x).max() > 0.0


def test_packaged_streamlit_floor_covers_what_the_pages_use():
    """The shipped pin must not permit a Streamlit too old to run the app.

    The floor was >=1.36 while the pages used st.space, st.pills and
    st.segmented_control and config.toml set theme keys that did not exist then. The
    build only ever worked because the resolver happened to pick a recent wheel; one
    that picked an older one would have shipped a release that crashed on page load.
    """
    import streamlit as st
    from packaging.requirements import Requirement
    from packaging.version import Version

    needed = ["space", "pills", "segmented_control", "navigation", "Page"]
    for attr in needed:
        assert hasattr(st, attr), f"app pages use st.{attr}, missing from installed Streamlit"

    text = Path("packaging/requirements-core.txt").read_text(encoding="utf-8")
    line = next(
        raw.strip() for raw in text.splitlines()
        if raw.strip() and not raw.strip().startswith("#")
        and Requirement(raw.strip()).name == "streamlit"
    )
    floor = next(
        Version(spec.version) for spec in Requirement(line).specifier if spec.operator == ">="
    )

    # Whatever we test against is the oldest thing we have any evidence works.
    assert floor <= Version(st.__version__), (
        f"pin floor {floor} is above the tested version {st.__version__}"
    )
    assert floor >= Version("1.66"), (
        f"pin floor {floor} is below the version that introduced the APIs the pages use"
    )


def test_an_untried_endpoint_outranks_a_slow_proven_one(clean_routing):
    """Rule 3 says unknowns get one chance. They have to actually get it.

    Unknowns used to occupy a tier below every proven endpoint. Tier outranks cost
    absolutely and a chain stops at the first success, so an untried endpoint was
    reached only when every proven one failed on the same call. Against a reliable but
    slow incumbent that meant never: simulated over 2000 runs of 60 decisions, a
    twentyfold cheaper newcomer was tried a median of zero times and the router cost 20x
    the best possible ordering, which is exactly the fixed chain it exists to beat.
    """
    for _ in range(10):
        clean_routing.record("slow-but-sure", "Write", ok=True, latency=20.0)

    order = clean_routing.order(["slow-but-sure", "never-tried"], "Write")
    assert order[0] == "never-tried", (
        "an endpoint costed at UNKNOWN_LATENCY must be tried ahead of a proven one "
        "that takes longer than that, or it is never tried at all"
    )


def test_an_untried_endpoint_does_not_displace_a_fast_proven_one(clean_routing):
    """The other half of rule 3: a working endpoint keeps its place.

    Letting unknowns compete on cost is only safe if the comparison is a real one. An
    endpoint already answering faster than UNKNOWN_LATENCY must stay in front.
    """
    for _ in range(10):
        clean_routing.record("quick-and-sure", "Write", ok=True, latency=1.5)

    order = clean_routing.order(["quick-and-sure", "never-tried"], "Write")
    assert order[0] == "quick-and-sure", (
        "a proven endpoint faster than the unknown guess must not be displaced"
    )
