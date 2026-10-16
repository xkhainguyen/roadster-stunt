import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "audio"
FINAL = ROOT / "out" / "kickflip_explainer.mp4"


def test_sections_timed(tmp_path):
    from video.tts import Section, synthesize
    script = tmp_path / "s.md"
    script.write_text("## one\nFirst sentence here. Second one.\n\n## two\nThird sentence.\n")
    secs = synthesize(str(script), str(tmp_path / "a"))
    assert [s.name for s in secs] == ["one", "two"]
    for s in secs:
        assert s.duration > 0 and len(s.sentences) >= 1 and Path(s.wav).exists()
        t = [x for _, t0, t1 in s.sentences for x in (t0, t1)]
        assert t == sorted(t) and t[-1] <= s.duration + 1e-6
    data = json.loads((tmp_path / "a" / "sections.json").read_text())
    assert data[0]["name"] == "one"


def test_assemble_rejects_too_long(tmp_path):
    from video.assemble import assemble
    from video.tts import Section
    secs = [Section(name=f"s{i}", wav="x.wav", duration=47.0, sentences=[("x", 0.0, 1.0)]) for i in range(3)]
    with pytest.raises(ValueError):
        assemble(secs, {}, str(tmp_path / "out.mp4"))


def test_final_video_props():
    if not FINAL.exists():
        pytest.skip("video not built; run python -m video.assemble")
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json",
         str(FINAL)], capture_output=True, text=True, check=True).stdout)
    kinds = {s["codec_type"] for s in info["streams"]}
    v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"]) == (1920, 1080)
    assert "audio" in kinds
    assert 55 <= float(info["format"]["duration"]) <= 140


def test_main_checks_cap_before_rendering(monkeypatch):
    import video.assemble as A
    from video.tts import Section
    secs = [Section(name=n, wav="x.wav", duration=24.0, sentences=[("x", 0.0, 1.0)]) for n in A.ORDER]
    monkeypatch.setattr(A, "load_sections", lambda _: secs)

    def boom(*a, **k):
        raise AssertionError("rendered before checking the cap")
    monkeypatch.setattr(A, "build_clips", boom)
    with pytest.raises(ValueError):
        A.main()


def test_final_video_loudness():
    """Social feeds play around -14 to -16 LUFS; a quiet narration sounds broken next to other posts."""
    if not FINAL.exists():
        pytest.skip("video not built; run python -m video.assemble")
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(FINAL), "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = log[log.rfind("Summary:"):]
    lufs = float(summary.split("I:")[1].split("LUFS")[0])
    peak = float(summary.split("Peak:")[1].split("dBFS")[0])
    assert -17.5 <= lufs <= -14.5, lufs
    assert peak <= -1.0, peak


def test_parse_voice_blend():
    from video.tts import parse_voice
    assert parse_voice("am_michael") == [("am_michael", 1.0)]
    assert parse_voice("am_eric:0.4,zm_yunxi:0.6") == [("am_eric", 0.4), ("zm_yunxi", 0.6)]
    with pytest.raises(ValueError):
        parse_voice("am_eric:0.4,zm_yunxi:0.4")  # weights must sum to 1


def test_repo_qr_decodes(tmp_path):
    import zxingcpp
    from PIL import Image
    from video.qr import REPO_URL, make_qr
    png = make_qr(str(tmp_path / "qr.png"))
    hits = zxingcpp.read_barcodes(Image.open(png))
    assert [h.text for h in hits] == [REPO_URL]


def test_end_card_qr_scans_in_final_video(tmp_path):
    """The QR must survive Manim, scaling and H.264: decode it from a real frame of the end card."""
    import zxingcpp
    from PIL import Image
    from video.qr import REPO_URL
    if not FINAL.exists():
        pytest.skip("video not built; run python -m video.assemble")
    frame = tmp_path / "end.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-1.0", "-i", str(FINAL), "-frames:v", "1",
                    str(frame)], check=True)
    assert REPO_URL in [h.text for h in zxingcpp.read_barcodes(Image.open(frame))]


def test_pronunciation_only_changes_audio_text():
    from video.tts import spoken
    assert spoken("Here I use TinyMPC, and TinyMPC lands it.") == "Here I use Tiny M P C, and Tiny M P C lands it."
    assert spoken("No change here.") == "No change here."


def test_x_clips_match_sections(tmp_path):
    if not FINAL.exists():
        pytest.skip("video not built; run python -m video.assemble")
    from video.export_clips import CLIPS_FOR_POSTS, export_x_clips
    from video.tts import load_sections
    out = export_x_clips(str(FINAL), str(tmp_path))
    secs = {s.name: s for s in load_sections(str(AUDIO))}
    assert [p.stem for p in out] == list(CLIPS_FOR_POSTS.values())
    for name, path in zip(CLIPS_FOR_POSTS, out):
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json",
             str(path)], capture_output=True, text=True, check=True).stdout)
        v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
        assert (v["width"], v["height"]) == (1920, 1080) and "audio" in {s["codec_type"] for s in info["streams"]}
        assert float(info["format"]["duration"]) == pytest.approx(secs[name].duration, abs=0.15)
