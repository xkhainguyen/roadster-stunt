"""Build the final explainer: sim clips timed to the narration, Manim overlays, captions, audio."""
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import soundfile as sf

from video.tts import SR, Section, load_sections

ROOT = Path(__file__).resolve().parents[1]
DATA, CLIPS, OUT, AUDIO = ROOT / "data", ROOT / "clips", ROOT / "out", ROOT / "audio"
MEDIA = ROOT / "media"
MAX_S = 140.0
END_HOLD_S = 3.0  # extra still time on the end card so viewers can scan the QR
FADE_S = 0.2      # fade through black at section boundaries
FPS = 30
ORDER = ["hook", "jump", "planning", "hold", "landing", "ramp", "semi", "end"]

# Sim-time windows (seconds relative to launch) shown in each sim section.
# The hook opens mid-air with jets firing; the hold window puts the gust under the sentence that names it.
WINDOWS = {"hook": (0.35, 3.0), "jump": (-0.8, 3.6), "hold": (-0.5, 4.4), "landing": (1.6, 3.4), "ramp": (-3.9, 2.8), "semi": (-0.8, 3.8),
           "hook_ramp": (-1.0, 2.0)}


def _run(cmd):
    subprocess.run(cmd, check=True, cwd=ROOT)


def _ffmpeg(*args):
    _run(["ffmpeg", "-y", "-loglevel", "error", *args])


def _encode_args():
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS)]


def manim(scene: str, transparent: bool) -> Path:
    cmd = [shutil.which("manim") or "manim", "render", "-r", "1920,1080", "--fps", str(FPS),
           "--media_dir", str(MEDIA), "--disable_caching", "-v", "WARNING"]
    if transparent:
        cmd.append("-t")
    _run(cmd + ["video/scenes.py", scene])
    ext = ".mov" if transparent else ".mp4"
    hits = sorted(MEDIA.glob(f"videos/scenes/**/{scene}{ext}"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"manim output for {scene}")
    return hits[-1]


def overlay(base: Path, over: Path, out: Path) -> Path:
    _ffmpeg("-i", str(base), "-i", str(over), "-filter_complex", "[0:v][1:v]overlay=eof_action=pass:format=auto",
            *_encode_args(), "-an", str(out))
    return out


def build_clips(sections: dict[str, Section]) -> dict[str, Path]:
    from video.render import render_clip
    CLIPS.mkdir(exist_ok=True)
    timing = {}
    for name in ("hook", "jump", "hold", "landing", "ramp", "semi"):
        dur = sections[name].duration
        if name == "landing":
            dur = sections[name].sentences[2][1]  # card takes over at the TinyMPC sentence
        t0, t1 = WINDOWS[name]
        timing[name] = {"t0": t0, "t1": t1, "slowmo": (t1 - t0) / dur, "duration": dur}
    t0, t1 = WINDOWS["hook_ramp"]
    timing["hook_ramp"] = {"t0": t0, "t1": t1, "slowmo": (t1 - t0) / timing["hook"]["duration"],
                           "duration": timing["hook"]["duration"]}
    (DATA / "clip_timing.json").write_text(json.dumps(timing, indent=1))

    def rc(log, out, name, **kw):
        tm = timing[name]
        return Path(render_clip(str(DATA / log), str(CLIPS / out), t_range=(tm["t0"], tm["t1"]),
                                slowmo=tm["slowmo"], **kw))

    base = {
        "hook_kickflip": rc("log_nominal_cl.npz", "hook_kickflip.mp4", "hook", camera="follow", size=(960, 1080)),
        "hook_ramp": rc("log_ramp_nominal_cl.npz", "hook_ramp.mp4", "hook_ramp", camera=("track", 9.2, 23.0, 90.0, -4.0),
                        size=(960, 1080)),
        "jump": rc("log_nominal_cl.npz", "jump_base.mp4", "jump", camera="side", show_trail=True),
        "landing": rc("log_nominal_cl.npz", "landing_base.mp4", "landing", camera="follow", show_mpc=True),
        "ramp": rc("log_ramp_nominal_cl.npz", "ramp_base.mp4", "ramp", camera="side", show_trail=True),
        "semi": rc("log_semi_cl.npz", "semi_base.mp4", "semi", camera=("track", 2.2, 14.0, 30.0, -13.0), show_mpc=True),
    }
    ol = rc("log_gust_model_error_ol_breakable.npz", "hold_ol.mp4", "hold", camera="follow", size=(960, 1080))
    cl = rc("log_gust_model_error_cl.npz", "hold_cl.mp4", "hold", camera="follow", size=(960, 1080))
    base["hook"] = CLIPS / "hook_base.mp4"
    _ffmpeg("-i", str(base["hook_kickflip"]), "-i", str(base["hook_ramp"]), "-filter_complex",
            "[0:v][1:v]hstack=inputs=2", *_encode_args(), str(base["hook"]))
    base["hold"] = CLIPS / "hold_base.mp4"
    _ffmpeg("-i", str(ol), "-i", str(cl), "-filter_complex", "[0:v][1:v]hstack=inputs=2", *_encode_args(),
            str(base["hold"]))

    clips = {}
    for name, scene in (("hook", "HookTag"), ("jump", "JumpOverlay"), ("hold", "HoldOverlay"),
                        ("landing", "LandingOverlay"), ("ramp", "RampTag"),
                        ("semi", "SemiTag")):
        clips[name] = overlay(base[name], manim(scene, transparent=True), CLIPS / f"{name}.mp4")
    card = manim("LandingCard", transparent=False)
    d_land = timing["landing"]["duration"]
    _ffmpeg("-i", str(clips["landing"]), "-i", str(card), "-filter_complex",
            f"[0:v]fade=t=out:st={d_land - FADE_S:.3f}:d={FADE_S}[a];[1:v]fade=t=in:st=0:d={FADE_S}[b];"
            "[a][b]concat=n=2:v=1:a=0",
            *_encode_args(), str(CLIPS / "landing_full.mp4"))
    clips["landing"] = CLIPS / "landing_full.mp4"
    clips["planning"] = manim("Planning", transparent=False)
    clips["end"] = manim("EndCard", transparent=False)
    return clips


def _srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_captions(sections: list[Section], path: Path):
    lines, k, offset = [], 1, 0.0
    for s in sections:
        for text, t0, t1 in s.sentences:
            lines += [str(k), f"{_srt_time(offset + t0)} --> {_srt_time(offset + t1 + 0.15)}", text, ""]
            k += 1
        offset += s.duration
    path.write_text("\n".join(lines))


def check_length(sections: list[Section]) -> float:
    total = sum(s.duration for s in sections)
    if total > MAX_S:
        raise ValueError(f"narration is {total:.1f}s, over the {MAX_S:.0f}s cap")
    return total


def assemble(sections: list[Section], clips: dict, out: str = str(OUT / "kickflip_explainer.mp4")) -> str:
    check_length(sections)
    OUT.mkdir(exist_ok=True)
    sections = [Section(s.name, s.wav, s.duration + (END_HOLD_S if s is sections[-1] else 0.0), s.sentences)
                for s in sections]
    narration = np.concatenate([sf.read(s.wav)[0] for s in sections] + [np.zeros(int(END_HOLD_S * SR))]
                               ).astype(np.float32)
    sf.write(OUT / "narration.wav", narration, SR)
    srt = OUT / "captions.srt"
    write_captions(sections, srt)

    inputs, parts = [], []
    for i, s in enumerate(sections):
        inputs += ["-i", str(clips[s.name])]
        # pad or trim every section to its narration length so audio and captions stay in sync
        fades = ""
        if i > 0:
            fades += f",fade=t=in:st=0:d={FADE_S}"
        if i < len(sections) - 1:
            fades += f",fade=t=out:st={s.duration - FADE_S:.3f}:d={FADE_S}"
        parts.append(f"[{i}:v]scale=1920:1080,fps={FPS},tpad=stop_mode=clone:stop_duration=5,"
                     f"trim=duration={s.duration:.3f},setpts=PTS-STARTPTS{fades}[v{i}]")
    n = len(sections)
    # one box behind the whole caption (libass BorderStyle=4; style 3 boxes each line separately), with a
    # transparent letter outline so the glyphs get no halo
    style = ("FontName=DejaVu Sans,FontSize=15,PrimaryColour=&H00FFFFFF,OutlineColour=&HFF000000,"
             "BackColour=&H50000000,BorderStyle=4,Outline=4,Shadow=0,MarginV=18")
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(n)) + \
        f"concat=n={n}:v=1:a=0[cat];[cat]subtitles={srt}:force_style='{style}'[v]"
    _ffmpeg(*inputs, "-i", str(OUT / "narration.wav"), "-filter_complex", graph, "-map", "[v]",
            "-map", f"{n}:a", *_encode_args(), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", out)
    return out


def main():
    secs = load_sections(str(AUDIO))
    by_name = {s.name: s for s in secs}
    secs = [by_name[n] for n in ORDER]
    check_length(secs)  # fail before ~10 minutes of rendering
    clips = build_clips(by_name)
    print(assemble(secs, clips))


if __name__ == "__main__":
    main()
