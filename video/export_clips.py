"""Cut the finished video into one clip per X post (captions and narration included)."""
import subprocess
from pathlib import Path

from video.tts import load_sections

ROOT = Path(__file__).resolve().parents[1]

# section name -> clip file name; the number is the post it goes with
CLIPS_FOR_POSTS = {"jump": "2_jump", "planning": "3_planning", "hold": "4_feedback", "landing": "5_landing",
                   "ramp": "6_ramp", "semi": "7_semi"}


def export_x_clips(final: str, out_dir: str) -> list[Path]:
    secs = load_sections(str(ROOT / "audio"))
    starts, t = {}, 0.0
    for s in secs:
        starts[s.name] = (t, s.duration)
        t += s.duration
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, stem in CLIPS_FOR_POSTS.items():
        t0, dur = starts[name]
        path = out / f"{stem}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", final, "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
                        "-af", f"afade=t=in:d=0.15,afade=t=out:st={dur - 0.2:.3f}:d=0.2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", str(path)], check=True)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in export_x_clips(str(ROOT / "out" / "kickflip_explainer.mp4"), str(ROOT / "out" / "x_clips")):
        print(p)
