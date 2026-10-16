import json
import os
import subprocess

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from roadster.flight import DATA, simulate
from video.render import render_clip


@pytest.fixture(scope="module")
def log_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("logs") / "log.npz"
    simulate("nominal").save(p)
    return str(p)


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def test_render_frame_count(log_path, tmp_path):
    out = render_clip(log_path, str(tmp_path / "clip.mp4"), camera="side", t_range=(1.5, 2.5))
    s = _probe(out)
    assert (s["width"], s["height"]) == (1920, 1080)
    assert int(s["nb_read_frames"]) == 30


def test_render_slowmo_frames(log_path, tmp_path):
    out = render_clip(log_path, str(tmp_path / "slow.mp4"), camera="chase", t_range=(2.0, 2.5), slowmo=0.5,
                      size=(640, 360))
    assert int(_probe(out)["nb_read_frames"]) == 30


def test_render_ramp_scene(tmp_path):
    from roadster.ramp import simulate as simulate_ramp
    p = tmp_path / "ramp.npz"
    simulate_ramp("nominal").save(p)
    out = render_clip(str(p), str(tmp_path / "ramp.mp4"), camera="side", t_range=(-0.5, 0.0), size=(640, 360))
    assert int(_probe(out)["nb_read_frames"]) == 15


def test_render_tracking_side_camera(log_path, tmp_path):
    out = render_clip(log_path, str(tmp_path / "track.mp4"), camera=("track", 4.0, 14.0, 90.0, -3.0),
                      t_range=(0.0, 0.5), size=(320, 360))
    s = _probe(out)
    assert (s["width"], s["height"]) == (320, 360) and int(s["nb_read_frames"]) == 15


def test_semi_scene_near_clip_is_close():
    """MuJoCo scales the near clip plane by the model extent; a long road must not push it out to meters."""
    from roadster.model import load
    m, _ = load("semi")
    assert m.vis.map.znear * m.stat.extent < 2.0
