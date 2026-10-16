import re
from pathlib import Path

import numpy as np
import pytest

from roadster import params as P
from roadster.flight import DATA, Log, get_plan, simulate
from roadster.trajopt import Plan
from video import facts

ROOT = Path(__file__).resolve().parents[1]


def _log(scenario):
    path = DATA / f"log_{scenario}_cl.npz"
    if not path.exists():  # fresh clone: generate instead of requiring scripts/run_kickflip.py first
        DATA.mkdir(exist_ok=True)
        simulate(scenario).save(path)
    return Log.load(path)


@pytest.fixture(scope="module")
def plan():
    return get_plan()


@pytest.fixture(scope="module")
def log():
    return _log("nominal")


def test_air_fraction_matches_plan(plan):
    impulse = float((plan.U.sum(axis=1) * np.diff(plan.t)).sum())
    assert facts.air_fraction(plan) == pytest.approx(impulse / P.IMPULSE_BUDGET)


def test_burn_end_is_when_launch_thrust_stops(plan):
    t = facts.burn_end(plan)
    k = np.searchsorted(plan.t, t)
    assert plan.U[:k].sum(axis=1).min() >= 0.01 * plan.U.sum(axis=1).max()
    assert plan.U[k:k + 3].sum(axis=1).max() < 0.01 * plan.U.sum(axis=1).max()


def test_solve_stats_come_from_logs():
    logs = [_log(s) for s in ("nominal", "gust", "fast_launch")]
    med, worst = facts.solve_ms(logs)
    allt = np.concatenate([l.mpc_solve_ms for l in logs])
    assert med == pytest.approx(np.median(allt)) and worst == pytest.approx(allt.max())


def test_narration_height_matches_apex(log):
    apex, _ = facts.apex_and_airtime(log)
    script = (ROOT / "video" / "script.md").read_text()
    if "six and a half meters" in script:
        assert 6.25 <= apex < 6.75, apex
    else:
        assert "over six meters" in script and 6.0 <= apex < 6.5, apex


def test_narration_crash_speed_matches_open_loop():
    from roadster.flight import touchdown_metrics
    path = DATA / "log_gust_model_error_ol.npz"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        simulate("gust_model_error", closed_loop=False).save(path)
    vz = touchdown_metrics(Log.load(path))["touchdown_vz"]
    script = (ROOT / "video" / "script.md").read_text()
    assert "eight meters per second" in script
    assert round(vz) == 8, vz


def test_ramp_narration_matches_log():
    from scipy.spatial.transform import Rotation as Rot
    path = DATA / "log_ramp_nominal_cl.npz"
    if not path.exists():
        pytest.skip("ramp log not generated; run python scripts/run_ramp.py")
    log = Log.load(path)
    q = log.qpos[:, 3:7]
    R = Rot.from_quat(np.c_[q[:, 1], q[:, 2], q[:, 3], q[:, 0]]).as_matrix()
    lip = int(np.flatnonzero(log.phase == "flight")[0])
    td = int(np.flatnonzero(log.phase == "rollout")[0])
    script = (ROOT / "video" / "script.md").read_text()
    assert "leaves the ramp upside down" in script and R[lip][2, 2] < 0          # roof facing down at the lip
    assert "keeping its heading" in script
    nose_x = R[lip:td, 0, 0]
    assert np.all(nose_x < 0)                                                    # nose never swings back toward the ramp
    assert R[td][2, 2] > 0.98                                                    # upright at touchdown
    assert set(log.mpc_solver) == {"tinympc"}


def test_fan_suction_is_attributed_to_tesla():
    """The suction system is Tesla's reported claim, not something I measured: the narration must say so."""
    script = (ROOT / "video" / "script.md").read_text()
    ramp = script.split("## ramp")[1].split("##")[0]
    assert "fans" in ramp and "Tesla says fans" in ramp


def test_semi_narration_matches_the_trailer_and_log():
    path = DATA / "log_semi_cl.npz"
    if not path.exists():
        pytest.skip("semi log not generated; run python scripts/run_kickflip.py")
    from roadster.flight import touchdown_metrics
    script = (ROOT / "video" / "script.md").read_text()
    semi = script.split("## semi")[1].split("##")[0]
    assert "sixty centimeters wider than the car" in semi
    assert round(P.DECK_W - 2 * 0.98, 1) == 0.6
    m = touchdown_metrics(Log.load(path))
    assert m["on_deck"] and m["touchdown_vz"] < 1.5
    vz, y_cm = facts.semi_touchdown(Log.load(path))
    assert vz == pytest.approx(m["touchdown_vz"]) and y_cm == pytest.approx(100 * abs(m["touchdown_y"]))
