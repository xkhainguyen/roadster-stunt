"""My own stunt: jump onto the flatbed trailer of a moving Semi."""
import mujoco
import numpy as np
import pytest

from roadster import params as P
from roadster.model import load


def test_semi_scene_has_a_trailer_deck_at_the_right_height():
    m, d = load("semi")
    deck = m.geom("deck")
    top = d.geom("deck").xpos[2] + deck.size[2]
    assert top == pytest.approx(P.DECK_H, abs=1e-6)
    assert 2 * deck.size[1] == pytest.approx(P.DECK_W)
    assert 2 * deck.size[0] == pytest.approx(P.DECK_LEN)


def test_semi_keeps_its_speed_when_driven():
    from roadster.flight import drive_semi
    m, d = load("semi")
    x0 = d.body("semi").xpos[0]
    for _ in range(1000):
        drive_semi(m, d, P.V_SEMI)
        mujoco.mj_step(m, d)
    assert d.qvel[m.jnt_dofadr[m.joint("semi_x").id]] == pytest.approx(P.V_SEMI, rel=1e-3)
    assert d.body("semi").xpos[0] - x0 == pytest.approx(P.V_SEMI * 1.0, rel=0.02)


def test_car_starts_in_the_next_lane_level_with_the_trailer_middle():
    m, d = load("semi")
    car, semi = d.body("car").xpos, d.body("semi").xpos
    assert car[1] == pytest.approx(P.SEMI_LANE_Y)
    assert car[0] == pytest.approx(semi[0], abs=0.01)  # same speed from here on, so it stays level


@pytest.fixture(scope="module")
def semi_plan():
    from roadster.trajopt import plan_semi
    return plan_semi()


def test_semi_plan_lands_on_the_deck(semi_plan):
    y, z, vy, vz, phi, om = semi_plan.X[-1]
    assert z == pytest.approx(P.DECK_H + P.Z_TOUCH, abs=0.01)
    assert phi == pytest.approx(2 * np.pi, abs=0.01) and abs(om) < 1e-3 and abs(y) < 1e-3
    assert -0.6 - 1e-6 <= vz <= 0


def test_semi_plan_starts_in_the_next_lane_and_stays_above_the_deck(semi_plan):
    assert semi_plan.X[0, 0] == pytest.approx(P.SEMI_LANE_Y)
    y, z = semi_plan.X[:, 0], semi_plan.X[:, 1]
    over = np.abs(y) - 0.98 < P.DECK_W / 2  # some part of the car is above the deck
    assert over.any()
    assert np.all(z[over] >= P.DECK_H + P.Z_TOUCH - 0.02)


def test_kickflip_plan_unchanged_by_refactor():
    from roadster.trajopt import plan_kickflip
    p = plan_kickflip()
    assert p.T == pytest.approx(2.637, abs=0.005)
    assert p.X[:, 1].max() == pytest.approx(6.33, abs=0.02)


CAR_HALF_W = 0.98


@pytest.fixture(scope="module")
def runs():
    from roadster.flight import simulate
    return {("semi", True): simulate("semi"),
            ("semi_gust_model_error", True): simulate("semi_gust_model_error"),
            ("semi_gust_model_error", False): simulate("semi_gust_model_error", closed_loop=False)}


def _on_deck_ok(m):
    return (m["on_deck"] and m["touchdown_vz"] <= 1.5 and max(m["roll_err_deg"], m["pitch_err_deg"]) <= 10
            and abs(m["touchdown_y"]) <= P.DECK_W / 2 - CAR_HALF_W and not m["hit_obstacle"])


@pytest.mark.parametrize("scenario", ["semi", "semi_gust_model_error"])
def test_feedback_lands_on_the_moving_deck(runs, scenario):
    from roadster.flight import touchdown_metrics
    m = touchdown_metrics(runs[(scenario, True)])
    assert _on_deck_ok(m), m


def test_car_stays_on_the_trailer(runs):
    log = runs[("semi", True)]
    m, d = load("semi")
    d.qpos[:] = log.qpos[-1]
    mujoco.mj_kinematics(m, d)
    rear = d.body("semi").xpos[0] - P.DECK_LEN / 2
    x_rel = d.body("car").xpos[0] - rear
    assert 0.5 < x_rel < P.DECK_LEN - 0.5
    assert d.body("car").xpos[2] == pytest.approx(P.DECK_H + P.RIDE_Z, abs=0.15)


def test_no_feedback_misses_or_crashes(runs):
    from roadster.flight import touchdown_metrics
    m = touchdown_metrics(runs[("semi_gust_model_error", False)])
    assert not _on_deck_ok(m), m


def test_semi_landing_is_solved_by_tinympc(runs):
    for key in [("semi", True), ("semi_gust_model_error", True)]:
        assert len(runs[key].mpc_solver) > 10 and set(runs[key].mpc_solver) == {"tinympc"}


def test_car_owns_the_first_state_slots():
    """Code reads the car's pose from qpos[0:7]; scenery with joints must come after it."""
    m, _ = load("semi")
    assert m.jnt_qposadr[m.joint("car").id] == 0


def test_semi_plan_lines_up_before_coming_down(semi_plan):
    """Rocket-style: over the centerline and nearly done moving sideways for the last part of the descent."""
    last = semi_plan.t >= 0.8 * semi_plan.T
    assert np.all(np.abs(semi_plan.X[last, 0]) <= 0.1 + 1e-6)
    assert np.all(np.abs(semi_plan.X[last, 2]) <= 0.3 + 1e-6)
