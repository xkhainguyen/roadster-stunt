"""Closed-loop ramp stunt in MuJoCo: drive up a curved ramp pressed on by fan suction, leave the overhanging
lip belly-up, roll upright about the long axis with the nose kept pointing away from the ramp (plan
feedforward + LQR on SO(3)), then land nose-first on TinyMPC."""
import time
from pathlib import Path

import mujoco
import numpy as np

from roadster import params as P
from roadster.control import AttitudeLQR, LandingMPC
from roadster.flight import DATA, MPC_EVERY, ROLLOUT_S, SETTLE_S, Log, _quiet_stdout
from roadster.model import load
from roadster.thrusters import AirTank, Valves, allocate, allocation_matrix
from roadster.ramp_plan import R_UPRIGHT, RampPlan, attitude, body_rate, plan_ramp

DRIVE_GAIN = 600.0   # wheel motor torque per rad/s of wheel speed error [N m s]
LAUNCH_DEBOUNCE = 0.02  # no tire on the ramp for this long, high up, counts as leaving the lip [s]

SCENARIOS = {
    "nominal": dict(v_ramp=P.V_RAMP, gust=None),
    # body torque [N m], world force [N], start [s after the lip], duration [s]
    "gust": dict(v_ramp=P.V_RAMP, gust=(np.array([3e3, 12e3, 0.0]), np.array([0.0, 4e3, 0.0]), 0.25, 0.15)),
    # motors hold 0.5 m/s too slow: the car leaves the lip off the plan
    "slow_ramp": dict(v_ramp=P.V_RAMP - 0.5, gust=None),
}


def pitch_of(R: np.ndarray) -> float:
    """Planar pitch angle about world y: body x = (cos b, 0, -sin b)."""
    return float(np.arctan2(-R[2, 0], R[0, 0]))


class _Sim:
    """Model, ids and the per-step contact checks shared by the approach run and the full stunt."""

    def __init__(self):
        self.m, self.d = load("ramp")
        m = self.m
        self.car = m.body("car").id
        self.floor = m.geom("floor").id
        self.tires = {m.geom(f"tire_{n}").id for n in ("fl", "fr", "rl", "rr")}
        self.ramp = {i for i in range(m.ngeom) if m.geom(i).name.startswith("ramp_")}
        self.body_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == self.car and m.geom_contype[i]}
        self.spin_dofs = [m.jnt_dofadr[m.joint(f"spin_{n}").id] for n in ("fl", "fr", "rl", "rr")]

    def contacts(self):
        """(tire on ramp, anything of the car on the floor, chassis or cabin on the ramp)."""
        on_ramp = ground = body_ramp = False
        car_geoms = self.tires | self.body_geoms
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.ramp and g2 in self.tires) or (g2 in self.ramp and g1 in self.tires):
                on_ramp = True
            if (g1 in self.ramp and g2 in self.body_geoms) or (g2 in self.ramp and g1 in self.body_geoms):
                body_ramp = True
            if self.floor in (g1, g2) and (g1 in car_geoms or g2 in car_geoms):
                ground = True
        return on_ramp, ground, body_ramp

    def drive(self, v_des: float):
        """Wheel motors: per-wheel speed loop with a torque limit (traction limits the rest)."""
        w = self.d.qvel[self.spin_dofs]
        tau = np.clip(DRIVE_GAIN * (v_des / P.WHEEL_RADIUS - w), -P.DRIVE_TAU_MAX, P.DRIVE_TAU_MAX)
        self.d.qfrc_applied[self.spin_dofs] = tau

    def planar_state(self) -> np.ndarray:
        """[x, z, vx, vz, pitch, pitch rate] of the car in the x-z plane."""
        d, car = self.d, self.car
        R = d.xmat[car].reshape(3, 3)
        return np.array([d.xpos[car][0], d.xpos[car][2], d.qvel[0], d.qvel[2], pitch_of(R), d.qvel[4]])


def lip_state(v_ramp: float = P.V_RAMP) -> np.ndarray:
    """Drive the approach and the ramp with no thrusters; planar state at the moment the car leaves the lip."""
    s = _Sim()
    m, d = s.m, s.d
    lip_z = 0.6 * P.RAMP_R
    steps_settle = int(SETTLE_S / P.DT_SIM)
    last_on, state_at_last_on = None, None
    for step in range(int((SETTLE_S + 10.0) / P.DT_SIM)):
        t = step * P.DT_SIM
        on_ramp, _, _ = s.contacts()
        if step == steps_settle:
            d.qvel[0] = v_ramp
            d.qvel[s.spin_dofs] = v_ramp / P.WHEEL_RADIUS
        d.xfrc_applied[s.car] = 0.0
        d.qfrc_applied[:] = 0.0
        if step >= steps_settle:
            s.drive(v_ramp)
            if on_ramp:
                d.xfrc_applied[s.car, :3] = -P.F_SUCTION * d.xmat[s.car].reshape(3, 3)[:, 2]
                last_on, state_at_last_on = t, s.planar_state()
            elif last_on is not None and d.xpos[s.car][2] > lip_z and t - last_on >= LAUNCH_DEBOUNCE:
                return state_at_last_on
        mujoco.mj_step(m, d)
    raise RuntimeError("car never left the lip")


def get_plan() -> RampPlan:
    path = DATA / "ramp_plan.npz"
    if path.exists():
        return RampPlan.load(path)
    plan = plan_ramp(lip_state())
    DATA.mkdir(exist_ok=True)
    plan.save(path)
    return plan


def _landing_ref(plan: RampPlan, tp: float, mpc: LandingMPC) -> np.ndarray:
    """Plan's [y, z, vy, vz] over the horizon (x rolls free); past
    the plan end, descend at 0.5 m/s but stop 5 cm under touchdown height so the reference never sits on the
    MPC's state floor (that stalls TinyMPC's ADMM)."""
    ref = np.zeros((4, mpc.N))
    for k in range(mpc.N):
        tk = tp + k * mpc.dt
        if tk <= plan.T:
            ref[:, k] = plan.at(tk)[0][[1, 2, 4, 5]]
        else:
            ref[:, k] = [0.0, max(P.Z_TOUCH - 0.5 * (tk - plan.T), P.Z_TOUCH - 0.05), 0.0, -0.5]
    return ref


def simulate(scenario: str, closed_loop: bool = True, seed: int = 0) -> Log:
    """Phases: settle, ramp (drive + suction), flight (pitch flip), landing (MPC), coast (open loop after the
    plan ends), rollout (valves off once the car touches down). obstacle_contact records any contact with the
    ramp after leaving the lip, or of the chassis with the ramp at any time."""
    cfg = SCENARIOS[scenario]
    plan = get_plan()
    s = _Sim()
    m, d, car = s.m, s.d, s.car
    A = allocation_matrix(m)
    fmax = m.actuator_ctrlrange[:, 1].copy()
    valves, tank = Valves(fmax), AirTank()
    lqr, mpc = AttitudeLQR(P.INERTIA), LandingMPC()
    J = np.diag(P.INERTIA)
    lip_z = 0.6 * P.RAMP_R

    rec = {k: [] for k in ("t", "qpos", "qvel", "thrust", "phase", "ground_contact", "obstacle_contact",
                           "tank_remaining", "gust_on")}
    mpc_t, mpc_paths, mpc_solver, mpc_solve_ms = [], [], [], []

    phase, launch_t, rollout_t, last_on = "settle", None, None, None
    forces = np.zeros(m.nu)
    mpc_F, tick = np.zeros(2), 0
    steps_per_ctrl = int(round(P.DT_CTRL / P.DT_SIM))
    step = 0
    while True:
        t = step * P.DT_SIM
        on_ramp, ground, body_ramp = s.contacts()

        # phase transitions
        if phase == "settle" and t >= SETTLE_S:
            phase = "ramp"
            d.qvel[0] = cfg["v_ramp"]
            d.qvel[s.spin_dofs] = cfg["v_ramp"] / P.WHEEL_RADIUS
        elif phase == "ramp":
            if on_ramp:
                last_on = t
            elif last_on is not None and d.xpos[car][2] > lip_z and t - last_on >= LAUNCH_DEBOUNCE:
                phase, launch_t = "flight", last_on
        tp = t - launch_t if launch_t is not None else 0.0
        if phase in ("flight", "landing", "coast") and tp > 0.3 and (ground or on_ramp):
            phase, rollout_t = "rollout", t
        if phase == "flight":
            if closed_loop and tp >= plan.t_flip:
                phase = "landing"
            elif not closed_loop and tp >= plan.T:
                phase = "coast"
        if phase == "rollout" and t - rollout_t >= ROLLOUT_S:
            break
        if t > SETTLE_S + 15.0:
            break

        # drive and fan suction while on the ramp; thrusters at 100 Hz once airborne
        d.qfrc_applied[:] = 0.0
        suction = np.zeros(3)
        if phase == "ramp":
            s.drive(cfg["v_ramp"])
            if on_ramp:
                suction = -P.F_SUCTION * d.xmat[car].reshape(3, 3)[:, 2]

        if step % steps_per_ctrl == 0:
            R = d.xmat[car].reshape(3, 3).copy()
            w = d.qvel[3:6].copy()  # free joint angular velocity, body frame
            wrench = None
            if phase == "flight":
                x_ref, u_ref = plan.at(tp)
                wrench_ff = A @ u_ref[:12]
                if closed_loop:
                    p, dp = x_ref[6], x_ref[7]
                    w_des, dw_des = body_rate(p, dp, u_ref[12], plan.beta0)
                    tau = lqr.torque(R, w, attitude(p, plan.beta0), np.array(w_des), J @ np.array(dw_des))
                    wrench = np.array([*wrench_ff[:3], *tau])
                else:
                    wrench = wrench_ff
            elif phase == "landing":
                if tick % MPC_EVERY == 0:
                    x_now = np.array([d.xpos[car][1], d.xpos[car][2], d.qvel[1], d.qvel[2]])
                    ref = _landing_ref(plan, tp, mpc)
                    with _quiet_stdout():
                        t_solve = time.perf_counter()
                        mpc_F, path = mpc.force(x_now, ref)
                        mpc_solve_ms.append((time.perf_counter() - t_solve) * 1e3)
                    mpc_t.append(t)
                    mpc_paths.append(np.asarray(path)[:mpc.N])
                    mpc_solver.append(mpc.solver_used)
                F_body = R.T @ np.array([0.0, mpc_F[0], mpc_F[1]])
                tau = lqr.torque(R, w, R_UPRIGHT, np.zeros(3), np.zeros(3))
                wrench = np.array([*F_body, *tau])
            if wrench is None:
                forces = np.zeros(m.nu)
            else:
                forces = valves.step(allocate(wrench, A, fmax))
            tick += 1

        applied = tank.draw(forces, P.DT_SIM)
        d.ctrl[:] = applied

        # disturbance
        gust_on = False
        if cfg["gust"] is not None and launch_t is not None:
            tau_b, f_w, t0, dur = cfg["gust"]
            gust_on = t0 <= tp < t0 + dur
        d.xfrc_applied[car] = 0.0
        d.xfrc_applied[car, :3] = suction
        if gust_on:
            d.xfrc_applied[car, :3] += f_w
            d.xfrc_applied[car, 3:] = d.xmat[car].reshape(3, 3) @ tau_b

        rec["t"].append(t)
        rec["qpos"].append(d.qpos.copy())
        rec["qvel"].append(d.qvel.copy())
        rec["thrust"].append(applied.copy())
        rec["phase"].append(phase)
        rec["ground_contact"].append(ground)
        rec["obstacle_contact"].append(body_ramp or (launch_t is not None and on_ramp))
        rec["tank_remaining"].append(tank.remaining)
        rec["gust_on"].append(gust_on)

        mujoco.mj_step(m, d)
        step += 1

    return Log(**{k: np.array(v) for k, v in rec.items()},
               mpc_t=np.array(mpc_t),
               mpc_paths=np.array(mpc_paths) if mpc_paths else np.zeros((0, mpc.N, 4)),
               mpc_solver=np.array(mpc_solver, dtype=str),
               mpc_solve_ms=np.array(mpc_solve_ms),
               launch_t=float(launch_t if launch_t is not None else 0.0),
               scene="ramp")
