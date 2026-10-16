"""Closed-loop kickflip in MuJoCo: plan feedforward + LQR attitude, TinyMPC landing, on/off valves."""
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from roadster import params as P
from roadster.control import AttitudeLQR, LandingMPC
from roadster.model import load
from roadster.thrusters import AirTank, Valves, allocate, allocation_matrix
from roadster.trajopt import Plan, plan_kickflip, plan_semi, reduced_wrench

DATA = Path(__file__).resolve().parents[1] / "data"
SETTLE_S = 1.0
ROLLOUT_S = 1.5
MPC_EVERY = 2  # control ticks per MPC solve (50 Hz)
LANDING_SWITCH_RAD = 0.6  # hand over to the MPC this close to finishing the roll (rad); earlier = more time to brake

SCENARIOS = {
    "nominal": dict(vx0=P.VX0, gust=None),
    # body torque [N m], world force [N], start as a fraction of plan time, duration [s]
    "gust": dict(vx0=P.VX0, gust=(np.array([6e3, 3e3, 0.0]), np.array([0.0, 4e3, 0.0]), 0.45, 0.15)),
    "fast_launch": dict(vx0=P.VX0 + 1.5, gust=None),
}
# My own stunt: land on the flatbed of a Semi driving ahead at V_SEMI.
SCENARIOS["semi"] = dict(vx0=P.VX0, gust=None, scene="semi")

# Model uncertainty: the simulated car differs from the model the plan and controllers use.
MODEL_ERROR = dict(mass=1.04, inertia=1.25, com_y=0.06, thrust=0.97)  # 4% heavier, 25% more roll inertia, CoM 6 cm off, jets 3% weaker
SCENARIOS["model_error"] = dict(vx0=P.VX0, gust=None, model_error=MODEL_ERROR)
SCENARIOS["gust_model_error"] = dict(vx0=P.VX0, gust=SCENARIOS["gust"]["gust"], model_error=MODEL_ERROR)
SCENARIOS["semi_gust_model_error"] = dict(vx0=P.VX0, gust=SCENARIOS["gust"]["gust"], model_error=MODEL_ERROR,
                                          scene="semi")


def drive_semi(m, d, speed: float):
    """The truck holds its speed regardless of the car landing on it (an infinitely strong drivetrain)."""
    d.qvel[m.jnt_dofadr[m.joint("semi_x").id]] = speed


BRAKE_DAMPING = 3_000.0  # [N m s] wheel spin damping once the car is down on the trailer
BREAK_SPEED = 5.0  # [m/s] ground impact speed above which the breakable car's welds release


def apply_model_error(m, err: dict):
    """Change the simulated car only: chassis mass and inertia scale, lateral CoM shift, thruster strength."""
    car = m.body("car").id
    m.body_mass[car] *= err["mass"]
    for i in range(m.nbody):  # breakable panels carry part of the car's mass; scale them the same way
        if m.body(i).name.startswith("piece_"):
            m.body_mass[i] *= err["mass"]
    m.body_inertia[car] *= err["inertia"]
    m.body_ipos[car][1] += err["com_y"]
    m.actuator_gear[:, :3] *= err["thrust"]


@dataclass
class Log:
    t: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    thrust: np.ndarray
    phase: np.ndarray
    ground_contact: np.ndarray
    obstacle_contact: np.ndarray
    tank_remaining: np.ndarray
    gust_on: np.ndarray
    deck_contact: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    mpc_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mpc_paths: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 4)))
    mpc_solver: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=str))
    mpc_solve_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    launch_t: float = 0.0
    scene: str = "kickflip"
    broken_at: float = float("nan")  # sim time the breakable car's welds released (nan: never)

    def save(self, path):
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self.__dict__.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        kw = {k: z[k] for k in z.files}
        kw["launch_t"] = float(kw["launch_t"])
        if "broken_at" in kw:
            kw["broken_at"] = float(kw["broken_at"])
        if "scene" in kw:
            kw["scene"] = str(kw["scene"])
        return cls(**kw)


@contextmanager
def _quiet_stdout():
    """TinyMPC prints from C++ on every solve; silence fd 1 around it."""
    fd = os.dup(1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(fd, 1)
            os.close(fd)


HEIGHT_KP, HEIGHT_KD = 16.0, 8.0  # vertical tracking, natural frequency 4 rad/s, critically damped


def height_feedback(R: np.ndarray, z: float, vz: float, x_ref: np.ndarray) -> float:
    """Extra body-z force that tracks the plan's height and vertical speed. Only acts while the car is
    roughly upright (body z within 60 deg of vertical), because upside down the lift jets push it down."""
    up = R[2, 2]
    if up < 0.5:
        return 0.0
    a_cmd = HEIGHT_KP * (x_ref[1] - z) + HEIGHT_KD * (x_ref[3] - vz)
    return P.MASS * a_cmd / up


MAX_LANDING_TILT = np.radians(25.0)


def landing_attitude(F_world_yz: np.ndarray, height: float = np.inf) -> np.ndarray:
    """Thrust vectoring for the landing: lean the car so its lift jets point along the MPC's desired force
    (world [Fy, Fz]), up to MAX_LANDING_TILT, fading to level over the last meter above the touchdown height
    so the car meets the surface flat. A roll of phi sends lift toward world -y * sin(phi)."""
    Fy, Fz = F_world_yz
    cap = MAX_LANDING_TILT * float(np.clip(height, 0.0, 1.0))
    phi = np.clip(np.arctan2(-Fy, max(Fz, 1.0)), -cap, cap)
    return Rot.from_euler("x", phi).as_matrix()


def get_plan(scene: str = "kickflip") -> Plan:
    path = DATA / f"{scene}_plan.npz"
    if path.exists():
        return Plan.load(path)
    plan = plan_semi() if scene == "semi" else plan_kickflip()
    DATA.mkdir(exist_ok=True)
    plan.save(path)
    return plan


def _landing_ref(plan: Plan, tp: float, mpc: LandingMPC, z_touch: float = P.Z_TOUCH) -> np.ndarray:
    """Plan's [y, z, vy, vz] over the horizon; past the plan end, descend at 0.5 m/s but stop the reference
    5 cm under touchdown height so it never sits on the MPC's state floor (that stalls TinyMPC's ADMM)."""
    ref = np.zeros((4, mpc.N))
    for k in range(mpc.N):
        tk = tp + k * mpc.dt
        if tk <= plan.T:
            ref[:, k] = plan.at(tk)[0][:4]
        else:
            ref[:, k] = [0.0, max(z_touch - 0.5 * (tk - plan.T), z_touch - 0.05), 0.0, -0.5]
    return ref


def simulate(scenario: str, closed_loop: bool = True, seed: int = 0, breakable: bool = False) -> Log:
    cfg = SCENARIOS[scenario]
    base = cfg.get("scene", "kickflip")
    plan = get_plan(base)
    scene = f"{base}:breakable" if breakable else base
    m, d = load(scene)
    on_semi = base == "semi"
    z_touch = P.Z_TOUCH + (P.DECK_H if on_semi else 0.0)
    car = m.body("car").id
    A = allocation_matrix(m)  # controllers keep the nominal model
    fmax = m.actuator_ctrlrange[:, 1].copy()
    if "model_error" in cfg:
        apply_model_error(m, cfg["model_error"])
    valves, tank = Valves(fmax), AirTank()
    # on the semi the landing needs real sideways authority: side thrusters plus leaning the lift jets (a box
    # approximation of a thrust cone, the constraint Conic-TinyMPC would handle exactly)
    fy_max = 2 * P.F_SIDE_MAX + (np.tan(MAX_LANDING_TILT) * P.MASS * P.G if on_semi else 0.0)
    lqr, mpc = AttitudeLQR(P.INERTIA), LandingMPC(z_touch=z_touch, fy_max=fy_max)

    floor = m.geom("floor").id
    tires = {m.geom(f"tire_{n}").id for n in ("fl", "fr", "rl", "rr")}
    body_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == car and m.geom_contype[i]}
    obstacles = {m.geom("cab").id} if on_semi else {m.geom(f"obstacle_{i}").id for i in range(5)}
    surfaces = {floor, m.geom("deck").id} if on_semi else {floor}
    deck = m.geom("deck").id if on_semi else -1
    spin_dofs = [m.jnt_dofadr[m.joint(f"spin_{n}").id] for n in ("fl", "fr", "rl", "rr")]
    piece_geoms = [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("piece_")]
    piece_bodies = [i for i in range(m.nbody) if m.body(i).name.startswith("piece_")]
    broken_at = float("nan")

    rec = {k: [] for k in ("t", "qpos", "qvel", "thrust", "phase", "ground_contact", "obstacle_contact",
                           "tank_remaining", "gust_on", "deck_contact")}
    mpc_t, mpc_paths, mpc_solver, mpc_solve_ms = [], [], [], []

    phase, launch_t, rollout_t = "settle", None, None
    forces = np.zeros(m.nu)
    mpc_F, tick = np.zeros(2), 0
    steps_per_ctrl = int(round(P.DT_CTRL / P.DT_SIM))
    step = 0
    while True:
        t = step * P.DT_SIM
        # contacts
        ground = obst = on_deck = False
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (g1 in surfaces or g2 in surfaces) and (g1 in tires or g2 in tires or g1 in body_geoms or g2 in body_geoms):
                ground = True
                on_deck = on_deck or deck in (g1, g2)
            if (g1 in obstacles or g2 in obstacles) and (g1 in body_geoms | tires or g2 in body_geoms | tires):
                obst = True

        # a hard ground impact tears the breakable car apart: release the welds, let the parts collide
        if breakable and np.isnan(broken_at) and launch_t is not None and ground and abs(d.qvel[2]) > BREAK_SPEED:
            broken_at = t
            d.eq_active[:] = 0
            m.geom_contype[piece_geoms] = 1
            m.geom_conaffinity[piece_geoms] = 1
            m.body_contype[piece_bodies] = 1  # MuJoCo also filters collisions by these per-body masks
            m.body_conaffinity[piece_bodies] = 1
            m.geom_contype[list(tires)] = 0  # the visible tires came off; the chassis now rests on its own body
            m.geom_conaffinity[list(tires)] = 0
            m.geom_contype[m.geom("cabin_col").id] = m.geom_conaffinity[m.geom("cabin_col").id] = 0  # cabin is gone

        # phase transitions
        if on_semi:
            drive_semi(m, d, P.V_SEMI if phase != "settle" else 0.0)
        if phase == "settle" and t >= SETTLE_S:
            phase = "runup"
            d.qvel[0] = cfg["vx0"]
            d.qvel[spin_dofs] = cfg["vx0"] / P.WHEEL_RADIUS
            for b in piece_bodies:  # welded parts start the run-up moving with the car
                adr = m.jnt_dofadr[m.body_jntadr[b]]
                d.qvel[adr] = cfg["vx0"]
                if m.body(b).name.startswith("piece_wheel_"):
                    d.qvel[adr + 4] = cfg["vx0"] / P.WHEEL_RADIUS  # tire shells spin with the wheels
        elif phase == "runup" and d.xpos[car][0] >= 0.0:
            phase, launch_t = "flight", t
        tp = t - launch_t if launch_t is not None else 0.0
        if phase in ("flight", "landing", "coast") and tp > 0.3 and ground:
            phase, rollout_t = "rollout", t
            if on_semi:
                m.dof_damping[spin_dofs] = BRAKE_DAMPING  # brake: the car arrives faster than the truck
        if phase == "flight":
            phi_ref = plan.at(tp)[0][4]
            if closed_loop and phi_ref >= 2 * np.pi - LANDING_SWITCH_RAD and plan.at(tp)[0][3] < 0:
                phase = "landing"
            elif not closed_loop and tp >= plan.T:
                phase = "coast"
        if phase == "rollout" and t - rollout_t >= ROLLOUT_S:
            break
        if t > SETTLE_S + 10.0:
            break

        # control at 100 Hz
        if step % steps_per_ctrl == 0:
            R = d.xmat[car].reshape(3, 3).copy()
            w = d.qvel[3:6].copy()  # free joint angular velocity, body frame
            wrench = None
            if phase == "flight":
                x_ref, u_ref = plan.at(tp)
                Fz_ff, tx_ff = reduced_wrench(u_ref)
                if closed_loop:
                    R_des = Rot.from_euler("x", x_ref[4]).as_matrix()
                    tau = lqr.torque(R, w, R_des, np.array([x_ref[5], 0, 0]), np.array([tx_ff, 0, 0]))
                    Fz = Fz_ff + height_feedback(R, d.xpos[car][2], d.qvel[2], x_ref)
                    wrench = np.array([0, 0, Fz, *tau])
                else:
                    wrench = np.array([0, 0, Fz_ff, tx_ff, 0, 0])
            elif phase == "landing":
                if tick % MPC_EVERY == 0:
                    x_now = np.array([d.xpos[car][1], d.xpos[car][2], d.qvel[1], d.qvel[2]])
                    ref = _landing_ref(plan, tp, mpc, z_touch)
                    with _quiet_stdout():
                        t_solve = time.perf_counter()
                        mpc_F, path = mpc.force(x_now, ref)
                        mpc_solve_ms.append((time.perf_counter() - t_solve) * 1e3)
                    mpc_t.append(t)
                    mpc_paths.append(np.asarray(path)[:mpc.N])
                    mpc_solver.append(mpc.solver_used)
                F_body = R.T @ np.array([0.0, mpc_F[0], mpc_F[1]])
                tau = lqr.torque(R, w, landing_attitude(mpc_F, d.xpos[car][2] - z_touch), np.zeros(3), np.zeros(3))
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
            tau_b, f_w, frac, dur = cfg["gust"]
            t0 = frac * plan.T
            gust_on = t0 <= tp < t0 + dur
        d.xfrc_applied[car] = 0.0
        if gust_on:
            d.xfrc_applied[car, :3] = f_w
            d.xfrc_applied[car, 3:] = d.xmat[car].reshape(3, 3) @ tau_b

        rec["t"].append(t)
        rec["qpos"].append(d.qpos.copy())
        rec["qvel"].append(d.qvel.copy())
        rec["thrust"].append(applied.copy())
        rec["phase"].append(phase)
        rec["ground_contact"].append(ground)
        rec["obstacle_contact"].append(obst)
        rec["tank_remaining"].append(tank.remaining)
        rec["gust_on"].append(gust_on)
        rec["deck_contact"].append(on_deck)

        mujoco.mj_step(m, d)
        step += 1

    log = Log(**{k: np.array(v) for k, v in rec.items()},
              mpc_t=np.array(mpc_t),
              mpc_paths=np.array(mpc_paths) if mpc_paths else np.zeros((0, mpc.N, 4)),
              mpc_solver=np.array(mpc_solver, dtype=str),
              mpc_solve_ms=np.array(mpc_solve_ms),
              launch_t=float(launch_t if launch_t is not None else 0.0), scene=scene, broken_at=broken_at)
    return log


def touchdown_metrics(log: Log) -> dict:
    airborne = np.isin(log.phase, ["flight", "landing", "coast"])
    idx = np.flatnonzero(log.phase == "rollout")
    if len(idx) == 0 or not airborne.any():
        return dict(touchdown_vz=np.inf, roll_err_deg=180.0, pitch_err_deg=180.0, hit_obstacle=True, landed=False,
                    on_deck=False, touchdown_y=np.nan)
    i = idx[0]
    q = log.qpos[i, 3:7]  # w, x, y, z
    roll, pitch, _ = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)
    # also check the car did not end up on its roof or side after settling
    qe = log.qpos[-1, 3:7]
    up = Rot.from_quat([qe[1], qe[2], qe[3], qe[0]]).as_matrix()[2, 2]
    return dict(touchdown_vz=float(abs(log.qvel[i, 2])), roll_err_deg=float(abs(roll)),
                pitch_err_deg=float(abs(pitch)), hit_obstacle=bool(log.obstacle_contact.any()),
                landed=True, final_upright=float(up), touchdown_t=float(log.t[i] - log.launch_t),
                touchdown_x=float(log.qpos[i, 0]), tank_used=float(P.IMPULSE_BUDGET - log.tank_remaining[-1]),
                on_deck=bool(len(log.deck_contact) and log.deck_contact[i]), touchdown_y=float(log.qpos[i, 1]))
