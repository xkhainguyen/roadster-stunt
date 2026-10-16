"""Thruster allocation, on/off valve modulation, and the air tank."""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from roadster import params as P


def allocation_matrix(model: mujoco.MjModel) -> np.ndarray:
    """6 x nu map from thruster forces to the body wrench [F; tau] about the car's CoM (body frame)."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    car = model.body("car").id
    R = d.xmat[car].reshape(3, 3)
    com = R.T @ (d.subtree_com[car] - d.xpos[car])
    A = np.zeros((6, model.nu))
    for i in range(model.nu):
        site = model.actuator_trnid[i, 0]
        p = model.site_pos[site] - com
        f = model.actuator_gear[i, :3]
        A[:3, i] = f
        A[3:, i] = np.cross(p, f)
    return A


def allocate(wrench: np.ndarray, A: np.ndarray, f_max: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Bounded least squares: closest achievable wrench with 0 <= f <= f_max, small force penalty."""
    n = A.shape[1]
    A_aug = np.vstack([A, np.sqrt(reg) * np.eye(n)])
    b_aug = np.concatenate([wrench, np.zeros(n)])
    res = lsq_linear(A_aug, b_aug, bounds=(np.zeros(n), f_max), method="bvls")
    return np.clip(res.x, 0.0, f_max)


class Valves:
    """Per-thruster first-order sigma-delta: each output is 0 or f_max, averaging to the command."""

    def __init__(self, f_max: np.ndarray):
        self.f_max = np.asarray(f_max, dtype=float)
        self.acc = np.zeros_like(self.f_max)

    def step(self, f_cmd: np.ndarray) -> np.ndarray:
        cmd = np.clip(f_cmd, 0.0, self.f_max)
        self.acc += cmd
        on = self.acc >= 0.5 * self.f_max
        out = np.where(on, self.f_max, 0.0)
        self.acc = np.clip(self.acc - out, -self.f_max, self.f_max)
        return out


class AirTank:
    """Tracks the remaining total impulse; zeroes thrust once empty."""

    def __init__(self, budget: float = P.IMPULSE_BUDGET):
        self.remaining = float(budget)
        self.t = 0.0
        self.empty_at = None

    def draw(self, forces: np.ndarray, dt: float) -> np.ndarray:
        self.t += dt
        need = float(np.sum(forces)) * dt
        if need <= self.remaining:
            self.remaining -= need
            return np.asarray(forces, dtype=float)
        scale = self.remaining / need if need > 0 else 0.0
        self.remaining = 0.0
        if self.empty_at is None:
            self.empty_at = self.t
        return np.asarray(forces, dtype=float) * scale
