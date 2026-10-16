"""Simulate every ramp scenario, open and closed loop, and save logs to data/."""
import mujoco
import numpy as np

from roadster import params as P
from roadster.flight import DATA, touchdown_metrics
from roadster.ramp import SCENARIOS, get_plan, simulate


def main():
    DATA.mkdir(exist_ok=True)
    plan = get_plan()
    imp = float((plan.U[:, :12].sum(axis=1) * np.diff(plan.t)).sum())
    print(f"plan: lip z={plan.X[0, 2]:.2f} m pitch={np.degrees(plan.beta0):.0f} deg, upright after "
          f"{plan.t_flip:.2f} s, touchdown {plan.T:.2f} s, apex {plan.X[:, 2].max():.2f} m, "
          f"air {imp / 1e3:.1f} kNs ({imp / P.IMPULSE_BUDGET:.0%})")
    for scenario in SCENARIOS:
        for closed in (True, False):
            log = simulate(scenario, closed_loop=closed)
            tag = "cl" if closed else "ol"
            log.save(DATA / f"log_ramp_{scenario}_{tag}.npz")
            m = touchdown_metrics(log)
            i = np.flatnonzero(log.phase == "rollout")[0]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, log.qpos[i, 3:7])
            nose = R.reshape(3, 3)[:, 0]
            heading = np.degrees(np.arctan2(nose[1], nose[0]))
            solve = f"  tinympc {np.median(log.mpc_solve_ms):.2f} ms median" if len(log.mpc_solve_ms) else ""
            print(f"{scenario:10s} {tag}  vz={m['touchdown_vz']:.2f} m/s  roll={m['roll_err_deg']:.1f} deg  "
                  f"pitch={m['pitch_err_deg']:.1f} deg  heading={heading:+.0f} deg  upright={m.get('final_upright', 0):+.2f}  "
                  f"air={m.get('tank_used', 0) / 1e3:.1f} kNs{solve}")


if __name__ == "__main__":
    main()
