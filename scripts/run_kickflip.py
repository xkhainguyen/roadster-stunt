"""Simulate every kickflip scenario, open and closed loop, and save logs to data/."""
from roadster.flight import DATA, SCENARIOS, simulate, touchdown_metrics


def main():
    DATA.mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        for closed in (True, False):
            log = simulate(scenario, closed_loop=closed)
            tag = "cl" if closed else "ol"
            log.save(DATA / f"log_{scenario}_{tag}.npz")
            m = touchdown_metrics(log)
            print(f"{scenario:12s} {tag}  vz={m['touchdown_vz']:.2f} m/s  roll={m['roll_err_deg']:.1f} deg  "
                  f"pitch={m['pitch_err_deg']:.1f} deg  upright={m.get('final_upright', 0):+.2f}  "
                  f"air={m.get('tank_used', 0) / 1e3:.1f} kNs")


def save_breakable_crash():
    """The no-feedback run shown in the video, with a car that breaks apart on a hard impact."""
    log = simulate("gust_model_error", closed_loop=False, breakable=True)
    log.save(DATA / "log_gust_model_error_ol_breakable.npz")


if __name__ == "__main__":
    main()
    save_breakable_crash()
