"""Render a flight log to an MP4 with MuJoCo's offscreen renderer."""
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from roadster.flight import Log
from roadster.model import load

PLUME_ALPHA = 0.9
MPC_RGBA = np.array([1.0, 0.85, 0.2, 0.9], dtype=np.float32)
TRAIL_RGBA = np.array([0.55, 0.8, 1.0, 0.8], dtype=np.float32)


def _add_sphere(scene, pos, radius, rgba, emission=0.0):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    g.emission = emission
    scene.ngeom += 1


JET = {"lift": (2.8, 0.10), "roof": (1.7, 0.08), "side": (1.3, 0.07)}  # length, nozzle radius


def _add_jet(scene, nozzle, exhaust_dir, kind, strength, phase):
    """A soft cold-gas jet: a chain of translucent spheres that widen and fade along the exhaust."""
    length, r0 = JET[kind]
    n = 16
    for k in range(n):
        s = (k + 0.5) / n
        flicker = 1.0 + 0.12 * np.sin(37.0 * phase + 5.0 * k)
        pos = nozzle + exhaust_dir * length * s * flicker
        radius = r0 * (1.0 + 3.5 * s)
        alpha = strength * 0.38 * (1.0 - s) ** 1.5
        _add_sphere(scene, pos, radius, [0.85, 0.93, 1.0, alpha], emission=0.6)


def render_clip(log_path: str, out: str, camera: str = "chase", fps: int = 30, size=(1920, 1080),
                show_mpc: bool = False, slowmo: float = 1.0, t_range: tuple | None = None,
                show_trail: bool = False) -> str:
    """t_range is in seconds relative to launch; slowmo < 1 slows playback (0.5 = half speed)."""
    log = Log.load(log_path)
    m, d = load(log.scene)
    car = m.body("car").id
    pieces = [b for b in range(m.nbody) if m.body(b).name.startswith("piece_")]
    fmax = m.actuator_ctrlrange[:, 1]
    plume_ids = [m.geom(f"plume_{m.actuator(j).name}").id for j in range(m.nu)]
    base_rgba = m.geom_rgba[plume_ids].copy()

    tp = log.t - log.launch_t
    t0, t1 = t_range if t_range is not None else (-1.0, tp[-1])
    n_frames = int(round((t1 - t0) / slowmo * fps))
    w, h = size
    r = mujoco.Renderer(m, h, w)
    cam = None
    track_z = None
    if camera == "follow":
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 12.0, 125.0, -6.0
    elif isinstance(camera, tuple) and camera[0] == "track":
        # ("track", z, distance, azimuth, elevation): pans with the car horizontally, height held fixed
        _, track_z, dist, az, el = camera
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = dist, az, el

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "18", "-preset", "medium", out], stdin=subprocess.PIPE)
    frame_dt = slowmo / fps
    trail = []
    for f in range(n_frames):
        t = t0 + f * frame_dt
        i = int(np.clip(np.searchsorted(tp, t), 0, len(tp) - 1))
        d.qpos[:] = log.qpos[i]
        d.qvel[:] = log.qvel[i]
        mujoco.mj_forward(m, d)

        # plume opacity follows each valve's duty cycle over this frame's slice of sim time
        j0 = int(np.clip(np.searchsorted(tp, t - frame_dt / 2), 0, len(tp) - 1))
        j1 = int(np.clip(np.searchsorted(tp, t + frame_dt / 2), j0 + 1, len(tp)))
        duty = np.clip(log.thrust[j0:j1].mean(axis=0) / fmax, 0, 1)
        m.geom_rgba[plume_ids, 3] = 0.0  # model capsules stay hidden; jets are drawn below

        broken = np.isfinite(getattr(log, "broken_at", np.nan)) and log.t[i] >= log.broken_at
        if cam is not None and track_z is not None:
            cam.lookat[:] = [d.xpos[car][0], d.xpos[car][1], track_z]
        elif cam is not None and broken:
            # after a breakup, frame the whole debris field instead of the bare chassis
            pts = np.array([d.xpos[car]] + [d.xpos[b] for b in pieces])
            cam.lookat[:] = pts.mean(axis=0) + np.array([0, 0, 0.5])
            spread = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()
            cam.distance = max(12.0, 2.2 * spread + 6.0)
        elif cam is not None:
            cam.lookat[:] = d.xpos[car] + np.array([0, 0, 0.5])
        r.update_scene(d, cam if cam is not None else camera)

        R = d.xmat[car].reshape(3, 3)
        for j in range(m.nu):
            if duty[j] < 0.02:
                continue
            site = m.actuator_trnid[j, 0]
            exhaust = -R @ m.actuator_gear[j, :3]
            kind = m.actuator(j).name.split("_")[0]
            _add_jet(r.scene, d.site_xpos[site], exhaust, kind, PLUME_ALPHA * duty[j], t + 0.37 * j)

        if show_trail and t >= 0:
            trail.append(d.xpos[car].copy())
            for p in trail[::2]:
                _add_sphere(r.scene, p, 0.06, TRAIL_RGBA)
        if show_mpc and len(log.mpc_t):
            k = np.searchsorted(log.mpc_t, log.t[i], side="right") - 1
            if k >= 0 and log.t[i] - log.mpc_t[k] < 0.05 and log.phase[i] == "landing":
                x_now, vx = d.xpos[car][0], log.qvel[i, 0]
                for n, s in enumerate(log.mpc_paths[k][::2]):
                    _add_sphere(r.scene, [x_now + vx * 0.04 * n, s[0], s[1]], 0.09, MPC_RGBA)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    ff.wait()
    r.close()
    if ff.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")
    return out
