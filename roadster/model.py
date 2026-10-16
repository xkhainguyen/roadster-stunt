"""MJCF for a generic wedge sports car with cold-gas thrusters, built in code."""
import mujoco
import numpy as np

from roadster import params as P

CORNERS = {"fl": (1, 1), "fr": (1, -1), "rl": (-1, 1), "rr": (-1, -1)}
WHEEL_X, WHEEL_Y, WHEEL_Z = 1.4, 0.9, -0.29

BODY_RGBA = "0.90 0.91 0.93 1"
GLASS_RGBA = "0.05 0.06 0.08 1"
PLUME_RGBA = "0.80 0.90 1.00 0"

# Body panels that can break off in a hard crash: (piece, mass [kg], geoms as (half size, pos, euler, rgba)).
PANELS = [
    ("hood", 40.0, [((0.95, 0.96, 0.06), (1.25, 0, 0.10), (0, 9, 0), BODY_RGBA)]),
    ("cabin", 60.0, [((0.75, 0.80, 0.12), (-0.25, 0, 0.28), (0, 0, 0), GLASS_RGBA),
                     ((0.55, 0.80, 0.03), (0.62, 0, 0.30), (0, 24, 0), GLASS_RGBA),
                     ((0.70, 0.82, 0.03), (-0.25, 0, 0.42), (0, 0, 0), BODY_RGBA)]),
    ("rear", 40.0, [((0.70, 0.94, 0.10), (-1.45, 0, 0.18), (0, -6, 0), BODY_RGBA)]),
]
WHEEL_SHELL_MASS = 5.0  # visible tire and hub, welded to each real wheel
BREAK_PIECES = [n for n, _, _ in PANELS] + [f"wheel_{n}" for n in CORNERS]


def thrusters():
    """(name, site position, unit force direction in body frame, max force)."""
    out = []
    for n, (sx, sy) in CORNERS.items():
        out.append((f"lift_{n}", (sx * P.THRUSTER_X, sy * P.THRUSTER_Y, -0.30), (0, 0, 1), P.F_LIFT_MAX))
    for n, (sx, sy) in CORNERS.items():
        out.append((f"roof_{n}", (sx * P.THRUSTER_X, sy * P.THRUSTER_Y, 0.30), (0, 0, -1), P.F_ROOF_MAX))
    for end, sx in (("f", 1), ("r", -1)):
        out.append((f"side_{end}_pos", (sx * P.SIDE_X, -P.SIDE_Y, 0.0), (0, 1, 0), P.F_SIDE_MAX))
        out.append((f"side_{end}_neg", (sx * P.SIDE_X, P.SIDE_Y, 0.0), (0, -1, 0), P.F_SIDE_MAX))
    return out


def _f(v):
    return " ".join(f"{x:.4g}" for x in v)


def _box_inertia(m, half):
    a, b, c = (2 * np.asarray(half)) ** 2
    return m / 12 * np.array([b + c, a + c, a + b])


def _panel_geoms(geoms, origin, breakable, prefix):
    out = []
    for k, (half, pos, euler, rgba) in enumerate(geoms):
        rel = np.asarray(pos, float) - np.asarray(origin, float)
        col = 'contype="0" conaffinity="0"'  # breakable pieces get collisions only once they come loose
        name = f'name="{prefix}_g{k}" ' if breakable else ""
        out.append(f'<geom {name}type="box" size="{_f(half)}" pos="{_f(rel)}" euler="{_f(euler)}" rgba="{rgba}" {col}/>')
    return out


def _car_xml(start, k_spring=44_000.0, c_damp=3_750.0, breakable=False):
    m_chassis = P.MASS - 4 * P.WHEEL_MASS
    I = m_chassis / P.MASS * P.INERTIA
    pieces_xml = []
    if breakable:
        # move the panels' and tire shells' mass and inertia out of the chassis so the whole car is unchanged
        for name, mass, geoms in PANELS:
            r = np.asarray(geoms[0][1], float)
            I = I - (_box_inertia(mass, geoms[0][0]) + mass * (r @ r - r ** 2))
            m_chassis -= mass
        m_chassis -= 4 * WHEEL_SHELL_MASS
        for name, mass, geoms in PANELS:
            origin = np.asarray(geoms[0][1], float)
            pieces_xml += [f'<body name="piece_{name}" pos="{_f(np.asarray(start) + origin)}">', '<freejoint/>',
                           f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{_f(_box_inertia(mass, geoms[0][0]))}"/>',
                           *_panel_geoms(geoms, origin, True, f"piece_{name}"), '</body>']
        for n, (sx, sy) in CORNERS.items():
            wpos = np.asarray(start) + np.array([sx * WHEEL_X, sy * WHEEL_Y, WHEEL_Z])
            pieces_xml += [f'<body name="piece_wheel_{n}" pos="{_f(wpos)}">', '<freejoint/>',
                           f'<inertial pos="0 0 0" mass="{WHEEL_SHELL_MASS}" diaginertia="0.2 0.3 0.2"/>',
                           f'<geom name="piece_wheel_{n}_g0" type="cylinder" size="{P.WHEEL_RADIUS} 0.14" euler="90 0 0" '
                           'rgba="0.08 0.08 0.09 1" contype="0" conaffinity="0"/>',
                           f'<geom type="cylinder" size="0.22 0.145" euler="90 0 0" rgba="0.55 0.56 0.6 1" '
                           'contype="0" conaffinity="0"/>', '</body>']
    parts = [f'<body name="car" pos="{_f(start)}">',
             '<freejoint name="car"/>',
             f'<inertial pos="0 0 0" mass="{m_chassis}" diaginertia="{_f(I)}"/>',
             # collision: chassis slab and cabin
             # breakable car: the chassis collides as the visible slab, since the panels above it can come off
             (f'<geom name="chassis" type="box" size="2.15 0.98 0.20" pos="0 0 -0.08" rgba="{BODY_RGBA}" group="3"/>'
              if breakable else f'<geom name="chassis" type="box" size="2.15 0.95 0.30" rgba="{BODY_RGBA}" group="3"/>'),
             f'<geom name="cabin_col" type="box" size="0.9 0.75 0.15" pos="-0.2 0 0.42" group="3"/>',
             # visuals: wedge body from primitives
             f'<geom type="box" size="2.15 0.98 0.20" pos="0 0 -0.08" rgba="{BODY_RGBA}" contype="0" conaffinity="0"/>',
             '<geom type="box" size="0.02 0.85 0.02" pos="2.16 0 0.02" rgba="1 1 1 1" contype="0" conaffinity="0"/>',
             '<geom type="box" size="0.02 0.90 0.025" pos="-2.16 0 0.10" rgba="1 0.08 0.05 1" contype="0" conaffinity="0"/>',
             ]
    if not breakable:
        for name, mass, geoms in PANELS:
            parts += _panel_geoms(geoms, (0, 0, 0), False, name)

    for name, pos, d, fmax in thrusters():
        parts.append(f'<site name="{name}" pos="{_f(pos)}" size="0.06" rgba="0.3 0.3 0.35 1"/>')
        length = {"lift": 1.4, "roof": 0.9, "side": 0.7}[name.split("_")[0]]
        radius = {"lift": 0.13, "roof": 0.09, "side": 0.07}[name.split("_")[0]]
        tip = np.array(pos) - length * np.array(d)  # exhaust leaves opposite to the force
        parts.append(f'<geom name="plume_{name}" type="capsule" fromto="{_f(pos)} {_f(tip)}" size="{radius}" '
                     f'rgba="{PLUME_RGBA}" contype="0" conaffinity="0" group="2"/>')
    for n, (sx, sy) in CORNERS.items():
        parts += [f'<body name="susp_{n}" pos="{_f((sx * WHEEL_X, sy * WHEEL_Y, WHEEL_Z))}">',
                  f'<joint name="susp_{n}" type="slide" axis="0 0 1" stiffness="{k_spring}" damping="{c_damp}" '
                  'range="-0.10 0.22" limited="true" armature="1"/>',
                  '<inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>',
                  f'<body name="wheel_{n}">',
                  f'<joint name="spin_{n}" type="hinge" axis="0 1 0" damping="0.5" armature="0.5"/>',
                  f'<inertial pos="0 0 0" mass="{P.WHEEL_MASS - 1}" diaginertia="1.5 2.5 1.5"/>',
                  f'<geom name="tire_{n}" type="sphere" size="{P.WHEEL_RADIUS}" group="3" friction="1.0 0.005 0.0001"/>']
        if not breakable:
            parts += [f'<geom type="cylinder" size="{P.WHEEL_RADIUS} 0.14" euler="90 0 0" rgba="0.08 0.08 0.09 1" contype="0" conaffinity="0"/>',
                      f'<geom type="cylinder" size="0.22 0.145" euler="90 0 0" rgba="0.55 0.56 0.6 1" contype="0" conaffinity="0"/>']
        parts += ['</body></body>']
    parts.append('<camera name="chase" mode="trackcom" pos="-10 -7 3.5" xyaxes="0.57 -0.82 0 0.17 0.12 0.98" fovy="50"/>')
    parts.append('</body>')
    return "\n".join(parts + pieces_xml)


def _obstacles_xml():
    """A row of parked vehicles as plain blocks, long axis across the jump path."""
    x0 = P.ROW_START
    # (center x offset, half x, half y, height, rgba)
    slot = P.ROW_LEN / 5
    row = [(0.5 * slot, 1.0, 2.4, 1.5, "0.35 0.37 0.40 1"),
           (1.5 * slot, 1.0, 2.5, 1.9, "0.55 0.10 0.10 1"),
           (2.5 * slot, 1.3, 3.2, P.ROW_HEIGHT, "0.82 0.83 0.85 1"),
           (3.5 * slot, 1.0, 2.6, 2.0, "0.25 0.26 0.28 1"),
           (4.5 * slot, 1.0, 2.4, 1.6, "0.70 0.71 0.74 1")]
    out = []
    for i, (cx, hx, hy, h, rgba) in enumerate(row):
        out.append(f'<geom name="obstacle_{i}" type="box" size="{hx} {hy} {h / 2}" pos="{x0 + cx} 0 {h / 2}" rgba="{rgba}"/>')
    return "\n".join(out)


def ramp_point(theta: float) -> np.ndarray:
    """Ramp surface point (x, z) at arc angle theta [rad]; 0 is the bottom, pi/2 the vertical wall."""
    return np.array([P.RAMP_X0 + P.RAMP_R * np.sin(theta), P.RAMP_R * (1 - np.cos(theta))])


def _ramp_xml():
    """The curved surface as a chain of thin boxes whose top faces are chords of the arc."""
    n = int(round(P.RAMP_LIP_DEG / P.RAMP_SEG_DEG))
    th = np.radians(np.linspace(0.0, P.RAMP_LIP_DEG, n + 1))
    h = P.RAMP_THICK / 2
    out = []
    for i in range(n):
        a, b = ramp_point(th[i]), ramp_point(th[i + 1])
        tm = 0.5 * (th[i] + th[i + 1])
        inward = np.array([-np.sin(tm), np.cos(tm)])
        c = 0.5 * (a + b) - h * inward
        half = 0.5 * np.linalg.norm(b - a) + 0.02  # overlap hides seams behind the next facet
        out.append(f'<geom name="ramp_{i}" type="box" size="{half:.4f} {P.RAMP_WIDTH / 2} {h}" '
                   f'pos="{c[0]:.4f} 0 {c[1]:.4f}" euler="0 {-np.degrees(tm):.4f} 0" material="ramp"/>')
    # visual backing: a tower behind the wall and side rails along the arc
    top = ramp_point(np.radians(P.RAMP_LIP_DEG))[1]
    xb = P.RAMP_X0 + P.RAMP_R + 0.9
    out.append(f'<geom type="box" size="0.6 {P.RAMP_WIDTH / 2 + 0.3} {top / 2 + 0.3}" pos="{xb} 0 {top / 2 + 0.3}" '
               'rgba="0.16 0.17 0.19 1" contype="0" conaffinity="0"/>')
    return "\n".join(out)


def _semi_xml(x_rear):
    """A Semi pulling a flatbed, on a slide joint along the road; its rear deck edge starts at x_rear."""
    L, W, H = P.DECK_LEN, P.DECK_W, P.DECK_H
    cx = x_rear + L / 2
    out = [f'<body name="semi" pos="{cx} 0 0">',
           '<joint name="semi_x" type="slide" axis="1 0 0" damping="0"/>',
           '<inertial pos="0 0 1" mass="15000" diaginertia="2e4 2e5 2e5"/>',
           # flatbed deck: the landing surface
           f'<geom name="deck" type="box" size="{L / 2} {W / 2} 0.1" pos="0 0 {H - 0.1}" rgba="0.22 0.23 0.26 1" '
           'friction="1.0 0.005 0.0001"/>',
           f'<geom type="box" size="{L / 2} 0.05 0.12" pos="0 {W / 2} {H - 0.12}" rgba="0.75 0.12 0.1 1" contype="0" conaffinity="0"/>',
           f'<geom type="box" size="{L / 2} 0.05 0.12" pos="0 {-W / 2} {H - 0.12}" rgba="0.75 0.12 0.1 1" contype="0" conaffinity="0"/>',
           f'<geom type="box" size="{L / 2 - 0.5} 0.4 {(H - 0.7) / 2}" pos="0 0 {0.5 + (H - 0.7) / 2}" rgba="0.12 0.12 0.13 1" contype="0" conaffinity="0"/>',
           ]
    for wx in (-L / 2 + 1.2, -L / 2 + 2.6, L / 2 - 1.5):
        for wy in (-1.0, 1.0):
            out.append(f'<geom type="cylinder" size="0.5 0.25" pos="{wx} {wy} 0.5" euler="90 0 0" rgba="0.06 0.06 0.07 1" '
                       'contype="0" conaffinity="0"/>')
    # headboard at the front of the flatbed
    out.append(f'<geom name="headboard" type="box" size="0.06 {W / 2} 0.55" pos="{L / 2 - 0.06} 0 {H + 0.55}" '
               'rgba="0.30 0.31 0.34 1"/>')
    # tractor in front of the trailer: a hidden collision box plus a generic cab-over-hood shape (no branding)
    tx = L / 2 + P.CAB_LEN / 2 + 0.3
    out.append(f'<geom name="cab" type="box" size="{P.CAB_LEN / 2} 1.25 {P.CAB_H / 2}" pos="{tx} 0 {P.CAB_H / 2 + 0.4}" '
               'group="3"/>')
    white, dark, glass, grey = "0.88 0.89 0.91 1", "0.10 0.10 0.11 1", "0.05 0.06 0.08 1", "0.45 0.46 0.49 1"
    vis = 'contype="0" conaffinity="0"'
    parts = [  # (half size, center offset from tx, rgba, euler)
        ((2.9, 0.45, 0.12), (0.0, 0, 0.95), dark, (0, 0, 0)),         # frame rails
        ((0.85, 1.25, 1.4), (-1.9, 0, 2.3), white, (0, 0, 0)),        # sleeper
        ((0.75, 1.25, 1.05), (-0.25, 0, 2.05), white, (0, 0, 0)),     # cab
        ((0.03, 1.12, 0.42), (0.52, 0, 2.55), glass, (0, -12, 0)),    # windshield
        ((0.95, 1.1, 0.5), (1.45, 0, 1.5), white, (0, 4, 0)),         # hood, slightly sloped
        ((0.03, 0.85, 0.42), (2.42, 0, 1.4), grey, (0, 0, 0)),        # grille
        ((0.12, 1.25, 0.14), (2.5, 0, 0.75), grey, (0, 0, 0)),        # bumper
        ((0.05, 0.07, 0.22), (0.35, 1.42, 2.55), dark, (0, 0, 0)),    # mirrors
        ((0.05, 0.07, 0.22), (0.35, -1.42, 2.55), dark, (0, 0, 0)),
        ((0.02, 1.2, 0.04), (2.47, 0, 1.95), "1 0.95 0.75 1", (0, 0, 0)),  # light bar
    ]
    for half, off, rgba, euler in parts:
        out.append(f'<geom type="box" size="{_f(half)}" pos="{_f(np.array(off) + [tx, 0, 0])}" euler="{_f(euler)}" '
                   f'rgba="{rgba}" {vis}/>')
    for sy in (-1, 1):  # fuel tanks and exhaust stacks
        out.append(f'<geom type="cylinder" size="0.34 0.6" pos="{tx - 0.55} {sy * 1.05} 0.95" euler="0 90 0" '
                   f'rgba="{grey}" {vis}/>')
        out.append(f'<geom type="cylinder" size="0.08 0.9" pos="{tx - 1.0} {sy * 1.2} 3.1" rgba="{grey}" {vis}/>')
    for wx in (tx - 2.3, tx - 1.1, tx + 1.6):
        for wy in (-1.0, 1.0):
            out.append(f'<geom type="cylinder" size="0.5 0.25" pos="{wx} {wy} 0.5" euler="90 0 0" rgba="0.06 0.06 0.07 1" '
                       f'{vis}/>')
    out.append('</body>')
    return "\n".join(out)


def _road_xml():
    """A two-lane road for the semi stunt: asphalt, solid edge lines and a dashed line between the lanes."""
    y_mid = P.SEMI_LANE_Y / 2
    half = P.SEMI_LANE_Y / 2 + 2.0
    out = [f'<geom type="box" size="200 {half} 0.002" pos="120 {y_mid} 0.001" rgba="0.15 0.15 0.16 1" '
           'contype="0" conaffinity="0"/>']
    for y in (y_mid - half + 0.2, y_mid + half - 0.2):
        out.append(f'<geom type="box" size="200 0.08 0.002" pos="120 {y} 0.003" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>')
    for x in np.arange(-60.0, 320.0, 12.0):
        out.append(f'<geom type="box" size="1.5 0.08 0.002" pos="{x} {y_mid} 0.003" rgba="0.95 0.85 0.3 1" contype="0" conaffinity="0"/>')
    return "\n".join(out)


def semi_start_x() -> float:
    """Where the trailer's rear edge starts so that the trailer's middle is level with the car when it launches at
    x=0. Both start moving when the run-up begins; the car reaches x=0 after RUNUP / VX0 seconds."""
    return -P.DECK_LEN / 2 - P.V_SEMI * P.RUNUP / P.VX0


def build_mjcf(scene: str = "kickflip") -> str:
    """scene is 'kickflip' or 'ramp', optionally with ':breakable' (panels and tires welded on, released on impact)."""
    scene, _, option = scene.partition(":")
    breakable = option == "breakable"
    if scene not in ("kickflip", "ramp", "semi") or option not in ("", "breakable"):
        raise ValueError(f"unknown scene {scene!r}:{option!r}")
    if scene == "semi":
        start = (-P.RUNUP, P.SEMI_LANE_Y, P.RIDE_Z + 0.02)
        car = _car_xml(start, breakable=breakable)
        props = _semi_xml(semi_start_x()) + "\n" + _road_xml()
        mid, side_y, side_z = 10.0, -34, 4.5
    elif scene == "kickflip":
        start = (-P.RUNUP, 0.0, P.RIDE_Z + 0.02)
        car = _car_xml(start, breakable=breakable)
        props = _obstacles_xml()
        mid, side_y, side_z = P.ROW_START + P.ROW_LEN / 2, -34, 4.5
    else:
        start = (P.RAMP_X0 - P.RAMP_RUNUP, 0.0, P.Z_TOUCH - 0.03)
        car = _car_xml(start, P.SUSP_K_RAMP, P.SUSP_C_RAMP, breakable=breakable)
        props = _ramp_xml()
        mid, side_y, side_z = P.RAMP_X0 - 3, -28, 8.5
    acts = "\n".join(
        f'<general name="{n}" site="{n}" gear="{_f(d)} 0 0 0" ctrllimited="true" ctrlrange="0 {fmax}"/>'
        for n, _, d, fmax in thrusters())
    excl = "\n".join(f'<exclude body1="car" body2="wheel_{n}"/><exclude body1="car" body2="susp_{n}"/>'
                     for n in CORNERS)
    equality = ""
    if breakable:
        car_bodies = ["car"] + [f"wheel_{n}" for n in CORNERS] + [f"susp_{n}" for n in CORNERS]
        pieces = [f"piece_{p}" for p in BREAK_PIECES]
        excl += "\n" + "\n".join(f'<exclude body1="{p}" body2="{b}"/>' for p in pieces for b in car_bodies)
        excl += "\n" + "\n".join(f'<exclude body1="{a}" body2="{b}"/>' for i, a in enumerate(pieces) for b in pieces[i + 1:])
        welds = [f'<weld name="weld_{p}" body1="piece_{p}" body2="{p if p.startswith("wheel_") else "car"}" '
                 'solref="0.004 1"/>' for p in BREAK_PIECES]
        equality = "<equality>" + "\n".join(welds) + "</equality>"
    return f"""
<mujoco model="roadster_{scene}">
  <compiler angle="degree" inertiafromgeom="false"/>
  <option timestep="{P.DT_SIM}" integrator="implicitfast" gravity="0 0 -{P.G}"/>
  {'<statistic extent="30" center="0 0 2"/>' if scene == "semi" else ""}
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.25 0.25 0.28" diffuse="0.4 0.4 0.4"/>
    <map znear="0.05" zfar="500"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.02 0.025 0.04" rgb2="0 0 0" width="512" height="512"/>
    <texture name="ground" type="2d" builtin="checker" rgb1="0.11 0.11 0.12" rgb2="0.13 0.13 0.14" width="512" height="512"/>
    <material name="ground" texture="ground" texrepeat="40 40" reflectance="0.15"/>
    <material name="ramp" rgba="0.30 0.31 0.34 1" specular="0.2"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 30" dir="0.2 0.3 -1" diffuse="0.55 0.55 0.6" castshadow="true"/>
    <light pos="{mid} -15 12" dir="0 1 -0.6" diffuse="0.5 0.5 0.55" cutoff="60" castshadow="false"/>
    <geom name="floor" type="plane" size="200 200 0.1" material="ground"/>
    {'' if scene == "semi" else f'<geom type="box" size="60 0.06 0.002" pos="{mid} -6 0.002" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>'}
    {car}
    <camera name="side" pos="{mid - 2} {side_y} {side_z}" xyaxes="1 0 0 0 0.05 1" fovy="45"/>
    <camera name="top" pos="{mid} 0 45" xyaxes="1 0 0 0 1 0" fovy="50"/>
    {props}
  </worldbody>
  <contact>{excl}</contact>
  {equality}
  <actuator>
    {acts}
  </actuator>
</mujoco>"""


def load(scene: str = "kickflip"):
    m = mujoco.MjModel.from_xml_string(build_mjcf(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d
