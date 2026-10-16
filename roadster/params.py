"""Physical and scene constants. Every number here is an estimate, not a Tesla figure."""
import numpy as np

G = 9.81

# Car (estimate): mass including a full air tank, solid-box inertia.
MASS = 2000.0
BOX = (4.3, 2.0, 1.15)  # length, width, height [m]
WHEEL_MASS = 25.0
WHEEL_RADIUS = 0.35
INERTIA = np.array([
    MASS / 12 * (BOX[1] ** 2 + BOX[2] ** 2),
    MASS / 12 * (BOX[0] ** 2 + BOX[2] ** 2),
    MASS / 12 * (BOX[0] ** 2 + BOX[1] ** 2),
])

# CoM height at rest on the wheels, measured from the model (tests/test_model.py).
RIDE_Z = 0.5328
# Chassis height when the tires first touch the ground with the suspension relaxed (free fall):
# wheel center 0.29 m below the chassis origin plus the tire radius.
Z_TOUCH = 0.29 + 0.35

# Thrusters (estimate). Body frame: x forward, y left, z up.
THRUSTER_X = 1.6
THRUSTER_Y = 0.85
F_LIFT_MAX = 15_000.0
F_ROOF_MAX = 5_000.0
F_SIDE_MAX = 3_000.0
SIDE_X = 1.8
SIDE_Y = 1.0

# 260 kg of air at a cold-gas Isp of about 70 s.
IMPULSE_BUDGET = 180_000.0

DT_SIM = 0.001
DT_CTRL = 0.01

# Kickflip scene.
VX0 = 18.0           # run-up speed [m/s]
RUNUP = 12.0         # distance driven on the ground before the launch point [m]
ROW_START = 16.0     # obstacle row starts this far past the launch point [m]
ROW_LEN = 14.0
ROW_HEIGHT = 4.0     # tallest vehicle (semi)
CLEARANCE_RADIUS = 1.2  # half-diagonal of the car cross-section, rounded up

# Touchdown limits.
TD_VZ_MAX = 1.5
TD_ANGLE_MAX_DEG = 10.0

# Ramp scene (estimates). A circular quarter pipe that keeps curving past vertical to an overhanging lip.
RAMP_X0 = 0.0          # where the curve starts on the flat [m]
RAMP_R = 7.0           # radius of the curve [m]
RAMP_LIP_DEG = 160.0   # arc angle of the lip, 70 deg past vertical (close to a half circle)
RAMP_SEG_DEG = 1.5     # facet size of the polygonal surface
RAMP_WIDTH = 6.0
RAMP_THICK = 0.3
RAMP_RUNUP = 18.0      # flat approach before the curve [m]
V_RAMP = 9.0           # speed the wheel motors hold on the approach and up the ramp [m/s]
DRIVE_TAU_MAX = 3_000.0   # per wheel [N m]
F_SUCTION = 20_000.0      # fan suction pressing the car into the ramp, about 2 t (estimate)
SUSP_K_RAMP = 150_000.0   # stiffer stunt suspension for the ramp (suction plus ~1 g of turn) [N/m]
SUSP_C_RAMP = 7_000.0
RAMP_LAND_GAP = 3.5    # plan lands the CoM at least this far before the curve starts [m]

# My own stunt (estimates): from the next lane, jump sideways onto the flatbed of a Semi driving alongside.
V_SEMI = VX0           # truck speed [m/s]; the car matches it, level with the middle of the trailer
SEMI_LANE_Y = 4.0      # the car's lane, measured from the trailer centerline [m]
DECK_H = 1.5           # trailer deck top above the road [m]
DECK_LEN = 14.0        # flatbed length [m]
DECK_W = 2.6           # flatbed width [m]; the car body is 1.96 m wide
CAB_LEN, CAB_H = 6.0, 3.6   # tractor ahead of the trailer [m]
SEMI_MIN_FLIGHT = 2.6    # [s] minimum planned flight time, so feedback has time to correct
