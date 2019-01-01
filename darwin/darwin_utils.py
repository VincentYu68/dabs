import numpy as np


############### Darwin utilization functions ###############################
### Conversion between units
def VAL2RADIAN(val):
    return (val - 2048) * 0.088 * np.pi / 180

def RADIAN2VAL(rad):
    return rad * 180 / np.pi / 0.088 + 2048

# speed
def SPEED_HW2SIM(val):  # from value to radians per sec
    rpm = np.zeros_like(val)
    rpm[val >= 1024] = -(val[val >= 1024] - 1024) * 0.1111
    rpm[val < 1024] = val[val < 1024] * 0.1111
    return rpm / 60.0 * np.pi * 2

# gyro
def VAL2RPS(val):
    return (val-512)/512.0*500/180*np.pi

def RPS2VAL(rps):
    return [int(v) for v in (rps/np.pi*180.0/500*512+512)]

### Conversion between darwin and simulation indices
SIM2HW_JOINT_INDEX = [3,0,4,1,5,2,14,8,15,9,16,10,17,11,18,12,19,13,6,7]
HW2SIM_JOINT_INDEX = np.argsort(SIM2HW_JOINT_INDEX).astype(int).tolist()

def SIM2HW_INDEX(input):
    return (np.array(input)[SIM2HW_JOINT_INDEX])

def HW2SIM_INDEX(input):
    return (np.array(input)[HW2SIM_JOINT_INDEX])


### Motor angle bounds in simulation and hardware
SIM_JOINT_LOW_BOUND_VAL = np.array([
    0, 700, 1400, 0, 1600, 1000,
    1346, 1850,
    1800, 1400, 1800, 648, 1241, 1850,   1800, 2048, 1000, 2048, 1500, 1700
])

SIM_JOINT_UP_BOUND_VAL = np.array([
    4095, 2400, 3000, 4095, 3400, 2800,
    2632, 2600,
    2200, 2048, 3100, 2048, 2500, 2400,   2200, 2600, 2300, 3448, 2855, 2300
])

SIM_JOINT_LOW_BOUND_RAD = VAL2RADIAN(SIM_JOINT_LOW_BOUND_VAL)
SIM_JOINT_UP_BOUND_RAD = VAL2RADIAN(SIM_JOINT_UP_BOUND_VAL)

HW_JOINT_LOW_BOUND_VAL = SIM2HW_INDEX(SIM_JOINT_LOW_BOUND_VAL)
HW_JOINT_UP_BOUND_VAL = SIM2HW_INDEX(SIM_JOINT_UP_BOUND_VAL)
HW_JOINT_LOW_BOUND_RAD = SIM2HW_INDEX(SIM_JOINT_LOW_BOUND_RAD)
HW_JOINT_UP_BOUND_RAD = SIM2HW_INDEX(SIM_JOINT_UP_BOUND_RAD)


###################### predefined poses #######################################
pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                               2047, 2171,
                               2032, 2039, 2795, 648, 1241, 2040,   2041, 2060, 1281, 3448, 2855, 2073])
pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                           2048, 2048,
                           2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])
pose_squat = VAL2RADIAN(pose_squat_val)
pose_stand = VAL2RADIAN(pose_stand_val)

###############################################################################


################################ control bounds ###############################
# Joint control limits
SIM_CONTROL_LOW_BOUND_VAL = np.array([
    1500, 1500, 1600, 1500, 1800, 1400,
    2000, 2000,
    1950, 2048, 1800, 648, 1241, 1850,   1800, 2048, 1000, 2048, 1500, 1700
])

SIM_CONTROL_UP_BOUND_VAL = np.array([
    2500, 2200, 2600, 2500, 2600, 2600,
    2100, 2100,
    2200, 2048, 3100, 2048, 2500, 2400,   2200, 2200, 2300, 3448, 2855, 2300
])

SIM_CONTROL_LOW_BOUND_RAD = VAL2RADIAN(SIM_CONTROL_LOW_BOUND_VAL)

SIM_CONTROL_UP_BOUND_RAD = VAL2RADIAN(SIM_CONTROL_UP_BOUND_VAL)

SIM_CONTROL_LOW_BOUND_NEWFOOT_VAL = SIM_CONTROL_LOW_BOUND_VAL[[0,1,2,3,4,5,6,7,  8,9,10,11, 14,15,16,17]]

SIM_CONTROL_UP_BOUND_NEWFOOT_VAL = SIM_CONTROL_UP_BOUND_VAL[[0,1,2,3,4,5,6,7,  8,9,10,11, 14,15,16,17]]

SIM_CONTROL_LOW_BOUND_NEWFOOT_RAD = VAL2RADIAN(SIM_CONTROL_LOW_BOUND_NEWFOOT_VAL)

SIM_CONTROL_UP_BOUND_NEWFOOT_RAD = VAL2RADIAN(SIM_CONTROL_UP_BOUND_NEWFOOT_VAL)


################################ obs and action permutation indices for mirror symmetry ######################
def make_mirror_perm_indices(imu_input_step, accum_imu_input, include_accelerometer, UP_dim):
    obs_perm_base = np.array(
        [-3, -4, -5, -0.0001, -1, -2, 6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13,
         -23, -24, -25, -20, -21, -22, 26, 27, -34, -35, -36, -37, -38, -39, -28, -29, -30, -31, -32, -33])
    act_perm_base = np.array(
        [-3, -4, -5, -0.0001, -1, -2, 6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13])

    for i in range(imu_input_step):
        beginid = len(obs_perm_base)
        if include_accelerometer:
            obs_perm_base = np.concatenate(
                [obs_perm_base, [-beginid, beginid + 1, beginid + 2, -beginid - 3, beginid + 4, -beginid - 5]])
        else:
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1, -beginid - 2]])
    if accum_imu_input:
        beginid = len(obs_perm_base)
        if include_accelerometer:
            obs_perm_base = np.concatenate(
                [obs_perm_base, [-beginid, beginid + 1, beginid + 2, -beginid - 3, beginid + 4, beginid + 5,
                                 -beginid - 6, beginid + 7, -beginid - 8]])
        else:
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1, -beginid - 2]])

    if UP_dim > 0:
        obs_perm_base = np.concatenate([obs_perm_base, np.arange(len(obs_perm_base), len(obs_perm_base) + UP_dim)])

    return np.copy(obs_perm_base), np.copy(act_perm_base)