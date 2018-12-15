import numpy as np


############### Darwin utilization functions ###############################
### Conversion between units
def VAL2RADIAN(val):
    return (val - 2048) * 0.088 * np.pi / 180

def RADIAN2VAL(rad):
    return rad * 180 / np.pi / 0.088 + 2048

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


###############################################################################