import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from darwin.np_policy import *
import joblib
from dabs import *
import time
import os, errno

if __name__ == "__main__":

    pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                               2048, 2048,
                               2048, 2048, 2048, 2048, 2048, 2048,    2048, 2048, 2048, 2048, 2048, 2048])

    pose_stand = VAL2RADIAN(pose_stand_val)

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 8)

    motor_pose = darwin.read_motor_positions()

    darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(pose_stand)))

    time.sleep(5)

    current_jt_id = 1
    current_pose = np.copy(pose_stand)

    while True:
        current_pose[current_jt_id - 1] = pose_stand[current_jt_id - 1]
        darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(current_pose)))
        cmd = input('Enter joint id in simulation to examine joint range. Enter \'exit\' to close:\n')
        if cmd == 'exit':
            break

        if type(cmd) is int and cmd <= 20 and cmd >= 1:
            current_jt_id = cmd
            max_val = SIM_JOINT_UP_BOUND_RAD[current_jt_id-1]
            min_val = SIM_JOINT_LOW_BOUND_RAD[current_jt_id - 1]
            for i in range(4):
                if i % 2 == 0:
                    current_pose[current_jt_id-1] = max_val
                    darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(current_pose)))
                else:
                    current_pose[current_jt_id-1] = min_val
                    darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(current_pose)))
                time.sleep(2)

        else:
            print('Format incorrect!')


    darwin.disconnect()














