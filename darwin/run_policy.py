import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from darwin.np_policy import *
import joblib
from dabs import *
import time

if __name__ == "__main__":
    filename = 'darwin_squatstand_policy.pkl'

    pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                               2047, 2171,
                               2032, 2039, 2795, 568, 1231, 2040, 2041, 2060, 1281, 3525, 2855, 2073])
    pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                               2048, 2048,
                               2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])

    pose_squat = VAL2RADIAN(pose_squat_val)
    pose_stand = VAL2RADIAN(pose_stand_val)

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                  [1.5, pose_squat],
                  [2.5, pose_stand],
                  [3.0, pose_squat],
                  [3.3, pose_stand],
                  [3.6, pose_squat], ]
    policy = NP_Policy(interp_sch, 'data/darwin_squatstand_policy.pkl', discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.3)

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    motor_pose = darwin.read_motor_positions()

    darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(policy.get_initial_state())))

    time.sleep(5)

    prev_motor_pose = darwin.read_motor_positions()
    current_step = 0
    initial_time = time.monotonic()
    prev_time = time.monotonic()
    while current_step < 500:
        if time.monotonic() - prev_time >= 0.05:  # control every 50 ms
            tdif = time.monotonic() - prev_time
            prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)
            motor_pose = darwin.read_motor_positions()
            est_vel = (motor_pose - prev_motor_pose) / tdif
            act = policy.act(VAL2RADIAN(np.concatenate(HW2SIM_INDEX(motor_pose), HW2SIM_INDEX(est_vel))), time.monotonic() - initial_time)
            darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(act)))

            current_step += 1

    darwin.disconnect()














