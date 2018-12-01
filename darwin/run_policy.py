import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from darwin.np_policy import *
import joblib
from dabs import *
import time
import os, errno

if __name__ == "__main__":
    filename = 'squatstand_notl.pkl'

    savename = 'ground'

    pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                               2047, 2171,
                               2032, 2039, 2795, 648, 1231, 2040, 2041, 2060, 1281, 3448, 2855, 2073])
    pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                               2048, 2048,
                               2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])

    pose_squat = VAL2RADIAN(pose_squat_val)
    pose_stand = VAL2RADIAN(pose_stand_val)

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_squat],
                  [3.0, pose_stand],
                  [4.0, pose_stand], ]
                  #[3.0, pose_squat],
                  #[3.3, pose_stand],
                  #[3.6, pose_squat], ]
    policy = NP_Policy(interp_sch, 'data/'+filename, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.3)

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    motor_pose = darwin.read_motor_positions()

    darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(policy.get_initial_state())))

    time.sleep(5)

    prev_motor_pose = np.array(darwin.read_motor_positions())
    current_step = 0
    initial_time = time.monotonic()
    prev_time = time.monotonic()
    all_inputs = []
    all_time = []
    all_actions = []
    while current_step < 200:
        if time.monotonic() - prev_time >= 0.05:  # control every 50 ms
            #tdif = time.monotonic() - prev_time
            prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)
            motor_pose = np.array(darwin.read_motor_positions())
            #est_vel = (motor_pose - prev_motor_pose) / tdif
            obs_input = VAL2RADIAN(np.concatenate([HW2SIM_INDEX(prev_motor_pose), HW2SIM_INDEX(motor_pose)]))
            ct = time.monotonic() - initial_time
            act = policy.act(obs_input, ct)
            darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(act)))

            prev_motor_pose = np.copy(motor_pose)

            current_step += 1
            all_actions.append(act)
            all_inputs.append(obs_input)
            all_time.append(ct)

    all_inputs = np.array(all_inputs)
    all_time = np.array(all_time)
    all_actions = np.array(all_actions)

    try:
        os.makedirs('data/hw_data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    np.savetxt('data/hw_data/'+savename+'_saved_obs.txt', all_inputs)
    np.savetxt('data/hw_data/'+savename+'_saved_time.txt', all_time)
    np.savetxt('data/hw_data/'+savename+'_saved_action.txt', all_actions)

    darwin.disconnect()














