import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from darwin.np_policy import *
import joblib
from dabs import *
import time
import os, errno

if __name__ == "__main__":
    filename = 'sqstsqst_newkpkd_lowfric.pkl'

    savename = 'ground'

    walk_motion = False
    singlefoot_motion = False
    crawl_motion = False

    savename += '_walk' if walk_motion else ''
    savename += '_singlefoot' if singlefoot_motion else ''
    savename += '_crawl' if crawl_motion else ''

    gyro_input = 0

    pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                               2047, 2171,
                               2032, 2039, 2795, 648, 1231, 2040, 2041, 2060, 1281, 3448, 2855, 2073])
    pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                               2048, 2048,
                               2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])

    pose_squat = VAL2RADIAN(pose_squat_val)
    pose_stand = VAL2RADIAN(pose_stand_val)

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                           [2.0, pose_squat],
                           [4.0, pose_stand],
                           [5.0, pose_stand],
                           [5.5, pose_squat],
                           [7.5, pose_squat],
                           [8.0, pose_stand],
                           [10.0, pose_stand],
                           ]

    if walk_motion or crawl_motion:
        if walk_motion:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe.txt')
        else:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_crawl.txt')
        interp_sch = [[0.0, rig_keyframe[0]]]
        interp_time = 0.5
        for i in range(10):
            for k in range(1, len(rig_keyframe)):
                interp_sch.append([interp_time, rig_keyframe[k]])
                interp_time += 0.5
        interp_sch.append([interp_time, rig_keyframe[0]])

    if singlefoot_motion:
        rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe2.txt')
        interp_sch = [[0.0, rig_keyframe[0]],
                      [2.0, rig_keyframe[1]],
                      [6.0, rig_keyframe[1]]]

    policy = NP_Policy(interp_sch, 'data/'+filename, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.3, action_filter_size=5)

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 8)

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
    all_gyros = []
    all_orientations = []
    cur_orientation = np.zeros(3)
    while current_step < 200:
        if time.monotonic() - prev_time >= 0.05:  # control every 50 ms
            tdif = time.monotonic() - prev_time
            prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)
            motor_pose = np.array(darwin.read_motor_positions())
            gyro = darwin.read_gyro()
            #est_vel = (motor_pose - prev_motor_pose) / tdif
            obs_input = VAL2RADIAN(np.concatenate([HW2SIM_INDEX(prev_motor_pose), HW2SIM_INDEX(motor_pose)]))
            if gyro_input > 0:
                obs_input = np.concatenate([obs_input, VAL2RPS(gyro)])
            cur_orientation += VAL2RPS(gyro) * tdif

            ct = time.monotonic() - initial_time
            act = policy.act(obs_input, ct)
            darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(act)))

            prev_motor_pose = np.copy(motor_pose)

            current_step += 1
            all_actions.append(act)
            all_inputs.append(obs_input)
            all_time.append(ct)
            all_gyros.append(VAL2RPS(gyro))
            all_orientations.append(cur_orientation)

    all_inputs = np.array(all_inputs)
    all_time = np.array(all_time)
    all_actions = np.array(all_actions)
    all_orientations = np.array(all_orientations)

    try:
        os.makedirs('data/hw_data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    np.savetxt('data/hw_data/'+savename+'_saved_obs.txt', all_inputs)
    np.savetxt('data/hw_data/'+savename+'_saved_time.txt', all_time)
    np.savetxt('data/hw_data/'+savename+'_saved_action.txt', all_actions)
    np.savetxt('data/hw_data/' + savename + '_saved_gyro.txt', all_gyros)
    np.savetxt('data/hw_data/' + savename + '_saved_orientation.txt', all_orientations)

    darwin.disconnect()














