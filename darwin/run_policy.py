import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from darwin.np_policy import *
import joblib
from dabs import *
import time
import os, errno

if __name__ == "__main__":
    #filename = 'sqstsq_nolimvel_UP4d.pkl'
    folder = 'step_policies/'
    filename = '02action_fwd_oriinxy_up5d.pkl'

    savename = 'ground'+filename.split('.')[0]

    walk_motion = False
    singlefoot_motion = False
    crawl_motion = False
    lift_motion = False
    step_motion = True
    shake_motion = False

    direct_walk = False

    obs_app = [0.9, 0.05, 0.9, 0.0, 0.2]#[0.05945156, 0.73512937, 0.76391359, 0.41831418]

    control_timestep = 0.05  # time interval between control signals
    if direct_walk:
        control_timestep = 0.03

    delta_action = 0.2

    if shake_motion:
        delta_action = 0.0
        savename = 'ground_shake'

    bno055_input = True
    bno055_angvel_input = False
    gyro_input = 0
    gyro_accum_input = False

    savename += '_walk' if walk_motion else ''
    savename += '_singlefoot' if singlefoot_motion else ''
    savename += '_crawl' if crawl_motion else ''
    savename += '_lift' if lift_motion else ''
    savename += '_step_motion' if step_motion else ''

    savename += '_direct_walk' if direct_walk else ''


    pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                               2047, 2171,
                               2032, 2039, 2795, 648, 1241, 2040,    2041, 2060, 1281, 3448, 2855, 2073])
    pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                               2048, 2048,
                               2048, 2048, 2048, 2048, 2048, 2048,    2048, 2048, 2048, 2048, 2048, 2048])

    pose_squat = VAL2RADIAN(pose_squat_val)
    pose_stand = VAL2RADIAN(pose_stand_val)

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                           [2.0, pose_squat],
                           [3.5, pose_squat],
                           [4.0, pose_stand],
                           [5.0, pose_stand],
                           [7.0, pose_squat],
                           ]

    if walk_motion or crawl_motion or lift_motion or step_motion or shake_motion:
        if walk_motion:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe.txt')
        elif lift_motion:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_lift.txt')
        elif step_motion:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_step.txt')
        elif shake_motion:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_shakelr.txt')
        else:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_crawl.txt')
        interp_sch = [[0.0, 0.5*(pose_squat+pose_stand)]]
        interp_time = 0.2
        for i in range(1):
            for k in range(0, len(rig_keyframe)):
                interp_sch.append([interp_time, rig_keyframe[k]])
                interp_time += 0.03
        interp_sch.append([interp_time, rig_keyframe[0]])

        if shake_motion:
            interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.4
            for i in range(1):
                for k in range(0, len(rig_keyframe)):
                    interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.4
            interp_sch.append([interp_time, rig_keyframe[0]])

    if singlefoot_motion:
        rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe2.txt')
        interp_sch = [[0.0, rig_keyframe[0]],
                      [2.0, rig_keyframe[1]],
                      [6.0, rig_keyframe[1]]]


    if direct_walk:
        interp_sch = None

    if not direct_walk:
        policy = NP_Policy(interp_sch, 'data/'+folder+filename, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=delta_action, action_filter_size=5)
    else:
        obs_perm, act_perm = make_mirror_perm_indices(gyro_input, gyro_accum_input, False, len(obs_app), bno055_input)
        policy = NP_Policy(None, 'data/' + folder+filename, discrete_action=True,
                           action_bins=np.array([11] * 20), delta_angle_scale=delta_action, action_filter_size=5,
                           obs_perm=obs_perm, act_perm=act_perm)

    darwin = BasicDarwin(use_bno055=bno055_input)

    darwin.connect()

    darwin.write_torque_enable(True)
    darwin.write_pid(32, 0, 16)

    motor_pose = darwin.read_motor_positions()

    if not direct_walk:
        darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(policy.get_initial_state())))
    else:
        darwin.write_motor_goal(SIM2HW_INDEX(0.5*(pose_squat_val + pose_stand_val)))

    time.sleep(5)

    max_step = 200
    if interp_sch is not None:
        max_step = int(interp_sch[-1][0] / control_timestep)

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
    while current_step < max_step:
        if time.monotonic() - prev_time >= control_timestep:  # control every 50 ms
            tdif = time.monotonic() - prev_time
            prev_time = time.monotonic() - ((time.monotonic() - prev_time) - control_timestep)
            motor_pose = np.array(darwin.read_motor_positions())
            #gyro = darwin.read_gyro()
            #cur_orientation += VAL2RPS(gyro) * tdif
            #est_vel = (motor_pose - prev_motor_pose) / tdif
            obs_input = VAL2RADIAN(np.concatenate([HW2SIM_INDEX(prev_motor_pose), HW2SIM_INDEX(motor_pose)]))
            #if gyro_input > 0:
            #    obs_input = np.concatenate([obs_input, VAL2RPS(gyro)])
            #if gyro_accum_input:
            #    obs_input = np.concatenate([obs_input, cur_orientation])
            if bno055_input:
                gyro = darwin.read_bno055_gyro()
                if bno055_angvel_input:
                    gyro[3:] = 0.0
                obs_input = np.concatenate([obs_input, gyro])
                all_gyros.append(gyro)

            if len(obs_app) > 0:
                obs_input = np.concatenate([obs_input, obs_app])

            ct = time.monotonic() - initial_time
            act = policy.act(obs_input, ct - control_timestep)
            darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(act)))

            prev_motor_pose = np.copy(motor_pose)

            current_step += 1
            all_actions.append(act)
            all_inputs.append(obs_input)
            all_time.append(ct - control_timestep)

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

    darwin.disconnect()














