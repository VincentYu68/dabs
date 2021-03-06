#######################################
# test the file in pydart2 simulation #
#######################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pydart2 as pydart
import gym, joblib
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.np_policy import *
import time

if __name__ == "__main__":
    policy_path = 'data/step_policies/04action_fwd_gyroinxy_up5d.pkl'
    #policy_path = 'data/walk_up5d_02stride_fixgain_squatinit_gyroin.pkl'
    fixed_root = False
    action_path = 'data/hw_data/squat_stand.txt'
    run_policy = True

    walk_motion = False
    singlefoot_motion = False
    crawl_motion = False
    lift_motion = False
    step_motion = False
    shake_motion = True

    direct_walk = False

    obs_app = [0.9, 0.05, 0.9, 0.0, 0.2]

    gyro_input = True

    delta_action = 0.4
    if shake_motion:
        delta_action = 0.0

    control_timestep = 0.05  # time interval between control signals
    if direct_walk:
        control_timestep = 0.03

    # initialize policy

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                           [2.0, pose_squat],
                           [3.5, pose_squat],
                           [4.0, pose_stand],
                           [5.0, pose_stand],
                           [7.0, pose_squat],
                           ]

    '''rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_crawl.txt')
    interp_sch = [[0.0, rig_keyframe[0]],
                  [2.0, rig_keyframe[1]],
                  [6.0, rig_keyframe[1]]]'''

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

        interp_sch = [[0.0, 0.5*(pose_stand + pose_squat)]]
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
        policy = NP_Policy(interp_sch, policy_path, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=delta_action, action_filter_size=5)
    else:
        obs_perm, act_perm = make_mirror_perm_indices(0, False, False, len(obs_app), gyro_input)
        policy = NP_Policy(None, policy_path, discrete_action=True,
                           action_bins=np.array([11] * 20), delta_angle_scale=0.0, action_filter_size=5,
                           obs_perm=obs_perm, act_perm=act_perm)

    # load actions
    hw_actions = np.loadtxt(action_path)

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(fixed_root)
    #darwinenv.simenv.env.param_manager.set_simulator_parameters(obs_app)

    if shake_motion:
        darwinenv.set_root_dof(darwinenv.get_root_dof() + np.array([0, -0.1, 0, 0, 0, -0.075]))

    opt_result = joblib.load('data/sysid_data/generic_motion/' + 'all_vel0_nn_hid5' + '.pkl')['all_sol']
    #darwinenv.set_mu(opt_result[0])

    '''darwinenv.set_mu(np.array([4.295156336729233360e-01, 9.547139638558959085e-01, 6.929434610954511298e-01,\
                               9.782717037252172121e-01, 9.990063426489504961e-01, 9.983547461764588071e-01,\
                               5.049677514992344518e-01, 1.721375253280012230e-02, 3.108910579911322580e-01,\
                               2.352015444571272651e-01, 1.889580853785498005e-01, 2.318264830448395486e-01,\
                               2.195776263584806737e-03, 4.484429587055664967e-01]))'''

    darwinenv.simenv.env.interp_sch = interp_sch

    darwinenv.simenv.env.frame_skip = int(control_timestep / darwinenv.simenv.env.sim_dt)

    darwinenv.reset()
    if not direct_walk:
        darwinenv.set_pose(policy.get_initial_state())

    #hw_poses = np.loadtxt('data/hw_data/ground_saved_obs.txt')
    if not run_policy:
        darwinenv.set_pose(hw_actions[0])


    sim_poses = []
    sim_vels = []
    sim_actions = []
    sim_times = []
    sim_gyro = []
    sim_orientation = []
    prev_obs = darwinenv.get_motor_pose()
    max_step = 200
    if interp_sch is not None:
        max_step = int(interp_sch[-1][0] / control_timestep)
    if not run_policy:
        max_step = len(hw_actions)
    for i in range(max_step):
        current_obs = darwinenv.get_motor_pose()
        current_vel = darwinenv.get_motor_velocity()
        input_obs = np.concatenate([prev_obs, current_obs])

        #if direct_walk:
        #    input_obs = np.concatenate([input_obs, darwinenv.get_gyro_data(), darwinenv.accum_orientation])

        if gyro_input:
            input_obs = np.concatenate([input_obs, darwinenv.get_gyro_data()])
        if len(obs_app) > 0:
            input_obs = np.concatenate([input_obs, obs_app])
        if run_policy:
            act = policy.act(input_obs, darwinenv.time)
        else:
            act = hw_actions[i]

        darwinenv.step(act)
        sim_actions.append(act)
        sim_times.append(darwinenv.time)
        darwinenv.render()
        time.sleep(control_timestep)
        sim_poses.append(input_obs)
        sim_vels.append(current_vel)
        prev_obs = current_obs
        imu_data = darwinenv.get_gyro_data()
        sim_gyro.append(imu_data)


    sim_poses = np.array(sim_poses)
    sim_vels = np.array(sim_vels)
    sim_actions = np.array(sim_actions)
    sim_times = np.array(sim_times)
    sim_gyro = np.array(sim_gyro)
    sim_orientation = np.array(sim_orientation)

    savename = 'sim_saved'
    if run_policy:
        savename = 'pol_' + savename
    else:
        savename = 'traj_' + savename
    if fixed_root:
        savename = 'fixed_' + savename
    else:
        savename = 'ground_' + savename
    #savename = 'sim_saved_obs_walk.txt'
    if shake_motion:
        savename = 'ground_shake'

    if run_policy:
        savename += policy_path.split('/')[-1].split('.')[0]
    else:
        savename += action_path.split('/')[-1].split('.')[0]
    np.savetxt('data/sim_data/' + savename + '_obs.txt', sim_poses)
    np.savetxt('data/sim_data/' + savename + '_vels.txt', sim_vels)
    np.savetxt('data/sim_data/' + savename + '_action.txt', sim_actions)
    np.savetxt('data/sim_data/' + savename + '_time.txt', sim_times)
    np.savetxt('data/sim_data/' + savename + '_gyro.txt', sim_gyro)


