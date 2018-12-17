#######################################
# test the file in pydart2 simulation #
#######################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.np_policy import *
import time

if __name__ == "__main__":
    policy_path = 'data/sqstsq_weakknee_modcom.pkl'
    fixed_root = False
    action_path = 'data/hw_data/ground_saved_action.txt'
    run_policy = True

    # initialize policy

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                           [2.0, pose_squat],
                           [3.5, pose_squat],
                           [4.0, pose_stand],
                           [5.0, pose_stand],
                           [7.0, pose_squat],
                           ]

    rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_crawl.txt')
    interp_sch = [[0.0, rig_keyframe[0]],
                  [2.0, rig_keyframe[1]],
                  [6.0, rig_keyframe[1]]]

    interp_sch = []
    interp_time = 0.0
    for i in range(10):
        for k in range(1, len(rig_keyframe)):
            interp_sch.append([interp_time, rig_keyframe[k]])
            interp_time += 0.5
    interp_sch.append([interp_time, rig_keyframe[0]])

    policy = NP_Policy(interp_sch, policy_path, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.0, action_filter_size=5)

    # load actions
    hw_actions = np.loadtxt(action_path)

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(fixed_root)

    darwinenv.simenv.env.interp_sch = interp_sch

    darwinenv.reset()
    darwinenv.set_pose(policy.get_initial_state())

    #hw_poses = np.loadtxt('data/hw_data/ground_saved_obs.txt')
    #darwinenv.set_pose(hw_poses[0][0:20])

    sim_poses = []
    sim_actions = []
    sim_times = []
    sim_gyro = []
    sim_orientation = []
    prev_obs = darwinenv.get_motor_pose()
    for i in range(200):
        current_obs = darwinenv.get_motor_pose()
        input_obs = np.concatenate([prev_obs, current_obs])
        if run_policy:
            act = policy.act(input_obs, darwinenv.time)
        else:
            act = hw_actions[i]

        darwinenv.step(act)
        sim_actions.append(act)
        sim_times.append(darwinenv.time)
        darwinenv.render()
        time.sleep(0.05)
        sim_poses.append(input_obs)
        prev_obs = current_obs
        imu_data = darwinenv.get_imu_reading()
        darwinenv.integrate_imu_reading()
        sim_gyro.append(imu_data)
        sim_orientation.append(darwinenv.get_integrated_imu())

    sim_poses = np.array(sim_poses)
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
    np.savetxt('data/sim_data/' + savename + '_obs.txt', sim_poses)
    np.savetxt('data/sim_data/' + savename + '_action.txt', sim_actions)
    np.savetxt('data/sim_data/' + savename + '_time.txt', sim_times)
    np.savetxt('data/sim_data/' + savename + '_gyro.txt', sim_gyro)
    np.savetxt('data/sim_data/' + savename + '_orientation.txt', sim_orientation)


