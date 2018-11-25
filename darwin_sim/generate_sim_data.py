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
    policy_path = 'data/darwin_standsquat_policy_conseq_obs_warmstart.pkl'
    fixed_root = False
    action_path = 'data/fixed_saved_action.txt'
    run_policy = True

    # initialize policy
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
                  [1.5, pose_squat],
                  [2.5, pose_stand],
                  [3.0, pose_squat],
                  [3.3, pose_stand],
                  [3.6, pose_squat], ]
    policy = NP_Policy(interp_sch, policy_path, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.0)

    # load actions
    hw_actions = np.loadtxt(action_path)

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(fixed_root)

    darwinenv.simenv.env.interp_sch = interp_sch

    darwinenv.reset()
    darwinenv.set_pose(policy.get_initial_state())

    sim_poses = []
    prev_obs = darwinenv.get_motor_pose()
    for i in range(200):
        current_obs = darwinenv.get_motor_pose()
        # WARNING: the order of current obs and prev obs is flipped, need to fix later, but use this for now as the hw data is also flpped
        input_obs = np.concatenate([current_obs, prev_obs])
        if run_policy:
            darwinenv.step(policy.act(input_obs, darwinenv.time))
        else:
            darwinenv.step(hw_actions[i])
        darwinenv.render()
        time.sleep(0.05)
        sim_poses.append(input_obs)

    sim_poses = np.array(sim_poses)

    savename = 'sim_saved_obs.txt'
    if run_policy:
        savename = 'pol_' + savename
    else:
        savename = 'traj_' + savename
    if fixed_root:
        savename = 'fixed_' + savename
    else:
        savename = 'ground_' + savename
    savename = 'sim_saved_obs.txt'
    np.savetxt('data/' + savename, sim_poses)


