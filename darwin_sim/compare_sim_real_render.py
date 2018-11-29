import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import pydart2 as pydart
import gym
from darwin_sim.darwin_env_plain import *
import time
from darwin.np_policy import *



if __name__ == "__main__":
    poses1 = np.loadtxt('data/sim_data/ground_traj_sim_saved_obs.txt')
    poses2 = np.loadtxt('data/hw_data/ground_saved_obs.txt')
    #poses2 = np.loadtxt('data/hw_data/ground_saved_obs.txt')

    actions1 = np.loadtxt('data/sim_data/ground_traj_sim_saved_action.txt')
    actions2 = np.loadtxt('data/hw_data/ground_saved_action.txt')
    #actions2 = np.loadtxt('data/hw_data/ground_saved_action.txt')

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
    interp_sch = [[0.0, pose_squat],
                  [3.0, pose_stand],
                  [4.0, pose_stand], ]
    policy_path = 'data/squatstand_selfcol.pkl'
    policy = NP_Policy(interp_sch, policy_path, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.3)

    loop_size = np.min([len(poses1), len(poses2)])

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)

    darwinenv.reset()

    current_step = 0

    plt.ion()
    plt.show()

    while True:
        darwinenv.set_pose(poses1[current_step][0:20])
        darwinenv.set_dup_pose(poses2[current_step][0:20])

        darwinenv.render()

        if darwinenv.simenv.env._get_viewer() is not None:
            if hasattr(darwinenv.simenv.env._get_viewer(), 'key_being_pressed'):
                if darwinenv.simenv.env._get_viewer().key_being_pressed is not None:
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'-':
                        current_step -= 1
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'+':
                        current_step += 1
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'q':
                        current_step -= 5
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'e':
                        current_step += 5
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'<':
                        current_step -= 10
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'>':
                        current_step += 10
                    if current_step > loop_size - 1:
                        current_step = 0
                    if current_step < 0:
                        current_step = loop_size-1



                    plt.clf()
                    plt.subplot(2,1,1)
                    plt.plot(np.arange(40), poses1[current_step], label='pose1')
                    plt.plot(np.arange(40), poses2[current_step], label='pose2')
                    plt.legend()
                    plt.subplot(2, 1, 2)
                    plt.plot(np.arange(20), actions1[current_step], label='action1')
                    plt.plot(np.arange(20), actions2[current_step], label='action2')
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                    time.sleep(0.1)

