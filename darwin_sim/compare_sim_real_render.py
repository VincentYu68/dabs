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
    poses1 = np.loadtxt('data/sim_data/ground_pol_sim_savedwalk_tl10_vrew10_limvel_obs.txt')
    poses2 = np.loadtxt('data/hw_data/groundwalk_tl10_vrew10_limvel_direct_walk_saved_obs.txt')
    #poses1 = np.loadtxt('data/hw_data/fixed_saved_obs.txt')

    actions1 = np.loadtxt('data/sim_data/ground_pol_sim_savedwalk_tl10_vrew10_limvel_action.txt')
    actions2 = np.loadtxt('data/hw_data/groundwalk_tl10_vrew10_limvel_direct_walk_saved_action.txt')
    #actions1 = np.loadtxt('data/hw_data/fixed_saved_action.txt')

    times1 = np.loadtxt('data/sim_data/ground_pol_sim_savedwalk_tl10_vrew10_limvel_time.txt')
    times2 = np.loadtxt('data/hw_data/groundwalk_tl10_vrew10_limvel_direct_walk_saved_time.txt')

    loop_size = np.min([len(poses1), len(poses2)])

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                  [2.0, pose_squat],
                  [3.5, pose_squat],
                  [4.0, pose_stand],
                  [5.0, pose_stand],
                  [7.0, pose_squat],
                  ]

    policy = NP_Policy(interp_sch, 'data/sqstsq_weakknee.pkl', discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.3, action_filter_size=5)


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

        act1 = None
        act2 = None

        if darwinenv.simenv.env._get_viewer() is not None:
            if hasattr(darwinenv.simenv.env._get_viewer(), 'key_being_pressed'):
                if darwinenv.simenv.env._get_viewer().key_being_pressed is not None:
                    act1 = None
                    act2 = None

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

                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'p':
                        for i in range(11):
                            if current_step - 10 + i < 0:
                                continue
                            act1 = policy.act(poses1[current_step - 10 + i], times1[current_step - 10 + i]-darwinenv.simenv.env.dt)
                        for i in range(11):
                            if current_step - 10 + i < 0:
                                continue
                            act2 = policy.act(poses2[current_step - 10 + i], times2[current_step - 10 + i])
                        print('time1: ', times1[current_step]-darwinenv.simenv.env.dt)
                        print('time2: ', times2[current_step])

                        if current_step > 1:
                            vel1 = (poses1[current_step][0:20] - poses1[current_step][20:]) / (
                                    times1[current_step] - times1[current_step - 1])
                            vel2 = (poses2[current_step][0:20] - poses2[current_step][20:]) / (
                                    times2[current_step] - times2[current_step - 1])
                            print('velocity 1: ', vel1)
                            print('velocity 2: ', vel2)


                    plt.clf()
                    plt.subplot(2,1,1)
                    plt.plot(np.arange(len(poses1[0])), poses1[current_step], label='pose1')
                    plt.plot(np.arange(len(poses1[0])), poses2[current_step], label='pose2')
                    plt.legend()
                    plt.subplot(2, 1, 2)
                    plt.plot(np.arange(20), actions1[current_step], label='action1')
                    plt.plot(np.arange(20), actions2[current_step], label='action2')

                    if act1 is not None:
                        plt.plot(np.arange(20), act1, label='action1_rep')
                        plt.plot(np.arange(20), act2, label='action2_rep')
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                    time.sleep(0.1)

