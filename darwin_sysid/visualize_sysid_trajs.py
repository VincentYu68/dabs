import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.np_policy import *
import time
import cma, os, sys, joblib

from darwin_sysid.optimize_simulation import *

if __name__ == "__main__":
    data_dir = 'data/sysid_data/generic_motion/'
    specific_data = 'vel01_0.1only'
    all_trajs = [joblib.load(data_dir + file) for file in os.listdir(data_dir) if
                      '.pkl' in file]

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)
    darwinenv.reset()

    opt_result = np.loadtxt(data_dir+'/opt_result'+specific_data+'.txt')
    print('Use mu of: ', opt_result[1])

    darwinenv.set_mu(np.array([0.5, 0.5, 0.5, 0.5, 0.5,\
                               0.1, 0.1, 0.1, 0.1, 0.1,\
                               0.5, 0.5, 0.0, 0.5]))
    #darwinenv.set_mu(opt_result[0])

    sysid_optimizer = SysIDOptimizer('data/sysid_data/generic_motion/', velocity_weight=0.05, specific_data='',
                                     save_app='vel005')

    total_positional_error = 0
    total_velocity_error = 0
    total_step = 0
    for i, traj in enumerate(all_trajs):
        control_dt = traj['control_dt']
        keyframes = traj['keyframes']
        traj_time = traj['total_time']

        hw_pose_data = traj['pose_data']
        hw_vel_data = traj['vel_data']
        darwinenv.set_control_timestep(control_dt)

        darwinenv.reset()
        darwinenv.set_pose(keyframes[0][1])
        step = 0
        actions = []
        sim_poses = [darwinenv.get_motor_pose()]
        sim_vels = [darwinenv.get_motor_velocity()]


        while darwinenv.time <= traj_time + 0.004:
            act = keyframes[0][1]
            for kf in keyframes:
                if darwinenv.time >= kf[0]-0.00001:
                    act = kf[1]
            actions.append(act)
            sim_poses.append(darwinenv.get_motor_pose())
            sim_vels.append(darwinenv.get_motor_velocity())
            darwinenv.set_dup_pose(hw_pose_data[step])
            darwinenv.render()
            darwinenv.step(act)
            step += 1
            time.sleep(0.1)

        max_step = np.min([len(sim_poses), len(hw_pose_data)])
        total_step += max_step
        total_positional_error += np.sum(
            np.abs(np.array(hw_pose_data)[1:max_step] - np.array(sim_poses)[1:max_step]))
        total_velocity_error += np.sum(
            np.abs(np.array(hw_vel_data)[1:max_step] - np.array(sim_vels)[1:max_step]))

        traj['pose_data'] = sim_poses
        traj['vel_data'] = sim_vels

        '''plt.figure()
        actions = np.array(actions)
        sim_poses = np.array(sim_poses)
        hw_pose_data = np.array(hw_pose_data)
        plt.plot(actions[:, -3], label='sim action')
        plt.plot(sim_poses[:, -3], label='sim pose')
        plt.plot(hw_pose_data[:, -3], label='hw pose')
        plt.title(str(i))
        plt.legend()
        plt.show()'''

        #allfiles = [data_dir.replace('generic', 'synthetic') + file for file in os.listdir(data_dir) if
        # '.pkl' in file]
        #joblib.dump(traj, allfiles[i], compress=True)


    loss = (total_positional_error + total_velocity_error * 0.05) / total_step + 0.001 * np.sum(opt_result[0] ** 2)

    print('Loss: ', loss)






