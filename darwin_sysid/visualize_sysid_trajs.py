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


if __name__ == "__main__":
    data_dir = 'data/sysid_data/generic_motion/'
    specific_data = 'all_vel0_nn_standup'
    all_trajs = [joblib.load(data_dir + file) for file in os.listdir(data_dir) if
                      '.pkl' in file and 'path' in file and 'standup' in file]

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)
    darwinenv.reset()

    #opt_result = np.loadtxt('data/sysid_data/generic_motion/opt_result'+specific_data+'.txt')
    opt_result = joblib.load('data/sysid_data/generic_motion/' + specific_data + '.pkl')['all_sol']
    print('Use mu of: ', opt_result[1])

    #darwinenv.set_mu(np.array([0.03216347, 0.34971422, 0.50214142, 0.94386206, 0.47390177,
    #   0.12861344, 0.96039561, 0.56407919, 0.72965827, 0.84092037,
    #   0.07445393, 0.98530918, 0.07949251]))
    #opt_result[0][0:5] += 0.3
    darwinenv.set_mu(opt_result[0])


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

        fix_root = True
        if 'fix_root' in traj:
            if not traj['fix_root']:
                fix_root = False

        darwinenv.toggle_fix_root(fix_root)

        darwinenv.reset()
        darwinenv.set_pose(keyframes[0][1])
        if not fix_root:
            darwinenv.set_root_dof(darwinenv.get_root_dof() + np.array([0, 0, 0, 0, 0, -0.075]))

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
            sim_vels.append(darwinenv.get_closest_motor_velocity(hw_vel_data[step+1], 'l1', np.arange(5)))
            darwinenv.set_dup_pose(hw_pose_data[step])
            darwinenv.render()
            darwinenv.step(act)
            step += 1
            time.sleep(0.1)

            if not fix_root:
                # penalize offset in x direction for now, in general should use imu reading
                total_positional_error += (np.clip(np.abs(darwinenv.robot.C[0]), 0.06, 100) - 0.06) * 20
        if not fix_root and darwinenv.check_collision(['MP_ANKLE2_L', 'MP_ANKLE2_R']):
            total_positional_error += 10
        max_step = np.min([len(sim_poses), len(hw_pose_data)])
        total_step += max_step
        total_positional_error += np.sum(
            np.abs(np.array(hw_pose_data)[1:max_step] - np.array(sim_poses)[1:max_step]))
        total_velocity_error += np.sum(
            np.abs(np.array(hw_vel_data)[1:max_step] - np.array(sim_vels)[1:max_step]))


        traj['pose_data'] = sim_poses
        traj['vel_data'] = sim_vels

        print((total_positional_error) / total_step + 0.001 * np.sum(opt_result[0] ** 2))

        '''plt.figure()
        actions = np.array(actions)
        sim_poses = np.array(sim_poses)
        hw_pose_data = np.array(hw_pose_data)
        sim_vels = np.array(sim_vels)
        hw_vel_data = np.array(hw_vel_data)
        plt.plot(actions[:, 3], label='sim action')
        plt.plot(sim_poses[:, 3], label='sim pose')
        plt.plot(hw_pose_data[:, 3], label='hw pose')
        plt.title(str(i) + '_3')
        plt.legend()

        plt.figure()
        plt.plot(actions[:, 13], label='sim action')
        plt.plot(sim_poses[:, 13], label='sim pose')
        plt.plot(hw_pose_data[:, 13], label='hw pose')
        plt.title(str(i) + '_13')
        plt.legend()
        plt.show()'''

        #allfiles = [data_dir.replace('generic', 'synthetic') + file for file in os.listdir(data_dir) if
        # '.pkl' in file]
        #joblib.dump(traj, allfiles[i], compress=True)



    loss = (total_positional_error) / total_step + 0.001 * np.sum(opt_result[0] ** 2)

    print('Loss: ', loss)






