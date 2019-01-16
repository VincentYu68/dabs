# propse a set of system id actions such that no self collision is filtered

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.darwin_utils import *
import time, sys, joblib

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--control_dt', help='control timestep', type=float, default=0.05)
    parser.add_argument('--move_size', help='movement size', type=float, default=0.1)
    parser.add_argument('--num_move', help='number of movements', type=int, default=4)
    parser.add_argument('--num_trajs', help='number of trajectories', type=int, default=5)
    parser.add_argument('--traj_time', help='total time for each traj', type=float, default=1.0)
    parser.add_argument('--visualize', help='whether to visualize each traj', type=str, default='False')
    args = parser.parse_args()

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)
    darwinenv.reset()
    darwinenv.set_control_timestep(args.control_dt)

    generated_trajs = 0

    move_interval = args.traj_time / args.num_move

    while generated_trajs < args.num_trajs:
        one_traj = {}
        while True:
            traj = []
            current_time = 0.0
            rand_pose_coef = np.random.random(20)
            rand_pose = SIM_CONTROL_LOW_BOUND_RAD * rand_pose_coef + SIM_CONTROL_UP_BOUND_RAD * (1-rand_pose_coef)

            darwinenv.set_pose(rand_pose)
            if darwinenv.check_self_collision():
                continue

            traj.append([current_time, rand_pose])

            move_direction = np.random.randint(0, 2, len(rand_pose)) * 2 - 1
            next_pose = np.copy(rand_pose)
            for i in range(args.num_move):
                next_pose = np.clip(next_pose + move_direction * args.move_size, SIM_CONTROL_LOW_BOUND_RAD, SIM_CONTROL_UP_BOUND_RAD)
                traj.append([current_time, np.copy(next_pose)])
                current_time += move_interval
                move_direction[(next_pose==SIM_CONTROL_LOW_BOUND_RAD) + (next_pose==SIM_CONTROL_UP_BOUND_RAD)] *= -1
                #move_direction *= -1
                #move_direction = np.random.randint(0, 2, len(rand_pose)) * 2 - 1

            darwinenv.reset()
            darwinenv.set_pose(traj[0][1])
            self_col = False
            while darwinenv.time <= args.traj_time:
                act = traj[0][1]
                for kf in traj:
                    if darwinenv.time >= kf[0]:
                        act = kf[1]
                darwinenv.step(act)
                if darwinenv.check_self_collision():
                    self_col = True
                    break
            if self_col:
                continue

            if args.visualize == 'True':
                darwinenv.reset()
                darwinenv.set_pose(traj[0][1])
                while darwinenv.time <= args.traj_time:
                    act = traj[0][1]
                    for kf in traj:
                        if darwinenv.time >= kf[0]:
                            act = kf[1]
                    darwinenv.step(act)
                    darwinenv.render()
                    time.sleep(0.01)


            one_traj['control_dt'] = args.control_dt
            one_traj['keyframes'] = traj
            one_traj['total_time'] = args.traj_time
            break

        fname = 'sysidpath_' + str(args.control_dt) + '_' + str(args.move_size) + '_' + str(generated_trajs) + '.pkl'
        joblib.dump(one_traj, 'data/sysid_data/generic_motion_test/'+fname, compress=True)
        generated_trajs += 1






