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
    parser.add_argument('--traj_time', help='total time for each traj', type=float, default=3.0)
    args = parser.parse_args()

    one_traj = {}

    traj = []
    current_time = 0.0
    traj.append([current_time, pose_squat])
    current_time += 0.5
    traj.append([current_time, pose_squat])

    while current_time < args.traj_time:
        current_time += args.control_dt
        ratio = current_time / args.traj_time

        traj.append([current_time, ratio * pose_stand + (1-ratio) * pose_squat])

    current_time += 0.5
    traj.append([current_time, pose_stand])

    one_traj['control_dt'] = args.control_dt
    one_traj['keyframes'] = traj
    one_traj['total_time'] = args.traj_time + 1
    one_traj['fix_root'] = False

    fname = 'sysidpath_' + str(args.control_dt) + '_' + str(args.traj_time) + '_standup.pkl'
    joblib.dump(one_traj, 'data/sysid_data/generic_motion/'+fname, compress=True)






