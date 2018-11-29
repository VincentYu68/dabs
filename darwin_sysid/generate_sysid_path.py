# propse a set of system id actions such that no self collision is filtered

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.darwin_utils import *
import time


if __name__ == "__main__":
    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)

    darwinenv.reset()

    num_single_pose = 5
    num_double_pose = 5
    double_pose_interp = 11

    single_poses = []
    double_poses = []

    while len(single_poses) < num_single_pose:
        rand_pose_coef = np.random.random(20)
        rand_pose = SIM_JOINT_LOW_BOUND_RAD * rand_pose_coef + SIM_JOINT_UP_BOUND_RAD * (1-rand_pose_coef)

        darwinenv.set_pose(rand_pose)
        if not darwinenv.check_self_collision():
            single_poses.append(rand_pose)

    while len(double_poses) < num_double_pose:
        rand_pose_coef1 = np.random.random(20)
        rand_pose1 = SIM_JOINT_LOW_BOUND_RAD * rand_pose_coef1 + SIM_JOINT_UP_BOUND_RAD * (1 - rand_pose_coef1)

        rand_pose_coef2 = np.random.random(20)
        rand_pose2 = SIM_JOINT_LOW_BOUND_RAD * rand_pose_coef2 + SIM_JOINT_UP_BOUND_RAD * (1 - rand_pose_coef2)

        self_collided = False
        for i in range(double_pose_interp):
            darwinenv.set_pose(rand_pose1 * i * 1.0 / (double_pose_interp-1) + rand_pose2 * (1.0 - i * 1.0 / (double_pose_interp-1)))
            if darwinenv.check_self_collision():
                self_collided = True

        if not self_collided:
            double_poses.append(np.concatenate([rand_pose1, rand_pose2]))

    np.savetxt('data/sysid_data/single_poses.txt', single_poses)
    np.savetxt('data/sysid_data/double_poses.txt', double_poses)

