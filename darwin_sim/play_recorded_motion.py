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
import time

if __name__ == "__main__":
    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)

    poses = np.loadtxt('data/saved_obs.txt')

    darwinenv.reset()
    sim_poses = []
    for i in range(200):
        darwinenv.step(poses[i][0:20]*0)
        darwinenv.render()
        time.sleep(0.05)
        sim_poses.append(darwinenv.get_motor_pose())

    sim_poses = np.array(sim_poses)
    plt.figure()
    plt.plot(sim_poses[:, -3])
    plt.show()


