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
    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(False)

    interp_sch = [[0.0, np.zeros(20)],
                  [10.0, np.zeros(20)]]
    darwinenv.simenv.env.interp_sch = interp_sch

    darwinenv.reset()
    q = darwinenv.robot.q
    q[1] = 0.3
    q[5] = -0.3
    darwinenv.robot.q = q

    sim_poses = []
    prev_obs = darwinenv.get_motor_pose()
    for i in range(200):
        current_obs = darwinenv.get_motor_pose()
        input_obs = np.concatenate([prev_obs, current_obs])

        darwinenv.step(np.zeros(20))
        darwinenv.render()
        time.sleep(0.05)
        sim_poses.append(input_obs)
        prev_obs = current_obs



    sim_poses = np.array(sim_poses)




