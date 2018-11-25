import numpy as np

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
import time

if __name__ == "__main__":
    poses1 = np.loadtxt('data/sim_saved_obs.txt')
    poses2 = np.loadtxt('data/fixed_ref_saved_obs.txt')

    loop_size = np.min([len(poses1), len(poses2)])

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)

    darwinenv.reset()

    current_step = 0
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
                    time.sleep(0.1)

