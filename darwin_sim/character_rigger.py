import numpy as np

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
import time, os, errno
from darwin.darwin_utils import *

if __name__ == "__main__":
    save_path = 'data/rig_data'

    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    saved_poses = []

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)

    darwinenv.reset()

    current_pose = darwinenv.get_motor_pose()
    current_dof = 0

    while True:
        darwinenv.set_pose(current_pose)

        darwinenv.render()

        if darwinenv.simenv.env._get_viewer() is not None:
            if hasattr(darwinenv.simenv.env._get_viewer(), 'key_being_pressed'):
                if darwinenv.simenv.env._get_viewer().key_being_pressed is not None:
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'-':
                        current_dof -= 1
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'+':
                        current_dof += 1
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'<':
                        current_pose[current_dof] -= 0.008 * 10
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'>':
                        current_pose[current_dof] += 0.008 * 10

                    if current_dof < 0:
                        current_dof = 19
                    if current_dof > 19:
                        current_dof = 0
                    current_pose = np.clip(current_pose, SIM_JOINT_LOW_BOUND_RAD, SIM_JOINT_UP_BOUND_RAD)
                    time.sleep(0.1)

