import numpy as np

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
import time, os, errno
from darwin.darwin_utils import *

if __name__ == "__main__":
    save_path = 'data/rig_data'

    preset_poses = [VAL2RADIAN(np.array([2509, 2297, 1714, 1508, 1816, 2376,
                                    2047, 2171,
                                    2032, 2039, 2795, 648, 1231, 2040, 2041, 2060, 1281, 3448, 2855, 2073])),
                    VAL2RADIAN(np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                    2048, 2048,
                                    2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])),
                    VAL2RADIAN(np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                         2048, 2048,
                                         2032, 2039, 2795, 568, 1231, 2040, 2048, 2048, 2048, 2048, 2048, 2048])),
                    VAL2RADIAN(np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                          2048, 2048,
                                          2048, 2048, 2048, 2048, 2048, 2048, 2041, 2060, 1281, 3525, 2855, 2073]))]


    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if os.path.exists(save_path + '/rig_keyframe2.txt'):
        preset_poses += np.loadtxt(save_path + '/rig_keyframe2.txt').tolist()

    saved_poses = []

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)

    darwinenv.reset()

    current_pose = darwinenv.get_motor_pose()
    current_dof = 0

    current_preset = 0

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
                    com = darwinenv.robot.dofs[current_dof+6].joint.position_in_world_frame()
                    darwinenv.simenv.env.dart_world.skeletons[2].q = [0, 0, 0, com[0], com[1], com[2]]
                    current_pose = np.clip(current_pose, SIM_JOINT_LOW_BOUND_RAD, SIM_JOINT_UP_BOUND_RAD)


                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'a':
                        saved_poses.append(current_pose)
                        preset_poses.append(current_pose)
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'd':
                        if len(saved_poses) > 0:
                            saved_poses.pop(len(saved_poses)-1)
                            preset_poses.pop(len(preset_poses)-1)

                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'p':
                        # start playing the motion
                        for i in range(len(saved_poses)-1):
                            interv = 50
                            for j in range(interv):
                                darwinenv.set_pose(saved_poses[i] * (1-j*1.0/interv) + saved_poses[i+1] * j * 1.0 / interv)
                                darwinenv.render()
                                time.sleep(0.05)
                        # resume the pose
                        darwinenv.set_pose(current_pose)

                    # save the motion
                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b's':
                        np.savetxt(save_path+'/rig_keyframe2.txt', np.array(saved_poses))


                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b't':
                        current_pose = preset_poses[current_preset]
                        current_preset += 1
                        if current_preset >= len(preset_poses):
                            current_preset = 0

                    if darwinenv.simenv.env._get_viewer().key_being_pressed == b'm':
                        # mirror the current pose
                        current_pose = np.dot(darwinenv.pose_mirror_matrix, current_pose)

                    time.sleep(0.2)



