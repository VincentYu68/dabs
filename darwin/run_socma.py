# optimize the strategy using cma

import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from darwin.np_policy import *
import joblib
from dabs import *
import time
import os, errno
from darwin.strategy_optimizer import *

if __name__ == "__main__":
    filename = 'sqstsq_nolimvel_UP4d.pkl'

    walk_motion = False
    singlefoot_motion = False
    crawl_motion = False

    gyro_input = 0

    pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                               2047, 2171,
                               2032, 2039, 2795, 648, 1241, 2040, 2041, 2060, 1281, 3448, 2855, 2073])
    pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                               2048, 2048,
                               2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])

    pose_squat = VAL2RADIAN(pose_squat_val)
    pose_stand = VAL2RADIAN(pose_stand_val)

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                  [2.0, pose_squat],
                  [3.5, pose_squat],
                  [4.0, pose_stand],
                  [5.0, pose_stand],
                  [7.0, pose_squat],
                  ]

    if walk_motion or crawl_motion:
        if walk_motion:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe.txt')
        else:
            rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe_crawl.txt')
        interp_sch = [[0.0, rig_keyframe[0]]]
        interp_time = 0.5
        for i in range(10):
            for k in range(1, len(rig_keyframe)):
                interp_sch.append([interp_time, rig_keyframe[k]])
                interp_time += 0.5
        interp_sch.append([interp_time, rig_keyframe[0]])

    if singlefoot_motion:
        rig_keyframe = np.loadtxt('data/rig_data/rig_keyframe2.txt')
        interp_sch = [[0.0, rig_keyframe[0]],
                      [2.0, rig_keyframe[1]],
                      [6.0, rig_keyframe[1]]]

    policy = NP_Policy(interp_sch, 'data/'+filename, discrete_action=True,
                       action_bins=np.array([11] * 20), delta_angle_scale=0.3, action_filter_size=5)

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    optimizer = StrategyOptimizer(darwin, policy, 4, 1)

    optimizer.optimize()

    darwin.disconnect()














