import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from dabs import *
import time

if __name__ == "__main__":
    SQUAT_POSE = SIM2HW_INDEX(pose_squat_val)
    STAND_POSE = SIM2HW_INDEX(pose_stand_val)
    MID_POSE = [int(v) for v in 0.5 * (SQUAT_POSE + STAND_POSE)]

    darwin = BasicDarwin()

    darwin.connect()

    init_pose = darwin.read_motor_positions()

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 8)

    darwin.write_motor_goal(SQUAT_POSE)
    current_pose = np.copy(SQUAT_POSE)

    time.sleep(5)

    current_step = 0
    prev_time = time.monotonic()
    while True:
        cmd = input("Choose initial pose: 1: Squat  2: Stand   3: Mid\nInput \'exit\' to close\n")

        if cmd == 'exit':
            break

        try:
            cmd = int(cmd)
        except ValueError:
            print('Format incorrect!')
            continue

        if cmd <= 3 and cmd >= 1:
            if cmd == 1:
                darwin.write_motor_goal(SQUAT_POSE)
                current_pose = np.copy(SQUAT_POSE)
            elif cmd == 2:
                darwin.write_motor_goal(STAND_POSE)
                current_pose = np.copy(STAND_POSE)
            elif cmd == 3:
                darwin.write_motor_goal(MID_POSE)
                current_pose = np.copy(MID_POSE)
            time.sleep(5)

            while True:
                if cmd == 1:
                    darwin.write_motor_goal(SQUAT_POSE)
                    current_pose = np.copy(SQUAT_POSE)
                elif cmd == 2:
                    darwin.write_motor_goal(STAND_POSE)
                    current_pose = np.copy(STAND_POSE)
                elif cmd == 3:
                    darwin.write_motor_goal(MID_POSE)
                    current_pose = np.copy(MID_POSE)

                cmd_sub = input("Choose pertubing dofs: 1: shoulder  2: thigh   3: ankle\nInput \'up\' to go to upper-level menu\n")
                if cmd_sub == 'up':
                    break

                try:
                    cmd_sub = int(cmd_sub)
                except ValueError:
                    print('Format incorrect!')
                    continue

                if cmd_sub <= 3 and cmd_sub >= 1:
                    if cmd_sub == 1:
                        plus_dofs = [0,5]
                        minus_dofs = [1,4]
                    elif cmd_sub == 2:
                        plus_dofs = [10]
                        minus_dofs = [11]
                    elif cmd_sub == 3:
                        plus_dofs = [14]
                        minus_dofs = [15]
                    while True:
                        cmd_sub_sub = input(
                            "Use \'+\' or \'-\' to control the dofs\nInput \'up\' to go to upper-level menu\n")
                        if cmd_sub_sub == 'up':
                            break
                        if len(cmd_sub_sub) == 0:
                            continue
                        if cmd_sub_sub[0] == '+':
                            rep = 1
                            if len(cmd_sub_sub) > 1:
                                rep = int(cmd_sub_sub[1:])
                            current_pose[plus_dofs] += 10 * rep
                            current_pose[minus_dofs] -= 10 * rep
                        if cmd_sub_sub[0] == '-':
                            rep = 1
                            if len(cmd_sub_sub) > 1:
                                rep = int(cmd_sub_sub[1:])
                            current_pose[plus_dofs] -= 10 * rep
                            current_pose[minus_dofs] += 10 * rep
                        darwin.write_motor_goal(current_pose)
                else:
                    print('Format incorrect!')

        else:
            print('Format incorrect!')
            continue



    darwin.disconnect()