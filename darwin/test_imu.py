from darwin.basic_darwin import *
from darwin.darwin_utils import *
import numpy as np
from bno055_usb_stick_py import BnoUsbStick



if __name__ == '__main__':
    poses_to_test = []
    poses_to_test.append(RADIAN2VAL(np.zeros(20)))
    poses_to_test.append(SIM2HW_INDEX(0.5*(pose_squat_val + pose_stand_val)))
    poses_to_test.append(SIM2HW_INDEX(pose_squat_val))

    bno_usb_stick = BnoUsbStick()
    bno_usb_stick.activate_streaming()

    pose_id = 0

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_torque_enable(True)
    darwin.write_pid(32, 0, 16)

    motor_pose = darwin.read_motor_positions()

    darwin.write_motor_goal(SIM2HW_INDEX(pose_squat_val))

    while True:
        cmd = input('Input command. n: next pose, g: print current gyro reading, c: print gyro reading for 100 steps')
        if cmd == 'n':
            darwin.write_motor_goal(poses_to_test[pose_id])
            pose_id += 1
        if cmd == 'g':
            print(darwin.read_bno055_gyro())
        if cmd == 'c':
            for i in range(100):
                print(darwin.read_bno055_gyro())
