import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from dabs import *
import time
import os, errno

if __name__ == "__main__":
    try:
        os.makedirs('data/voltage_data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_motor_delay([0] * 20)
    delay = darwin.read_motor_delay()
    print('Delay: ', delay)

    init_pose = darwin.read_motor_positions()
    print(init_pose)

    neutral_pose = SIM2HW_INDEX(np.array([2048]*20))

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    # from squat to stand
    poses_squatstand = []
    voltages_squatstand = []
    darwin.write_motor_goal(neutral_pose)
    for i in range(100):
        poses_squatstand.append(darwin.read_motor_positions())
        voltages_squatstand.append(darwin.read_motor_voltages())
        time.sleep(0.005)

    neutral_pose[0] += 300
    poses_armmove = []
    voltages_armmove = []
    darwin.write_motor_goal(neutral_pose)
    for i in range(100):
        poses_armmove.append(darwin.read_motor_positions())
        voltages_armmove.append(darwin.read_motor_voltages())
        time.sleep(0.005)

    poses_nopid = []
    voltages_nopid = []
    darwin.write_pid(0, 0, 0)
    for i in range(100):
        poses_nopid.append(darwin.read_motor_positions())
        voltages_nopid.append(darwin.read_motor_voltages())
        time.sleep(0.005)

    np.savetxt('data/voltage_data/poses1.txt', poses_squatstand)
    np.savetxt('data/voltage_data/voltage1.txt', voltages_squatstand)
    np.savetxt('data/voltage_data/poses2.txt', poses_armmove)
    np.savetxt('data/voltage_data/voltage2.txt', voltages_armmove)
    np.savetxt('data/voltage_data/poses3.txt', poses_nopid)
    np.savetxt('data/voltage_data/voltage3.txt', voltages_nopid)

    darwin.disconnect()