import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from dabs import *
import time

if __name__ == "__main__":
    filename = 'feedforward_target_hwindex.txt'

    feedforward_goals = np.loadtxt('data/'+filename)

    darwin = BasicDarwin(True)

    darwin.connect()

    darwin.write_motor_delay([0] * 20)
    delay = darwin.read_motor_delay()
    print('Delay: ', delay)

    init_pose = darwin.read_motor_positions()
    print(init_pose)

    squat_pose = SIM2HW_INDEX(np.array([2509, 2297, 1714, 1508, 1816, 2376,
                                    2047, 2171,
                                    2032, 2039, 2795, 648, 1241, 2040,   2041, 2060, 1281, 3448, 2855, 2073]))

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    t1=time.monotonic()
    for i in range(10):
        darwin.write_motor_goal(squat_pose)
    t2 = time.monotonic()

    print('10 multi write operation: ', t2-t1)

    '''time.sleep(5)
    darwin.write_torque_limit([0] * 20)
    time.sleep(5)
    darwin.write_torque_limit([1023] * 20)'''

    t1 = time.monotonic()
    for i in range(10):
        darwin.read_motor_positions()
    t2 = time.monotonic()

    print('10 multi read operation: ', t2 - t1)

    times = []
    for i in range(50):
        t1 = time.monotonic()
        darwin.read_motor_positions()
        darwin.write_motor_goal(squat_pose)
        darwin.read_bno055_gyro()
        #darwin.read_motor_voltages()
        t2 = time.monotonic()
        times.append(t2-t1)

    #print('List of all time intervals ', times)
    print('Average time interval: ', np.mean(times))
    print('Std time interval: ', np.std(times))

    times = []
    for i in range(50):
        t1 = time.monotonic()
        darwin.read_bno055_gyro()
        t2 = time.monotonic()
        times.append(t2 - t1)

    print('Average time for gyro read: ', np.mean(times))
    print('Std time for gyro read: ', np.std(times))

    darwin.disconnect()