import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from dabs import *
import time

if __name__ == "__main__":
    filename = 'feedforward_target_hwindex.txt'

    feedforward_goals = np.loadtxt('data/'+filename)

    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_motor_delay([0] * 20)
    delay = darwin.read_motor_delay()
    print('Delay: ', delay)

    init_pose = darwin.read_motor_positions()
    print(init_pose)

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    t1=time.monotonic()
    for i in range(10):
        darwin.write_motor_goal(feedforward_goals[0])
    t2 = time.monotonic()

    print('10 multi write operation: ', t2-t1)

    t1 = time.monotonic()
    for i in range(10):
        darwin.read_motor_positions()
    t2 = time.monotonic()

    print('10 multi read operation: ', t2 - t1)



    darwin.disconnect()