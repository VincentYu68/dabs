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

    init_pose = darwin.read_motor_positions()
    print(init_pose)

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    darwin.write_motor_goal(feedforward_goals[0])

    time.sleep(5)

    current_step = 0
    prev_time = time.monotonic()
    while current_step < len(feedforward_goals):
        if time.monotonic() - prev_time >= 0.05: # control every 50 ms
            prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)
            darwin.write_motor_goal(feedforward_goals[current_step])
            current_step += 1

    darwin.disconnect()