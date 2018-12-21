import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from dabs import *
import time

if __name__ == "__main__":
    filename = 'sysid_data/velocity_test.txt'

    feedforward_goals = np.loadtxt('data/'+filename)

    savename = 'ff_' + filename.split('/')[-1].split('.')[0]

    darwin = BasicDarwin()

    darwin.connect()

    init_pose = darwin.read_motor_positions()
    print(init_pose)

    darwin.write_torque_enable(True)

    darwin.write_pid(32, 0, 16)

    darwin.write_motor_goal(SIM2HW_INDEX(RADIAN2VAL(feedforward_goals[0])))

    time.sleep(5)

    current_step = 0
    initial_time = time.monotonic()
    prev_time = time.monotonic()
    all_inputs = []
    all_velocities = []
    all_time = []
    all_actions = []
    all_gyros = []
    all_orientations = []
    cur_orientation = np.zeros(3)
    while current_step < len(feedforward_goals):
        if time.monotonic() - prev_time >= 0.05: # control every 50 ms
            tdif = time.monotonic() - prev_time
            prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)
            motor_pose = np.array(darwin.read_motor_positions())
            gyro = darwin.read_gyro()
            motor_velocity = np.array(darwin.read_motor_velocities())
            act = SIM2HW_INDEX(RADIAN2VAL(feedforward_goals[current_step]))
            darwin.write_motor_goal(act)
            current_step += 1
            ct = time.monotonic() - initial_time
            cur_orientation += VAL2RPS(gyro) * tdif

            all_actions.append(feedforward_goals[current_step])
            all_inputs.append(VAL2RADIAN(np.concatenate([HW2SIM_INDEX(motor_pose), HW2SIM_INDEX(motor_pose)])))  # to match the dimension in the normal policy
            all_velocities.append(SPEED_HW2SIM(HW2SIM_INDEX(motor_velocity)))
            all_time.append(ct - 0.05)
            all_gyros.append(VAL2RPS(gyro))
            all_orientations.append(np.copy(cur_orientation))

    darwin.disconnect()

    np.savetxt('data/hw_data/' + savename + '_saved_obs.txt', all_inputs)
    np.savetxt('data/hw_data/' + savename + '_saved_vels.txt', all_velocities)
    np.savetxt('data/hw_data/' + savename + '_saved_time.txt', all_time)
    np.savetxt('data/hw_data/' + savename + '_saved_action.txt', all_actions)
    np.savetxt('data/hw_data/' + savename + '_saved_gyro.txt', all_gyros)
    np.savetxt('data/hw_data/' + savename + '_saved_orientation.txt', all_orientations)