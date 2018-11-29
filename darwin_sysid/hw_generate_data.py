import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from dabs import *
import time

if __name__ == "__main__":
    darwin = BasicDarwin()

    # load data
    single_poses = np.loadtxt('data/sysid_data/single_poses.txt')
    double_poses = np.loadtxt('data/sysid_data/double_poses.txt')

    # hyperparameters
    num_steps_per_trial = 100

    darwin.connect()

    darwin.write_motor_delay([0] * 20)
    delay = darwin.read_motor_delay()

    darwin.write_torque_enable(True)

    print('Start in 5 seconds ...')
    time.sleep(5)

    for i in range(len(single_poses)):
        darwin.write_pid(32, 0, 16)
        pose = RADIAN2VAL(SIM2HW_INDEX(single_poses[i]))
        darwin.write_motor_goal(pose)
        time.sleep(2)

        prev_time = time.monotonic()
        darwin.write_pid(0, 0, 0)
        pose_data = [VAL2RADIAN(HW2SIM_INDEX(np.array(darwin.read_motor_positions())))]
        while len(pose_data) < num_steps_per_trial:
            if time.monotonic() - prev_time >= 0.05:  # control every 50 ms
                prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)

                motor_pose = np.array(darwin.read_motor_positions())
                motor_pose_sim = VAL2RADIAN(HW2SIM_INDEX(motor_pose))
                pose_data.append(motor_pose_sim)
        np.savetxt('data/sysid_data/single_pose/result_'+str(i)+'.txt', pose_data)

    print('Finished single poses.')

    darwin.write_pid(32, 0, 16)
    for i in range(len(double_poses)):
        pose1 = RADIAN2VAL(SIM2HW_INDEX(double_poses[i][0:20]))
        pose2 = RADIAN2VAL(SIM2HW_INDEX(double_poses[i][20:]))
        darwin.write_motor_goal(pose1)
        time.sleep(2)

        pose_data = [VAL2RADIAN(HW2SIM_INDEX(np.array(darwin.read_motor_positions())))]
        action_data = [double_poses[i][0:20]]
        darwin.write_motor_goal(pose1)
        prev_time = time.monotonic()
        current_step = 1
        while current_step < num_steps_per_trial:
            if time.monotonic() - prev_time >= 0.05:  # control every 50 ms
                prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)

                motor_pose = np.array(darwin.read_motor_positions())
                motor_pose_sim = VAL2RADIAN(HW2SIM_INDEX(motor_pose))
                pose_data.append(motor_pose_sim)

                next_pose = pose1 * (1-current_step*1.0/(num_steps_per_trial-1)) + pose2 * (current_step*1.0/(num_steps_per_trial-1))
                next_pose = np.clip(next_pose + np.random.uniform(-20, 20, 20), HW_JOINT_LOW_BOUND_VAL, HW_JOINT_UP_BOUND_VAL)
                action_data.append(VAL2RADIAN(HW2SIM_INDEX(next_pose)))
                darwin.write_motor_goal(next_pose)

        np.savetxt('data/sysid_data/double_pose/result_pose_'+str(i)+'.txt', pose_data)
        np.savetxt('data/sysid_data/double_pose/result_action_' + str(i) + '.txt', action_data)

    print('Finished double poses.')

    darwin.disconnect()