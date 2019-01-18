import numpy as np
from darwin.darwin_utils import *
from darwin.basic_darwin import *
from dabs import *
import time, os, joblib

if __name__ == "__main__":
    darwin = BasicDarwin()

    # load data
    data_dir = 'data/sysid_data/generic_motion/'
    specific_traj = 'standup'
    all_trajs = [joblib.load(data_dir + file) for file in os.listdir(data_dir) if '.pkl' in file and 'path' in file and specific_traj in file]
    all_files = [data_dir + file for file in os.listdir(data_dir) if '.pkl' in file and 'path' in file and specific_traj in file]

    darwin.connect()

    darwin.write_torque_enable(True)

    input('Press enter to start ...')
    time.sleep(1)

    for i in range(len(all_trajs)):
        control_dt = all_trajs[i]['control_dt']
        keyframes = all_trajs[i]['keyframes']
        traj_time = all_trajs[i]['total_time']
        darwin.write_pid(32, 0, 16)   # subject to change

        init_pose = RADIAN2VAL(SIM2HW_INDEX(keyframes[0][1]))
        darwin.write_motor_goal(init_pose)
        time.sleep(2)

        pose_data = [VAL2RADIAN(HW2SIM_INDEX(np.array(darwin.read_motor_positions())))]
        vel_data = [SPEED_HW2SIM(HW2SIM_INDEX(np.array(darwin.read_motor_velocities())))]
        gyro_data = [VAL2RPS(np.array(darwin.read_gyro()))]
        prev_time = time.monotonic() - control_dt # minus control dt so that the first loop would take an action
        initial_time = time.monotonic()
        while time.monotonic() - initial_time < traj_time:
            if time.monotonic() - prev_time >= control_dt:  # control every control_dt ms
                prev_time = time.monotonic() - ((time.monotonic() - prev_time) - control_dt)

                action = keyframes[0][1]
                for kf in keyframes:
                    if time.monotonic() - initial_time >= kf[0]:
                        action = kf[1]

                darwin.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(action)))

                motor_pos_velocity = np.array(darwin.read_motor_positionvelocities())
                #motor_velocity = SPEED_HW2SIM(HW2SIM_INDEX(np.array(darwin.read_motor_velocities())))
                #motor_pose = VAL2RADIAN(HW2SIM_INDEX(np.array(darwin.read_motor_positions())))
                gyro = VAL2RPS(np.array(darwin.read_gyro()))

                motor_pose = VAL2RADIAN(HW2SIM_INDEX(motor_pos_velocity[::2]))
                motor_velocity = SPEED_HW2SIM(HW2SIM_INDEX(motor_pos_velocity[1::2]))

                pose_data.append(motor_pose)
                vel_data.append(motor_velocity)
                gyro_data.append(gyro)
        all_trajs[i]['pose_data'] = pose_data
        all_trajs[i]['vel_data'] = vel_data
        all_trajs[i]['gyro_data'] = gyro_data

        joblib.dump(all_trajs[i], all_files[i], compress=True)

        print('Start next trajectory in 3 seconds ...')
        time.sleep(3)
        if specific_traj == 'standup':
            input('Press enter to start next traj...')

    print('Finished single poses.')


    darwin.disconnect()