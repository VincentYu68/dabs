import cma, sys, joblib
import numpy as np
from darwin.darwin_utils import *
import time
import os, errno

class StrategyOptimizer:
    def __init__(self, robot, policy, strategy_dim, timestep, eval_num = 2, save_dir = None, bno055_input = False):
        self.robot = robot
        self.policy = policy
        self.strategy_dim = strategy_dim
        self.timestep = timestep
        self.eval_num = eval_num
        self.rollout_num = 0
        self.bno055_input = bno055_input

        self.all_samples = []
        self.all_fitness = []
        self.best_x_hist = []
        self.best_f_hist = []
        self.best_f = 100000
        self.best_x = None

        self.save_dir = save_dir
        if save_dir is not None:
            try:
                os.makedirs(save_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def reset(self):
        self.rollout_num = 0

        self.best_f = 100000
        self.best_x = None
        self.all_samples = []
        self.all_fitness = []
        self.best_x_hist = []
        self.best_f_hist = []

    def cames_callback(self, es):
        self.best_x_hist.append(self.best_x)
        self.best_f_hist.append(self.best_f)
        if self.save_dir is not None:
            np.savetxt(self.save_dir + '/best_x_hist.txt', np.array(self.best_x_hist))
            np.savetxt(self.save_dir + '/best_f_hist.txt', np.array(self.best_f_hist))


    def fitness(self, x):
        app = np.copy(x)
        avg_perf = []
        print("Start a new trial with parameter: ", x)
        for _ in range(self.eval_num):
            dummpy_input = input("press any key to start trial " + str(_))

            self.robot.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(self.policy.get_initial_state())))

            time.sleep(2)
            gyro = self.robot.read_bno055_gyro()
            while np.any(np.abs(gyro[0:2]-np.array([0, 0.16])) > 0.1): # should be around 0.16 for half squat pose
                gyro = self.robot.read_bno055_gyro()
                print(gyro)
            print('Gyro initial state good, start in 4 second ...')
            time.sleep(4)

            prev_motor_pose = np.array(self.robot.read_motor_positions())
            current_step = 0
            initial_time = time.monotonic()
            prev_time = time.monotonic()
            max_step = 200
            if self.policy.interp_sch is not None:
                max_step = int(self.policy.interp_sch[-1][0] / self.timestep)
            while current_step < max_step:
                if time.monotonic() - prev_time >= self.timestep:  # control every 50 ms
                    # tdif = time.monotonic() - prev_time
                    prev_time = time.monotonic() - ((time.monotonic() - prev_time) - self.timestep)
                    motor_pose = np.array(self.robot.read_motor_positions())
                    obs_input = VAL2RADIAN(np.concatenate([HW2SIM_INDEX(prev_motor_pose), HW2SIM_INDEX(motor_pose)]))
                    ct = time.monotonic() - initial_time
                    gyro = self.robot.read_bno055_gyro()
                    if self.bno055_input:
                        obs_input = np.concatenate([obs_input, gyro])
                    obs_input = np.concatenate([obs_input, app])
                    act = self.policy.act(obs_input, ct)
                    self.robot.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(act)))

                    prev_motor_pose = np.copy(motor_pose)

                    current_step += 1
                    if np.any(np.abs(gyro[0:2]-np.array([0, 0.16])) > 1.0): # early terminate
                        print('Gyro: ', gyro[0:2])
                        break
            self.robot.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(self.policy.get_initial_state())))
            valid = False
            while not valid:
                try:
                    rollout_rew = float(input("Rollout terminated after " + str(current_step) + " steps.\nInput the estimated reward for this rollout: "))
                    valid = True
                except ValueError:
                    print('invalid reward')

            self.rollout_num += 1
            avg_perf.append(rollout_rew)

        if -np.mean(avg_perf) < self.best_f:
            self.best_x = np.copy(x)
            self.best_f = -np.mean(avg_perf)
        print('Sampled perf: ', np.mean(avg_perf))
        self.all_samples.append(x)
        self.all_fitness.append(np.mean(avg_perf))

        if self.save_dir is not None:
            # save all samples and other info
            np.savetxt(self.save_dir + '/all_samples.txt', np.array(self.all_samples))
            np.savetxt(self.save_dir + '/all_fitness.txt', np.array(self.all_fitness))

        return -np.mean(avg_perf)



    def optimize(self, maxiter = 20):
        init_guess = np.random.random(self.strategy_dim)
        init_std = 0.5
        bound = [0.0, 1.0]

        xopt, es = cma.fmin2(self.fitness, init_guess, init_std, options={'bounds': bound, 'maxiter': maxiter}
                             , callback=self.cames_callback)

        print('optimized: ', repr(xopt))
        print('Total rollout: ', self.rollout_num)

        return xopt