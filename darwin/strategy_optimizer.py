import cma, sys, joblib
import numpy as np
from darwin.darwin_utils import *
import time

class StrategyOptimizer:
    def __init__(self, robot, policy, strategy_dim, eval_num = 2):
        self.robot = robot
        self.policy = policy
        self.strategy_dim = strategy_dim
        self.eval_num = eval_num
        self.sample_num = 0
        self.rollout_num = 0

        self.solution_history = []
        self.best_f = 100000
        self.best_x = None

    def reset(self):
        self.sample_num = 0
        self.rollout_num = 0

        self.best_f = 100000
        self.best_x = None
        self.solution_history = []

    def fitness(self, x):
        app = np.copy(x)
        avg_perf = []
        print("Start a new trial with parameter: ", x)
        for _ in range(self.eval_num):
            dummpy_input = input("press any key to start trial " + str(_))

            self.robot.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(self.policy.get_initial_state())))

            time.sleep(5)

            prev_motor_pose = np.array(self.robot.read_motor_positions())
            current_step = 0
            initial_time = time.monotonic()
            prev_time = time.monotonic()
            while current_step < 200:
                if time.monotonic() - prev_time >= 0.05:  # control every 50 ms
                    # tdif = time.monotonic() - prev_time
                    prev_time = time.monotonic() - ((time.monotonic() - prev_time) - 0.05)
                    motor_pose = np.array(self.robot.read_motor_positions())
                    obs_input = VAL2RADIAN(np.concatenate([HW2SIM_INDEX(prev_motor_pose), HW2SIM_INDEX(motor_pose)]))
                    ct = time.monotonic() - initial_time
                    act = self.policy.act(np.concatenate([obs_input, app]), ct)
                    self.robot.write_motor_goal(RADIAN2VAL(SIM2HW_INDEX(act)))

                    prev_motor_pose = np.copy(motor_pose)

                    current_step += 1

            valid = False
            while not valid:
                try:
                    rollout_rew = float(input("Input the estimated reward for this rollout"))
                    valid = True
                except ValueError:
                    print('invalid reward')

            self.rollout_num += 1
            avg_perf.append(rollout_rew)

        if -np.mean(avg_perf) < self.best_f:
            self.best_x = np.copy(x)
            self.best_f = -np.mean(avg_perf)
        print('Sampled perf: ', np.mean(avg_perf))
        return -np.mean(avg_perf)



    def optimize(self, maxiter = 20):
        init_guess = np.random.random(self.strategy_dim)
        init_std = 0.5
        bound = [0.0, 1.0]

        xopt, es = cma.fmin2(self.fitness, init_guess, init_std, options={'bounds': bound, 'maxiter': maxiter})

        print('optimized: ', repr(xopt))
        print('Total rollout: ', self.rollout_num)

        return xopt