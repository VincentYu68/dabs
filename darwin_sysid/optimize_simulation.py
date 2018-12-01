import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.np_policy import *
import time
import cma, os, sys

class SysIDOptimizer:
    def __init__(self):
        self.darwinenv = DarwinPlain()
        self.darwinenv.toggle_fix_root(True)
        self.darwinenv.reset()

        self.solution_history = []
        self.value_history = []
        self.best_f = 10000000
        self.best_x = None

        # single pose results
        self.single_pose_sequences = []
        for i in range(100):
            if os.path.exists('data/sysid_data/single_pose/result_'+str(i)+'.txt'):
                data = np.loadtxt('data/sysid_data/single_pose/result_'+str(i)+'.txt')
                self.single_pose_sequences.append(data)

        # double pose results
        self.double_pose_sequences = []
        self.double_pose_actions = []
        for i in range(100):
            if os.path.exists('data/sysid_data/double_pose/result_pose_'+str(i)+'.txt'):
                pose_data = np.loadtxt('data/sysid_data/double_pose/result_pose_'+str(i)+'.txt')
                self.double_pose_sequences.append(pose_data)
                action_data = np.loadtxt('data/sysid_data/double_pose/result_action_' + str(i) + '.txt')
                self.double_pose_actions.append(action_data)

        self.optimize_dimension = len(self.darwinenv.simenv.env.param_manager.activated_param)

    def fitness(self, x):
        self.darwinenv.simenv.env.param_manager.set_simulator_parameters(x)

        total_single_error = 0
        total_single_size = 0
        for test_data in self.single_pose_sequences:
            error, size = self.evaluate_onepose_traj(test_data)
            total_single_error += error
            total_single_size += size

        total_double_error = 0
        total_double_size = 0
        for test_data, test_action in zip(self.double_pose_sequences, self.double_pose_actions):
            error, size = self.evaluate_doublepose_traj(test_data, test_action)
            total_double_error += error
            total_double_size += size

        total_single_error /= total_single_size
        total_double_error /= total_double_size

        loss = total_single_error + total_double_error

        if loss < self.best_f:
            self.best_x = np.copy(x)
            self.best_f = loss
        return loss

    def evaluate_onepose_traj(self, pose_data, render=False):
        self.darwinenv.set_pose(pose_data[0])
        sim_data = []
        total_square_error = 0
        for i in range(len(pose_data)-1):
            self.darwinenv.passive_step()
            if render:
                self.darwinenv.render()
            pose = np.array(self.darwinenv.get_motor_pose())
            sim_data.append(pose)
            total_square_error += np.sum(np.square(pose-pose_data[i+1]))
        return total_square_error, len(sim_data)

    def evaluate_doublepose_traj(self, pose_data, action_data, render=False):
        self.darwinenv.set_pose(pose_data[0])
        sim_data = []
        total_square_error = 0
        for i in range(len(action_data)-1):
            self.darwinenv.step(action_data[i])
            if render:
                self.darwinenv.render()
            pose = np.array(self.darwinenv.get_motor_pose())
            sim_data.append(pose)
            total_square_error += np.sum(np.square(pose - pose_data[i + 1]))
        return total_square_error, len(sim_data)

    def cmaes_callback(self, es):
        self.solution_history.append(self.best_x)
        self.value_history.append(self.best_f)

        print('Current best: ', repr(self.best_x), self.best_f)

    def optimize(self, maxiter = 50):
        init_guess = [0.5] * self.optimize_dimension
        init_std = 0.5

        bound = [0.0, 1.0]

        xopt, es = cma.fmin2(self.fitness, init_guess, init_std, options={'bounds': bound, 'maxiter': maxiter,
                                                                          }
                             , callback=self.cmaes_callback)

        print('optimized: ', repr(xopt))

if __name__ == "__main__":
    sysid_optimizer = SysIDOptimizer()

    sysid_optimizer.optimize()

    plt.plot(sysid_optimizer.best_f)
    plt.show()


