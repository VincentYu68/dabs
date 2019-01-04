import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.np_policy import *
import time
import cma, os, sys, joblib
from mpi4py import MPI

class SysIDOptimizer:
    def __init__(self, data_dir, velocity_weight = 1.0, regularization = 0.001):
        self.data_dir = data_dir
        self.velocity_weight = velocity_weight
        self.regularization = regularization
        self.all_trajs = [joblib.load(data_dir + file) for file in os.listdir(data_dir) if '.pkl' in file]

        self.darwinenv = DarwinPlain()
        self.darwinenv.toggle_fix_root(True)
        self.darwinenv.reset()

        self.solution_history = []
        self.value_history = []
        self.best_f = 10000000
        self.best_x = None

        self.optimize_dimension = np.sum(self.darwinenv.MU_DIMS[self.darwinenv.ACTIVE_MUS])

        self.thread_id = MPI.COMM_WORLD.Get_rank()
        self.total_threads = MPI.COMM_WORLD.Get_size()

    def fitness(self, x):
        self.darwinenv.set_mu(x)

        total_positional_error = 0
        total_velocity_error = 0

        total_step = 0
        for traj in self.all_trajs:
            control_dt = traj['control_dt']
            keyframes = traj['keyframes']
            traj_time = traj['total_time']

            hw_pose_data = traj['pose_data']
            hw_vel_data = traj['vel_data']
            self.darwinenv.set_control_timestep(control_dt)

            self.darwinenv.reset()
            self.darwinenv.set_pose(keyframes[0][1])
            sim_poses = [self.darwinenv.get_motor_pose()]
            sim_vels = [self.darwinenv.get_motor_velocity()]
            while self.darwinenv.time <= traj_time:
                act = keyframes[0][1]
                for kf in traj:
                    if self.darwinenv.time >= kf[0]:
                        act = kf[1]
                self.darwinenv.step(act)
                pose = self.darwinenv.get_motor_pose()
                vel = self.darwinenv.get_motor_velocity()
                sim_poses.append(pose)
                sim_vels.append(vel)
            max_step = np.max([len(sim_poses), len(hw_pose_data)])
            total_step += max_step
            total_positional_error += np.sum(np.square(np.array(hw_pose_data)[1:max_step] - np.array(sim_poses)[1:max_step]))
            total_velocity_error += np.sum(
                np.square(np.array(hw_vel_data)[1:max_step] - np.array(sim_vels)[1:max_step]))

        loss = (total_positional_error + total_velocity_error * self.velocity_weight) / total_step +
                self.regularization * np.linalg.norm(x)

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

        es = cma.CMAEvolutionStrategy(init_guess, init_std, options={'bounds': bound, 'maxiter': maxiter,})
        while not es.stop():
            solutions = es.ask()
            # evaluate fitness for each sample
            evaluated_fitness = {str(self.thread_id): []}
            num_per_thread = int(np.ceil(len(solutions) / self.total_threads))
            for i in range(num_per_thread):
                sol_id = self.thread_id * num_per_thread + i
                if sol_id >= len(solutions):
                    break
                evaluated_fitness[str(self.thread_id)].append(self.fitness(solutions[sol_id]))

            all_evaluated_fitness = {k: v for d in MPI.COMM_WORLD.allgather(evaluated_fitness) for k, v in d.items()}
            merged_evaluated_fitness = []
            for i in range(self.total_threads):
                merged_evaluated_fitness += all_evaluated_fitness[str(i)]

            es.tell(solutions, merged_evaluated_fitness)
            es.logger.add()  # write data to disc to be plotted
            es.disp()
            self.cmaes_callback(es)
        es.result_pretty()
        xopt = es.best.x

        print('optimized: ', repr(xopt))

if __name__ == "__main__":
    sysid_optimizer = SysIDOptimizer('data/sysid_data/generic_motion/')

    sysid_optimizer.optimize()

    plt.plot(sysid_optimizer.value_history)
    plt.show()


