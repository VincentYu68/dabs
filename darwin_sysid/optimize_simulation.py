import matplotlib
matplotlib.use('Agg')
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
    def __init__(self, env, data_dir, velocity_weight = 1.0, regularization = 0.001, specific_data = '.', save_dict='', save_name='',
                 minibatch = 0, init_guess = None, random_subset = None):
        self.data_dir = data_dir
        self.save_dict = save_dict
        self.save_name = save_name
        self.velocity_weight = velocity_weight
        self.regularization = regularization
        self.all_trajs = [joblib.load(data_dir + file) for file in os.listdir(data_dir) if '.pkl' in file and 'path' in file and specific_data in file]

        self.darwinenv = env

        self.minibatch = minibatch
        self.random_subset = random_subset

        if self.random_subset is not None:
            self.all_trajs = np.random.permutation(self.all_trajs)
            self.all_trajs = self.all_trajs[0:int(self.random_subset * len(self.all_trajs))]

        self.init_guess = init_guess

        self.solution_history = []
        self.value_history = []
        self.best_f = 10000000
        self.best_x = None

        self.optimize_dimension = np.sum(self.darwinenv.MU_DIMS[self.darwinenv.ACTIVE_MUS])

        self.thread_id = MPI.COMM_WORLD.Get_rank()
        self.total_threads = MPI.COMM_WORLD.Get_size()

    def meta_fitness(self, x): # x would be mean and std of cma
        pass

    def fitness(self, x, iter_num = -1):
        self.darwinenv.set_mu(x)

        total_positional_error = 0
        total_velocity_error = 0

        total_step = 0
        if self.minibatch == 0 or iter_num == -1:
            traj_to_test = self.all_trajs
        else:
            traj_to_test = self.all_trajs[self.minibatch * iter_num:np.min([self.minibatch*(iter_num+1), len(self.all_trajs)])]
        for i, traj in enumerate(traj_to_test):
            control_dt = traj['control_dt']
            keyframes = traj['keyframes']
            traj_time = traj['total_time']

            hw_pose_data = traj['pose_data']
            hw_vel_data = traj['vel_data']
            self.darwinenv.set_control_timestep(control_dt)

            fix_root = True
            if 'fix_root' in traj:
                if not traj['fix_root']:
                    fix_root = False

            self.darwinenv.toggle_fix_root(fix_root)

            self.darwinenv.reset()
            if not fix_root:
                self.darwinenv.set_root_dof(self.darwinenv.get_root_dof() + np.array([0, 0, 0, 0, 0, -0.075]))
            self.darwinenv.set_pose(keyframes[0][1])

            sim_poses = [self.darwinenv.get_motor_pose()]
            sim_vels = [self.darwinenv.get_motor_velocity()]
            step = 0
            while self.darwinenv.time <= traj_time+0.004:
                act = keyframes[0][1]
                for kf in keyframes:
                    if self.darwinenv.time >= kf[0] - 0.00001:
                        act = kf[1]
                pose = self.darwinenv.get_motor_pose()
                self.darwinenv.step(act)
                sim_poses.append(pose)
                step += 1
                if not fix_root:
                    # penalize offset in x direction for now, in general should use imu reading
                    total_positional_error += (np.clip(np.abs(darwinenv.robot.C[0]), 0.06, 100) - 0.06) * 20
            if not fix_root and self.darwinenv.check_collision(['MP_ANKLE2_L', 'MP_ANKLE2_R']):
                total_positional_error += 10
            max_step = np.min([len(sim_poses), len(hw_pose_data)])
            total_step += max_step
            total_positional_error += np.clip(np.sum(
                np.abs(np.array(hw_pose_data)[1:max_step] - np.array(sim_poses)[1:max_step])), 0, 1000)
            total_velocity_error += 0
        #print('total step: ', total_step)
        loss = (total_positional_error + total_velocity_error * self.velocity_weight) / total_step + \
                self.regularization * np.sum(x**2)

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
        all_best_xs = MPI.COMM_WORLD.allgather(self.best_x)
        self.best_f = 100000
        for x in all_best_xs:
            eval = self.fitness(x)
            if eval < self.best_f:
                self.best_x = np.copy(x)
                self.best_f = eval

        self.solution_history.append(self.best_x)
        self.value_history.append(self.best_f)

        print('Current best: ', repr(self.best_x), self.best_f)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.darwinenv.set_mu(self.best_x)
            print('optimized: ', self.darwinenv.MU_UNSCALED, self.best_f)
            opt_result = [self.best_x, self.darwinenv.MU_UNSCALED]
            #np.savetxt(self.data_dir+'/opt_result' + self.save_app + '.txt', opt_result)

    def optimize(self, maxiter = 100):
        if self.init_guess is None:
            init_guess = [0.5] * self.optimize_dimension
            init_std = 0.25
        else:
            init_guess = np.copy(self.init_guess)
            init_std = 0.05

        bound = [0.0, 1.0]

        num_segs = 1
        if self.minibatch > 0:
            num_segs = int(np.ceil(len(self.all_trajs) / self.minibatch))
            maxiter *= num_segs

        es = cma.CMAEvolutionStrategy(init_guess, init_std, {'bounds': bound, 'maxiter': maxiter,})

        pop_size = es.popsize
        sol_id_to_evaluate = []
        cur_id = 0
        while cur_id < pop_size:
            if cur_id % (self.total_threads) == self.thread_id:
                sol_id_to_evaluate.append(cur_id)
            cur_id += 1
        print('Thread ' + str(self.thread_id) + ' evaluates: ' + str(sol_id_to_evaluate))

        iter_num = 0
        while not es.stop():
            if self.minibatch > 0 and iter_num % num_segs == 0:
                # reshuffle the data
                self.all_trajs = np.random.permutation(self.all_trajs)
            solutions = es.ask()
            solutions = MPI.COMM_WORLD.bcast(solutions, root=0)
            # evaluate fitness for each sample, spread out to multiple threads
            evaluated_fitness = {str(self.thread_id): []}
            for sol_id in sol_id_to_evaluate:
                eval = self.fitness(solutions[sol_id], iter_num%num_segs)
                evaluated_fitness[str(self.thread_id)].append(eval)

            all_evaluated_fitness = {k: v for d in MPI.COMM_WORLD.allgather(evaluated_fitness) for k, v in d.items()}
            merged_evaluated_fitness = []
            for i in range(self.total_threads):
                merged_evaluated_fitness += all_evaluated_fitness[str(i)]

            es.tell(solutions, merged_evaluated_fitness)
            es.logger.add()  # write data to disc to be plotted
            es.disp()
            if iter_num % (num_segs * 20) == 0:
                self.cmaes_callback(es)
            iter_num += 1
        es.result_pretty()
        xopt = self.best_x

        self.darwinenv.set_mu(xopt)
        print('optimized: ', self.darwinenv.MU_UNSCALED, self.best_f)
        opt_result = [xopt, self.darwinenv.MU_UNSCALED]
        return opt_result

if __name__ == "__main__":
    data_dir = 'data/sysid_data/generic_motion/'
    #savename = '01only_vel0_minibatch3_NNmotor'

    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(True)
    darwinenv.reset()

    group_run_result = {}
    group_run_result['variations'] = [darwinenv.VARIATIONS[i] for i in darwinenv.ACTIVE_MUS]

    all_savename = 'all_vel0_pid_standup'
    sysid_optimizer = SysIDOptimizer(darwinenv, data_dir, velocity_weight=0.0, specific_data='.', save_dict=all_savename, save_name = 'all',
                                     minibatch=0)
    result_all = sysid_optimizer.optimize(maxiter=500)
    group_run_result['all_sol'] = result_all
    group_run_result['all_lc'] = sysid_optimizer.value_history

    for i in range(0):
        savename = '1o3subset_vel0_pid_warmstartall_' + str(i)
        sysid_optimizer = SysIDOptimizer(darwinenv, data_dir, velocity_weight=0.0, specific_data='.', save_app=savename,
                                         minibatch=0,
                                         init_guess=result_all[0], random_subset=0.1)
        result = sysid_optimizer.optimize(maxiter=100)

        group_run_result['subset_'+str(i)+'_sol'] = result
        group_run_result['subset_'+str(i)+'_lc'] = sysid_optimizer.value_history

    joblib.dump(group_run_result, data_dir+all_savename+'.pkl', compress=True)

