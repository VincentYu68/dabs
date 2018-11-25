################################################################################
#           Controller of the Darwin Squat-Stand task using numpy              #
#           Note: all joint data used in this file uses the dof indexing with  #
#                 from the simulation environment, not the hardware.           #
################################################################################

import joblib
import numpy as np

from darwin.darwin_utils import *

# Class for a neural network model in numpy
class NP_Net:
    def __init__(self, nvec = None):
        self.obrms_mean = None    # for observation running mean std
        self.obrms_std = None     # for observation running mean std
        self.nn_params = []       # stores the neural net parameters in the form of [[W0, b0], [W1, b1], ... [Wn, bn]]
        self.nvec = nvec          # None if continuous action, otherwise discrete action in the form of
                                  # [numbins, numbins, ... numbins]

    def load_from_file(self, fname):
        params = joblib.load(fname)

        pol_scope = list(params.keys())[0][0:list(params.keys())[0].find('/')]
        obrms_runningsumsq = params[pol_scope+'/obfilter/runningsumsq:0']
        obrms_count = params[pol_scope+'/obfilter/count:0']
        obrms_runningsum = params[pol_scope+'/obfilter/runningsum:0']

        self.obrms_mean = obrms_runningsum / obrms_count
        self.obrms_std = np.sqrt(np.clip(obrms_runningsumsq / obrms_count - (self.obrms_mean**2), 1e-2, 1000000))

        for i in range(10): # assume maximum layer size of 10
            if pol_scope+'/pol/fc'+str(i)+'/kernel:0' in params:
                W = params[pol_scope+'/pol/fc'+str(i)+'/kernel:0']
                b = params[pol_scope+'/pol/fc'+str(i)+'/bias:0']
                self.nn_params.append([W, b])
        W_final = params[pol_scope + '/pol/final/kernel:0']
        b_final = params[pol_scope + '/pol/final/bias:0']
        self.nn_params.append([W_final, b_final])

    def get_output(self, input, activation = np.tanh):
        assert self.obrms_mean is not None

        last_out = np.clip((input - self.obrms_mean) / self.obrms_std, -5.0, 5.0)

        for i in range(len(self.nn_params)-1):
            last_out = activation(np.dot(self.nn_params[i][0].T, last_out) + self.nn_params[i][1])
        out = np.dot(self.nn_params[-1][0].T, last_out) + self.nn_params[-1][1]

        if self.nvec is None:
            return out
        else:
            # convert for discrete output
            splitted_out = np.split(out, np.cumsum(self.nvec)[0:-1])
            discrete_out = np.array([np.argmax(prob) for prob in splitted_out])
            return discrete_out

# Class for a neural network policy in numpy
# Includes the action filtering and pose interpolation
class NP_Policy:
    # interp_sch makes the feed-forward motion
    # interp_sch contains the timing and pose id throughout the trajectory
    def __init__(self, interp_sch, param_file, discrete_action, action_bins, delta_angle_scale):
        self.interp_sch = interp_sch
        self.obs_cache = []
        self.action_cache = []
        self.action_filter_size = 5
        self.net = NP_Net()
        self.net.load_from_file(param_file)
        self.discrete_action = discrete_action
        self.delta_angle_scale = delta_angle_scale
        if discrete_action:
            self.net.nvec = action_bins

    # Get the initial state for the robot
    # RETURN: a 20d vector for the robot pose
    def get_initial_state(self):
        return self.interp_sch[0][1]

    # Reset the state of the policy
    # This is needed because the action cache essentially forms a memory in the policy
    def reset(self):
        self.action_cache = []

    # Return the action to be taken by the robot given the observation and current time
    # INPUT: o, a 40d vector containing the pose and velocity of the robot
    #        t, current time in seconds, used to get the reference pose
    # RETURN: a 20d vector containing the target angle (in radians) for the robot joints
    def act(self, o, t):
        # get network output action
        new_action = self.net.get_output(o)
        if self.discrete_action:
            new_action = new_action * 1.0 / np.floor(self.net.nvec/2.0) - 1.0

        self.action_cache.append(new_action)
        if len(self.action_cache) > self.action_filter_size:
            self.action_cache.pop(0)
        filtered_action = np.mean(self.action_cache, axis=0)

        # get feedforward action
        clamped_control = np.clip(filtered_action, -1, 1)
        self.ref_target = self.interp_sch[0][1]
        for i in range(len(self.interp_sch) - 1):
            if t >= self.interp_sch[i][0] and t < self.interp_sch[i + 1][0]:
                ratio = (t - self.interp_sch[i][0]) / (self.interp_sch[i + 1][0] - self.interp_sch[i][0])
                self.ref_target = ratio * self.interp_sch[i + 1][1] + (1 - ratio) * self.interp_sch[i][1]
        if t > self.interp_sch[-1][0]:
            self.ref_target = self.interp_sch[-1][1]

        # combine policy output and keyframe interpolation to get the target joint positions
        target_pose = self.ref_target + clamped_control * self.delta_angle_scale
        target_pose = np.clip(target_pose, SIM_JOINT_LOW_BOUND_RAD, SIM_JOINT_UP_BOUND_RAD)

        return target_pose




def toRobot(positions):
    # reorder joints
    index = [3,0,4,1,5,2,14,8,15,9,16,10,17,11,18,12,19,13,6,7]
    # convert from radians to int
    robotState = np.zeros(len(positions))
    for i in range(len(positions)):
        robotState[i] = int(positions[i]*180*(1/(np.pi*0.088))) + 2048

    return robotState[index].astype(int)


#######################################
# test the file in pydart2 simulation #
#######################################
if __name__ == "__main__":
    import pydart2 as pydart
    import gym

    env = gym.make('DartDarwinSquat-v1') # use the dart_world in the gym environment to avoid copying the data
    env.reset()
    dart_world = env.env.dart_world

    class Controller(object):
        def __init__(self, world, policy):
            self.world = world
            self.target = None
            self.kp = np.array([2.1, 1.79, 4.93,
                                2.0, 2.02, 1.98,
                                2.2, 2.06,
                                148, 152, 150, 136, 153, 102,
                                151, 151.4, 150.45, 151.36, 154, 105.2])
            self.kd = np.array([0.21, 0.23, 0.22,
                                0.25, 0.21, 0.26,
                                0.28, 0.213
                                   , 0.192, 0.198, 0.22, 0.199, 0.02, 0.01,
                                0.53, 0.27, 0.21, 0.205, 0.022, 0.056])
            self.step = 0
            self.frameskip = 25
            self.fulltau = np.zeros(26)
            self.np_policy = policy
            self.target_sim_cache = []
            self.target_hw_cache = []


        def compute(self):
            if self.step % self.frameskip == 0:
                o = np.concatenate([self.world.skeletons[-1].q[6:], self.world.skeletons[-1].dq[6:]])
                self.target = self.np_policy.act(o, self.world.time())
                self.target_hw_cache.append(toRobot(self.target))
                self.target_sim_cache.append(RADIAN2VAL(self.target))
                np.savetxt('darwin/feedforward_target_simindex.txt', np.array(self.target_sim_cache, dtype=np.int))
                np.savetxt('darwin/feedforward_target_hwindex.txt', np.array(self.target_hw_cache, dtype=np.int))
            tau = -self.kp * (self.world.skeletons[-1].q[6:] - self.target) - self.kd * self.world.skeletons[-1].dq[6:]
            self.fulltau = np.concatenate([np.zeros(6), tau])
            self.step += 1
            return np.clip(self.fulltau, -3.5, 3.5) # torque limit of 3.5 Nm


    # Set joint damping
    for i in range(6, dart_world.skeletons[-1].ndofs):
        j = dart_world.skeletons[-1].dof(i)
        j.set_damping_coefficient(0.515)

    dart_world.set_gravity([0, 0, -9.81])
    dart_world.skeletons[1].set_mobile(False)
    dart_world.skeletons[1].q = dart_world.skeletons[1].q + 100
    dart_world.set_collision_detector(0)
    dart_world.skeletons[-1].set_self_collision_check(False)

    dart_world.skeletons[0].bodynodes[0].set_friction_coeff(5.0)
    for bn in dart_world.skeletons[-1].bodynodes:
        bn.set_friction_coeff(5.0)

    ############################################################################
    #### Setup the policy from file                                         ####
    #### refer to this part for construction of policy to be run on hardware ###
    ############################################################################
    pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                                    2047, 2171,
                                    2032, 2039, 2795, 648, 1231, 2040, 2041, 2060, 1281, 3448, 2855, 2073])
    pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                    2048, 2048,
                                    2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])

    pose_squat = VAL2RADIAN(pose_squat_val)
    pose_stand = VAL2RADIAN(pose_stand_val)

    # keyframe scheduling for squat stand task
    interp_sch = [[0.0, pose_stand],
                   [1.5, pose_squat],
                   [2.5, pose_stand],
                   [3.0, pose_squat],
                   [3.3, pose_stand],
                   [3.6, pose_squat], ]
    policy = NP_Policy(interp_sch, 'data/darwin_standsquat_policy_conseq_obs_warmstart.pkl', discrete_action=True,
              action_bins=np.array([11] * 20), delta_angle_scale=0.3)
    ############################################################################
    # End of setup for policy
    # policy should be used for executing on other environments
    ############################################################################

    # Initialize the controller
    controller = Controller(dart_world, policy)
    dart_world.skeletons[-1].set_controller(controller)
    print('create controller OK')

    pydart.gui.viewer.launch(dart_world,
                             default_camera=1)  # Use Z-up camera













