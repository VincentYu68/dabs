import numpy as np
import gym
from darwin.darwin_utils import *
from pydart2.constraints import BallJointConstraint

class DarwinPlain:
    def __init__(self):
        self.simenv = gym.make('DartDarwinSquat-v1')
        self.simenv.reset()
        self.simenv.env.disableViewer = False
        self.robot = self.simenv.env.robot_skeleton
        self.dup_robot = self.simenv.env.dart_world.skeletons[1]
        self.dart_world = self.simenv.env.dart_world

        # need to use fixed=True for initializing the constraints
        self.fixed_root = True
        self.reset()
        C = self.dart_world.skeletons[-1].bodynode('MP_BODY').C
        bc = BallJointConstraint(self.dart_world.skeletons[-1].bodynode('MP_BODY'),
                                 self.dart_world.skeletons[0].bodynodes[0], C)
        self.fix_root_constraints = []
        self.fix_root_constraints.append(bc)

        C2 = C + np.array([0, 0.1, 0])
        bc2 = BallJointConstraint(self.dart_world.skeletons[-1].bodynode('MP_BODY'),
                                  self.dart_world.skeletons[0].bodynodes[0], C2)
        self.fix_root_constraints.append(bc2)

        C3 = C + np.array([0, 0.0, 0.1])
        bc3 = BallJointConstraint(self.dart_world.skeletons[-1].bodynode('MP_BODY'),
                                  self.dart_world.skeletons[0].bodynodes[0], C3)
        self.fix_root_constraints.append(bc3)

        self.fixed_root = False
        self.reset()

        self.pose_mirror_matrix = np.zeros((self.robot.num_dofs()-6, self.robot.num_dofs()-6))
        for i, perm in enumerate(self.simenv.env.act_perm):
            self.pose_mirror_matrix[i][int(np.abs(perm))] = np.sign(perm)

        self.time = 0

    def render(self):
        self.simenv.render()

    def set_pose(self, pose):
        q = self.robot.q
        q[6:] = pose
        self.robot.q = q

    def set_dup_pose(self, pose):
        q = self.dup_robot.q
        q[6:] = pose
        self.dup_robot.q = q

    # whether fix root in the air
    def toggle_fix_root(self, is_fix):
        self.fixed_root = is_fix
        self.reset()

        if is_fix:
            for bc in self.fix_root_constraints:
                bc.add_to_world(self.dart_world)
        else:
            self.dart_world.remove_all_constraints()


    def step(self, target):
        self.time += self.simenv.env.dt

        self.simenv.env.target[6:] = target
        self.simenv.env.target[6:] = np.clip(self.simenv.env.target[6:], SIM_JOINT_LOW_BOUND_RAD, SIM_JOINT_UP_BOUND_RAD)

        tau = np.zeros(26)
        for i in range(self.simenv.env.frame_skip):
            #self.robot.bodynode('MP_ANKLE2_L').add_ext_force(np.array([-20, 0, 0]), np.array([0.0, 0.0, 0.0]))
            tau[6:] = self.simenv.env.PID()
            self.robot.set_forces(tau)
            self.dart_world.step()

    def passive_step(self): # advance simualtion without control
        for i in range(self.simenv.env.frame_skip):
            self.dart_world.step()

    def get_motor_pose(self):
        return self.robot.q[6:]

    def reset(self):
        self.simenv.reset()
        q = self.robot.q
        if self.fixed_root:
            q[5] = 0

        self.robot.q = q

        dq = self.robot.dq * 0
        #dq[0] += 2
        self.robot.dq = dq

        q[4] = 0.4
        #q[5] = 0.35
        self.dup_robot.set_positions(q)

        self.time = 0

    def check_self_collision(self):
        self.dart_world.check_collision()
        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.bodynode1.skel == contact.bodynode2.skel.id and contact.bodynode2.skel.id > 0:
                return True
        return False

    def get_imu_reading(self):
        return self.simenv.env.get_imu_data()