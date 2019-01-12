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

        self.max_so_far = None
        self.max_id = None

        self.accum_orientation = np.zeros(3)

        self.sub_step_velocities = []

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

    def get_root_dof(self):
        return np.array(self.robot.q[0:6])

    def set_root_dof(self, rootdof):
        q = self.robot.q
        q[0:6] = rootdof
        self.robot.q = q

    # whether fix root in the air
    def toggle_fix_root(self, is_fix):
        self.fixed_root = is_fix
        self.reset()

        if is_fix:
            for bc in self.fix_root_constraints:
                bc.add_to_world(self.dart_world)
        else:
            self.dart_world.remove_all_constraints()

    def set_control_timestep(self, control_dt):
        self.simenv.env.frame_skip = int(control_dt/self.simenv.env.sim_dt)

    def step(self, target):
        self.time += self.simenv.env.dt

        self.simenv.env.target[6:] = target
        self.simenv.env.target[6:] = np.clip(self.simenv.env.target[6:], SIM_JOINT_LOW_BOUND_RAD, SIM_JOINT_UP_BOUND_RAD)

        tau = np.zeros(26)

        self.sub_step_velocities = []
        for i in range(self.simenv.env.frame_skip):
            #self.robot.bodynode('MP_ANKLE2_L').add_ext_force(np.array([-20, 0, 0]), np.array([0.0, 0.0, 0.0]))
            tau[6:] = self.simenv.env.PID()
            #tau[(np.abs(self.robot.dq) > 2.0) * (np.sign(self.robot.dq) == np.sign(tau))] = 0
            self.robot.set_forces(tau)
            self.dart_world.step()
            self.sub_step_velocities.append(np.array(self.robot.dq)[6:])
            '''if self.time > 0.1:
                if self.max_so_far is None:
                    self.max_so_far = np.max(np.abs(np.array(self.robot.dq)[6:]))
                    self.max_id = np.argmax(np.abs(np.array(self.robot.dq)[6:]))
                    print(self.max_so_far, self.max_id)
                else:
                    if np.max(np.abs(np.array(self.robot.dq)[6:])) > self.max_so_far:
                        self.max_so_far = np.max(np.abs(np.array(self.robot.dq)[6:]))
                        self.max_id = np.argmax(np.abs(np.array(self.robot.dq)[6:]))
                        print(self.max_so_far, self.max_id)'''
        self.accum_orientation += self.get_gyro_data() * self.simenv.env.dt


    def get_gyro_data(self):
        return np.array(self.simenv.env.get_imu_data()[-3:])

    def passive_step(self): # advance simualtion without control
        self.time += self.simenv.env.dt
        for i in range(self.simenv.env.frame_skip):
            self.dart_world.step()

    def get_motor_pose(self):
        return np.array(self.robot.q)[6:]

    def get_motor_velocity(self):
        return np.array(self.robot.dq)[6:]

    def get_closest_motor_velocity(self, ref, metric, search_range):
        if len(self.sub_step_velocities) == 0:
            return self.get_motor_velocity()
        evals = []
        for i in search_range:
            if metric == 'l1':
                val = np.sum(np.abs(ref-self.sub_step_velocities[i]))
            elif metric == 'l2':
                val = np.sum(np.square(ref-self.sub_step_velocities[i]))
            else:
                print('Unknown metric!')
                val = 0
            evals.append(val)
        return self.sub_step_velocities[np.argmin(evals)]


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

        self.accum_orientation = np.zeros(3)

        self.time = 0

    def check_self_collision(self):
        self.dart_world.check_collision()
        self.contacts = self.dart_world.collision_result.contacts
        for contact in self.contacts:
            if contact.bodynode1.skel == contact.bodynode2.skel and contact.bodynode2.skel.id > 0:
                return True
        return False

    def check_collision(self, permitted_body_names):
        self.dart_world.check_collision()
        self.contacts = self.dart_world.collision_result.contacts
        for contact in self.contacts:
            if contact.bodynode1.name not in permitted_body_names and contact.bodynode2.name not in permitted_body_names:
                return True
        return False

    def create_contact_constraint(self):
        self.check_self_collision()
        self.contact_constraints = []
        for contact in self.contacts:
            C = contact.point
            bc = BallJointConstraint(contact.bodynode1, contact.bodynode2, C)
            self.contact_constraints.append(bc)

    def toggle_contact_constraint(self, is_enforced):
        self.contact_constraint_enforced = is_enforced

        if is_enforced:
            for bc in self.contact_constraints:
                bc.add_to_world(self.dart_world)
        else:
            self.dart_world.remove_all_constraints()

    def integrate_imu_reading(self):
        self.simenv.env.integrate_imu_data()

    def get_integrated_imu(self):
        return np.array(self.simenv.env.accumulated_imu_info)

    def get_imu_reading(self):
        return self.simenv.env.get_imu_data()


    ####################################
    ##### parameters for system id #####
    ####################################
    KP, KD, KC, VEL_LIM, JOINT_DAMPING, JOINT_FRICTION, TORQUE_LIM = list(range(7))
    MU_DIMS = np.array([5, 5, 5, 1, 1, 1, 1])
    MU_UP_BOUNDS = [[200, 200, 200, 200, 200], [1,1,1,1,1], [10,10,10,10,10], [15], [1], [1], [20.0]]
    MU_LOW_BOUNDS = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [2.0], [0], [0], [3.0]]
    ACTIVE_MUS = [KP, KD, VEL_LIM, JOINT_DAMPING, TORQUE_LIM]
    MU_UNSCALED = None # unscaled version of mu

    def set_mu(self, x):
        assert(len(x) == np.sum(self.MU_DIMS[self.ACTIVE_MUS]))

        self.MU_UNSCALED = np.zeros(len(x))
        current_id = 0
        for mu in self.ACTIVE_MUS:
            self.MU_UNSCALED[current_id:current_id + self.MU_DIMS[mu]] = \
                np.array(x[current_id:current_id + self.MU_DIMS[mu]]) * \
                (np.array(self.MU_UP_BOUNDS[mu]) - np.array(self.MU_LOW_BOUNDS[mu])) + \
                np.array(self.MU_LOW_BOUNDS[mu])
            current_id += self.MU_DIMS[mu]

        current_id = 0
        if self.KP in self.ACTIVE_MUS:
            self.simenv.env.kp = np.zeros(20)
            # arms
            self.simenv.env.kp[0:6] = self.MU_UNSCALED[current_id]
            # head
            self.simenv.env.kp[6:8] = self.MU_UNSCALED[current_id + 1]
            # hip
            self.simenv.env.kp[8:11] = self.MU_UNSCALED[current_id + 2]
            self.simenv.env.kp[14:17] = self.MU_UNSCALED[current_id + 2]
            # knee
            self.simenv.env.kp[11] = self.MU_UNSCALED[current_id + 3]
            self.simenv.env.kp[17] = self.MU_UNSCALED[current_id + 3]
            # ankle
            self.simenv.env.kp[12:14] = self.MU_UNSCALED[current_id + 4]
            self.simenv.env.kp[18:20] = self.MU_UNSCALED[current_id + 4]

            current_id += self.MU_DIMS[self.KP]

        if self.KD in self.ACTIVE_MUS:
            self.simenv.env.kd = np.zeros(20)
            # arms
            self.simenv.env.kd[0:6] = self.MU_UNSCALED[current_id]
            # head
            self.simenv.env.kd[6:8] = self.MU_UNSCALED[current_id + 1]
            # hip
            self.simenv.env.kd[8:11] = self.MU_UNSCALED[current_id + 2]
            self.simenv.env.kd[14:17] = self.MU_UNSCALED[current_id + 2]
            # knee
            self.simenv.env.kd[11] = self.MU_UNSCALED[current_id + 3]
            self.simenv.env.kd[17] = self.MU_UNSCALED[current_id + 3]
            # ankle
            self.simenv.env.kd[12:14] = self.MU_UNSCALED[current_id + 4]
            self.simenv.env.kd[18:20] = self.MU_UNSCALED[current_id + 4]
            current_id += self.MU_DIMS[self.KD]

        if self.KC in self.ACTIVE_MUS:
            self.simenv.env.kc = np.zeros(20)
            # arms
            self.simenv.env.kc[0:6] = self.MU_UNSCALED[current_id]
            # head
            self.simenv.env.kc[6:8] = self.MU_UNSCALED[current_id + 1]
            # hip
            self.simenv.env.kc[8:11] = self.MU_UNSCALED[current_id + 2]
            self.simenv.env.kc[14:17] = self.MU_UNSCALED[current_id + 2]
            # knee
            self.simenv.env.kc[11] = self.MU_UNSCALED[current_id + 3]
            self.simenv.env.kc[17] = self.MU_UNSCALED[current_id + 3]
            # ankle
            self.simenv.env.kc[12:14] = self.MU_UNSCALED[current_id + 4]
            self.simenv.env.kc[18:20] = self.MU_UNSCALED[current_id + 4]
            current_id += self.MU_DIMS[self.KC]

        if self.VEL_LIM in self.ACTIVE_MUS:
            self.simenv.env.joint_vel_limit = self.MU_UNSCALED[current_id]
            current_id += self.MU_DIMS[self.VEL_LIM]

        if self.JOINT_DAMPING in self.ACTIVE_MUS:
            joint_damping = self.MU_UNSCALED[current_id]
            for i in range(6, self.robot.ndofs):
                j = self.robot.dof(i)
                j.set_damping_coefficient(joint_damping)

            current_id += self.MU_DIMS[self.JOINT_DAMPING]

        if self.JOINT_FRICTION in self.ACTIVE_MUS:
            joint_friction = self.MU_UNSCALED[current_id]
            for i in range(6, self.robot.ndofs):
                j = self.robot.dof(i)
                j.set_coulomb_friction(joint_friction)
            current_id += self.MU_DIMS[self.JOINT_FRICTION]

        if self.TORQUE_LIM in self.ACTIVE_MUS:
            self.simenv.env.torqueLimits = self.MU_UNSCALED[current_id]
            current_id += self.MU_DIMS[self.TORQUE_LIM]










