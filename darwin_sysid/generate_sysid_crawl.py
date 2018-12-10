# propse a set of system id actions such that no self collision is filtered

import pydart2 as pydart
import gym
import numpy as np
from darwin_sim.darwin_env_plain import *
from darwin.darwin_utils import *
import time
import time

if __name__ == "__main__":
    playback = True
    darwinenv = DarwinPlain()
    darwinenv.toggle_fix_root(False)

    ###################################################
    ################# Set up the environment ##########
    ###################################################
    darwinenv.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(10.0)
    for bn in darwinenv.robot.bodynodes:
        bn.set_friction_coeff(10.0)

    darwinenv.reset()
    CRAWL_POSE = np.array([-8.416675784817453376e-01, 4.800000000000000377e-01, 0.000000000000000000e+00, 8.416675784817453376e-01, -4.800000000000000377e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 4.800000000000000377e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 7.999999999999999334e-01, -4.000000000000000222e-01, -1.599999999999999756e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -7.999999999999999334e-01, 4.000000000000000222e-01, 1.599999999999999756e-01, 0.000000000000000000e+00
])
    darwinenv.set_pose(CRAWL_POSE)
    darwinenv.set_root_dof(darwinenv.get_root_dof() + np.array([0.0, 1.35, 0.0, 0.0, 0.0, 0.0]))
    q = darwinenv.get_root_dof()
    q[5] += -0.315 - np.min([darwinenv.robot.bodynodes[-1].C[2], darwinenv.robot.bodynodes[-8].C[2]])
    darwinenv.set_root_dof(q)
    for i in range(20):
        darwinenv.step(CRAWL_POSE)
    init_root = darwinenv.get_root_dof()

    num_unconstrained_pose = 10
    num_constrained_pose = 10
    sim_length = 100
    trial_num = 5  # repeat time for each new pose

    unconstrained_poses = []
    constrained_poses = []

    unconst_actions = []
    const_actions = []

    #####################################################
    ############# Start generating data #################
    #####################################################

    unconstrained_poses.append(CRAWL_POSE)
    prev_root = darwinenv.get_root_dof()
    unconst_actions.append(CRAWL_POSE)
    while len(unconstrained_poses) < num_unconstrained_pose:
        rand_pose_coef = np.random.random(20)
        rand_pose_coef[0] = np.clip(rand_pose_coef[0], 0.3, 0.7)
        rand_pose_coef[3] = np.clip(rand_pose_coef[3], 0.3, 0.7)
        rand_pose = SIM_JOINT_LOW_BOUND_RAD * rand_pose_coef + SIM_JOINT_UP_BOUND_RAD * (1-rand_pose_coef)

        rand_pose = 0.5 * (rand_pose + unconstrained_poses[-1])

        valid = True
        for trial in range(trial_num):
            darwinenv.reset()
            darwinenv.set_pose(unconstrained_poses[-1])
            darwinenv.set_root_dof(prev_root)
            for i in range(sim_length):
                perc = np.min([((int(i / 10) + 1) * 15) / sim_length, 1.0])
                darwinenv.step(rand_pose * perc + (1-perc) * unconstrained_poses[-1])
                if darwinenv.check_collision(['l_hand', 'r_hand', 'MP_ANKLE2_R', 'MP_ANKLE2_L']) or darwinenv.check_self_collision():
                    valid = False
                    break
            if not valid:
                break

        if valid and len(darwinenv.contacts) >= 3:
            darwinenv.set_pose(unconstrained_poses[-1])
            darwinenv.set_root_dof(prev_root)
            for i in range(sim_length):
                perc = np.min([((int(i / 10) + 1) * 15) / sim_length, 1.0])
                darwinenv.step(rand_pose * perc + (1 - perc) * unconstrained_poses[-1])
                unconst_actions.append(rand_pose * perc + (1 - perc) * unconstrained_poses[-1])
                if playback:
                    darwinenv.render()
                    time.sleep(0.01)
            unconstrained_poses.append(rand_pose)
            prev_root = darwinenv.get_root_dof()
    print("Unconstrained crawl pose generated")

    ############################################
    ####### try to maintain the contacts ######
    ############################################
    constrained_poses.append(CRAWL_POSE)
    const_actions.append(CRAWL_POSE)
    prev_root = init_root
    darwinenv.reset()
    darwinenv.set_pose(constrained_poses[-1])
    darwinenv.set_root_dof(prev_root)
    for i in range(10):
        darwinenv.step(CRAWL_POSE)
        darwinenv.render()

    darwinenv.check_collision([])
    collision_constraints = {}
    for contact in darwinenv.contacts:
        key = contact.bodynode1.name+contact.bodynode2.name if contact.bodynode1.id > contact.bodynode2.id \
            else contact.bodynode2.name+contact.bodynode1.name
        if key in collision_constraints:   # if duplicated, use the point largest in y
            if contact.point[1] > collision_constraints[key][1]:
                collision_constraints[key] = contact.point
        else:
            collision_constraints[key] = contact.point
    darwinenv.create_contact_constraint()

    while len(constrained_poses) < num_constrained_pose:
        rand_pose_coef = np.random.random(20)
        rand_pose_coef[0] = np.clip(rand_pose_coef[0], 0.3, 0.7)
        rand_pose_coef[3] = np.clip(rand_pose_coef[3], 0.3, 0.7)
        rand_pose = SIM_JOINT_LOW_BOUND_RAD * rand_pose_coef + SIM_JOINT_UP_BOUND_RAD * (1 - rand_pose_coef)

        rand_pose = 0.5 * (rand_pose + constrained_poses[-1])

        # use contact constraints to make the guessed pose closer to a valid pose
        darwinenv.reset()
        darwinenv.set_pose(constrained_poses[-1])
        darwinenv.set_root_dof(prev_root)
        darwinenv.toggle_contact_constraint(True)
        for i in range(sim_length):
            perc = np.min([((int(i / 10) + 1) * 15) / sim_length, 1.0])
            darwinenv.step(rand_pose * perc + (1 - perc) * constrained_poses[-1])
        darwinenv.toggle_contact_constraint(False)
        rand_pose = darwinenv.get_motor_pose()

        valid = True
        for trial in range(trial_num):
            darwinenv.reset()
            darwinenv.set_pose(constrained_poses[-1])
            darwinenv.set_root_dof(prev_root)
            for i in range(sim_length):
                perc = np.min([((int(i / 10) + 1) * 15) / sim_length, 1.0])
                darwinenv.step(rand_pose * perc + (1 - perc) * constrained_poses[-1])
                if darwinenv.check_collision(
                        ['l_hand', 'r_hand', 'MP_ANKLE2_R', 'MP_ANKLE2_L']) or darwinenv.check_self_collision():
                    valid = False
                    break
            if not valid:
                break

        if valid and len(darwinenv.contacts) >= 3:
            contact_match = True
            current_contacts = {}
            for contact in darwinenv.contacts:
                key = contact.bodynode1.name + contact.bodynode2.name if contact.bodynode1.id > contact.bodynode2.id \
                    else contact.bodynode2.name + contact.bodynode1.name
                if key in current_contacts:  # if duplicated, use the point largest in y
                    if contact.point[1] > current_contacts[key][1]:
                        current_contacts[key] = contact.point
                else:
                    current_contacts[key] = contact.point
            if len(current_contacts.keys()) != len(collision_constraints.keys()):
                contact_match = False
            else:
                largest_error = -1
                for k in current_contacts.keys():
                    if k not in collision_constraints:
                        contact_match = False
                        break
                    error = np.linalg.norm(current_contacts[k] - collision_constraints[k])
                    if error > largest_error:
                        largest_error = error
                if largest_error > 0.02:
                    contact_match = False

            if contact_match:
                darwinenv.set_pose(constrained_poses[-1])
                darwinenv.set_root_dof(prev_root)
                for i in range(sim_length):
                    perc = np.min([((int(i / 10) + 1) * 15) / sim_length, 1.0])
                    darwinenv.step(rand_pose * perc + (1 - perc) * constrained_poses[-1])
                    const_actions.append(rand_pose * perc + (1 - perc) * constrained_poses[-1])
                    if playback:
                        darwinenv.render()
                        time.sleep(0.01)
                constrained_poses.append(rand_pose)
                prev_root = darwinenv.get_root_dof()
    print("Constrained crawl pose generated")

    np.savetxt('data/sysid_data/unconstrained_poses.txt', unconstrained_poses)
    np.savetxt('data/sysid_data/constrained_poses.txt', constrained_poses)
    np.savetxt('data/sysid_data/unconstrained_actions.txt', unconst_actions)
    np.savetxt('data/sysid_data/constrained_actions.txt', const_actions)

