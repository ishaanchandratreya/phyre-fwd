import math
import random
random.seed(1)
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from phyre.action_simulator import initialize_simulator_without_compile
from phyre.creator import constants
import gym
from gym import spaces
from phyre.interface.scene import ttypes as scene_if
import phyre
from gym import utils
from gym.utils import seeding
import neural_agent

from phyre.action_mappers import Unscaler
np.random.seed(1)


def make_new_task_from_prev_and_feat(prev_task, featurized_objects, idx=-1):
    '''Issue: unit difference between quantities

    Scene being constructured from featurized objects values, hence difference in units

    '''

    prev_scene = prev_task.scene
    prev_bodies = prev_scene.bodies
    new_task = deepcopy(prev_task)

    prev_diameters = []
    for body_idx in range(len(prev_bodies)):

        body = prev_bodies[body_idx]

        prev_diameters.append(body.diameter)
        if body.bodyType == 2: #move over information for dynamic objects

            feat_obj_idx = -1
            cur_idx = -1
            for diam in featurized_objects.diameters.tolist():

                cur_idx += 1
                if body.diameter == diam * constants.SCENE_WIDTH:
                    feat_obj_idx = cur_idx

            if feat_obj_idx == -1:
                raise Exception('body not found')

            new_task.scene.bodies[body_idx].position.x = featurized_objects.xs[idx, feat_obj_idx] * constants.SCENE_WIDTH
            new_task.scene.bodies[body_idx].position.y = featurized_objects.ys[idx, feat_obj_idx] * constants.SCENE_HEIGHT

            if new_task.scene.bodies[body_idx].velocity is None:
                new_task.scene.bodies[body_idx].velocity = scene_if.Vector(featurized_objects.vxs[idx, feat_obj_idx] * constants.SCENE_WIDTH
                                                                           , featurized_objects.vys[idx, feat_obj_idx] * constants.SCENE_HEIGHT)

            else:
                new_task.scene.bodies[body_idx].velocity.x = featurized_objects.vxs[idx, feat_obj_idx] * constants.SCENE_WIDTH
                new_task.scene.bodies[body_idx].velocity.y = featurized_objects.vys[idx, feat_obj_idx] * constants.SCENE_HEIGHT

            if new_task.scene.bodies[body_idx].angular_velocity is None:
                new_task.scene.bodies[body_idx].angular_velocity = featurized_objects.avs[idx, feat_obj_idx]

            new_task.scene.bodies[body_idx].angle = featurized_objects.angles[idx, feat_obj_idx]


    all_obj_diam = featurized_objects.diameters.tolist()
    action_params = []
    for action_body_idx in range(len(all_obj_diam)):

        current_diam = all_obj_diam[action_body_idx] * constants.SCENE_WIDTH
        if current_diam not in prev_diameters:

            print('Printing colors one hot')
            print(featurized_objects.colors_one_hot[action_body_idx])

            x, y, radius, vx, vy, av = Unscaler.unscale_with_velocities(featurized_objects.xs[idx, action_body_idx] * constants.SCENE_WIDTH,
                                           featurized_objects.ys[idx, action_body_idx] * constants.SCENE_HEIGHT,
                                            featurized_objects.diameters[action_body_idx] * constants.SCENE_WIDTH / 2,
                                           featurized_objects.vxs[idx, action_body_idx] * constants.SCENE_WIDTH,
                                           featurized_objects.vys[idx, action_body_idx] * constants.SCENE_HEIGHT,
                                           featurized_objects.avs[idx, action_body_idx])
            action_params.append(
                [x, y, radius, vx, vy, av]
            )




    return new_task, action_params


#print('All eval setups:', *phyre.MAIN_EVAL_SETUPS)
current_eval_setup = 'ball_cross_template'
fold_id = 0

train_tasks, dev_tasks, test_tasks = phyre.get_fold(current_eval_setup, fold_id)
#print(*[len(each_task) for each_task in [train_tasks, dev_tasks, test_tasks]], sep=',')

action_tier = phyre.eval_setup_to_action_tier(eval_setup_name=current_eval_setup)
if action_tier == 'ball':
    action_tier = 'ball_v'

tasks = dev_tasks[:1]

print(tasks)
#Check how to complexify the action tiers
simulator = phyre.initialize_simulator(tasks, action_tier)
task_idx = 0
initial_scene = simulator.initial_scenes[task_idx]

np.set_printoptions(precision=3)
actions = simulator.build_discrete_action_space(max_actions=100)
action = random.choice(actions)

action[3] = 0.0
action[4] = 0.0 #set initial velocity to 0
action[5] = 0.0 #set angular velocity to 0


print(simulator.initial_featurized_objects[0].features)

im = simulator.initial_scenes[0]
im_init = phyre.observations_to_float_rgb(im)
fig, ax = plt.subplots()
i = ax.imshow(im_init)
plt.show()


simulation = simulator.simulate_action(task_idx, action, need_images=True, need_featurized_objects=True)
featurized_objects = simulation.featurized_objects

print('Later')
print(featurized_objects.features[0])
print('Later End')
new_task, action_params = make_new_task_from_prev_and_feat(simulator._tasks[-1], featurized_objects, 1)


new_tasks = [new_task]


for image in simulation.images[:5]:
    img = phyre.observations_to_float_rgb(image)
    fig, ax = plt.subplots()
    i = ax.imshow(img)
    plt.show()


simulator = initialize_simulator_without_compile(new_tasks, action_tier)
new_action = action_params[0]
new_action[2] = action[2]

#new_action[3] = 0.0
#new_action[4] = 0.0 #set initial velocity to 0
#new_action[5] = 0.0 #set angular velocity to 0


print(simulator.initial_featurized_objects[0].features)

im = simulator.initial_scenes[0]
im_init = phyre.observations_to_float_rgb(im)
fig, ax = plt.subplots()
i = ax.imshow(im_init)
plt.show()


simulation = simulator.simulate_action(task_idx, np.asarray(new_action), need_images=True, need_featurized_objects=True)
featurized_objects = simulation.featurized_objects
print('Later')
print(featurized_objects.features[0])
print('Later end')


for image in simulation.images[:5]:
    img = phyre.observations_to_float_rgb(image)
    fig, ax = plt.subplots()
    i = ax.imshow(img)

    plt.show()
