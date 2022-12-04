import math
import random
random.seed(1)
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import imageio
from phyre.action_simulator import initialize_simulator_without_compile
from phyre.creator import constants
import gym
from gym import spaces
from gym_utils import play_gif, gif_create
from phyre.interface.scene import ttypes as scene_if
from train2 import get_train_test
import phyre
import os
from gym import utils
import concurrent.futures
from gym.utils import seeding
import neural_agent
import matplotlib.animation as animation
import matplotlib.cm as cm


from phyre.action_mappers import Unscaler
np.random.seed(1)


def break_img(img):




    outcome_img = np.copy(img) * 255
    action_img = np.ones_like(img) * 255
    action_indices = np.where((outcome_img[:, :, 0] == 243) & (outcome_img[:, :, 1] == 79) & (outcome_img[:, :, 2] == 70))

    outcome_img[action_indices] = np.array([255, 255, 255])
    action_img[action_indices] = np.array([243, 79, 70])

    return action_img / 255 , outcome_img / 255


class PhyreSingleTaskEnv():

    metadata = {'render.modes': ['human']}
    SINGLE_ENV_TASK_IDX = 0
    DEFAULT_MAX_STEPS = 1800
    SCENE_WIDTH = constants.SCENE_WIDTH
    SCENE_HEIGHT = constants.SCENE_HEIGHT
    TIER = 'ball_v'

    def __init__(self, physical_mode,   #currently set to 'physics'
                 action_mode,           #currently set to 0
                 observation_mode,      #currently set to 0
                 task_string,
                 allow_interventional_action=True,
                 init_from_sim_cache= True,
                 max_actions=100000,
                 stride=60,
                 specific_task=True,
                 stable_touch=False,
                 seed=42):
        '''
        physical_mode ['physics', 'cc']: tells us whether to use previous action from physics or try a new action
        action_mode [0, 1, 2]: tells us whether or not to allow velocity and angular velocity setting
        observation_mode [0, 1]: currently only supports 0 (image space) #TODO mode 1 (feature space)
        task_string: PHYRE task string (eg. 00000:01 for the task to solve)
        TODO:stochastic_mode [0, 1, 2, 3]: Different kinds of uncertainty inside what is otherwise a static physical environment
        allow_interventional_action [bool]: Marks whether or not a non-physically progressing action is allowed to happen
        in the current scenario
        init_from_sim_cache [bool]: Indicates whether or not the first action chosen should be picked from PHYRE's
        default simulation cache. If not it will manually check for whether or not chosen action is valid, and keep
        resampling until closest action to the requested action is found.
        max_actions:
        specific_task: indicates if particular instance of this task is being released
        '''

        #TODO change this call when everthing is added
        #utils.EzPickle.__init__(self)
        np.set_printoptions(precision=3)
        self.rng = self.seed(seed=seed)

        self.physical_mode = self._convert_mode_to_index(physical_mode)
        self.base_string = task_string
        self.specific_task = specific_task
        self.task_template = task_string.split(':')[0]
        self.root_cache = phyre.get_default_100k_cache(self.TIER.split('_')[0])
        self._set_steps_and_initial_task(self.base_string, self.specific_task)
        self.rand_name = 0


        self.default_info                      = {
        '_NUM_FEATURES'       : 17,
        '_X_INDEX'            : 0,
        '_Y_INDEX'            : 1,
        '_ANGLE_INDEX '       : 2,
        '_DIAMETER_INDEX'     : 3,
        '_SHAPE_START_INDEX'  : 4,
        '_SHAPE_END_INDEX'    : 8,
        '_COLOR_START_INDEX'  : 8,
        '_COLOR_END_INDEX'    : 14,
        '_VX_INDEX'           : 14,
        '_VY_INDEX'           : 15,
        '_AV_INDEX'           : 16,

        }

        self.action_mode                       = action_mode
        self.stride                            = stride
        self.stable_touch                      = stable_touch
        self.action_space, self._action_append = self._init_action_space(action_mode)
        self.observation_space                 = self._init_obs_space(observation_mode)



        self.intervene                         = allow_interventional_action
        self.init_method                       = init_from_sim_cache
        self.max_actions                       = max_actions
        self.occlusions_allowed                = False


        self._set_initial_values()

    def _set_steps_and_initial_task(self, task_string, specific_task):

        if self.physical_mode == 0:
            '''USed to keep track of the change in the index from the simulation'''
            self.step_calls_in_bank = -1
            self.step_idx = 0
            self.accepting_actions = True
            self.step_call = 0

        else:

            self.step_calls_in_bank = -1
            self.step_idx = None
            self.accepting_actions = True
            self.step_call = 0

        if specific_task:
            self.root_task                     = task_string
            print(f'Root task has been set to {self.root_task}')
        else:
            self.root_task                     = self.sample_random_task_from_template(self.task_template)
            print(f'Root task has been set to {self.root_task}')

        self.simulator = self._get_simulator(self.root_task)


    def _set_initial_values(self):

        self.rec_init_action = self._get_final_action(self.get_first_action())

        self.initial_viz                    = self.simulator.initial_scenes[self.SINGLE_ENV_TASK_IDX]
        self.initial_obs                    = self.simulator.initial_featurized_objects[self.SINGLE_ENV_TASK_IDX]
        self.task_class                     = self.simulator._tasks[self.SINGLE_ENV_TASK_IDX]

        if not self.stable_touch:
            self.switch_to_momentary_touching_goal()

        self.scene_without_user_input       = self.task_class.scene
        self.all_bodies_without_user_input  = self.scene_without_user_input.bodies
        self.full_base_body_info            = self.resolve_bodies(self.all_bodies_without_user_input)

        self.current_task = self.task_class
        self.current_simulation = None

    def sample_random_task_from_template(self, task_template):

        return self.rng.choice(create_available_task_list(task_template, self.root_cache), 1)[0]

    def switch_to_momentary_touching_goal(self):

        if self.task_class.relationships == [6]:
            self.task_class.relationships = [5]

    def show_obs(self, obs, name=None):

        from PIL import Image
        im_init = phyre.observations_to_float_rgb(obs)
        print(type(im_init))
        print(im_init.dtype)
        print(im_init)
        save_init = im_init.copy() * 255
        save_init = save_init.astype('uint8')
        img = Image.fromarray(save_init)
        if name is not None:
            print('Saving')
            img.save(os.path.join('presentation_example', f'{name}.png'))

        print(type(im_init))
        fig, ax = plt.subplots()
        i = ax.imshow(im_init)
        plt.show()

    def get_first_action(self, solves_task=True):
        '''Sets the first action inside the phyre scene. Checks if action is valid for the task, and only then
        allows it to get through. Can the check for whether it is valid or not be made end-to-end differentiable?
        '''

        if self.init_method:

            data = self.root_cache.get_sample([self.root_task], self.max_actions)

            task_indices, is_solved, actions, _, _ = (
                neural_agent.compact_simulation_data_to_trainset(
                    self.TIER, **data))

            self.all_possible_actions = actions

            indices = np.arange(len(is_solved))
            solved_mask = is_solved > 0

            if solves_task:
                final_indices = indices[solved_mask]
            else:
                final_indices = indices[~solved_mask]


            self.possible_solve_indices = indices[solved_mask]
            self.possible_non_solve_indices = indices[~solved_mask]

            chosen_idx = self.rng.choice(final_indices, size=1)
            return actions[chosen_idx]

        else:
            #TODO Implement select action and find nearest neighbour pipeline in configuration space
            raise NotImplementedError("Random actions in space are so far not implemented (but should be soon!)")


    def sample_action(self, solve=False):

        if np.random.uniform() < 0.8:
            solve = not solve

        if solve:
            chosen_idx = self.rng.choice(self.possible_solve_indices, size=1)
        else:
            chosen_idx = self.rng.choice(self.possible_non_solve_indices, size=1)

        return self._get_final_action(self.all_possible_actions[chosen_idx])

    def is_invalid(self, action):
        '''Given an action that is inside of the action space, check whether the given action is valid for the task'''

        #TODO use the radius of the object to quickly calculate a configuration space and check if (x,y) is valid
        pass

    def _get_simulator(self, task_string):

        return phyre.initialize_simulator([task_string], self.TIER)


    def seed(self, seed=42):

        #torch.cuda.set_manual_seed
        #rng.seed
        #torch.seed
        return np.random.RandomState(seed)

    def _get_final_action(self, action):

        if isinstance(action, np.ndarray):
            assert len(action) + len(self._action_append) == 6

        return np.concatenate([np.squeeze(action), self._action_append])

    def _init_obs_space(self, observation_mode):

        self.screen_width, self.screen_height = phyre.creator.SCENE_WIDTH, phyre.creator.SCENE_HEIGHT
        if observation_mode == 0:
            return spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)

        elif observation_mode == 1:
            raise NotImplementedError("Feature space mode not yet supported")

        else:
            raise EnvironmentError("Choose between 0 (img space) and 1 (feature space)")


    def _init_action_space(self, action_mode):

        if action_mode == 0:
            return spaces.Box(low=0, high=1, shape=(6,), dtype=np.float16), np.zeros((3,)) + 0.5

        elif action_mode == 1:
            raise NotImplementedError("Start velocity mode not yet supported yet")
            #return spaces.Box(low=0, high=1, shape=(5,), dtype=np.float16), np.zeros((1,))

        else:
            raise NotImplementedError("Start velocity+angular velocity mode not yet supported yet")
            #return spaces.Box(low=0, high=1, shape=(6,), dtype=np.float16), None


    def resolve_bodies(self, bodies):
        '''Given a set of bodies in the scene resolve the static and dynamic bodies at initialization to
        encourager faster task creation and resolution

        Goal example: running the following should give body indices with static motion and shape jar
        set(full_body_info['motion']['static']).intersection(set(full_body_info['shape'][0]))
        '''

        full_body_info = {}

        #make list of static and dynamic bodies
        motion = {'static': [], 'dynamic': []}
        for body_idx in range(len(bodies)):

            body = bodies[body_idx]
            if body.bodyType == 2:
                motion['dynamic'].append(body_idx)

            elif body.bodyType == 1:
                motion['static'].append(body_idx)

            else:
                raise Exception("This is an invalid kind of bodyType: check sim installation or modification")


        full_body_info['motion'] = motion

        #UNDEFINED = 0,
        #BALL = 1,
        #BAR = 2,
        #JAR = 3,
        #STANDINGSTICKS = 4

        shape = {str(i): [] for i in range(5)}
        for body_idx in range(len(bodies)):

            body = bodies[body_idx]
            try:
                shape[str(body.shapeType)].append(body_idx)

            except KeyError:
                raise Exception("This is an invalid kind of shapeType: check sim installation or modification")

        full_body_info['shape'] = shape

        #WHITE = 0,
        #BLACK = 6,
        #GRAY = 5,
        #GREEN = 2,
        #BLUE = 3,
        #PURPLE = 4,
        #RED = 1,
        #LIGHT_RED = 7,

        color = {str(i): [] for i in range(8)}
        for body_idx in range(len(bodies)):
            body = bodies[body_idx]
            try:
                color[str(body.color)].append(body_idx)
            except KeyError:
                raise Exception("This is an invalid kind of color. check sim installation or modification")

        full_body_info['color'] = color

        return full_body_info

    def update_task_after_step(self, featurized_objects):


        prev_task = self.task_class
        new_task = deepcopy(prev_task)


        base_dynamic_body_idx = self.full_base_body_info['motion']['dynamic']

        for body_idx in base_dynamic_body_idx:

            c_body  = self.all_bodies_without_user_input[body_idx]
            feat_obj_idx = self._find_featurized_obj_idx_with_color(featurized_objects, c_body.color)

            new_task.scene.bodies[body_idx].position.x = featurized_objects.xs[
                                                             -1, feat_obj_idx] * constants.SCENE_WIDTH
            new_task.scene.bodies[body_idx].position.y = featurized_objects.ys[
                                                             -1, feat_obj_idx] * constants.SCENE_HEIGHT

            if new_task.scene.bodies[body_idx].velocity is None:
                new_task.scene.bodies[body_idx].velocity = scene_if.Vector(
                    featurized_objects.vxs[1, feat_obj_idx] * constants.SCENE_WIDTH
                    , featurized_objects.vys[1, feat_obj_idx] * constants.SCENE_HEIGHT)

            else:
                new_task.scene.bodies[body_idx].velocity.x = featurized_objects.vxs[
                                                                 -1, feat_obj_idx] * constants.SCENE_WIDTH
                new_task.scene.bodies[body_idx].velocity.y = featurized_objects.vys[
                                                                 -1, feat_obj_idx] * constants.SCENE_HEIGHT


            new_task.scene.bodies[body_idx].angular_velocity = featurized_objects.avs[-1, feat_obj_idx] * 2 * math.pi

            new_task.scene.bodies[body_idx].angle = featurized_objects.angles[-1, feat_obj_idx] * 2 * math.pi


        return new_task

    def _find_featurized_obj_idx_with_color(self, featurized_objects, color):

        '''
        One hot encoding of object color, according to order:
                red, green, blue, purple, gray, black
        '''

        all_featurized_objs_with_color = []
        for all_body_idx in range(0, featurized_objects.features.shape[1]):

            if np.argmax(featurized_objects.colors_one_hot[all_body_idx]) + 1 == float(color):
                all_featurized_objs_with_color.append(all_body_idx)

        if len(all_featurized_objs_with_color) > 1:
            raise Exception("Multiple objects found with same color- change pipeline for identification")

        if len(all_featurized_objs_with_color) == 0:
            raise Exception("No objects found of this color- there appears to be a mismatch")

        return all_featurized_objs_with_color[0]


    def update_action_after_step(self, featurized_objects, given_idx=-1):

        #action objects are NOT in the dynamic objects of the task list
        #to get them we must query the featurized objects and see which of the objects are not already accounted for

        #however, a simple solution is to reserve a (color, shape) pair for a particular action

        #change this function with base body dynamic info
        base_dynamic_body_idx = self.full_base_body_info['motion']['dynamic']

        #action object identified by the fact that it is red and circle, #TODO make a generic criteria for this to work
        action_body_idx = -1
        for all_body_idx in range(0, featurized_objects.features.shape[1]):
            is_red = featurized_objects.colors_one_hot[all_body_idx][0] == 1.0
            is_circle = featurized_objects.shapes_one_hot[all_body_idx][0] == 1.0
            if is_red and is_circle:
                action_body_idx = all_body_idx
                break

        if action_body_idx == -1:
            raise Exception("Actionable object not found")


        action_params = []
        x, y, radius, vx, vy, av = Unscaler.unscale_with_velocities(
            featurized_objects.xs[given_idx, action_body_idx] * constants.SCENE_WIDTH,
            featurized_objects.ys[given_idx, action_body_idx] * constants.SCENE_HEIGHT,
            featurized_objects.diameters[action_body_idx] * constants.SCENE_WIDTH / 2,
            featurized_objects.vxs[given_idx, action_body_idx] * constants.SCENE_WIDTH,
            featurized_objects.vys[given_idx, action_body_idx] * constants.SCENE_HEIGHT,
            featurized_objects.avs[given_idx, action_body_idx] * 2 * math.pi)
        action_params.append(
            [x, y, radius, vx, vy, av]
        )

        return action_params

    def _convert_mode_to_index(self, mode):
        MODES = {'physics': 0,
                 'cc': 1}

        return MODES[mode]

    def reset_step_equivalences(self):

        self.step_idx = 0
        self.step_calls_in_bank = -1
        self.step_call = 0

    def step(self, action=None, nframes=None):
        #Executing a timestep

        if self.accepting_actions:

            if self.physical_mode == 0:
                self.reset_step_equivalences()

            if action is None:
                action = self.rec_init_action

            #TODO deal with this case better
            if action == 'blank':
                action = None

            if nframes is None:

                if self.physical_mode == 0:

                    nframes = self.DEFAULT_MAX_STEPS

                else:
                    nframes = self.stride



            self.simulator._action_mapper.OCCLUSIONS_ALLOWED = True

            self.current_simulation = self.simulator.simulate_action(self.SINGLE_ENV_TASK_IDX,
                                                                     action,
                                                                     need_images=True,
                                                                     need_featurized_objects=True,
                                                                     stride=self.stride,
                                                                     nframes=nframes)

            self.step_call += 1

            if self.physical_mode == 1:
                new_task   = self.update_task_after_step(self.current_simulation.featurized_objects)
                if action is not None:
                    new_action = self.update_action_after_step(self.current_simulation.featurized_objects)

                else:
                    new_action = 'blank'

                del self.simulator
                del self.current_task

                self.current_task = new_task
                self.simulator = initialize_simulator_without_compile([new_task], self.TIER)

                _, final_obs = break_img(phyre.observations_to_float_rgb(self.current_simulation.images[-1]))

                return final_obs, self.find_reward(), self.done(self.current_simulation.status), \
                       {'new_action': np.asarray(new_action[0]), 'sim_done': False, 'fo':  self.current_simulation.featurized_objects.full_features[-1]}

            else:

                if action is None:
                    action_body_idx = -1
                else:
                    action_body_idx = self._find_featurized_obj_idx_with_color(self.current_simulation.featurized_objects, 1)
                self.step_calls_in_bank = len(self.current_simulation.featurized_objects.full_features) - 1
                self.step_idx += 1

                done = False if self.step_idx < self.step_calls_in_bank  else self.done(self.current_simulation.status)
                sim_done = False if self.step_idx < self.step_calls_in_bank else True

                if self.step_idx < self.step_calls_in_bank:
                    self.accepting_actions = False

                else:

                    self.accepting_actions = True

                '''
                Examples
                (step_idx, step_calls)

                #Case total_timesteps = 2
                (1, 1)

                #case total timesteps = 3
                (2, 1)
                (2, 2)
                '''
                if action is not None:
                    new_action = self.update_action_after_step(self.current_simulation.featurized_objects, given_idx=self.step_idx)
                else:
                    new_action = 'blank'

                _, final_obs = break_img(phyre.observations_to_float_rgb(self.current_simulation.images[self.step_idx]))

                return final_obs, \
                       self.find_reward(), \
                       done, \
                       {'fo': self.current_simulation.featurized_objects.full_features[self.step_idx],
                        'new_action': np.asarray(new_action[0]),
                        'action_body_idx': action_body_idx,
                        'sim_done': sim_done}

        else:
            if self.physical_mode == 1:

                raise Exception("Something has gone terribly wrong as there is no reason for cc mode class to ever "
                                "NOT be accepting actions. This branch is supposed to be for physical mode only")


            action_body_idx = self._find_featurized_obj_idx_with_color(self.current_simulation.featurized_objects, 1)
            self.step_idx += 1

            done = False if self.step_idx < self.step_calls_in_bank else self.done(self.current_simulation.status)
            sim_done = False if self.step_idx < self.step_calls_in_bank else True

            if self.step_idx < self.step_calls_in_bank:
                self.accepting_actions = False

            else:

                self.accepting_actions = True

            new_action = self.update_action_after_step(self.current_simulation.featurized_objects,
                                                       given_idx=self.step_idx)

            _, final_obs = break_img(phyre.observations_to_float_rgb(self.current_simulation.images[self.step_idx]))

            return final_obs, \
                   self.find_reward(), \
                   done , \
                   {'fo': self.current_simulation.featurized_objects.full_features[self.step_idx],
                    'action_body_idx': action_body_idx,
                    'sim_done': sim_done,
                    'new_action': np.asarray(new_action[0])}

    def find_reward(self):

        return -1

    def done(self, is_solved):
        if is_solved == phyre.SimulationStatus.NOT_SOLVED:
            return False
        else:
            return True


    def reset(self):

        self._set_steps_and_initial_task(self.base_string, self.specific_task)
        self._set_initial_values()
        _, root_img = break_img(phyre.observations_to_float_rgb(self.simulator.initial_scenes[self.SINGLE_ENV_TASK_IDX]))

        return root_img, self.root_task, self.rec_init_action


    def render_at_one_idx(self, idx):

        if self.current_simulation is None:
            raise Exception("Cant render at timestep before atleast one step action")

        try:
            self.show_obs(self.current_simulation.images[idx])

        except:
            print('Out of bounds idx')


    def render_start_with_action(self, mode='human', full_render=False):

        if self.current_simulation is None:
            raise Exception("Noob... how can you render start with action if you havent set action -_-")


        else:

            if mode=='human':
                self.show_obs(self.current_simulation.images[0])
                return None

            elif mode=='rgb':
                return break_img(phyre.observations_to_float_rgb(self.current_simulation.images[0])) if not full_render else phyre.observations_to_float_rgb(self.current_simulation.images[0])

            else:
                raise Exception("Invalid choice of rendering")

    def render(self, mode='human', full_render=False):

        if self.current_simulation is None:
            print('Warning: you are rendering at a time where you really should not be rendering as the action has not'
                  'been placed yet')
            im = self.simulator.initial_scenes[self.SINGLE_ENV_TASK_IDX]
            if mode == 'human':

                self.show_obs(im)
                return None

            elif mode == 'rgb':

                return break_img(phyre.observations_to_float_rgb(im)) if not full_render else phyre.observations_to_float_rgb(im)

            else:
                raise NotImplementedError("This image has not been implemented")

        else:

            if self.physical_mode == 0:

                if mode == 'human':
                    self.show_obs(self.current_simulation.images[self.step_idx])
                    return None


                elif mode == 'rgb':

                    return break_img(phyre.observations_to_float_rgb(self.current_simulation.images[self.step_idx])) if not full_render \
                        else phyre.observations_to_float_rgb(self.current_simulation.images[self.step_idx])

                else:

                    raise Exception("This has not been implemented")

            else:

                if mode == 'human':

                    self.show_obs(self.current_simulation.images[-1], name=self.rand_name)
                    self.rand_name += 1
                    return None


                elif mode == 'rgb':

                    return break_img(phyre.observations_to_float_rgb(self.current_simulation.images[-1])) if not full_render \
                        else phyre.observations_to_float_rgb(self.current_simulation.images[-1])

                else:

                    raise Exception("This has not been implemented")

    def close(self):

        self.current_simulation = None



def generate_actions_for_env(env, required=10):


    solver_actions = []
    for idx in range(required//2):
        solver_actions.append(env.sample_action(solve=True))

    for idx in range(int(required - required//2)):
        solver_actions.append(env.sample_action(solve=False))


    return solver_actions

def test_limits_for_template(template='00000', solo=False):

    from itertools import repeat
    from copy import deepcopy

    env = PhyreSingleTaskEnv(physical_mode='physics',
                             action_mode=0,
                             observation_mode=0,
                             task_string=template,
                             specific_task=False,
                             stride=60)

    new_observation = env.reset()
    solver_actions = generate_actions_for_env(env, required=10)
    names = [f'{each}.gif' for each in range(len(solver_actions))]

    if not solo:
        with concurrent.futures.ProcessPoolExecutor(max_workers=80) as executor:
            out_generator = executor.map(test_simulator_reset_with_template, repeat(deepcopy(env)), solver_actions, names, repeat(True))

        vx_mins = []
        vx_maxs = []
        vy_mins = []
        vy_maxs = []
        av_mins = []
        av_maxs = []

        for vx_stats, vy_stats, av_stats in out_generator:

            vx_mins.append(vx_stats[0])
            vx_maxs.append(vx_stats[1])

            vy_mins.append(vy_stats[0])
            vy_maxs.append(vy_stats[1])

            av_mins.append(av_stats[0])
            av_maxs.append(av_stats[1])

        return min(vx_mins), max(vx_maxs), min(vy_mins), max(vy_maxs), min(av_mins), max(av_maxs)

    else:
        test_simulator_reset_with_template(env, solver_actions[0], 'solo.gif', True)


def create_available_task_list(task_template, cache):

    train_task_ids, eval_task_ids = get_train_test("ball_cross_template", fold_id=0, use_test_split=True)

    all_solvable_task_ids = train_task_ids + eval_task_ids
    task_relevant_ids = []
    for solvable_task in all_solvable_task_ids:
        if task_template in solvable_task and solvable_task in cache._statuses_per_task:
            task_relevant_ids.append(solvable_task)

    return task_relevant_ids


def any_time_action_demo():

    env_cc = PhyreSingleTaskEnv(physical_mode='cc',
                                action_mode=0,
                                observation_mode=0,
                                task_string='00016:010')

    done = False
    action = env_cc.rec_init_action

    steps = 0
    while not done:

        _, _, done, info = env_cc.step(action)
        print(done)
        action = info['new_action']
        if action == 'b':
            action = 'blank'
        env_cc.render()
        steps += 1

def midway_action_trial():

    env_cc = PhyreSingleTaskEnv(physical_mode='cc',
                                action_mode=0,
                                observation_mode=0,
                                task_string='00020:007',
                                stride=20)

    done = False
    action = 'blank'

    steps = 0
    while steps < 6:

        _, _, done, info = env_cc.step(action)
        print(done)
        action = info['new_action']
        if action == 'b':
            action = 'blank'
        env_cc.render()
        steps += 1

    steps = 0
    action = np.asarray([0.327, 0.38 , 0.4, 0.5, 0.5, 0.5])
    while steps < 10:
        _, _, done, info = env_cc.step(action)
        print(done)
        action = info['new_action']
        if action == 'b':
            action = 'blank'
        env_cc.render()
        steps += 1


def test_simulator_reset_with_template(env, action, name, render=False):

    done = False
    info = {'sim_done': False}

    vx_set = set()
    vy_set = set()
    av_set = set()


    loops = 0
    if render:
        images = []
        action_images = []
    while (not done and not info['sim_done']):
        observation, _, done, info = env.step(action)
        vx_ = set(info['fo'][:, env.default_info['_VX_INDEX']].tolist())
        vy_ = set(info['fo'][:, env.default_info['_VY_INDEX']].tolist())
        av_ = set(info['fo'][:, env.default_info['_AV_INDEX']].tolist())

        vx_set = vx_set.union(vx_)
        vy_set = vy_set.union(vy_)
        av_set = av_set.union(av_)

        if loops == 0 and render:
            action_img, root_img = env.render_start_with_action(mode='rgb')
            images.append(root_img)
            action_images.append(action_img)

        if render:
            action_img, root_img = env.render(mode='rgb')
            images.append(root_img)
            action_images.append(action_img)

        if loops > 200:
            print(f'Loop not terminating for some reason')
            break

        loops += 1

    if render:
        gif_create(images, name)
        gif_create(action_images, f'action_{name}')

    vx_list = list(vx_set)
    vy_list = list(vy_set)
    av_list = list(av_set)
    return [min(vx_list), max(vx_list)], [min(vy_list), max(vy_list)], [min(av_list), max(av_list)]


def test_equivalence():

    import time


    env = PhyreSingleTaskEnv(physical_mode='physics',
                             action_mode=0,
                             observation_mode=0,
                             task_string='00000:000')


    env_cc = PhyreSingleTaskEnv(physical_mode='cc',
                                action_mode=0,
                                observation_mode=0,
                                task_string='00016:010')

    env.render()
    env_cc.render()

    action = env.rec_init_action

    _, _, done, info = env.step(action)

    #env.render()
    time.sleep(1)

    #Issue with jar toppling (known issues list)
    #Does not handle collision events well- add pipeline to listen for collision events
    #Enable API to allow user to decide if they want to allow occlusions
    #Consider the full role of contact forces well

    #env_cc.render()
    done = False
    action = env_cc.rec_init_action

    steps = 0
    while not done:

        _, _, done, info = env_cc.step(action)
        print(done)
        action = info['new_action']
        #action[2] += 0.1
        env.render_at_one_idx(steps+1)
        env_cc.render()

        #env_cc.render()

        steps += 1

def trial_101():
    print('All eval setups:', *phyre.MAIN_EVAL_SETUPS)
    # For now, let's select cross template for ball tier.
    eval_setup = 'ball_within_template'
    fold_id = 0  # For simplicity, we will just use one fold for evaluation.
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    tasks = dev_tasks[:5]

    sim = phyre.initialize_simulator(tasks, action_tier)
    #actions = sim.build_discrete_action_space(max_actions=100)
    #action = random.choice(actions)

    simulation = sim.simulate_action(0, None, need_images=True, need_featurized_objects=True)
    for obs in simulation.images:
        im_init = phyre.observations_to_float_rgb(obs)
        fig, ax = plt.subplots()
        i = ax.imshow(im_init)
        plt.show()

def make_gif():

    filenames = sorted(os.listdir('presentation_example'))

    print(filenames)
    import imageio
    images = []
    for filename in ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png', '15.png']:
        images.append(imageio.imread(os.path.join('presentation_example', filename)))
    imageio.mimsave('presentation.gif', images)




if __name__ == '__main__':

    '''
    Full list of ideas to explore with this environment
    Project title: "Robustness of World Models in learning tasks requiring simple dynamic reasoning"
    
    Hypothesis: "When you train on high action space environment which is not explicitly trained for intuitive dynamics, generalization is increasingly difficult"
    We adapt world models to a physical intuitive setting and force the emergence of a representation that is particularly amenable to modification.
    
    Eg. If we reverse gravity, only the aspect of our transition which models gravity should need to change, and it should only need to be negated to produce this answer.
    Eg. Transitions and states should both have required disentanglement. States should be disentangled for physical properties. Transitions should be disentangled for laws of physics.
    
    
    1. The relevance of world model training to pure physical task (with MPC, Actor Critic Algorithms)
    2. Combining this notion of "underlying dynamics" and expanding to other simple tasks (learn dynamics in bias free way)
    3. Train with reward signals that indicate closeness to a critical meetup event (collision-idea)- idea of next best relief
    4. Learn to predict across time (time contrastive networks but at discrete timesteps- don't predict always. only predict the next certain events)
    5. Forcing disentanglement of determinstic/non-deterministic predictions and showing the emergence of physical relationships
    6. Getting a direct estimate of predictability/uncertainty and moving to resolve these ideas of uncertainty - combine with geometric notions
    
    Read report on epistemic and aleotoric uncertainty
    '''

    #KNOWN ISSUES

    #1. LOOK INTO SIMULATION OF CONTACT PHYSICS (ARE CONTACT FORCES CAUSING ERROR)
    #2. LOOK INTO SIMULATION OF CHANGE OF DIRECTION

    #test_limits_for_template('00000', solo=True)
    #os.makedirs('presentation_example')

    make_gif()