from math import inf
import torch
from torch import jit
from torch.nn import functional as F
import argparse
from torchvision.utils import make_grid, save_image
from utils import save_current_tasks_and_actions_for_phyre
import numpy as np
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher, PhyreEnvBatcher
from torch import nn, optim
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
import os

def get_args():
    parser = argparse.ArgumentParser(description='PlaNet')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--env', type=str, default='00000',
                        help='Phyre Suite environment')
    parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
    parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
    parser.add_argument('--experience-size', type=int, default=1000000, metavar='D',
                        help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F),
                        help='Model activation function')
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E',
                        help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
    parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
    parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
    parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
    parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
    parser.add_argument('--seed-episodes', type=int, default=20, metavar='S', help='Seed episodes')
    parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B', help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=10, metavar='L', help='Chunk size')
    parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D',
                        help='Latent overshooting distance/latent overshooting weight for t = 1')
    parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1',
                        help='Latent overshooting KL weight for t > 1 (0 to disable)')
    parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1',
                        help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
    parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
    parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
    parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
    parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
                        help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
    parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value')
    # Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
    parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
    parser.add_argument('--planning-horizon', type=int, default=12, metavar='H', help='Planning horizon distance')
    parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I',
                        help='Planning optimisation iterations')
    parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
    parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=1, metavar='E', help='Number of test episodes')
    parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I',
                        help='Checkpoint interval (episodes)')
    parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
    # --models results
    parser.add_argument('--models', type=str, default='/proj/vondrick/ishaan/phyre-fwd/world_models/PlaNet/results/phyre_00000_5/models_200.pth', metavar='M', help='Load model checkpoint')
    parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--t_viz', action='store_true',
                        help='Visualize the learnt transition model in the latent space')
    parser.add_argument('--stride', type=int, default=60, help='Stride of the simulation in moving forward')
    parser.add_argument('--context', type=int, default=3, help='Number of context before using transition dynamics')
    parser.add_argument('--force_action', type=int, help='If provided, set to exactly the particular action under task.'
                                                         'Note: if required to force exact pair, just set --env to what '
                                                         'you want as API has force pair option')
    parser.add_argument('--num_viz', type=int, default=5, help='Number of different visuals to compare and check')

    args = parser.parse_args()

    return args


def add_action_to_image(given_action, image):

    pass


class LookAheadVisualizer():

  def __init__(self, transition_model, observation_model, lookahead):

    super().__init__()
    self.transition_model = transition_model
    self.observation_model = observation_model
    self.lookahead = lookahead
    self.candidates = 1

  def forward(self, belief, state, action):

    self.transition_model.eval()
    self.observation_model.eval()

    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    beliefs, states, _, _ = self.transition_model(state, action.unsqueeze(dim=0), belief)

    observations = []
    for each_belief, each_state in zip(beliefs, states):
      observations.append(self.observation_model(each_belief, each_state).cpu())


    return observations[0], beliefs[0], states[0]


def produce_belief_state_and_action(env, default_env, transition_model, encoder, belief, posterior_state, action,
                                         observation, min_action=-inf, max_action=inf, info=None):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history


    next_belief, _, _, _, next_posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))
    next_belief, next_posterior_state = next_belief.squeeze(dim=0), next_posterior_state.squeeze(dim=0)

    if info is None:
        action = default_env.sample_random_action() if isinstance(env, PhyreEnvBatcher) else env.sample_random_action()  # Get action from planner(q(s_t|o≤t,a<t), p)
        init_sample_action = action
    else:
        action = info
        init_sample_action = None

    action = action.to(posterior_state.device)

    if isinstance(env, PhyreEnvBatcher):
        action = torch.stack([action] * env.n, axis=0)

    elif len(action.shape) == 1:
        action = action.unsqueeze(dim=0)
    else:
        print('Nothing to do here')

    action.clamp_(min=min_action, max=max_action)  # Clip action range

    next_observation, reward, done, info = env.step(action.cpu() if isinstance(env, PhyreEnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)

    return next_belief, next_posterior_state, action, next_observation, reward, done, info, init_sample_action

def transition_viz():

    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lookahead = 1
    test_env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, int(args.stride))

    default_env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat,
                      args.bit_depth, int(args.stride))

    transition_model = TransitionModel(args.belief_size, args.state_size, test_env.action_size, args.hidden_size,
                                       args.embedding_size, args.activation_function).to(device=args.device)
    observation_model = ObservationModel(args.symbolic_env, test_env.observation_size, args.belief_size, args.state_size,
                                         args.embedding_size, args.activation_function).to(device=args.device)

    encoder = Encoder(args.symbolic_env, test_env.observation_size, args.embedding_size, args.activation_function).to(
        device=args.device)

    param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(encoder.parameters())
    optimiser = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate,
                           eps=args.adam_epsilon)

    if args.models == '':
        raise Exception("For visualization, model path must be provided")


    naming_list = args.models.split('/')
    root_folder = '/'.join(naming_list[:-1])
    model_timing = naming_list[-1].split('.')[0]

    viz_folder = os.path.join(root_folder, 'transition_viz', model_timing)
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    if args.models is not '' and os.path.exists(args.models):
        model_dicts = torch.load(args.models)
        transition_model.load_state_dict(model_dicts['transition_model'])
        observation_model.load_state_dict(model_dicts['observation_model'])
        encoder.load_state_dict(model_dicts['encoder'])
        optimiser.load_state_dict(model_dicts['optimiser'])

    viz = LookAheadVisualizer(transition_model, observation_model, lookahead=int(args.lookahead))


    global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device),
                          torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
    free_nats = torch.full((1,), args.free_nats, dtype=torch.float32,
                           device=args.device)  #

    transition_model.eval()
    observation_model.eval()
    encoder.eval()
    # Initialise parallelised test environments

    with torch.no_grad():

        for viz_no in range(args.num_viz):
            # defaults, test_episodes 10,

            observation, total_rewards, video_frames = test_env.reset(), np.zeros((args.test_episodes,)), []
            current_tasks, _ = test_env.return_current_values()

            belief          = torch.zeros(args.test_episodes, args.belief_size, device=args.device)
            posterior_state = torch.zeros(args.test_episodes, args.state_size, device=args.device)
            action          = torch.zeros(args.test_episodes, test_env.action_size, device=args.device)
            info = None

            final_belief, final_posterior_state = None, None
            video_frames_context_real = []
            video_frames_context_reconstructed_and_transition = []

            next_observation = None
            data_tasks = [current_tasks]
            data_col_actions = []

            for i in range(0, int(args.context)):

                video_frames_context_real.append(observation+0.5)
                final_belief, final_posterior_state, action, next_observation, reward, done, info, init_action = produce_belief_state_and_action(
                    test_env,
                    default_env,
                    transition_model,
                    encoder, belief,
                    posterior_state,
                    action,
                    observation.to(
                        device=args.device),
                    test_env.action_range[
                        0],
                    test_env.action_range[
                        1],
                    info)

                if init_action is not None:
                    data_col_actions.append(init_action)

                video_frames_context_reconstructed_and_transition.append(observation_model(final_belief, final_posterior_state).cpu() + 0.5)
                observation = next_observation

            save_current_tasks_and_actions_for_phyre(data_tasks, data_col_actions, os.path.join(viz_folder, f'{viz_no}'), args)

            for i in range(0, args.chunk_size - int(args.context)):

                observation, final_belief, final_posterior_state = viz.forward(final_belief, final_posterior_state, action)
                video_frames_context_real.append(next_observation.cpu() + 0.5)
                video_frames_context_reconstructed_and_transition.append(observation.cpu() + 0.5)

                _, _, action, next_observation, _, _, _, _ = produce_belief_state_and_action(
                    test_env,
                    default_env,
                    transition_model,
                    encoder, belief,
                    posterior_state,
                    action,
                    observation.to(
                        device=args.device),
                    test_env.action_range[
                        0],
                    test_env.action_range[
                        1],
                    info)

            real_viz = torch.cat(video_frames_context_real, dim=3)
            imagined_and_predicted_viz = torch.cat(video_frames_context_reconstructed_and_transition, dim=3)

            full_viz = torch.cat([real_viz, imagined_and_predicted_viz], dim=2)

            save_image(full_viz, os.path.join(viz_folder, f'{viz_no}.png'))

            #TODO add API to jointly visualize action
            #regress with gtruth featurized values

    # Close test environments
    test_env.close()

if __name__=='__main__':

    transition_viz()






