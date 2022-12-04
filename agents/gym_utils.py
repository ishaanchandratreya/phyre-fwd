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
from phyre.interface.scene import ttypes as scene_if
from train2 import get_train_test
import phyre
from gym import utils
import concurrent.futures
from gym.utils import seeding
import neural_agent
import matplotlib.animation as animation
import matplotlib.cm as cm


from phyre.action_mappers import Unscaler
np.random.seed(1)


def gif_create_path(images, path):

    frames = []
    fig = plt.figure()
    for i in range(len(images)):
        frames.append([plt.imshow(images[i], cmap=cm.Greys_r, animated=True)])


    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(path)
    #play_gif('temp_views/example.gif')

def gif_create(images, name):

    frames = []
    fig = plt.figure()
    for i in range(len(images)):
        frames.append([plt.imshow(images[i], cmap=cm.Greys_r, animated=True)])


    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(f'temp_views/{name}')
    #play_gif('temp_views/example.gif')


def play_gif(path):

    from PIL import Image
    im = Image.open(path)
    im.show()
