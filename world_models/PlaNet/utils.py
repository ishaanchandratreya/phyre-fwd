import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import sys
sys.path.append('/proj/vondrick/ishaan/phyre-fwd/agents/')
from phyre_gym import PhyreSingleTaskEnv
from gym_utils import gif_create_path
from env import PhyreEnv


def save_current_tasks_and_actions_for_phyre(list_of_tasks, list_of_actions, f_path, args):



  if not os.path.exists(f_path):
    os.makedirs(f_path)

  with open(os.path.join(f_path, 'log.txt'), 'w') as f:
    for idx, (task, action) in enumerate(zip(list_of_tasks, list_of_actions)):


      _env = PhyreSingleTaskEnv(physical_mode='physics',
                                action_mode=0,
                                observation_mode=0,
                                task_string=task,
                                specific_task=True,
                                stride=60,
                                seed=args.seed)

      images = []
      done = False
      play_action = action
      step_count = 0
      while not done:

        observation, _, done, info = _env.step(play_action)
        if step_count == 0:
          images.append(_env.render_start_with_action('rgb', full_render=True))

        images.append(_env.render('rgb', full_render=True))

        done = done or info['sim_done'] or _env.step_idx == args.max_episode_length
        image_obs = observation * 255
        play_action = info['new_action']

        step_count += 1



      gif_create_path(images, os.path.join(f_path, f'viz_{idx}.gif'))
      f.write(f'At this stage, {task} was attempted with action {action}.\n')




# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode'):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
    ys = np.asarray(ys_population, dtype=np.float32)
    ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
    trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
    data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
  else:
    data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
  plotly.offline.plot({
    'data': data,
    'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)


def write_video(frames, title, path=''):
  frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
  _, H, W, _ = frames.shape
  writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
  for frame in frames:
    writer.write(frame)
  writer.release()


class Dummy():

  def __init__(self, seed, max_episode_length):

    self.seed = seed
    self.max_episode_length = max_episode_length


if __name__=='__main__':
  tasks = ['00000:000','00000:001', '00000:004']


  args = Dummy(42, 1000)
  envs = [PhyreSingleTaskEnv(physical_mode='physics',
                            action_mode=0,
                            observation_mode=0,
                            task_string=task,
                            specific_task=True,
                            stride=60,
                            seed=42) for task in tasks]

  actions = [env.rec_init_action for env in envs]
  save_current_tasks_and_actions_for_phyre(tasks, actions, 'ici', args)
