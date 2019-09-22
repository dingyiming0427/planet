# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from planet import control
from planet import tools


Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def dummy(config, params):
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = lambda: control.wrappers.ActionRepeat(
      control.DummyEnv, action_repeat)
  return Task('dummy', env_ctor, max_length, state_components)


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance',
      params)
  return Task('cartpole_balance', env_ctor, max_length, state_components)

def cartpole_balance_distractor(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cartpole_distractor', 'balance_distractor',
      params)
  return Task('cartpole_balance_distractor', env_ctor, max_length, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup',
      params)
  return Task('cartpole_swingup', env_ctor, max_length, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'finger', 'spin', params)
  return Task('finger_spin', env_ctor, max_length, state_components)

def finger_spin_distractor(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'finger_distractor', 'spin', params)
  return Task('finger_spin_distractor', env_ctor, max_length, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cheetah', 'run', params)
  return Task('cheetah_run', env_ctor, max_length, state_components)

def cheetah_run_distractor(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cheetah_distractor', 'run', params)
  return Task('cheetah_run_distractor', env_ctor, max_length, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch',
      params)
  return Task('cup_catch', env_ctor, max_length, state_components)

def cup_catch_distractor(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup_distractor', 'catch',
      params)
  return Task('cup_catch', env_ctor, max_length, state_components)

def hopper_hop(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'hopper', 'hop', params)
  return Task('hopper_hop', env_ctor, max_length, state_components)

def hopper_stand(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'hopper', 'stand', params)
  return Task('hopper_stand', env_ctor, max_length, state_components)

def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'walker', 'walk', params)
  return Task('walker_walk', env_ctor, max_length, state_components)

def walker_walk_distractor(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'walker_distractor', 'walk', params)
  return Task('walker_walk_distractor', env_ctor, max_length, state_components)


def reacher_easy(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'to_target']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'reacher', 'easy', params)
  return Task('reacher_easy', env_ctor, max_length, state_components)

def reacher_easy_distractor(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'to_target']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'reacher_distractor', 'easy', params)
  return Task('reacher_easy_distractor', env_ctor, max_length, state_components)

def pendulum_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'orientation', 'velocity']
  env_ctor = tools.bind(
    _dm_control_env, action_repeat, max_length, 'pendulum', 'swingup',
    params)
  return Task('pendulum_swingup', env_ctor, max_length, state_components)

def pointmass_easy(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
    _dm_control_env, action_repeat, max_length, 'point_mass', 'easy',
    params)
  return Task('pointmass_easy', env_ctor, max_length, state_components)

def pr2_reach(config, params):
  action_repeat = params.get('action_repeat', 1)
  max_length = 200 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'ee_goal']
  env_ctor = tools.bind(
    _dm_control_env, action_repeat, max_length, 'pr2_dm', 'reach',
    params, normalize=True)
  return Task('pr2_reach', env_ctor, max_length, state_components)

def gym_cheetah(config, params):
  # Works with `isolate_envs: process`.
  action_repeat = params.get('action_repeat', 1)
  max_length = 100 // action_repeat
  state_components = ['reward', 'state']
  env_ctor = tools.bind(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'HalfCheetah-v3')
  return Task('gym_cheetah', env_ctor, max_length, state_components)


def gym_racecar(config, params):
  # Works with `isolate_envs: thread`.
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = tools.bind(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'CarRacing-v0', obs_is_image=True)
  return Task('gym_racing', env_ctor, max_length, state_components)


def pt_dis(config, params):
  action_repeat = params.get('action_repeat', 1)
  max_length = 100
  state_components = ['reward', 'state']
  env_ctor = tools.bind(_gym_env, action_repeat, config.batch_shape[1], max_length, 'pt_dis', gym=False)
  return Task('pt_dis', env_ctor, max_length, state_components)


def _dm_control_env(
    action_repeat, max_length, domain, task, params, normalize=False,
    camera_id=None):
  if isinstance(domain, str):
    from dm_control import suite
    env = suite.load(domain, task)
  else:
    assert task is None
    env = domain()
  if camera_id is None:
    camera_id = int(params.get('camera_id', 0))
  env = control.wrappers.DeepMindWrapper(env, (64, 64), camera_id=camera_id)
  if normalize:
    env = control.wrappers.NormalizeActions(env)
  assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.MaximumDuration(env, max_length)
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env


def _gym_env(action_repeat, min_length, max_length, name, obs_is_image=False,
             gym=True):
  if gym == True:
    import gym
    env = gym.make(name)
  else:
    if name == 'pt_dis':
      from planet.envs.point_pos_distractor import PointEnv
      env = PointEnv(random_reset=False)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.NormalizeActions(env)
  env = control.wrappers.MinimumDuration(env, min_length)
  env = control.wrappers.MaximumDuration(env, max_length)
  if obs_is_image:
    env = control.wrappers.ObservationDict(env, 'image')
    env = control.wrappers.ObservationToRender(env)
  else:
    env = control.wrappers.ObservationDict(env, 'state')
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env
