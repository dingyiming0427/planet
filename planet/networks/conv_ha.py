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

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools


def encoder(obs, embedding_size=1024):
  """Extract deterministic features from an observation."""
  kwargs = dict(strides=2, activation=tf.nn.relu)
  obs_is_dict = isinstance(obs, dict)
  if obs_is_dict:
    hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())
  else:
    hidden = obs
  img_size = hidden.shape[2:].as_list()[0]
  if img_size == 64:
    hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
  hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs)
  hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs)
  hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs)
  hidden = tf.layers.flatten(hidden)
  assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
  if embedding_size != 1024:
    hidden = tf.layers.dense(hidden, units=embedding_size)
  if obs_is_dict:
    hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
        np.prod(hidden.shape[1:].as_list())])
  return hidden


def decoder(state, data_shape):
  """Compute the data distribution of an observation from its state."""
  kwargs = dict(strides=2, activation=tf.nn.relu)
  hidden = tf.layers.dense(state, 1024, None)
  hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
  hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs)
  if data_shape[0] == 64:
    hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
  hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
  mean = tf.layers.conv2d_transpose(hidden, 3, 6, strides=2)
  assert mean.shape[1:].as_list() == data_shape, mean.shape
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tfd.Normal(mean, 1.0)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
