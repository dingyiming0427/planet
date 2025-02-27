import os
from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from gym import utils
import tensorflow as tf


class PointEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, random_reset=True, ptsize=2):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets',
                                              'point_pos_distractor.xml'), 2)

        self.random_reset = random_reset

    def step(self, a):
        desired_pos = self.get_xy() + np.clip(a, -20, 20) / 30.
        desired_pos = np.clip(desired_pos, -2.8, 2.8)
        self.reset_model(pos=desired_pos)
        ob = self._get_obs().copy()

        reward = self.reward(None, None, ob)
        done = False
        return ob, reward, done, dict(distance=-reward)

    def reset_model(self, pos=None):
        if pos is None:
            if self.random_reset:
                qpos_pt = self.init_qpos[:2] + np.random.uniform(size=2, low=-2., high=2.)
            else:
                qpos_pt = self.init_qpos[:2] + np.random.uniform(size=2, low=-0.01, high=0.01)
        else:
            qpos_pt = self.init_qpos[:2] + pos
        qvel = self.init_qvel + np.random.uniform(size=self.model.nv, low=-0.01, high=0.01)

        qpos_distractors = np.random.uniform(-4, 4, size=4)
        while np.any(np.absolute(np.concatenate([qpos_pt, qpos_pt]) - qpos_distractors) < 0.3): # keep sampling until the distractor is far from the object
            qpos_distractors = np.random.uniform(-4, 4, size=4)
        qpos = np.concatenate([qpos_pt, qpos_distractors])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 2:
            return -np.linalg.norm(obs_next[:, :2] - np.array([2, 2]), axis=1)
        elif obs_next.ndim == 1:
            return -np.linalg.norm(obs_next[:2] - np.array([2, 2]))
        else:
            raise NotImplementedError

    def tf_reward(self, obs, act, obs_next):
        return - tf.norm(obs_next[:, :2] - np.array([2, 2]), axis=1)

    def _get_obs(self):
        return self.sim.data.qpos.ravel()

    def viewer_setup(self):
        pass

    def render(self, mode='human', width=100, height=100):
        if mode == 'human':
            super().render(mode=mode)
        else:
            data = self.sim.render(width, height, camera_name='main')
            return data[::-1, :, :]

    def get_xy(self):
        return self.sim.data.qpos.ravel()[:2]

    # def log_diagnostics(self, paths, prefix=''):
    #     """
    #     Logs env-specific diagnostic information
    #
    #     Args:
    #         paths (list) : list of all paths collected with this env during this iteration
    #         prefix (str) : prefix for logger
    #     """
    #     final_distance = [path['env_infos']['distance'] for path in paths]
    #     logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_distance))


if __name__ == "__main__":
    env = PointEnv()
    env.reset()
    for _ in range(1000):
        # import pdb; pdb.set_trace()
        env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
