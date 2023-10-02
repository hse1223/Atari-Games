from collections import deque

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ClipRewardEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor # stable_baselines3 requires gym==0.17.3


def make_atari_deepmind(env_id, max_episode_steps=None, scale_values=False, clip_rewards=True, seed=None): 

    # Wrappers here are all for gym.Wrapper, not VecEnvWrapper.
    # This reduces a lot of memory compared to AtariPreprocessing and FrameStack in gym.wrappers.

    env = gym.make(env_id)
    if seed is not None:
        env.seed(seed)

    env = NoopResetEnv(env, noop_max=30)

    if 'NoFrameskip' in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = EpisodicLifeEnv(env)

    env = WarpFrame(env)

    if scale_values:
        env = ScaledFloatFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)   # env.observation_space.sample().shape == (84-84-1)

    env = TransposeImageObs(env, op=[2, 0, 1])  # Convert to torch order (C, H, W) => env.observation_space.sample().shape == (1-84-84)

    env = Monitor(env)

    # if seed is not None:
    #     env.seed(seed)

    return env

class TransposeImageObs(gym.ObservationWrapper):
    def __init__(self, env, op):
        super().__init__(env)
        assert len(op) == 3, "Op must have 3 dimensions"

        self.op = op

        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [
                obs_shape[self.op[0]],
                obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(self.op[0], self.op[1], self.op[2])


class BatchedPytorchFrameStack(VecEnvWrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        self.k = k
        self.batch_stacks = [deque([], maxlen=k) for _ in range(env.num_envs)]
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=env.observation_space.low[0][0][0], high=env.observation_space.high[0][0][0], shape=((shp[0] * k,) + shp[1:]),
                                                dtype=env.observation_space.dtype)
        # self.observation_space = gym.spaces.Box(low=0., high=255., shape=((shp[0] * k,) + shp[1:]),
        #                                         dtype=env.observation_space.dtype)
        self.env = env

    def reset(self):
        obses = self.env.reset()
        for _ in range(self.k):
            for i, obs in enumerate(obses):
                self.batch_stacks[i].append(obs.copy())
        return self._get_ob()

    def step_wait(self):
        obses, reward, done, info = self.env.step_wait()
        for i, obs_frame in enumerate(obses):
            self.batch_stacks[i].append(obs_frame)

        ret_ob = self._get_ob()
        return ret_ob, reward, done, info

    def _get_ob(self):
        return [PytorchLazyFrames(list(batch_stack), axis=0) for batch_stack in self.batch_stacks]

    def _transform_batched_frame(self, frame):
        return [f for f in frame]

class PytorchLazyFrames(object):
    def __init__(self, frames, axis=0):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.axis = axis

    def __len__(self):
        return len(self.get_frames())

    def get_frames(self):
        """Get Numpy representation without dumping the frames."""
        return np.concatenate(self._frames, axis=self.axis)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)