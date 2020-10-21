import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

ENV_KEYS = ['cartpole', 'super-mario-bros', 'pong']


def make_env(key: str):
    if key == 'cartpole':
        name = 'CartPole-v0'
        return gym.make(name)
    elif key == 'pong':
        name = 'Pong-v0'
        return gym.make(name)
    elif key == 'super-mario-bros':
        name = 'SuperMarioBros-v0'
        env = gym.make(name)
        return JoypadSpace(env, SIMPLE_MOVEMENT)
    else:
        raise NotImplementedError
