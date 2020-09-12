import gym_maze
import gym
import sys
import numpy as np
import math
import random
from time import sleep


ACTION = ["U", "D", "R", "L"]
ACTION_TO_INT = {a: i for a, i in enumerate(ACTION)}
ACTION_INT = [i for i in range(len(ACTION))]


def print_action(a):
    print(f'action: {ACTION_TO_INT[a]}')


def get_action(s, Q, epsilon, p):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTION_INT, p=p)
    else:
        return np.argmax(Q[s])


# def sarsa(s, a, r, s_next, a_next, Q):
#     eta = 0.1
#     gamma = 0.9

#     if s_next == (4, 4):
#         Q[s, a] = Q[s, a] + eta *


if __name__ == "__main__":
    env = gym.make("maze-sample-5x5-v0", enable_render=False)
    s_shape = tuple((env.observation_space.high +
                     np.ones(env.observation_space.shape)).astype(int))
    n_action = env.action_space.n
    s = env.reset()

    try:
        while True:
            a = np.random.randint(0, n_action)
            print(a)
            s, r, done, info = env.step(a)
            # env.render()
            if done:
                # env.render(close=True)
                break
            print(f's: {s}')
            print_action(a)
            sleep(0.5)

    except KeyboardInterrupt:
        env.render(close=True)
        env.close()

    env.close()
