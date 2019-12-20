import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gym

env = gym.make("procgen:procgen-coinrun-v0")
print('here')

s = env.reset()
n = env.action_space.n

while True:
    env.render()
    a = np.random.randint(0, n-1, dtype=np.int32)
    s = env.step(a)

env.close()
