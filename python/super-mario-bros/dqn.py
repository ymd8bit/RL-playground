import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import collections
import random
from functools import reduce  # Required in Python 3
import operator

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 16


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=1024):
        super(MLP, self).__init__()
        self.obs_size = obs_size
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        x = x.reshape(-1, self.obs_size)
        return self.net(x)

    @torch.no_grad()
    def sample_action(self, obs, epsilon):
        if torch.cuda.is_available():
            obs = obs.cuda()
        action = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return action.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        if torch.cuda.is_available():
            s = s.cuda()
            s_prime = s_prime.cuda()
            done_mask = done_mask.cuda()
            r = r.cuda()
            a = a.cuda()

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    obs_shape = env.observation_space.shape
    obs_size = reduce(operator.mul, obs_shape, 1)
    action_size = env.action_space.n

    q = MLP(obs_size, action_size)
    q_target = MLP(obs_size, action_size)
    q_target.load_state_dict(q.state_dict())
    if torch.cuda.is_available():
        q = q.cuda()
        q_target = q_target.cuda()

    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)
                      )  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(np.array(s)).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()
