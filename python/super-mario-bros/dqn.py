import argparse
import collections
import random
from functools import reduce  # Required in Python 3
import operator

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, buffer_limit):
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
    def __init__(self, obs_size, action_size, hidden_size=128):
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

    def sample_action(self, obs, epsilon):
        action = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return action.argmax().item()


def train_network(q, q_target, memory, optimizer, args):
    loss_list = []

    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(args.batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + args.gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        loss_list.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.sum(torch.Tensor(loss_list))


def train(args):
    env = gym.make('CartPole-v1')
    obs_shape = env.observation_space.shape
    obs_size = reduce(operator.mul, obs_shape, 1)
    action_size = env.action_space.n
    q = MLP(obs_size, action_size)
    q_target = MLP(obs_size, action_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(args.buffer_limit)

    score = 0.0
    interval = args.sync_interval
    optimizer = optim.Adam(q.parameters(), lr=args.lr)

    for ep in range(args.max_epochs):
        # Linear annealing from 8% to 1%
        epsilon = max(0.01, 0.08 - 0.01*(ep/200))
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > args.init_steps:
            loss = train_network(q, q_target, memory, optimizer, args)
        else:
            loss = 0.0

        if ep % interval == 0 and ep != 0:
            q_target.load_state_dict(q.state_dict())
            print("episode :{}, score : {:.1f}, loss : {:.8f}, epsilon : {:.1f}%".format(
                ep, score/interval, loss, epsilon*100))
            score = 0.0

        if ep % args.save_interval == 0 and ep != 0:
            torch.save(q.state_dict(), 'out.pth')

    env.close()


def test(args):
    assert args.weight_path is not None

    env = gym.make('CartPole-v1')
    obs_shape = env.observation_space.shape
    obs_size = reduce(operator.mul, obs_shape, 1)
    action_size = env.action_space.n

    q = MLP(obs_size, action_size)
    q.load_state_dict(torch.load(args.weight_path))
    q.eval()

    interval = args.sync_interval

    for ep in range(args.max_epochs):
        epsilon = 0.01
        s = env.reset()
        done = False
        step = 0
        score = 0.0

        while not done:
            step += 1
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            env.render()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r

            if step % interval == 0 and ep != 0:
                print("episode :{}, step: {}, score : {:.1f}, epsilon : {:.1f}%".format(
                    ep, step, score, epsilon*100))

    env.close()


if __name__ == '__main__':
    torch.manual_seed(0)
    modes = ['train', 'test']

    p = argparse.ArgumentParser()
    p.add_argument('mode', type=str, choices=modes)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-epochs', type=int, default=3000)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--buffer-limit', type=int, default=50000)
    p.add_argument('--init-steps', type=int, default=2000)
    p.add_argument('--sync-interval', type=int, default=20)
    p.add_argument('--save-interval', type=int, default=100)
    p.add_argument('--weight-path', type=str, default=None)
    args = p.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        test(args)
