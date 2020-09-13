import os
import argparse
from typing import Tuple, List
import time
from collections import namedtuple, deque, OrderedDict
from functools import reduce  # Required in Python 3
import operator

import numpy as np

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class MLP(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(MLP, self).__init__()
        self.obs_size = obs_size
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        x = x.float().reshape(-1, self.obs_size)
        return self.net(x)


Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward',
                                     'done', 'new_state'])


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, sample_size: int) -> Tuple:
        indices = np.random.choice(
            len(self.buffer), sample_size, replace=False)
        s, a, r, done, next_s = zip(*[self.buffer[i] for i in indices])
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(done, dtype=np.bool), np.array(next_s))


class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, batch_size: int = 200) -> None:
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Tuple:
        s, a, r, done, next_s = self.buffer.sample(self.batch_size)
        for i in range(len(done)):
            yield s[i], a[i], r[i], done[i], next_s[i]


class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, reward_func=None) -> None:
        self.env = env
        self.reward_func = reward_func
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self) -> None:
        self.s = self.env.reset()

    def get_action(self, Q: nn.Module, epsilon: float, device: str = 'cpu') -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            s = torch.tensor([self.s])

            if device not in ['cpu']:
                s = s.cuda(device)

            values = Q(s)
            _, a = torch.max(values, dim=1)
            return int(a.item())

    # @torch.no_grad()
    def play_step(self, q: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        a = self.get_action(q, epsilon, device)

        next_s, r, done, _ = self.env.step(a)
        r = r / 100
        exp = Experience(self.s, a, r, done, next_s)
        self.replay_buffer.append(exp)
        self.s = next_s

        if done:
            self.reset()
        return r, done


def make_env(env_key: str):
    assert env_key in env_keys
    if env_key == 'cartpole':
        name = 'CartPole-v0'
        return gym.make(name)
    elif env_keys == 'super-mario-bros':
        name = 'SuperMarioBros-v0'
        env = gym.make(name)
        return JoypadSpace(env, SIMPLE_MOVEMENT)
    else:
        raise NotImplementedError


class DQNModule(pl.LightningModule):

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.env = make_env(self.hparams.env_key)

        obs_shape = self.env.observation_space.shape
        obs_size = reduce(operator.mul, obs_shape, 1)
        as_size = self.env.action_space.n

        self.Q = MLP(obs_size, as_size)
        self.Q_target = MLP(obs_size, as_size)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)

        self.epsilon = self.hparams.eps_max
        self.total_reward = 0
        self.episode_reward = 0

        # initially fill replay buffer
        for i in range(self.hparams.init_steps):
            self.agent.play_step(self.Q, epsilon=1.0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        out = self.Q(s)
        return out

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        s, a, r, done, next_s = batch
        # estimation
        # it picks up the Q values with taken actions in this state
        estimation = self.Q(s).gather(1, a.unsqueeze(-1)).squeeze(-1)

        # V(s') = max(a) { Q(s) } with 0 if done
        with torch.no_grad():
            V_prime = self.Q_target(next_s).max(1).values
            V_prime[done] = 0.0
            V_prime = V_prime.detach()

        # it's Q-learning so, target is R(s) + γ・V(s')
        target = r + self.hparams.gamma * V_prime

        # return nn.MSELoss()(estimation, target)
        return F.smooth_l1_loss(estimation, target)

    def on_epoch_start(self):
        self.epsilon = max(self.hparams.eps_min, self.hparams.eps_max -
                           self.current_epoch / self.hparams.eps_last_frame)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        device = self.get_device(batch)

        # step through environment with agent
        r, done = self.agent.play_step(self.Q, self.epsilon, device)
        self.episode_reward += r

        # calculates training loss
        loss = self.loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward += self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_interval == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        self.logger.experiment.add_scalar(
            'metrics/loss', loss, self.global_step)
        self.logger.experiment.add_scalar(
            'metrics/episode_reward', self.episode_reward, self.global_step)
        self.logger.experiment.add_scalar(
            'metrics/epsilon', self.epsilon, self.global_step)

        result = pl.TrainResult(minimize=loss)
        result.log('loss x 1000', loss * 1000, prog_bar=True, logger=True)
        result.log('episode_reward', self.episode_reward,
                   prog_bar=True, logger=True)
        result.log('epsilon', self.epsilon, prog_bar=True, logger=True)
        result.log('steps', self.global_step, prog_bar=True)

        return result

    def configure_optimizers(self) -> List[Optimizer]:
        optimizer = optim.Adam(self.Q.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        dataset = RLDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def get_device(self, batch) -> str:
        return batch[0].device.index if self.on_gpu else 'cpu'


def train(hparams) -> None:
    model = DQNModule(hparams)
    tb_logger = pl.loggers.TensorBoardLogger('logs/')
    # log_dir = tb_logger.log_dir
    # ckpt_dir = os.path.join(log_dir, 'checkpoints')
    # ckpt_path = os.path.join(ckpt_dir, '{epoch}_{val_loss:.2f}_{reward:.2f}')

    # ckpt_callback = ModelCheckpoint(
    #     filepath=ckpt_path,
    #     save_top_k=3,
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min',
    #     period=10,
    #     save_last=True
    # )

    trainer_args = {
        "logger": tb_logger,
        # "checkpoint_callback": ckpt_callback,
        "gpus": hparams.gpus,
        "max_epochs": hparams.max_epochs,
        "early_stop_callback": False,
        "val_check_interval": 10,
        # "distributed_backend='dp',
    }

    if hparams.ckpt_path != "":
        trainer_args['resume_from_checkpoint'] = hparams.ckpt_path

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)


def test(hparams) -> None:
    assert hparams.ckpt_path != '', 'you must take an path to checkpoint when test'

    model = DQNModule.load_from_checkpoint(hparams.ckpt_path)
    env = model.env
    Q = model.Q
    agent = model.agent
    epsilon = max(hparams.eps_min, hparams.eps_max -
                  model.global_step / hparams.eps_last_frame)
    device = 'cpu'
    interval = hparams.sync_interval

    for ep in range(hparams.max_epochs):
        epsilon = 0.01
        s = env.reset()
        done = False
        step = 0
        score = 0.0

        while not done:
            step += 1
            a = agent.get_action(Q, epsilon, device)
            env.render()
            next_s, r, done, _ = env.step(a)
            s = next_s
            score += r

            if done:
                print('done')

            if step % interval == 0 and ep != 0:
                print("episode :{}, step: {}, score : {:.1f}, epsilon : {:.1f}%".format(
                    ep, step, score, epsilon*100))

    agent.env.close()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    modes = ['train', 'test']
    env_keys = ['cartpole', 'super-mario-bros']

    p = argparse.ArgumentParser()
    p.add_argument('mode', type=str, choices=modes, help='mode to run')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-epochs', type=int, default=10000)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--env-key', type=str,
                   choices=env_keys, default=env_keys[0])
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--sync-interval', type=int, default=1000)
    p.add_argument('--replay-size', type=int, default=50000)
    p.add_argument('--init-steps', type=int, default=2000)
    p.add_argument('--eps-min', type=float, default=0.01)
    p.add_argument('--eps-max', type=float, default=0.08)
    p.add_argument('--eps-last-frame', type=int, default=5000)
    p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--ckpt-path', type=str, default='')
    hparams = p.parse_args()

    if hparams.mode == 'train':
        train(hparams)
    elif hparams.mode == 'test':
        test(hparams)
    else:
        raise NotImplementedError
