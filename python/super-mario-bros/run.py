import os
import argparse
from typing import Tuple, List
import time
import timeit
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

import onnx
import onnxruntime as ort

from utils import make_env, ENV_KEYS


class MLP(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 1024):
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
    def __init__(self, buffer: ReplayBuffer, batch_size: int, batch_per_epoch: int = 10) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch

    def __iter__(self) -> Tuple:
        for _ in range(self.batch_per_epoch):
            s, a, r, done, next_s = self.buffer.sample(self.batch_size)
            for i in range(len(done)):
                yield s[i], a[i], r[i], done[i], next_s[i]


class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def update_reward(self, r: float) -> float:
        return r / 100.0

    def reset(self) -> None:
        self.s = self.env.reset()

    def get_action(self, Q: nn.Module, epsilon: float, device: str = 'cpu') -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            s = torch.tensor([self.s])

            if device not in ['cpu']:
                s = s.cuda(device)

            _, a = torch.max(Q(s), dim=1)
            return int(a.item())

    @torch.no_grad()
    def play_step(self, q: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        a = self.get_action(q, epsilon, device)

        next_s, r, done, _ = self.env.step(a)
        r = self.update_reward(r)
        exp = Experience(self.s, a, r, done, next_s)
        self.replay_buffer.append(exp)
        self.s = next_s

        if done:
            self.reset()
        return r, done

    @torch.no_grad()
    def eval_step(self, q: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        a = self.get_action(q, epsilon, device)
        next_s, r, done, _ = self.env.step(a)
        self.s = next_s

        if done:
            self.reset()
        return r, done


class DQNModule(pl.LightningModule):

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.env = make_env(self.hparams.env_key)

        self.obs_shape = self.env.observation_space.shape
        self.obs_size = reduce(operator.mul, self.obs_shape, 1)
        self.as_size = self.env.action_space.n

        self.Q = MLP(self.obs_size, self.as_size)
        self.Q_target = MLP(self.obs_size, self.as_size)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)

        epsilon = self.hparams.eps_max
        self.episode_count = 0
        self.total_reward = 0
        self.episode_reward = 0

        # initially fill replay buffer
        self.fill_replay_buffer(self.hparams.num_init_states)

    def fill_replay_buffer(self, n: int, epsilon: float = 1.0) -> None:
        for _ in range(n):
            self.agent.play_step(self.Q, epsilon=1.0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.Q(s)

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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        device = self.get_device(batch)
        epsilon = max(self.hparams.eps_min, self.hparams.eps_max -
                      self.episode_count / self.hparams.eps_last_frame)

        # just fill_replay_buffer states while updating Q
        with torch.no_grad():
            self.fill_replay_buffer(100, epsilon)

        # step through environment with agent
        r, done = self.agent.play_step(self.Q, epsilon, device)

        self.episode_count += 1
        self.episode_reward += r

        # calculates training loss
        loss = self.loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward += self.episode_reward
            self.episode_count = 0
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_interval == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        metrics = OrderedDict({
            'loss': loss,
            'episode_reward': self.episode_reward,
            'episode_count': self.episode_count,
            'epsilon': epsilon,
        })
        self.save_metrics(metrics)

        result = pl.TrainResult(minimize=loss)
        result.log('episode_reward', self.episode_reward,
                   prog_bar=True, logger=True)
        result.log('episode_count', self.episode_count,
                   prog_bar=True, logger=True)
        result.log('epsilon', epsilon, prog_bar=True, logger=True)

        return result

    def save_metrics(self, metrics: OrderedDict):
        for k, v in metrics.items():
            key = f'metrics/{k}'
            self.logger.experiment.add_scalar(key, v, self.global_step)

    def configure_optimizers(self) -> List[Optimizer]:
        optimizer = optim.Adam(self.Q.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        dataset = RLDataset(
            self.buffer, self.hparams.batch_size, self.hparams.batch_per_epoch)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_cpu_threads)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def get_device(self, batch) -> str:
        return batch[0].device.index if self.on_gpu else 'cpu'


def train(hparams) -> None:
    model = DQNModule(hparams)
    tb_logger = pl.loggers.TensorBoardLogger(f'{hparams.env_key}_logs/')
    log_dir = tb_logger.log_dir
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, '{epoch}')

    ckpt_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=3,
        verbose=False,
        mode='min',
        period=hparams.save_interval,
        save_last=True
    )

    trainer_args = {
        "logger": tb_logger,
        "checkpoint_callback": ckpt_callback,
        "gpus": hparams.gpus,
        "max_epochs": hparams.max_epochs + 1,
        "early_stop_callback": False,
        # "distributed_backend='dp',
    }

    if hparams.ckpt_path != "":
        trainer_args['resume_from_checkpoint'] = hparams.ckpt_path

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)


def test(hparams) -> None:
    assert hparams.ckpt_path != '', 'you must take an path to checkpoint when test'

    model = DQNModule.load_from_checkpoint(hparams.ckpt_path)
    model.eval()
    model.cuda()

    env = model.env
    agent = model.agent
    Q = model.Q
    device = 'cuda:0'

    interval = hparams.sync_interval

    for ep in range(hparams.max_epochs):
        epsilon = 0.01
        step = 0
        score = 0.0

        while True:
            step += 1
            env.render()
            time
            r, done = agent.eval_step(Q, epsilon, device)
            score += r

            if step % interval == 0 and ep != 0:
                print("episode :{}, step: {}, score : {:.1f}, epsilon : {:.1f}%".format(
                    ep, step, score, epsilon*100))

            if done:
                print("episode :{} done, score : {:.1f}".format(ep, score))
                break

    model.agent.env.close()


def export(hparams) -> str:
    assert hparams.ckpt_path != '', 'you must take an path to checkpoint when test'
    model = DQNModule.load_from_checkpoint(hparams.ckpt_path)
    model.eval()
    input_shape = model.obs_shape
    dummy = torch.randn(*input_shape)
    export_name = 'Q.{}.onnx'.format(model.hparams.env_key)
    export_path = os.path.join(os.getcwd(), export_name)
    torch.onnx.export(model.Q, dummy, export_path, verbose=True)
    return export_path


def evaluate(hparams) -> None:
    assert hparams.ckpt_path != '', 'you must take an path to checkpoint when test'

    if hparams.onnx_path == "":
        onnx_path = export(hparams)
        print('onnx exported: ', onnx_path)
    else:
        if os.path.exists(hparams.onnx_path):
            onnx_path = hparams.onnx_path
        else:
            raise NotImplementedError

    model = DQNModule.load_from_checkpoint(hparams.ckpt_path)
    env = model.env

    sess = ort.InferenceSession(onnx_path)
    print(sess.get_providers())
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    in_name = inputs[0].name
    in_shape = inputs[0].shape

    def get_action(s, epsilon):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            s = np.array(s)
            q_value = sess.run(None, {in_name: s.astype(np.float32)})
            a = np.argmax(q_value[0])
            return a

    interval = hparams.sync_interval

    for ep in range(hparams.max_epochs):
        epsilon = 0.01
        step = 0
        score = 0.0
        s = env.reset()

        while True:
            step += 1
            env.render()
            a = get_action(s, epsilon)
            next_s, r, done, _ = env.step(a)
            score += r
            s = next_s

            if step % interval == 0 and ep != 0:
                print("episode :{}, step: {}, score : {:.1f}, epsilon : {:.1f}%".format(
                    ep, step, score, epsilon*100))

            if done:
                print("episode :{} done, score : {:.1f}".format(ep, score))
                break

            time.sleep(1.0 / float(hparams.fps))

    model.agent.env.close()


def profile(hparams) -> None:
    assert hparams.ckpt_path != '', 'you must take an path to checkpoint when test'

    if hparams.onnx_path == "":
        onnx_path = export(hparams)
        print('onnx exported: ', onnx_path)
    else:
        if os.path.exists(hparams.onnx_path):
            onnx_path = hparams.onnx_path
        else:
            raise NotImplementedError

    model = DQNModule.load_from_checkpoint(hparams.ckpt_path)
    model.eval()
    env = model.env
    Q = model.Q

    sess = ort.InferenceSession(onnx_path)
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    in_name = inputs[0].name
    in_shape = inputs[0].shape

    device = 'cuda:0'

    def run_pytorch(s):
        return Q(s)

    def run_pytorch_cuda(s):
        Q.cuda()
        s = s.cuda()
        return Q(s)

    def run_ort(s):
        return sess.run(None, {in_name: s})

    def measure_time(f, s):
        rep = 50
        num = 50
        costs = timeit.repeat(lambda: f(s), repeat=rep, number=num)
        costs = np.array(costs) / num
        std = np.std(costs) * 1000
        mean = np.mean(costs) * 1000
        return mean, std

    def run(tag, f, s):
        mean, std = measure_time(f, s)
        print(tag)
        print("performance: %.2fms (std = %.2f)" % (mean, std))

    s = env.reset()
    s = s.astype(np.float32)
    s_torch = torch.tensor(s)

    run("pytorch_cpu", run_pytorch, s_torch)

    run("pytorch_cuda", run_pytorch_cuda, s_torch)

    sess.set_providers(['CPUExecutionProvider'])
    run("onnxruntime_cpu", run_ort, s)

    sess.set_providers(['CUDAExecutionProvider'])
    run("onnxruntime_cuda", run_ort, s)

    sess.set_providers(['TensorrtExecutionProvider'])
    run("onnxruntime_tensorrt", run_ort, s)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    modes = ['train', 'test', 'export', 'evaluate', 'profile']

    p = argparse.ArgumentParser()
    p.add_argument('mode', type=str, choices=modes, help='mode to run')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-epochs', type=int, default=10000)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--env-key', type=str,
                   choices=ENV_KEYS, default=ENV_KEYS[0])
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--sync-interval', type=int, default=1000)
    p.add_argument('--save-interval', type=int, default=20)
    p.add_argument('--replay-size', type=int, default=50000)
    p.add_argument('--num-init-states', type=int, default=2000)
    p.add_argument('--batch-per-epoch', type=int, default=10)
    p.add_argument('--eps-min', type=float, default=0.01)
    p.add_argument('--eps-max', type=float, default=0.08)
    p.add_argument('--eps-last-frame', type=int, default=5000)
    p.add_argument('--ckpt-path', type=str, default='')
    p.add_argument('--onnx-path', type=str, default='')
    p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--num-cpu-threads', type=int, default=16)
    p.add_argument('--fps', type=int, default=100)
    hparams = p.parse_args()

    if hparams.mode == 'train':
        train(hparams)
    elif hparams.mode == 'test':
        test(hparams)
    elif hparams.mode == 'export':
        export(hparams)
    elif hparams.mode == 'evaluate':
        evaluate(hparams)
    elif hparams.mode == 'profile':
        profile(hparams)
    else:
        raise NotImplementedError
