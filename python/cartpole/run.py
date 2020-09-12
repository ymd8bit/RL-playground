import gym
import numpy as np
from collections import deque, namedtuple
from tensorboardX import SummaryWriter
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# for play
NUM_EPISODES = 500
MAX_STEPS = 200
GAMMA = 0.99
WARMUP = 10

# for search
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8
REWARD_STEPS = 10

# for memory
E_START = 1.0
E_STOP = 0.01
E_DECAY_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 32


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class Memory():
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class CartPole():

    Transition = namedtuple(
        'Transition', ('state', 'action', 'reward', 'next_state'))

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.n_state = self.env.observation_space.shape[0]  # 4
        self.n_action = self.env.action_space.n  # 2
        self.hidden_size = 16
        self.q = Network(self.n_state, self.hidden_size, self.n_action)
        self.optimizer = optim.Adam(self.q.parameters(), lr=0.00001)
        self.memory = Memory(MEMORY_SIZE)

    def Q(self, s):
        if isinstance(s, np.ndarray):
            if s.dtype != np.float32:
                s = s.astype(np.float32)
            y = self.q(torch.from_numpy(s))
        elif isinstance(s, torch.Tensor):
            y = self.q(s)
        else:
            raise NotImplementedError

        return y.detach().numpy()

    def update(self, done):
        inputs = np.zeros((BATCH_SIZE, self.n_state))
        targets = np.zeros((BATCH_SIZE, self.n_action))
        minibatch = self.memory.sample(BATCH_SIZE)

        for i, (s, a, r, s_next) in enumerate(minibatch):
            inputs[i] = s

            if not done:
                q = self.Q(s_next)
                value = r + GAMMA * np.amax(q[0])
            else:
                value = r

            targets[i] = self.Q(s)
            targets[i][a] = value

    def epsilon_greedy_action_choice(self, s, epsilon):
        if epsilon > np.random.rand():
            return self.env.action_space.sample()

        y = self.Q(s)
        return np.argmax(y)

    def make_state(self, s):
        return np.reshape(s, [1, self.n_state]).astype(np.float32)

    def play(self):

        total_step = 0
        success_count = 0
        for episode in range(1, NUM_EPISODES+1):
            step = 0
            s = self.make_state(self.env.reset())

            for _ in range(1, MAX_STEPS + 1):
                step += 1
                total_step += 1

                epsilon = E_STOP + (E_START - E_STOP) * \
                    np.exp(-E_DECAY_RATE * total_step)
                # print(f'epsilon: {epsilon}')
                # print(f's: {type(s)}{s}')
                # print(f'a: {type(a)}{a}')
                # print(f's_next: {type(s_next)}{s_next}')
                a = self.epsilon_greedy_action_choice(s, epsilon)
                s_next, _, done, _ = self.env.step(a)
                self.env.render()
                sleep(0.1)

                if step > WARMUP:
                    self.memory.add((s, a, r, s_next))

                if done:
                    if step >= 200:
                        success_count += 1
                        r = 1
                    else:
                        success_count = 0
                        r = 0

                    s_next = self.make_state(self.env.reset())

                else:
                    r = 0
                    if step > WARMUP:
                        self.memory.add((s, a, r, s_next))

                s = s_next

                # if len(self.memory) >= BATCH_SIZE:
                #     self.update(done)


if __name__ == "__main__":
    game = CartPole()
    game.play()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--baseline", default=False,
#                         action='store_true', help="Enable mean baseline")
#     args = parser.parse_args()

#     env = gym.make("CartPole-v0")
#     writer = SummaryWriter(comment="-cartpole-pg" +
#                            "-baseline=%s" % args.baseline)

#     net = PGN(env.observation_space.shape[0], env.action_space.n)
#     print(net)

#     agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
#                                    apply_softmax=True)
#     exp_source = ptan.experience.ExperienceSourceFirstLast(
#         env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

#     optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

#     total_rewards = []
#     step_rewards = []
#     step_idx = 0
#     done_episodes = 0
#     reward_sum = 0.0

#     batch_states, batch_actions, batch_scales = [], [], []

#     for step_idx, exp in enumerate(exp_source):
#         reward_sum += exp.reward
#         baseline = reward_sum / (step_idx + 1)
#         writer.add_scalar("baseline", baseline, step_idx)
#         batch_states.append(exp.state)
#         batch_actions.append(int(exp.action))
#         if args.baseline:
#             batch_scales.append(exp.reward - baseline)
#         else:
#             batch_scales.append(exp.reward)

#         # handle new rewards
#         new_rewards = exp_source.pop_total_rewards()
#         if new_rewards:
#             done_episodes += 1
#             reward = new_rewards[0]
#             total_rewards.append(reward)
#             mean_rewards = float(np.mean(total_rewards[-100:]))
#             print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
#                 step_idx, reward, mean_rewards, done_episodes))
#             writer.add_scalar("reward", reward, step_idx)
#             writer.add_scalar("reward_100", mean_rewards, step_idx)
#             writer.add_scalar("episodes", done_episodes, step_idx)
#             if mean_rewards > 195:
#                 print("Solved in %d steps and %d episodes!" %
#                       (step_idx, done_episodes))
#                 break

#         if len(batch_states) < BATCH_SIZE:
#             continue

#         states_v = torch.FloatTensor(batch_states)
#         batch_actions_t = torch.LongTensor(batch_actions)
#         batch_scale_v = torch.FloatTensor(batch_scales)

#         optimizer.zero_grad()
#         logits_v = net(states_v)
#         log_prob_v = F.log_softmax(logits_v, dim=1)
#         log_prob_actions_v = batch_scale_v * \
#             log_prob_v[range(BATCH_SIZE), batch_actions_t]
#         loss_policy_v = -log_prob_actions_v.mean()

#         loss_policy_v.backward(retain_graph=True)
#         grads = np.concatenate([p.grad.data.numpy().flatten()
#                                 for p in net.parameters()
#                                 if p.grad is not None])

#         prob_v = F.softmax(logits_v, dim=1)
#         entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
#         entropy_loss_v = -ENTROPY_BETA * entropy_v
#         entropy_loss_v.backward()
#         optimizer.step()

#         loss_v = loss_policy_v + entropy_loss_v

#         # calc KL-div
#         new_logits_v = net(states_v)
#         new_prob_v = F.softmax(new_logits_v, dim=1)
#         kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
#         writer.add_scalar("kl", kl_div_v.item(), step_idx)

#         writer.add_scalar("baseline", baseline, step_idx)
#         writer.add_scalar("entropy", entropy_v.item(), step_idx)
#         writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
#         writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
#         writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
#         writer.add_scalar("loss_total", loss_v.item(), step_idx)

#         writer.add_scalar("grad_l2", np.sqrt(
#             np.mean(np.square(grads))), step_idx)
#         writer.add_scalar("grad_max", np.max(np.abs(grads)), step_idx)
#         writer.add_scalar("grad_var", np.var(grads), step_idx)

#         batch_states.clear()
#         batch_actions.clear()
#         batch_scales.clear()

#     writer.close()
