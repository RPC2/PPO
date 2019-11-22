import torch
import numpy as np
import gym
import os
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AgentConfig, EnvConfig
from network import MlpPolicy
from ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig, EnvConfig):
    def __init__(self):
        self.env = gym.make(self.env_name)
        self.action_size = self.env.action_space.n  # 2 for cartpole
        if self.train_cartpole:
            self.policy_network = MlpPolicy(action_size=self.action_size).to(device)
            self.old_policy = MlpPolicy(action_size=self.action_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': []
        }

    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, info = self.env.step(action)
        return screen, reward, action, terminal

    def train(self):
        episode = 0
        step = 0
        reward_history = []

        if not os.path.exists("./GIF/"):
            os.makedirs("./GIF/")

        # A new episode
        while step < self.max_step:
            start_step = step
            episode += 1
            episode_length = 0
            total_episode_reward = 0
            frames_for_gif = []

            self.gif = True if episode % self.gif_every == 0 else False

            # Get initial state
            state, reward, action, terminal = self.new_random_game()
            current_state = state
            # current_state = np.stack((state, state, state, state))

            # A step in an episode
            while episode_length < self.max_episode_length:
                step += 1
                episode_length += 1

                # Choose action
                prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(device))
                action = random.randrange(self.action_size) if np.random.rand() < self.epsilon else \
                    torch.argmax(prob_a).item()

                # print(current_state)
                # print(self.policy_network(torch.FloatTensor(current_state).to(device)))

                # Act
                state, reward, terminal, _ = self.env.step(action)
                new_state = state
                # new_state = np.concatenate((current_state[1:], [state]))

                reward = -1 if terminal else reward

                self.add_memory(current_state, action, reward/10.0, new_state, terminal, prob_a[action])

                if self.gif:
                    frames_for_gif.append(new_state)

                current_state = new_state
                total_episode_reward += reward

                if terminal:
                    episode_length = step - start_step
                    reward_history.append(total_episode_reward)

                    print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                          'loss: %.4f, eps = %.2f' % (episode, step, episode_length, total_episode_reward,
                                                      self.loss, self.epsilon))

                    self.update_network()
                    self.empty_memory()

                    # print(prob_a)

                    self.env.reset()

                    if self.gif:
                        generate_gif(episode_length, frames_for_gif, total_episode_reward, "./GIF/", episode)

                    break

            if episode % self.plot_every == 0:
                plot_graph(reward_history)

            self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_minimum)

            # self.env.render()

        self.env.close()

    def update_network(self):
        # get delta
        td_target = torch.FloatTensor(self.memory['reward']).to(device) + \
                    self.gamma * self.policy_network.v(torch.FloatTensor(self.memory['state'])) * \
                    torch.FloatTensor(self.memory['terminal'])
        delta = td_target - self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        # print(delta)
        delta = delta.detach().numpy()
        # print('delta', delta)

        # get advantage
        advantages = []
        adv = 0.0
        # print('delta[::-1]', delta[::-1])
        for d in delta[::-1]:
            # print('d[0]', d[0])
            adv = self.gamma * self.lmbda * adv + d[0]
            # print(adv)
            advantages.append([adv])
        # print("advantages", advantages)
        advantages.reverse()
        advantages = torch.FloatTensor(advantages)
        # print('advantages', advantages)

        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(self.memory['state']).to(device))
        # print('pi', pi)
        # print(self.memory['action'])
        # print('action_prob', self.memory['action_prob'])
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action']))
        old_pi = self.old_policy.pi(torch.FloatTensor(self.memory['state']).to(device))
        old_probs_a = torch.gather(old_pi, 1, torch.tensor(self.memory['action']))
        # print('new_probs_a', new_probs_a)
        # ratio = torch.exp(torch.log(new_probs_a) - torch.log(torch.tensor(self.memory['action_prob'])))
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))
        # ratio = [rt.]
        # print('ratio', ratio)

        # surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        pred_v = self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        # print('surr1', surr1)
        # print('surr2', surr2)
        # print('pred_v', pred_v)
        # print('td_target', td_target)
        v_loss = 0.5 * (pred_v - td_target).pow(2)  # Huber loss
        # print('v_loss', v_loss)
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        # print('entropy', entropy)
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()
        # print('min of surr1, surr2', (-torch.min(surr1, surr2)).mean())
        # print('v loss', (self.v_coef * v_loss).mean())
        # print('entropy loss', (- self.entropy_coef * entropy).mean())
        # print('loss', self.loss)

        self.old_policy.load_state_dict(self.policy_network.state_dict())

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def add_memory(self, s, a, r, next_s, t, prob):
        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1-t])
        self.memory['action_prob'].append([prob])

    def empty_memory(self):
        self.memory['state'] = []
        self.memory['action'] = []
        self.memory['reward'] = []
        self.memory['next_state'] = []
        self.memory['terminal'] = []
        self.memory['action_prob'] = []


def plot_graph(reward_history):
    df = pd.DataFrame({'x': range(len(reward_history)), 'y': reward_history})
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
    plt.title("CartPole", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("score", fontsize=12)

    plt.savefig('score.png')
