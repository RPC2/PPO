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
import datetime

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
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1,
        #                                            gamma=0.98)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': []
        }
        f = open("log.txt", "w")
        f.write(str(datetime.datetime.now()) + "\n")
        f.close()

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
            while sum(reward_history[-100:-1])/100 < 194:
                step += 1
                episode_length += 1

                # Choose action
                prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(device))
                action = torch.distributions.Categorical(prob_a).sample().item()

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
                          'loss: %.4f' % (episode, step, episode_length, total_episode_reward, self.loss))

                    # print(prob_a)

                    self.env.reset()

                    if self.gif:
                        generate_gif(episode_length, frames_for_gif, total_episode_reward, "./GIF/", episode)

                    break

                if step % self.horizon == 0:
                    f = open("log.txt", "a+", buffering=1)
                    f.write(
                        "===================================== update %d ====================================\n" % (step // self.horizon))

                    for _ in range(self.k_epoch):
                        self.update_network()
                    self.empty_memory()

            if episode % self.plot_every == 0:
                plot_graph(reward_history)

            # self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_minimum)

            # self.env.render()

        self.env.close()

    def update_network(self):
        # get delta
        # print("current state", self.memory['state'])
        # print("next state", self.memory['next_state'])
        f = open("log.txt", "a+")
        td_target = torch.FloatTensor(self.memory['reward']).to(device) + \
                    self.gamma * self.policy_network.v(torch.FloatTensor(self.memory['next_state'])) * \
                    torch.FloatTensor(self.memory['terminal'])
        delta = td_target - self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        # print(delta)
        delta = delta.detach().numpy()
        f.write('delta: ' + str(delta) + "\n")

        # get advantage
        advantages = []
        adv = 0.0
        # print('delta[::-1]', delta[::-1])
        for d in delta[::-1]:
            # print('d[0]', d[0])
            adv = self.gamma * self.lmbda * adv + d[0]
            f.write("adv: " + str(adv) + '\n')
            advantages.append([adv])
        advantages.reverse()
        advantages = torch.FloatTensor(advantages)
        f.write('advantages: ' + str(advantages) + '\n')

        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(self.memory['state']).to(device))
        f.write('pi: ' + str(pi) + '\n')
        f.write('actions from memory: ' + str(self.memory['action']) + '\n')
        f.write('action_prob: ' + str(self.memory['action_prob']) + '\n')
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action']))
        old_pi = self.old_policy.pi(torch.FloatTensor(self.memory['state']).to(device))
        old_probs_a = torch.gather(old_pi, 1, torch.tensor(self.memory['action']))
        f.write('new_probs_a: ' + str(new_probs_a) + '\n')
        f.write('old_probs_a: ' + str(old_probs_a) + '\n')
        f.write("log new probs: " + str(torch.log(new_probs_a)) + '\n')
        f.write("log old probs: " + str(torch.log(old_probs_a)) + '\n')
        # ratio = torch.exp(torch.log(new_probs_a) - torch.log(torch.tensor(self.memory['action_prob'])))
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))
        f.write('ratio: ' + str(ratio) + '\n')

        # surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        pred_v = self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        f.write('surr1: ' + str(surr1) + '\n')
        f.write('surr2: ' + str(surr2) + '\n')
        f.write('min of surr1 and surr2: ' + str(-torch.min(surr1, surr2)) + '\n')
        f.write('pred_v: ' + str(pred_v) + '\n')
        f.write('td_target: ' + str(td_target) + '\n')
        v_loss = 0.5 * (pred_v - td_target).pow(2)  # Huber loss
        f.write('v_loss: ' + str(v_loss) + '\n')
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        f.write('entropy: ' + str(entropy) + '\n')
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()
        f.write('min of surr1, surr2: ' + str((-torch.min(surr1, surr2)).mean()) + '\n')
        f.write('v loss: ' + str((self.v_coef * v_loss).mean()) + '\n')
        f.write('entropy loss: ' + str((- self.entropy_coef * entropy).mean()) + '\n')
        f.write('loss: ' + str(self.loss) + '\n')

        self.old_policy.load_state_dict(self.policy_network.state_dict())

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

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
