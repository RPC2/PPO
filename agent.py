import torch
import itertools
from config import AgentConfig
from memory import Memory
from utils import plot_graph, save_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig):
    def __init__(self):
        AgentConfig.__init__(self)
        self.loss = 0
        self.memory = Memory(self.environment, self.memory_size, self.num_stack, self.frame_shape)

    def train(self):
        episode = 0
        reward_history = []
        solved = False

        # A new episode
        while not solved:
            episode += 1
            episode_length, episode_reward = 0, 0
            self.memory.reset()

            # A step in an episode
            while not solved:
                episode_length += 1
                reward, terminal = self.memory.step(self.policy_network.pi)
                episode_reward += reward

                if terminal:
                    reward_history.append(episode_reward)
                    self.finish_path(episode_length)
                    save_weights(self.policy_network)
                    solved = self.solve_criteria(reward_history)
                    print('episode: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                          'loss: %.4f, lr: %.4f' % (episode, episode_length, episode_reward, self.loss,
                                                    self.scheduler.get_lr()[0]))
                    # print(self.memory.buffer)
                    # print('len of memory', len(self.memory))
                    # print('len of obs', len(self.memory.buffer['obs']))
                    break

            if episode % self.update_freq == 0:
                for _ in range(self.k_epoch):
                    self.update_network()
            # if episode == 1:
            #     exit()

            if episode % self.plot_every == 0:
                plot_graph(reward_history)

    def update_network(self):
        start_idx, end_idx = 0, len(self.memory) if self.memory.step_count >= self.memory_size else self.memory.pointer
        self.memory.align_length()
        state = torch.FloatTensor(self.memory.get_state(start_idx=start_idx, end_idx=end_idx))
        action = torch.LongTensor([self.memory.buffer['action'][i] for i in range(start_idx, end_idx)])
        action_prob = torch.FloatTensor([self.memory.buffer['action_prob'][i] for i in range(start_idx, end_idx)])
        # print(self.memory.buffer['advantage'])
        advantage = torch.FloatTensor([self.memory.buffer['advantage'][i] for i in range(start_idx, end_idx)])
        td_target = torch.FloatTensor([self.memory.buffer['td_target'][i] for i in range(start_idx, end_idx)])
        # print('action shape', action.shape)
        # print('len memory', len(self.memory))
        # print('state shape', state.shape)
        # print('td_target shape', self.memory.buffer['td_target'].shape)
        # print(self.memory.buffer['terminal'])
        # print('state', state)

        # print('action', action)
        # print('action_prob', action_prob)
        # print('advantage', advantage)

        # get ratio
        pi = self.policy_network.pi(state)
        new_probs_a = torch.gather(pi, 1, action)
        old_probs_a = action_prob
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))
        # print(new_probs_a)
        # print(old_probs_a)
        # print(ratio)
        # surrogate loss
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        pred_v = self.policy_network.v(state)
        v_loss = 0.5 * (pred_v - td_target).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def finish_path(self, episode_length):
        # The information extracted includes start_idx but exclude end_idx
        start_idx = self.memory.pointer - episode_length
        end_idx = self.memory.pointer

        # print(f'in finish path, start_idx = {start_idx}. end_idx = {end_idx}')

        reward = torch.FloatTensor([self.memory.buffer['reward'][i] for i in range(start_idx, end_idx)])
        terminal = torch.FloatTensor([self.memory.buffer['terminal'][i] for i in range(start_idx, end_idx)])
        state = torch.FloatTensor(self.memory.get_state(start_idx=start_idx, end_idx=end_idx))
        next_state = torch.FloatTensor(self.memory.get_state(start_idx=start_idx, end_idx=end_idx, next_state=True))

        # print('terminal shape', terminal.shape)
        # print(self.memory.buffer)
        # # print(self.memory.buffer['action'])
        # print(state)
        # print(next_state)
        # print(len(self.memory))
        # print('state shape', state.shape)
        # print('next_state shape', next_state.shape)

        td_target = reward + self.gamma * self.policy_network.v(next_state) * terminal
        delta = td_target - self.policy_network.v(state)
        delta = delta.detach().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        for i, j in zip(range(start_idx, end_idx), range(len(td_target.data))):
            self.memory.buffer['td_target'][i] = td_target.data[j]
            self.memory.buffer['advantage'][i] = advantages[j]
        # self.memory.buffer['advantage'] += advantages

    def solve_criteria(self, reward_history):
        if self.environment == 'CartPole-v0':
            return len(reward_history) > 100 and sum(reward_history[-100:-1]) / 100 >= 195

