import torch
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
            episode_length = 0
            episode_reward = 0

            self.memory.reset()

            # A step in an episode
            while not solved:
                episode_length += 1

                obs = torch.FloatTensor(self.memory.get_state()).to(device)
                prob_a = self.policy_network.pi(obs)
                reward, terminal = self.memory.step(prob_a)
                episode_reward += reward

                if terminal:
                    reward_history.append(episode_reward)
                    self.finish_path(episode_length)
                    save_weights(self.policy_network)

                    solved = self.solve_criteria(reward_history)
                    print('episode: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                          'loss: %.4f, lr: %.4f' % (episode, episode_length, episode_reward, self.loss,
                                                    self.scheduler.get_lr()[0]))
                    break

            if episode % self.update_freq == 0:
                for _ in range(self.k_epoch):
                    self.update_network()

            if episode % self.plot_every == 0:
                plot_graph(reward_history)

    def update_network(self):
        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(self.memory['state']).to(device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action']))
        old_probs_a = torch.FloatTensor(self.memory['action_prob'])
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * torch.FloatTensor(self.memory['advantage'])
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * torch.FloatTensor(self.memory['advantage'])
        pred_v = self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        v_loss = 0.5 * (pred_v - self.memory['td_target']).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def finish_path(self, length):
        reward = self.memory['reward'][-length:]
        terminal = self.memory['terminal'][-length:]

        if self.environment == 'CartPole-v0':
            state = torch.FloatTensor(self.memory['state'][-length:-1])
            next_state = torch.FloatTensor(self.memory['state'][-length + 1:])
        else:
            state = torch.FloatTensor([
                self.stack_obs(idx) for idx in range(-length, -self.frame_stack)]
            ).reshape(length - self.frame_stack, self.frame_stack, self.height, self.width)
            next_state = torch.FloatTensor([
                self.stack_obs(idx) for idx in range(-length + 1, -self.frame_stack + 1)]
            ).reshape(length - self.frame_stack, self.frame_stack, self.height, self.width)

        td_target = torch.FloatTensor(reward) + \
                    self.gamma * self.policy_network.v(next_state) * torch.FloatTensor(terminal)
        delta = td_target - self.policy_network.v(state)
        delta = delta.detach().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'], td_target.data), dim=0)
        self.memory['advantage'] += advantages

    def solve_criteria(self, reward_history):
        if self.environment == 'CartPole-v0':
            return len(reward_history) > 100 and sum(reward_history[-100:-1]) / 100 >= 195

