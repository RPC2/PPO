import gym
import torch
import torch.optim as optim
from network import MlpPolicy, CNNPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgentConfig:
    def __init__(self):
        # Learning
        self.gamma = 0.99
        self.plot_every = 10
        self.update_freq = 1
        self.k_epoch = 1
        self.learning_rate = 0.02
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.v_coef = 1
        self.entropy_coef = 0.01
        self.memory_size = 400
        self.width = 84
        self.height = 84

        # Environment & Network
        self.environment = 'CartPole-v0'
        self.env = gym.make(self.environment)
        self.action_size = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape[0]
        self.policy_network = MlpPolicy(self.action_size, self.observation_size).to(device) \
            if self.environment == 'CartPole-v0' else CNNPolicy(self.action_size, self.num_stack).to(device)
        self.frame_shape = [4] if self.environment == 'CartPole-v0' else (self.height, self.width)
        self.num_stack = 1 if self.environment == 'CartPole-v0' else 4

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch,
                                                   gamma=0.999)
        self.criterion = torch.nn.MSELoss()

