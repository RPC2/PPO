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
        self.k_epoch = 3
        self.learning_rate = 0.02
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.v_coef = 1
        self.entropy_coef = 0.01
        self.memory_size = 400
        self.width = 84
        self.height = 84
        self.num_stack = 4
        self.frame_shape = [self.height, self.width]

        # Environment & Network
        self.environment = 'Breakout-v0'
        self.action_size = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape[0]
        self.policy_network = MlpPolicy(self.action_size, self.observation_size).to(device) \
            if self.environment == 'CartPole-v0' else CNNPolicy(self.action_size, self.frame_stack).to(device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch,
                                                   gamma=0.999)
        self.criterion = torch.nn.MSELoss()

