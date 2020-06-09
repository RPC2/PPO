import torch
import torch.nn as nn
from torchsummary import summary


class MlpPolicy(nn.Module):
    def __init__(self, action_size, input_size):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3_pi = nn.Linear(24, self.action_size)
        self.fc3_v = nn.Linear(24, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def pi(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_pi(x)
        return self.softmax(x)

    def v(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_v(x)
        return x


class CNNPolicy(nn.Module):
    def __init__(self, action_size, input_channel):
        super(CNNPolicy, self).__init__()
        self.action_size = action_size
        self.input_channel = input_channel
        self.network = nn.Sequential(
            nn.Conv2d(input_channel, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
            # nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Linear(3136, 512)
        self.fc_pi = nn.Linear(512, action_size)
        self.fc_v = nn.Linear(512, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def pi(self, x):
        x = self.network(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return self.softmax(self.fc_pi(x))

    def v(self, x):
        x = self.network(x)
        x = torch.flatten(x, 1)
        # print('x shape', x.shape)
        x = self.linear(x)
        return self.softmax(self.fc_v(x))

    def forward(self, x):
        x = self.network(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return self.fc_pi(x)


if __name__ == "__main__":
    net = CNNPolicy(4, 4)
    summary(net, (4, 84, 84))


