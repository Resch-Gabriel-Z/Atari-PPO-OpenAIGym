import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, in_channels, num_actions):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=-1)

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions),
        )

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        conv_layers = self.conv(x)
        fc_input = self.flatten(conv_layers)
        fc_input = fc_input.view(-1, 64 * 7 * 7)
        fc_output = self.fc(fc_input)
        action = self.softmax(fc_output)

        return action


class Critic(nn.Module):

    def __init__(self, in_channels):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        conv_layers = self.conv(x)
        fc_input = self.flatten(conv_layers)
        fc_input = fc_input.view(-1, 64 * 7 * 7)
        value = self.fc(fc_input)

        return value
