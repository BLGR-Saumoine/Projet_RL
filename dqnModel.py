import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_FrameStack, n_actions):
      super(DQN, self).__init__()

      self.model = nn.Sequential(
          nn.Conv2d(in_channels=n_FrameStack, out_channels=32, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
          nn.ReLU())

      self.stateValue = nn.Sequential(
          nn.Linear(3136,512),
          nn.ReLU(),
          nn.Linear(512,1))

      self.advantage = nn.Sequential(
          nn.Linear(3136,512),
          nn.ReLU(),
          nn.Linear(512,n_actions))


    def forward(self, x):
      temp = self.model(x)
      temp = torch.flatten(temp, start_dim=1, end_dim=-1)
      sValue = self.stateValue(temp)
      advantage = self.advantage(temp)

      return sValue + (advantage - torch.mean(advantage, dim=1, keepdim=True))