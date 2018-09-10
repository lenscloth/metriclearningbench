import torch
import torch.nn as nn


class SimpleSamplingNet(nn.Module):
    def __init__(self, n_feature):
        super(SimpleSamplingNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(n_feature, 256, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 256, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 256, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 2, 1))

    def forward(self, x):
        x = self.net(x)
        return
