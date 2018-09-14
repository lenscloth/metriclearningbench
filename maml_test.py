import torch
import torch.nn as nn

from model.maml.SGD import MamlSGD
from torch.optim.sgd import SGD

net = nn.Sequential(nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5))

optim = MamlSGD(net.parameters(), momentum=0.9, lr=0.1)
TrueOptim = SGD(net.parameters(), momentum=0.9, lr=0.1)

x = torch.randn(100, 10)

loss = net(x).sum()
loss.backward()
updated = optim.maml_step()
original = optim.maml_replace(net, updated)
loss = net(x).sum()
loss.backward()
TrueOptim.step()

print("hello")