import torch
import torch.nn as nn


class SubSpaceMLP(nn.Module):
    def __init__(self, n_group=3, n_input=1024, n_layer=3, n_hidden=1024):
        super(SubSpaceMLP, self).__init__()
        self.n_group = n_group
        self.n_input = n_input

        layers = []
        in_ = n_input
        for n in range(n_layer-1):
            layers.append(nn.Linear(in_, n_hidden))
            layers.append(nn.ReLU())
            in_ = n_hidden
        layers.append(nn.Linear(in_, n_group * n_input))
        self.net = nn.Sequential(*layers)

    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        return self.net(embeddings).view(batch_size, self.n_input, self.n_group)
