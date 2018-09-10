import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BaseEmbedding"]


class BaseEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=128, normalize=True):
        super(BaseEmbedding, self).__init__()
        self.base = base

        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize

    def forward(self, input):
        x = self.base(input)
        x = self.linear(x.view(x.size(0), -1))

        if self.normalize:
            return F.normalize(x, p=2, dim=1)
        else:
            return x


class NoEmbedding(nn.Module):
    def __init__(self, base, normalize=True):
        super(NoEmbedding, self).__init__()
        self.base = base
        self.normalize = normalize

    def forward(self, input):
        x = self.base(input)
        x = x.view(x.size(0), -1)

        if self.normalize:
            return F.normalize(x, p=2, dim=1)
        else:
            return x
