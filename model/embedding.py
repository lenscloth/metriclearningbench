import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LinearEmbedding", "ConvEmbedding"]


class ConvEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=128, normalize=True, return_base_embedding=False):
        super(ConvEmbedding, self).__init__()
        self.base = base
        self.embedding_size = embedding_size
        self.linear = nn.Conv1d(1, embedding_size, kernel_size=output_size-embedding_size+1)
        self.normalize = normalize
        self.return_base_embedding = return_base_embedding

    def forward(self, input):
        x = self.base(input).view(input.size(0), 1, -1)

        if self.return_base_embedding:
            if self.normalize:
                return F.normalize(x, p=2, dim=1)
            else:
                return x
        x = self.linear(x)[:, range(self.embedding_size), range(self.embedding_size)]
        if self.normalize:
            return F.normalize(x, p=2, dim=1)
        else:
            return x


class LinearEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=128, normalize=True, return_base_embedding=False):
        super(LinearEmbedding, self).__init__()
        self.base = base

        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize
        self.return_base_embedding = return_base_embedding

    def forward(self, input):
        b = self.base(input).view(input.size(0), -1)
        x = self.linear(b)

        if self.normalize:
            b = F.normalize(b, p=2, dim=1)
            x = F.normalize(x, p=2, dim=1)

        if self.return_base_embedding:
            return b
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
