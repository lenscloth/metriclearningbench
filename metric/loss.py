import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pdist
from metric.sampler.pair import RandomNegative


__all__ = ['Triplet']


class Triplet(nn.Module):
    def __init__(self, margin=0.2, squared=False, sampler=RandomNegative(), reduce=True, size_average=True):
        super(Triplet, self).__init__()
        self.margin = margin
        self.squared = squared
        self.sampler = sampler
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_index = self.sampler(embeddings, labels)

        d = pdist(embeddings, squared=self.squared)
        pos_val = d[anchor_idx, pos_idx]
        neg_val = d[anchor_idx, neg_index]

        loss = (pos_val - neg_val + self.margin).clamp(min=0)

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LearnableSamplingTriplet(nn.Module):
    def __init__(self, sampler, margin=0.2, reduce=True, size_average=True):
        super(LearnableSamplingTriplet, self).__init__()
        self.sampler = sampler
        self.margin = margin
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        embeddings.unsqueeze(0) - embeddings.unsqueeze(1)

        pos_index, neg_index = self.sampler(embeddings, labels)
