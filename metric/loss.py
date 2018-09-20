import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pdist
from metric.sampler.pair import RandomNegative, HardNegative


__all__ = ['L1Triplet', 'L2Triplet', 'CosineTriplet', 'RandomSubSpaceOnlySampleLoss']


class _Triplet(nn.Module):
    def __init__(self, margin=0.2, dist_func=None, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.margin = margin
        self.dist_func = dist_func

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = dist_func

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels, weight=None):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)
        d = self.dist_func(embeddings)

        pos_val = d[anchor_idx, pos_idx]
        neg_val = d[anchor_idx, neg_idx]

        loss = (pos_val - neg_val + self.margin).clamp(min=0)
        if weight is not None:
            pos_weight = weight[anchor_idx, pos_idx]
            neg_weight = weight[anchor_idx, neg_idx]
            pair_weight = (pos_weight + neg_weight) / 2.0
            loss = pair_weight * loss

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(margin=margin, dist_func=lambda e: pdist(e, squared=True), sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(margin=margin, dist_func=lambda e: pdist(e, squared=False), sampler=sampler)


class CosineTriplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        def min_cosine(e):
            norm = F.normalize(e, dim=1, p=2)
            return -1 * (norm @ norm.t())
        super().__init__(margin=margin, dist_func=lambda e: min_cosine(e), sampler=sampler)


class Noise(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, embeddings, labels):
        noise = embeddings.data.std(dim=0)
        gen = torch.distributions.normal.Normal(torch.zeros(noise.size(), device=embeddings.device), noise)

        l = []
        for _ in range(5):
            e = embeddings + 0.1 * gen.sample((embeddings.size(0),))
            e = F.normalize(e, dim=1, p=2)
            l.append(self.loss(e, labels))
        return torch.stack(l).mean()


class NaiveTriplet(nn.Module):
    def __init__(self, margin=0.2, squared=False, sampler=RandomNegative(), reduce=True, size_average=True):
        super(NaiveTriplet, self).__init__()
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


class AdverserialGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return -1 * grad_output


class OneHotCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_prob):
        m = torch.distributions.one_hot_categorical.OneHotCategorical(logits=log_prob)
        mask = AdverserialGradient.apply(m.sample())
        return mask

    @staticmethod
    def backward(ctx, mask_grad):
        return mask_grad


class RandomSubSpaceLoss(nn.Module):
    def __init__(self, loss_maker):
        super(RandomSubSpaceLoss, self).__init__()
        self.loss_maker = loss_maker
        self.sample_loss = L1Triplet(margin=0.2, sampler=HardNegative())

    def forward(self, embeddings, labels):
        loss = self.loss_maker(embeddings, labels)

        s_l = []
        for e in embeddings:
            e = e.unsqueeze(1)
            s_l.append(self.sample_loss(e, torch.randint(0, 6, (len(e),), device=embeddings.device)))
        loss += 2 * torch.stack(s_l).mean()
        return loss


class RandomSubSpaceOnlySampleLoss(nn.Module):
    def __init__(self, loss_maker, n_group=10):
        super(RandomSubSpaceOnlySampleLoss, self).__init__()
        self.loss_maker = loss_maker
        self.n_group = n_group

    def forward(self, embeddings, labels):
        embed_size = embeddings.size(1)

        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.ones((embed_size, self.n_group), device=embeddings.device))
        mask = m.sample()
        sampled_embedding = embeddings.unsqueeze(2) * mask.unsqueeze(0)

        a_is, p_is, n_is = [], [], []
        a_i, p_i, n_i = self.loss_maker.sampler(embeddings, labels)
        a_is.append(a_i)
        p_is.append(p_i)
        n_is.append(n_i)

        for k in range(self.n_group):
            e = sampled_embedding[..., k]
            a_i, p_i, n_i = self.loss_maker.sampler(e, labels)

            a_is.append(a_i)
            p_is.append(p_i)
            n_is.append(n_i)

        anchor_idx = torch.cat(a_is, dim=0)
        pos_idx = torch.cat(p_is, dim=0)
        neg_idx = torch.cat(n_is, dim=0)

        d = self.loss_maker.dist_func(embeddings)

        pos_val = d[anchor_idx, pos_idx]
        neg_val = d[anchor_idx, neg_idx]

        loss = (pos_val - neg_val + self.loss_maker.margin).clamp(min=0)
        return loss.mean()
