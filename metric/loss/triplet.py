import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pdist
from metric.sampler.pair import RandomNegative, HardNegative


__all__ = ['NaiveTriplet', 'LogTriplet', 'RandomSubSpaceLoss', 'RandomSubSpaceOnlySampleLoss']


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


class LogTriplet(nn.Module):
    def __init__(self, squared=False, sampler=RandomNegative(), reduce=True, size_average=True):
        super(LogTriplet, self).__init__()
        self.sampler = sampler
        self.squared = squared
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_index = self.sampler(embeddings, labels)

        d = pdist(embeddings, squared=self.squared)

        pos_val = d[anchor_idx, pos_idx]
        neg_val = d[anchor_idx, neg_index]

        loss = torch.log(1 + (pos_val - neg_val).exp())

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


class AdversarialSubSpaceLoss(nn.Module):
    def __init__(self, spacing_net, loss_maker, reg=0.0):
        super(AdversarialSubSpaceLoss, self).__init__()
        self.spacing_net = spacing_net
        self.loss_maker = NaiveTriplet(margin=0.2)
        self.reg = reg

    def forward(self, embeddings, labels):
        spacing_prob = F.log_softmax(self.spacing_net(embeddings.detach()), dim=2)
        mean_spacing_prob = torch.logsumexp(spacing_prob, dim=0) - \
                            (embeddings.size(0) * torch.ones(spacing_prob.size()[1:], device=spacing_prob.device)).log()

        mask = OneHotCategorical.apply(mean_spacing_prob)
        mask = AdverserialGradient.apply(mask)

        sampled_embedding = embeddings.unsqueeze(2) * mask.unsqueeze(0)

        loss = self.loss_maker(embeddings, labels)
        for k in range(self.spacing_net.n_group):
            e = sampled_embedding[..., k]
            loss += self.loss_maker(e, labels)

        loss += self.reg * (mean_spacing_prob.exp() * mean_spacing_prob.exp()).sum(dim=1).mean()
        return loss


# class RandomSubSpaceLoss(nn.Module):
#     def __init__(self, loss_maker, n_group=3):
#         super(RandomSubSpaceLoss, self).__init__()
#         self.loss_maker = loss_maker
#         self.n_group = n_group
#
#     def forward(self, embeddings, labels):
#         embed_size = embeddings.size(1)
#
#         m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.ones((embed_size, self.n_group),
#                                                                                        device=embeddings.device))
#         mask = m.sample()
#         sampled_embedding = embeddings.unsqueeze(2) * mask.unsqueeze(0)
#         loss = self.loss_maker(embeddings, labels)
#         e_l = []
#         for e in embeddings:
#             #e = embeddings[k]
#             e_l.append(self.loss_maker(e.unsqueeze(1), labels.repeat(10)[:e.size(0)]))
#
#         loss += torch.stack(e_l).mean()
#         # N X E x K
#
#         return loss

# class RandomSubSpaceLoss(nn.Module):
#     def __init__(self, loss_maker, n_group=3):
#         super(RandomSubSpaceLoss, self).__init__()
#         self.loss_maker = loss_maker
#         self.n_group = n_group
#
#     def forward(self, embeddings, labels):
#         embed_size = embeddings.size(1)
#
#         m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.ones((embed_size, self.n_group), device=embeddings.device))
#         mask = m.sample()
#         sampled_embedding = embeddings.unsqueeze(2) * mask.unsqueeze(0)
#
#         loss = self.loss_maker(embeddings, labels)
#         for k in range(self.n_group):
#             e = sampled_embedding[k]
#             loss += self.loss_maker(e, labels)
#         return loss


class RandomSubSpaceLoss(nn.Module):
    def __init__(self, loss_maker):
        super(RandomSubSpaceLoss, self).__init__()
        self.loss_maker = loss_maker
        self.sample_loss = NaiveTriplet(margin=loss_maker.margin, sampler=HardNegative(), squared=loss_maker.squared)

    def forward(self, embeddings, labels):
        loss = self.loss_maker(embeddings, labels)

        s_l = []
        for e in embeddings:
            s_l.append(self.sample_loss(e.unsqueeze(1), torch.randint(0, 2, (len(e), ), device=e.device)))
        loss += 2 *torch.stack(s_l).mean()
        return loss


class RandomSubSpaceOnlySampleLoss(nn.Module):
    def __init__(self, sampler=RandomNegative(), margin=0.2, n_group=3):
        super(RandomSubSpaceOnlySampleLoss, self).__init__()
        self.sampler = sampler
        self.n_group = n_group
        self.margin = margin

    def forward(self, embeddings, labels):
        embed_size = embeddings.size(1)

        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.ones((embed_size, self.n_group), device=embeddings.device))
        mask = m.sample()
        sampled_embedding = embeddings.unsqueeze(2) * mask.unsqueeze(0)

        a_is, p_is, n_is = [], [], []
        a_i, p_i, n_i = self.sampler(embeddings, labels)
        a_is.append(a_i)
        p_is.append(p_i)
        n_is.append(n_i)

        for k in range(self.n_group):
            e = sampled_embedding[..., k]
            a_i, p_i, n_i = self.sampler(e, labels)

            a_is.append(a_i)
            p_is.append(p_i)
            n_is.append(n_i)

        anchor_idx = torch.cat(a_is, dim=0)
        pos_idx = torch.cat(p_is, dim=0)
        neg_idx = torch.cat(n_is, dim=0)

        d = pdist(embeddings, squared=True)

        pos_val = d[anchor_idx, pos_idx]
        neg_val = d[anchor_idx, neg_idx]

        loss = (pos_val - neg_val + self.margin).clamp(min=0)
        return loss.mean()