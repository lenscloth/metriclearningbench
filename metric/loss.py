import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pdist
from metric.sampler.pair import RandomNegative, HardNegative, AllPairs


__all__ = ['L1Triplet', 'L2Triplet', 'DistillDistance', 'DistillAngle']


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class MarginLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=1.2, beta_classes=None, nu=0, sampler=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.sampler = sampler

        if beta_classes is not None:
            self.class_margin = nn.Parameter(beta_classes)
        else:
            self.class_margin = None

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        if self.class_margin is not None:
            beta = self.beta + self.class_margin[labels[anchor_idx]]
            beta_reg_loss = beta.sum() * self.nu
        else:
            beta = self.beta
            beta_reg_loss = 0

        anchor_embed = embeddings[anchor_idx]
        positve_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_dist = (anchor_embed-positve_embed).pow(2).sum(dim=1).sqrt()
        neg_dist = (anchor_embed-negative_embed).pow(2).sum(dim=1).sqrt()
        pos_loss = (self.alpha + (pos_dist - beta)).clamp(min=0)
        neg_loss = (self.alpha - (neg_dist - beta)).clamp(min=0)

        loss = (pos_loss.sum() + neg_loss.sum()) / ((pos_loss > 0).sum() + (neg_loss > 0).sum()).data.float()
        return loss


class HardDarkRank(nn.Module):
    def __init__(self, alpha=2, beta=2, permute_len=8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class DistillDistance(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
        d = pdist(student, squared=False)
        loss = F.smooth_l1_loss(d, self.alpha * t_d, reduction='elementwise_mean')
        return loss


class DistillAngle(nn.Module):
    def __init__(self, n_anchor=30):
        super().__init__()
        self.n_anchor = n_anchor

    def forward(self, student, teacher):
        batch_size = student.size(0)
        anchor_idx = torch.multinomial(torch.ones(batch_size, device=student.device), self.n_anchor,
                                       replacement=False)

        with torch.no_grad():
            td = torch.cat([teacher - teacher[i].unsqueeze(0) for i in anchor_idx], dim=0)
            norm_td = F.normalize(td, p=2, dim=1)
            t_angle = norm_td @ norm_td.t()

        sd = torch.cat([student - student[i].unsqueeze(0) for i in anchor_idx], dim=0)
        norm_sd = F.normalize(sd, p=2, dim=1)
        s_angle = norm_sd @ norm_sd.t()

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


# class RandomSubSpaceLoss(nn.Module):
#     def __init__(self, loss_maker):
#         super(RandomSubSpaceLoss, self).__init__()
#         self.loss_maker = loss_maker
#         self.embedding_per_group = 16
#         self.sample_loss = OnlyNegativeLoss(p=1, margin=0.2)
#         groups = torch.arange(0, 8).unsqueeze(1).repeat((1, 16)).view(-1)
#         self.register_buffer('groups', groups)
#
#     def forward(self, embeddings, labels):
#         loss = self.loss_maker(embeddings, labels)
#
#         s_l = []
#         for e in embeddings:
#             e = e.unsqueeze(1)
#             s_l.append(self.sample_loss(e, torch.ones(len(e), device=e.device)))
#         loss += 2 * torch.stack(s_l).mean()
#         return loss
#
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
#
#         sampled_embedding = embeddings.unsqueeze(2) * mask.unsqueeze(0)
#         loss = self.loss_maker(embeddings, labels)
#         for k in range(self.n_group):
#             e = sampled_embedding[k]
#             loss += self.loss_maker(e, labels)
#         return loss
#
#
#
# class RandomSubSpaceOnlySampleLoss(nn.Module):
#     def __init__(self, loss_maker, n_group=10):
#         super(RandomSubSpaceOnlySampleLoss, self).__init__()
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
#         a_is, p_is, n_is = [], [], []
#         a_i, p_i, n_i = self.loss_maker.sampler(embeddings, labels)
#         a_is.append(a_i)
#         p_is.append(p_i)
#         n_is.append(n_i)
#
#         for k in range(self.n_group):
#             e = sampled_embedding[..., k]
#             a_i, p_i, n_i = self.loss_maker.sampler(e, labels)
#
#             a_is.append(a_i)
#             p_is.append(p_i)
#             n_is.append(n_i)
#
#         anchor_idx = torch.cat(a_is, dim=0)
#         pos_idx = torch.cat(p_is, dim=0)
#         neg_idx = torch.cat(n_is, dim=0)
#
#         d = self.loss_maker.dist_func(embeddings)
#
#         pos_val = d[anchor_idx, pos_idx]
#         neg_val = d[anchor_idx, neg_idx]
#
#         loss = (pos_val - neg_val + self.loss_maker.margin).clamp(min=0)
#         return loss.mean()
