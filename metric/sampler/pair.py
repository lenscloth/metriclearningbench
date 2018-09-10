import torch
import torch.nn as nn

from utils import pdist

BIG_NUMBER = 1e12

__all__ = ['RandomNegative', 'HardNegative', 'SemiHardNegative', 'DistanceWeighted']


def pos_neg_mask(labels):
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))

    return pos_mask, neg_mask


class RandomNegative(nn.Module):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)

            pos_pair_index = pos_mask.nonzero()
            anchor_idx = pos_pair_index[:, 0]
            pos_idx = pos_pair_index[:, 1]
            neg_index = torch.multinomial(neg_mask.float()[anchor_idx], 1).squeeze(1)

        return anchor_idx, pos_idx, neg_index


class HardNegative(nn.Module):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)
            dist = pdist(embeddings)

            pos_pair_index = pos_mask.nonzero()
            anchor_idx = pos_pair_index[:, 0]
            pos_idx = pos_pair_index[:, 1]

            neg_dist = (neg_mask.float() * dist)
            neg_dist[neg_dist <= 0] = BIG_NUMBER
            neg_idx = neg_dist.argmin(dim=1)[anchor_idx]

        return anchor_idx, pos_idx, neg_idx


class SemiHardNegative(nn.Module):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            dist = pdist(embeddings)
            pos_mask, neg_mask = pos_neg_mask(labels)
            neg_dist = dist * neg_mask.float()

            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            tiled_negative = neg_dist[anchor_idx]
            satisfied_neg = (tiled_negative > dist[pos_mask].unsqueeze(1)) * neg_mask[anchor_idx]
            """
            When there is no negative pair that its distance bigger than positive pair, 
            then select negative pair with largest distance.
            """
            unsatisfied_neg = (satisfied_neg.sum(dim=1) == 0).unsqueeze(1) * neg_mask[anchor_idx]

            tiled_negative = (satisfied_neg.float() * tiled_negative) - (unsatisfied_neg.float() * tiled_negative)
            tiled_negative[tiled_negative == 0] = BIG_NUMBER
            neg_idx = tiled_negative.argmin(dim=1)

        return anchor_idx, pos_idx, neg_idx


class DistanceWeighted(nn.Module):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.4

    def forward(self, embeddings, labels):
        with torch.no_grad():

            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            d = embeddings.size(1)
            dist = pdist(embeddings, squared=False)
            dist = dist.clamp(min=self.cut_off)

            log_weight = -1 * ((d - 2.0) * dist.log() + ((d - 3.0)/2.0) * (1.0 - 0.25 * (dist * dist)).log())
            weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
            weight = ((dist < self.nonzero_loss_cutoff) * neg_mask).float() * weight

            """
            When all the negative pair's distance are larger thatn nonzero_loss_cutoff,
            We sample randomly among negative pairs
            """
            weight = weight + ((weight.sum(dim=1) == 0).unsqueeze(1) * neg_mask).float()
            weight = weight[anchor_idx]
            neg_idx = torch.multinomial(weight, 1).squeeze(1)

        return anchor_idx, pos_idx, neg_idx
