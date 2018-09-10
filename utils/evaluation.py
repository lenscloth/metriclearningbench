import torch

from .distance import pdist

__all__ = ['recall', 'topk_mask']


def recall(embeddings, labels, K=1):
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + K, dim=1, largest=False)[1][:, 1:]
    return (labels.unsqueeze(-1).expand_as(knn_inds) == labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)).max(1)[0].float().mean()


def topk_mask(input, dim, K=10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim=dim, **kwargs)[1]
    return torch.autograd.Variable(torch.zeros_like(input.data)).scatter(dim, index, 1.0)
