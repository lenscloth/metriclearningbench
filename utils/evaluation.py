import torch

from .distance import pdist

__all__ = ['recall', 'topk_mask']


def recall(embeddings, labels, K=1):
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + K, dim=1, largest=False)[1][:, 1:]
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels
    correct_samples = (correct_labels.sum(dim=1) > 0)
    return correct_samples.float().mean()


def topk_mask(input, dim, K=10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim=dim, **kwargs)[1]
    return torch.autograd.Variable(torch.zeros_like(input.data)).scatter(dim, index, 1.0)
