import torch

from .distance import pdist

__all__ = ['recall']


def recall(embeddings, labels, K=1):
    """
    Multiply by 10 for numerical stability.
    """
    D = pdist(embeddings * 10, squared=True)
    knn_inds = D.topk(1 + K, dim=1, largest=False)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []
    for k in range(1, K+1):
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k
