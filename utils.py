import torch
import torch.nn.functional as F


def topk_mask(input, dim, K=10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim=dim, **kwargs)[1]
    return torch.autograd.Variable(torch.zeros_like(input.data)).scatter(dim, index, 1.0)


def pdist(e, squared=False, eps=1e-4):
    prod = torch.mm(e, e.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()
