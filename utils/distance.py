import torch


__all__ = ['pdist']


def pdist(e, squared=False, eps=1e-12):
    e_square = (e * e).sum(dim=1)
    prod = torch.mm(e, e.t())
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res
