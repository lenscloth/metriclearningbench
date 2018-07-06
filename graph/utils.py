import torch
import torch.nn.functional as F

from utils import pdist


def manifold_dist(affinity, alpha=0.99):
    degree = 1 / affinity.sum(dim=1).diag()
    norm_aff = degree.sqrt() @ affinity @ degree.sqrt()
    identity = torch.eye(degree.size(0), device=degree.device)
    m_dist = (1 - alpha) * (identity - alpha * norm_aff).inverse()

    return m_dist


def exp_affinity(embeddings, sigma=3):
    return torch.exp((-1 * pdist(embeddings, squared=True)) / sigma)


def cosine_affinity(embeddings):
    norm_embd = F.normalize(embeddings, dim=1)
    return norm_embd @ norm_embd.t()