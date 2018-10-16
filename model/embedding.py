import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graph.gcn import GraphEdge, GraphConv

__all__ = ["LinearEmbedding", "NoEmbedding", "MultipleLinearEmbedding", "DistillLinearEmbedding", "GraphEmbedding",
           "SigmoidScaleEmbedding", "BatchNormScaleEmbedding", "MaxOutEmbedding", "SumLinearEmbedding",
           "PNormsEmbedding"]


class PNormsEmbedding(nn.Module):
    def __init__(self, base, n_layer=2, weighting=True, output_size=512, embedding_size=128, normalize=True, return_base=False):
        super().__init__()
        self.base = base
        self.linear = nn.ModuleList([nn.Linear(output_size, embedding_size) for _ in range(n_layer)])
        self.normalize = normalize
        self.return_base = return_base

        weight = nn.Parameter(torch.ones(n_layer))
        self.register_parameter('weight', weight)
        self.weighting = weighting

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        embed = torch.stack([F.normalize(l(feat), p=(i+2), dim=1) for i, l in enumerate(self.linear)])

        if self.weighting:
            embed = (self.weight.view(-1, 1, 1) * embed).sum(dim=0)

        if self.return_base:
            return embed, feat
        return embed


class SumLinearEmbedding(nn.Module):
    def __init__(self, base, n_layer=5, output_size=512, embedding_size=128, normalize=True, return_base=False):
        super().__init__()
        self.base = base
        self.linear = nn.ModuleList([nn.Linear(output_size, embedding_size) for _ in range(n_layer)])
        self.normalize = normalize
        self.return_base = return_base

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        embed = torch.stack([l(feat) for l in self.linear], dim=0) # K x N x E

        if self.normalize:
            embed = F.normalize(embed, p=2, dim=2)
        embed = embed.sum(dim=0)

        if self.return_base:
            return embed, feat
        return embed


"""
Training takes too long. 희망이 없어보임.
"""
class MaxOutEmbedding(nn.Module):
    def __init__(self, base, n_layer=5, output_size=512, embedding_size=128, normalize=True, return_base=False):
        super().__init__()
        self.base = base
        self.linear = nn.ModuleList([nn.Linear(output_size, embedding_size) for _ in range(n_layer)])
        self.normalize = normalize
        self.return_base = return_base

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        embed = torch.stack([l(feat) for l in self.linear], dim=0).max(dim=0)[0]

        if self.normalize:
            embed = F.normalize(embed, p=2, dim=1)
        if self.return_base:
            return embed, feat
        return embed


"""
안됨.
"""
class BatchNormScaleEmbedding(nn.Module):
    def __init__(self, base, amplify=2, output_size=512, embedding_size=128, normalize=True, return_base=False):
        super().__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.return_base = return_base
        self.norm_mean = 0
        self.norm_stdev = 1
        self.amplify = amplify

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        embed = self.linear(feat)

        norm = embed.pow(2).sum(dim=1).sqrt()

        if self.training:
            norm_mean = norm.mean()
            norm_stdev = norm.std()
            self.norm_mean = 0.9 * self.norm_mean + 0.1 * norm_mean.item()
            self.norm_stdev = 0.9 * self.norm_stdev + 0.1 * norm_stdev.item()
        else:
            norm_mean = self.norm_mean
            norm_stdev = self.norm_stdev

        scale = self.amplify * torch.sigmoid((norm - norm_mean) / norm_stdev).unsqueeze(1)
        embed = scale * F.normalize(embed, dim=1, p=2)

        if self.return_base:
            return embed, feat
        return embed

"""
Sigmoid 값이 1 근처로 나와서 됨...
"""
class SigmoidScaleEmbedding(nn.Module):
    def __init__(self, base, amplify=2, output_size=512, embedding_size=128, normalize=True, return_base=False):
        super().__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.return_base = return_base
        self.amplify = amplify

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        embed = self.linear(feat)
        scale = self.amplify * torch.sigmoid(embed.pow(2).sum(dim=1).sqrt()).unsqueeze(1)
        embed = scale * F.normalize(embed, dim=1, p=2)

        if self.return_base:
            return embed, feat
        return embed


class LinearEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=128, normalize=True, return_base=False):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize
        self.return_base = return_base

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        embed = self.linear(feat)

        if self.normalize:
            embed = F.normalize(embed, p=2, dim=1)
        if self.return_base:
            return embed, feat
        return embed


class MultipleLinearEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=[128], normalize=True):
        super().__init__()
        self.base = base
        self.linears = nn.ModuleList([nn.Linear(output_size, s) for s in embedding_size])
        self.normalize = normalize
        self.branch = None

    def set_branch(self, i):
        self.branch = i

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        embed = torch.stack([l(feat) for l in self.linears], dim=1)

        if self.normalize:
            embed = F.normalize(embed, p=2, dim=2)

        if self.branch is not None:
            return embed[:, self.branch]

        if self.training:
            return embed
        return embed.mean(dim=1)




class DistillLinearEmbedding(nn.Module):
    def __init__(self, base, distil_embedding_size=512, embedding_size=128, output_size=512, normalize=True):
        super().__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.big_linear = nn.Linear(output_size, distil_embedding_size)
        self.normalize = normalize

    def forward(self, input):
        feat = self.base(input).view(input.size(0), -1)
        distil_embed = self.big_linear(feat)
        target_embed = self.linear(feat)

        if self.normalize:
            distil_embed = F.normalize(distil_embed, p=2, dim=1)
            target_embed = F.normalize(target_embed, p=2, dim=1)

        if not self.training:
            return target_embed
        return distil_embed, target_embed


class GraphEmbedding(nn.Module):
    def __init__(self, in_feature=128, out_feature=128, n_layer=2, n_node=30, normalize=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.n_layer = n_layer
        self.normalize = normalize
        self.n_node = n_node

        channels = [in_feature for _ in range(n_layer)] + [out_feature]
        for n in range(n_layer):
            self.add_module('g_edge%d'%n, GraphEdge(channels[n], 1, operator="J2", activation="softmax", ratio=[2, 2]))
            self.add_module('g_conv%d'%n, GraphConv(channels[n], channels[n+1], 2, bn_bool=False))

    def forward(self, x):
        chunks = x.split(self.n_node, 0)
        last = chunks[-1]
        if last.size() != chunks[0].size():
            last_e = self.forward(last)
            x = torch.cat(chunks[:-1])
        else:
            last_e = None

        x = torch.stack(x.split(self.n_node, 0))
        for n in range(self.n_layer):
            e = self._modules['g_edge%d'%n](x)
            x = self._modules['g_conv%d'%n](e, x)

            if n < (self.n_layer-1):
                x = F.leaky_relu(x)

        x = x.view(-1, self.out_feature)
        if self.normalize:
            x = F.normalize(x, dim=1, p=2)

        if last_e is not None:
            x = torch.cat((x, last_e), dim=0)
        return x


class NoEmbedding(nn.Module):
    def __init__(self, base, normalize=True):
        super(NoEmbedding, self).__init__()
        self.base = base
        self.normalize = normalize

    def forward(self, input):
        x = self.base(input)
        x = x.view(x.size(0), -1)

        if self.normalize:
            return F.normalize(x, p=2, dim=1)
        else:
            return x

