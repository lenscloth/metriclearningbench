import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GConv"]


def gmul(node, edge):
    W, x = edge, node
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class GConv(nn.Module):
    def __init__(self, nf_input, nf_output, nf_edge, bn_bool=True):
        super(GConv, self).__init__()
        self.J = nf_edge
        self.num_inputs = nf_edge * nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, node, edge):
        x = gmul(node, edge) # out has size (bs, N, num_inputs)

        #if self.J == 1:
        #    x = torch.abs(x)

        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return edge, x