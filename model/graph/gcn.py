import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["GraphConv", "GraphEdge"]


def gmul(W, x):
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)  # bs, N, N, J => J x B
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class GraphConv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(GraphConv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, W, x):
        x = gmul(W, x) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return x


class GraphEdge(nn.Module):
    def __init__(self, input_features, output_feature, operator='J2', activation='softmax', ratio=[1, 1]):
        super(GraphEdge, self).__init__()

        modules = []
        in_channel = input_features
        for r in ratio:
            out_channel = int(r * input_features)
            modules.append(nn.Conv2d(in_channel, out_channel, 1))
            modules.append(nn.LeakyReLU())
            in_channel = out_channel
        modules.append(nn.Conv2d(in_channel, output_feature, 1))
        self.net = nn.Sequential(*modules)
        self.operator = operator
        self.activation = activation

    def forward(self, x):
        W_id = torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)

        W1 = x.unsqueeze(2)
        W2 = x.unsqueeze(1)
        W = torch.abs(W1 - W2) # bs x N x N x num_feature
        W = torch.transpose(W, 1, 3)
        W = self.net(W)
        W_new = torch.transpose(W, 1, 3)

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new, dim=1)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)

        return W_new
