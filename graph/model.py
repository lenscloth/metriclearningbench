import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, ChebConv


class MiningGraphNet(torch.nn.Module):
    def __init__(self, input_channel=1, num_class=2):
        super(MiningGraphNet, self).__init__()
        # self.conv1 = GCNConv(input_channel, 16)
        # self.conv2 = GCNConv(16, 32)
        # self.conv3 = GCNConv(32, 64)
        # self.cls = GCNConv(64, num_class)
        self.conv1 = ChebConv(input_channel, 16, K=3, bias=False)
        self.conv2 = ChebConv(16, 32, K=3, bias=False)
        self.conv3 = ChebConv(32, 64, K=3, bias=False)
        self.cls = ChebConv(64, num_class, K=3, bias=False)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        c = self.cls(x, edge_index, edge_attr=edge_attr)

        return c

#
# class SimpleNet(torch.nn.Module):
#     def __init__(self, input_channel=1):
#         super(SimpleNet, self).__init__()
#         self.conv1 = GCNConv(input_channel, 16)
#         self.conv2 = GCNConv(16, 2)
#
#     def forward(self):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)


class SimpleMiningGraphNet(torch.nn.Module):
    def __init__(self, input_channel=1, num_class=2):
        super(SimpleMiningGraphNet, self).__init__()
        self.conv1 = GCNConv(input_channel, num_class)

    def forward(self, x, edge_index=None, edge_attr=None):
        c = self.conv1(x, edge_index, edge_attr=edge_attr)
        return F.log_softmax(c, dim=1)


class KnnGraph(torch.nn.Module):
    def __init__(self, K=30):
        super(KnnGraph, self).__init__()
        self.k = K

    def forward(self, affinity, self_loop=False):
        knn_row = torch.zeros_like(affinity)
        knn_row.scatter_(1, affinity.topk(self.k+1, dim=1, largest=True, sorted=False)[1], 1)

        knn_col = torch.zeros_like(affinity)
        knn_col.scatter_(0, affinity.topk(self.k+1, dim=0, largest=True, sorted=False)[1], 1)
        knn_graph = affinity * knn_row * knn_col

        if not self_loop:
            knn_graph[range(len(knn_graph)), range(len(knn_graph))] = 0

        return knn_graph


class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(self, size_average=True, reduce=True, weight=None):
        super(SoftCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

        if weight is not None:
            self.weighted = True
            self.register_buffer("weight", weight.unsqueeze(0))
        else:
            self.weighted = False

    def forward(self, log_prob, soft_target):
        loss = -1 * log_prob * soft_target

        if self.weighted:
            loss = self.weight * loss

        loss = loss.sum(dim=1)

        if not self.reduce:
            return loss

        if not self.size_average:
            return loss.sum()

        return loss.mean()
