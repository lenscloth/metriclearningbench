import os
import argparse
import pickle
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F

import dataset
import model
import loss
import random
import utils
import graph.utils as graph_utils

from graph.model import KnnGraph, MiningGraphNet, SimpleMiningGraphNet


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset',
                    choices=dict(cub2011=dataset.CUB2011MetricLearning,
                                 cars196=dataset.Cars196MetricLearning,
                                 stanford_online_products=dataset.StanfordOnlineProducts),
                    default=dataset.CUB2011MetricLearning,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(inception_v1_googlenet=model.inception_v1_googlenet,
                                 resnet18=model.resnet18,
                                 resnet50=model.resnet50,
                                 vgg16bn=model.vgg16bn,
                                 vgg19bn=model.vgg19bn),
                    default=model.inception_v1_googlenet,
                    action=LookupChoices)

parser.add_argument('--affinity',
                    choices=dict(exp=graph_utils.exp_affinity,
                                 cos=graph_utils.cosine_affinity),
                    default=graph_utils.exp_affinity,
                    action=LookupChoices)

parser.add_argument('--data', default='data')
parser.add_argument('--log', default='data/log.txt')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--threads', default=16, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch', default=128, type=int)
opts = parser.parse_args()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)


def construct_embeddings(model, loader):
    embed = []
    labels = []
    for imgs, l in loader:
        embed.append(model(imgs.cuda()))
        labels.append(l.cuda())
    return torch.cat(embed), torch.cat(labels)


def construct_knn_graph(embeddings, K=60):
    aff = opts.affinity(embeddings)
    return KnnGraph(K=K)(aff)


def construct_feature(embeddings, knn_graph):
    cosine_aff = graph_utils.cosine_affinity(embeddings)
    #manifold_aff = graph_utils.manifold_dist(knn_graph)
    euclidian_aff = utils.pdist(embeddings)
    feat = torch.stack([cosine_aff, euclidian_aff, knn_graph], dim=2)
    feat = F.normalize(feat, dim=1)

    return feat


def eval_graph(scores, labels, K=1):
    pos_correct = 0
    neg_correct = 0
    total = K * len(scores)
    pos_select = 0
    for i, s in enumerate(scores):
        l = labels[i]
        pos = (labels == l)
        neg = (labels != l)
        s = F.softmax(s, dim=1)
        ind = s.max(dim=1)[1]
        pos_select += ind.sum().item()
        s[i] = -1000
        pos_correct += pos[torch.topk(s[:, 1], K)[1]].sum().item()
        neg_correct += neg[torch.topk(s[:, 0], K)[1]].sum().item()

    print("\n[Pos]: %d" % pos_select)
    return pos_correct, neg_correct, total


base_model = opts.base()
base_model_weights_path = os.path.join(opts.data, opts.base.__name__ + '.pkl')
if os.path.exists(base_model_weights_path):
    f = open(base_model_weights_path, 'rb')
    p = pickle.load(f, encoding='bytes')
    d = {k.decode('utf-8'): torch.from_numpy(p[k]) for k in p.keys()}
    base_model.load_state_dict(d)

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * base_model.rescale),
    transforms.Normalize(mean=base_model.rgb_mean, std=base_model.rgb_std),
    transforms.Lambda(lambda x: x[[2, 1, 0], ...])
])

dataset_train = opts.dataset(opts.data, train=True, transform=transforms.Compose([
    transforms.RandomSizedCrop(base_model.input_side),
    transforms.RandomHorizontalFlip(),
    normalize
]), download=True)

dataset_eval = opts.dataset(opts.data, train=False, transform=transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(base_model.input_side),
    normalize
]), download=True)

loader_train = torch.utils.data.DataLoader(dataset_train,
                                           shuffle=True,
                                           num_workers=opts.threads,
                                           batch_size=opts.batch,
                                           pin_memory=True)

loader_eval = torch.utils.data.DataLoader(dataset_eval,
                                          shuffle=True,
                                          num_workers=opts.threads,
                                          batch_size=opts.batch,
                                          pin_memory=True)

model = loss.Untrained(base_model, dataset_train.num_training_classes, embedding_size=128).cuda()
model.eval()

# Create Graph
with torch.no_grad():
    train_embedding, train_label = construct_embeddings(model, loader_train)
    train_graph = construct_knn_graph(train_embedding)
    train_feature = construct_feature(train_embedding, train_graph)
    train_edge_index, train_edge_attr = train_graph.triu().nonzero().t(), train_graph.triu()[train_graph.triu() > 0]

    test_embedding, test_label = construct_embeddings(model, loader_eval)
    test_graph = construct_knn_graph(test_embedding)
    test_feature = construct_feature(test_embedding, test_graph)
    test_edge_index, test_edge_attr = test_graph.triu().nonzero().t(), test_graph.triu()[test_graph.triu() > 0]

graph_net = MiningGraphNet(input_channel=train_feature.size(2)).cuda()
optimizer = torch.optim.SGD(graph_net.parameters(), lr=0.01, weight_decay=5e-4)

# Train & Test
for epoch in range(opts.epochs):
    is_train = epoch % 2 == 0
    embedding, graph, edge_index, edge_attr, feature, label =\
        (train_embedding, train_graph, train_edge_index, train_edge_attr, train_feature, train_label) if is_train else \
        (test_embedding, test_graph, test_edge_index, test_edge_attr, test_feature, test_label)

    scores = []
    for idx in range(len(embedding)):
        target = (label == label[idx]).long()
        if is_train:
            score = graph_net(feature[idx], edge_index=edge_index)
            loss = F.cross_entropy(score, target, weight=torch.tensor([1., 10.], device=score.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Train %d/%d, Loss:%.3f]" % (idx+1, len(embedding), loss.item()), end="\r")
        else:
            with torch.no_grad():
                score = graph_net(feature[idx], edge_index=edge_index)
            print("[Eval %d/%d]" % (idx+1, len(embedding)), end="\r")
        scores.append(score)

    if not is_train:
        pos_correct, neg_correct, total = eval_graph(scores, label, K=1)
        print("[Epoch %d][Positive %d/%d][Negative %d/%d]" % (epoch, pos_correct, total, neg_correct, total))
