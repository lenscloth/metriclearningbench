import os
import argparse
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import dataset
import model.backbone as backbone

import metric.loss as loss
import metric.sampler.pair as pair

from tqdm import tqdm
from utils import recall, pdist
from model import *
from metric.sampler.batch import NPairs, ExactNPairs
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--mode',
                    choices=["train", "eval"],
                    default="train")

parser.add_argument('--load',
                    default=None)

parser.add_argument('--dataset',
                    choices=dict(cub2011=dataset.CUB2011MetricLearning,
                                 cars196=dataset.Cars196MetricLearning,
                                 stanford_online_products=dataset.StanfordOnlineProducts),
                    default=dataset.CUB2011MetricLearning,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(inception_v1=backbone.GoogleNet,
                                 inception_v1bn=backbone.InceptionV1BN,
                                 resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50,
                                 vgg16=backbone.VGG16,
                                 vgg16bn=backbone.VGG16BN,
                                 vgg19bn=backbone.VGG19BN),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.AllPairs,
                    action=LookupChoices)

parser.add_argument('--loss',
                    choices=dict(l1_triplet=loss.L1Triplet,
                                 l2_triplet=loss.L2Triplet),
                    default=loss.L2Triplet,
                    action=LookupChoices)


parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--no_normalize', default=False, action='store_true')
parser.add_argument('--no_pretrained', default=False, action='store_true')

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_decay_epochs', type=int, default=[15, 20, 25, 30], nargs='+')
parser.add_argument('--lr_decay_gamma', default=0.25)
parser.add_argument('--seed', default=random.randint(1, 1000), type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch', default=150, type=int)
parser.add_argument('--n_node', default=None, type=int)
parser.add_argument('--num_image_per_class', default=5, type=int)
parser.add_argument('--optim', default="adam", choices=["adam", "sgd"])

parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--embedding_type', choices=["linear", "none"], default="linear")
parser.add_argument('--use_graph', default=False, action='store_true')

parser.add_argument('--data', default='data')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--tensorboard', type=str, default=None)

parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--iter_per_epoch', type=int, default=100)

opts = parser.parse_args()


if opts.tensorboard is not None:
    writer = SummaryWriter(log_dir='runs/%s'%opts.tensorboard)
else:
    writer = None


for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)

base_model = opts.base(pretrained=not opts.no_pretrained)
if isinstance(base_model, backbone.InceptionV1BN) or isinstance(base_model, backbone.GoogleNet):
    normalize = transforms.Compose([
        transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
        transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
    ])
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=True)
dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=True)

print("Number of images in Training Set: %d" % len(dataset_train))
print("Number of images in Test set: %d" % len(dataset_eval))

loader_train = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train,
                                                                     opts.batch,
                                                                     m=opts.num_image_per_class,
                                                                     iter_per_epoch=opts.iter_per_epoch),
                                        pin_memory=True, num_workers=opts.num_workers)

loader_train_eval = DataLoader(dataset_train, shuffle=True, batch_size=opts.batch, drop_last=False,
                               pin_memory=False, num_workers=opts.num_workers)
loader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=opts.batch, drop_last=False,
                         pin_memory=True, num_workers=opts.num_workers)

# loader_train_eval = DataLoader(dataset_train, batch_sampler=ExactNPairs(dataset_train,
#                                                                  opts.batch,
#                                                                  m=opts.num_image_per_class),
#                                         pin_memory=True, num_workers=opts.num_workers)
#
# loader_eval = DataLoader(dataset_eval, batch_sampler=ExactNPairs(dataset_eval,
#                                                                  opts.batch,
#                                                                  m=opts.num_image_per_class),
#                                         pin_memory=True, num_workers=opts.num_workers)

if opts.embedding_type == "none":
    model = NoEmbedding(base_model, normalize=not opts.no_normalize).cuda()
elif opts.embedding_type == "linear":
    model = LinearEmbedding(base_model,
                            output_size=base_model.output_size,
                            embedding_size=opts.embedding_size,
                            normalize=not opts.no_normalize).cuda()
if opts.n_node is None:
    opts.n_node = opts.batch

if opts.use_graph:
    graph_net = GraphEmbedding(in_feature=opts.embedding_size,
                               out_feature=opts.embedding_size, n_layer=2,
                               normalize=not opts.no_normalize,
                               n_node=opts.n_node).cuda()
else:
    graph_net = None


if opts.load is not None:
    model.load_state_dict(torch.load(opts.load))
    print("Loaded Model from %s" % opts.load)

criterion = opts.loss(sampler=opts.sample(), margin=opts.margin)
print(type(criterion))

parameters = list(model.parameters()) + (list(graph_net.parameters()) if graph_net is not None else [])
if opts.optim == "sgd":
    optimizer = optim.SGD(parameters, lr=opts.lr, momentum=0.9, weight_decay=1e-5)
elif opts.optim == "adam":
    optimizer = optim.Adam(parameters, lr=opts.lr, weight_decay=1e-5)

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)
n_iter = 0


def train(net, loader, ep, scheduler=None, writer=None):
    global n_iter
    if scheduler:
        scheduler.step()
    net.train()
    loss_all, norm_all = [], []
    train_iter = tqdm(loader)
    for images, labels in train_iter:
        n_iter += 1
        images, labels = images.cuda(), labels.cuda()
        embedding = net(images)

        loss = criterion(embedding, labels)
        if graph_net is not None:
            g_embedding = graph_net(embedding)
            loss += criterion(g_embedding, labels)
        loss_all.append(loss.item())

        if writer:
            writer.add_scalar('loss/train', loss.item(), n_iter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
    print('[Epoch %d] Loss: %.5f\n' % (ep, torch.Tensor(loss_all).mean()))


def eval(net, loader, ep):
    net.eval()
    n_node = graph_net.n_node
    graph_net.n_node = 1
    net.return_base = False
    test_iter = tqdm(loader)
    embeddings_all, labels_all = [], []

    test_iter.set_description("[Eval][Epoch %d]" % ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            if graph_net is not None:
                embedding = graph_net(embedding)

            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = recall(embeddings_all, labels_all)
        print('[Epoch %d] Recall@1: [%.4f]\n' % (ep, 100 * rec))
    graph_net.n_node = n_node
    return rec


def eval_graph(net, loader, ep):
    net.eval()
    graph_net.eval()
    test_iter = tqdm(loader)

    embeddings_all, labels_all = [], []
    test_iter.set_description("[Eval][Epoch %d]" % ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)
        embeddings_all = torch.cat(embeddings_all)
        labels_all = torch.cat(labels_all)

        d = pdist(embeddings_all)
        pos_idx = d.topk(11, dim=1, largest=False)[1][:, 1:]
        neg_idx = torch.randint(0, len(d), (len(d), 1), device=d.device, dtype=torch.int64)

        graph_embedding = []
        for i, e in enumerate(embeddings_all):
            pos_embedding = embeddings_all[pos_idx[i][1:]]
            neg_embedding = embeddings_all[torch.cat([pos_idx[j] for j in range(i-3, i-1)])]

            e = torch.cat((e.unsqueeze(0), pos_embedding, neg_embedding), dim=0)
            e = graph_net(e)
            graph_embedding.append(e[0])
        graph_embedding = torch.stack(graph_embedding, dim=0)

        rec = recall(graph_embedding, labels_all)
        print('[Epoch %d] Recall@1: [%.4f]\n' % (ep, 100 * rec))
    return rec


if opts.mode == "eval":
    eval(model, loader_eval, 0)
else:
    train_recall = eval(model, loader_train_eval, 0)
    val_recall = eval(model, loader_eval, 0)

    if writer:
        writer.add_scalar('recall@1/train', train_recall, 0)
        writer.add_scalar('recall@1/val', val_recall, 0)

    best_rec = val_recall
    for epoch in range(1, opts.epochs+1):
        train(model, loader_train, epoch, scheduler=lr_scheduler, writer=writer)
        train_recall = eval(model, loader_train_eval, epoch)
        val_recall = eval(model, loader_eval, epoch)

        if best_rec < val_recall:
            best_rec = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "best.pth"))
                if graph_net is not None:
                    torch.save(graph_net.state_dict(), "%s/%s" % (opts.save_dir, "best_graph.pth"))

        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "last.pth"))
            if graph_net is not None:
                torch.save(graph_net.state_dict(), "%s/%s" % (opts.save_dir, "last_graph.pth"))

            with open("%s/result.txt"%opts.save_dir, 'w') as f:
                f.write("Best Recall@1: %.4f\n" % (best_rec * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))

        print("Best Recall@1: %.4f" % best_rec)
        if writer:
            writer.add_scalar('recall@1/train', train_recall, epoch)
            writer.add_scalar('recall@1/val', val_recall, epoch)

if writer:
    writer.close()
