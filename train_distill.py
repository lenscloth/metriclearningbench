import os
import random
import argparse
import pickle
import dataset
import math
import model.backbone as backbone

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from metric.loss import HardDarkRank, DistillDistance, DistillRelativeDistance, DistillRelativeDistanceV2, DistillAngle, L1Triplet, L2Triplet
import metric.sampler.pair as pair

from tqdm import tqdm
from utils import recall, pdist
from model import *
from torch.utils.data import DataLoader
from metric.sampler.batch import NPairs


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset',
                    choices=dict(cub200=dataset.CUB2011MetricLearning,
                                 cars196=dataset.Cars196MetricLearning,
                                 stanford=dataset.StanfordOnlineProducts),
                    default=dataset.CUB2011MetricLearning,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(inception_v1=backbone.GoogleNet,
                                 inception_v1bn=backbone.InceptionV1BN,
                                 resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50,
                                 resnet50_v2=backbone.ResNet50_v2,
                                 vgg16=backbone.VGG16,
                                 vgg16bn=backbone.VGG16BN,
                                 vgg19bn=backbone.VGG19BN),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--base_teacher',
                    choices=dict(inception_v1=backbone.GoogleNet,
                                 inception_v1bn=backbone.InceptionV1BN,
                                 resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50,
                                 resnet50_v2=backbone.ResNet50_v2,
                                 vgg16=backbone.VGG16,
                                 vgg16bn=backbone.VGG16BN,
                                 vgg19bn=backbone.VGG19BN),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--triplet_ratio', default=0, type=float)
parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--angle_ratio', default=0, type=float)
parser.add_argument('--dark_ratio', default=0, type=float)

parser.add_argument('--aux_loss',
                    choices=dict(l1_triplet=L1Triplet,
                                 l2_triplet=L2Triplet),
                    default=L2Triplet,
                    action=LookupChoices)

parser.add_argument('--aux_sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.AllPairs,
                    action=LookupChoices)
parser.add_argument('--aux_margin', type=float, default=0.2)
parser.add_argument('--no_normalize', default=False, action='store_true')
parser.add_argument('--teacher_no_normalize', default=False, action='store_true')

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--iter_per_epoch', default=100, type=int)
parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--teacher_embedding_size', default=128, type=int)

parser.add_argument('--teacher_load', default=None)
parser.add_argument('--save_dir', default=None)

opts = parser.parse_args()
base_model = opts.base(pretrained=True)
teacher_base = opts.base_teacher(pretrained=False)

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
dataset_train_eval = opts.dataset(opts.data, train=True, transform=test_transform, download=True)
dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=True)

print("Number of images in Training Set: %d" % len(dataset_train))
print("Number of images in Test set: %d" % len(dataset_eval))

print(len(dataset_train))
print(len(dataset_eval))

loader_train_sample = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train,
                                                                     opts.batch,
                                                                     m=5,
                                                                     iter_per_epoch=opts.iter_per_epoch),
                                 pin_memory=True, num_workers=8)
loader_train_eval = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                               pin_memory=False, num_workers=8)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                         pin_memory=True, num_workers=8)

model = LinearEmbedding(base_model,
                        output_size=base_model.output_size,
                        embedding_size=opts.embedding_size,
                        normalize=not opts.no_normalize)

teacher = LinearEmbedding(teacher_base,
                          output_size=teacher_base.output_size,
                          embedding_size=opts.teacher_embedding_size,
                          normalize=not opts.teacher_no_normalize)

teacher.load_state_dict(torch.load(opts.teacher_load))

teacher = teacher.cuda()
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

dist_criterion = DistillRelativeDistanceV2() #DistillDistance()
angle_criterion = DistillAngle()
dark_criterion = HardDarkRank()
triplet_criterion = opts.aux_loss(sampler=opts.aux_sample(), margin=opts.aux_margin)


def train(net, loader, ep, scheduler=None):
    if scheduler is not None:
        scheduler.step()
    net.train()
    teacher.eval()

    dist_loss_all = []
    angle_loss_all = []
    dark_loss_all = []
    triplet_loss_all = []
    loss_all = []

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()

        e = model(images)
        with torch.no_grad():
            t_e = teacher(images)

        triplet_loss = opts.triplet_ratio * triplet_criterion(e, labels)
        dist_loss = opts.dist_ratio * dist_criterion(e, t_e)
        angle_loss = opts.angle_ratio * angle_criterion(e, t_e)
        dark_loss = opts.dark_ratio * dark_criterion(e, t_e)
        loss = triplet_loss + dist_loss + angle_loss + dark_loss

        triplet_loss_all.append(triplet_loss.item())
        dist_loss_all.append(dist_loss.item())
        angle_loss_all.append(angle_loss.item())
        dark_loss_all.append(dark_loss.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_iter.set_description("[Train][Epoch %d] Triplet: %.5f, Dist: %.5f, Angle: %.5f, Dark: %5f" %
                                   (ep, triplet_loss.item(), dist_loss.item(), angle_loss.item(), dark_loss.item()))
    print('[Epoch %d] Loss: %.5f, Triplet: %.5f, Dist: %.5f, Angle: %.5f, Dark: %.5f \n' %\
          (ep, torch.Tensor(loss_all).mean(), torch.Tensor(triplet_loss_all).mean(),
           torch.Tensor(dist_loss_all).mean(), torch.Tensor(angle_loss_all).mean(), torch.Tensor(dark_loss_all).mean()))


def eval(net, loader, ep):
    net.eval()
    test_iter = tqdm(loader)
    embeddings_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            output = net(images)
            embeddings_all.append(output.data)
            labels_all.append(labels.data)
            test_iter.set_description("[Eval][Epoch %d]" % ep)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = recall(embeddings_all, labels_all, K=4)

        for k, r in enumerate(rec):
            print('[Epoch %d] Recall@%d: [%.4f]\n' % (ep, k+1, 100 * r))

    return rec[0]


def mean_norm(net, loader):
    net.eval()

    with torch.no_grad():
        norms = []
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            output = net(images)
            norm = output.pow(2).sum(dim=1).sqrt().mean().item()
            norms.append(norm)
        mean_norm = sum(norms) / float(len(norms))
    print("Mean Norm : %f" % mean_norm)
    return mean_norm


# if opts.no_normalize and isinstance(model, LinearEmbedding):
#     print("Initializing Linear weight...")
#     mean_teacher = mean_norm(teacher, loader_train_eval)
#     mean_student = mean_norm(model, loader_train_eval)
#
#     model.linear.weight.data = model.linear.weight.data / (mean_student / mean_teacher)
#     model.linear.bias.data = model.linear.bias.data / (mean_student / mean_teacher)


eval(teacher, loader_train_eval, 0)
eval(teacher, loader_eval, 0)
best_train_rec = eval(model, loader_train_eval, 0)
best_val_rec = eval(model, loader_eval, 0)

for epoch in range(1, opts.epochs+1):
    train(model, loader_train_sample, epoch, scheduler=lr_scheduler)
    train_recall = eval(model, loader_train_eval, epoch)
    val_recall = eval(model, loader_eval, epoch)

    if best_train_rec < train_recall:
        best_train_rec = train_recall

    if best_val_rec < val_recall:
        best_val_rec = val_recall
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))
    if opts.save_dir is not None:
        if not os.path.isdir(opts.save_dir):
            os.mkdir(opts.save_dir)
        torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
        with open("%s/result.txt" % opts.save_dir, 'w') as f:
            f.write('Best Train Recall@1: %.4f\n' % (best_train_rec * 100))
            f.write("Best Test Recall@1: %.4f\n" % (best_val_rec * 100))
            f.write("Final Recall@1: %.4f\n" % (val_recall * 100))

    print("Best Train Recall: %.4f" % best_train_rec)
    print("Best Eval Recall: %.4f" % best_val_rec)
