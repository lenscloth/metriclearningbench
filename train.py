import os
import random
import argparse
import pickle
import dataset

import model.backbone as backbone

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from utils import recall
from metric import loss
from model import BaseEmbedding, NoEmbedding
from torch.utils.data import DataLoader
from metric.sampler.batch import NPairs
from metric.sampler.pair import RandomNegative, HardNegative, SemiHardNegative, DistanceWeighted


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset',
                    choices=dict(cub2011=dataset.CUB2011MetricLearning,
                                 cars196=dataset.Cars196MetricLearning,
                                 stanford_online_products=dataset.StanfordOnlineProducts),
                    default=dataset.CUB2011MetricLearning,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(inception_v1=backbone.inception_v1,
                                 resnet18=backbone.resnet18,
                                 resnet50=backbone.resnet50,
                                 vgg16bn=backbone.vgg16bn,
                                 vgg19bn=backbone.vgg19bn),
                    default=backbone.resnet50,
                    action=LookupChoices)

parser.add_argument('--sample',
                    choices=dict(random=RandomNegative,
                                 hard=HardNegative,
                                 semihard=SemiHardNegative,
                                 distance=DistanceWeighted),
                    default=RandomNegative,
                    action=LookupChoices)

parser.add_argument('--loss',
                    choices=dict(triplet=lambda *args, **kwargs:loss.Triplet(*args, squared=False, **kwargs),
                                 triplet_squared=lambda *args, **kwargs:loss.Triplet(*args, squared=True, **kwargs)),
                    default=loss.Triplet,
                    action=LookupChoices)

parser.add_argument('--no_pretrained', default=False, action='store_true')
parser.add_argument('--no_embedding', default=False, action='store_true')
parser.add_argument('--no_normalize', default=False, action='store_true')

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--save', default=None)
opts = parser.parse_args()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)


base_model = opts.base(pretrained=not opts.no_pretrained)

if isinstance(base_model, backbone.inception_v1):
    base_model_weights_path = os.path.join(opts.data, opts.base.__name__ + '.pkl')
    f = open(base_model_weights_path, 'rb')
    p = pickle.load(f, encoding='bytes')
    d = {k.decode('utf-8'): torch.from_numpy(p[k]) for k in p.keys()}
    base_model.load_state_dict(d)
    print("Inception V1: Loaded model at %s" % base_model_weights_path)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
        transforms.Normalize(mean=[122.7717, 115.9465, 102.9801], std=[1, 1, 1]),
        transforms.Lambda(lambda x: x[[2, 1, 0], ...])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
        transforms.Normalize(mean=[122.7717, 115.9465, 102.9801], std=[1, 1, 1]),
        transforms.Lambda(lambda x: x[[2, 1, 0], ...])
    ])

else:
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=True)
dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=True)

loader_train = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train, opts.batch, m=5),
                          num_workers=1, pin_memory=True)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                         num_workers=1, pin_memory=True)

if opts.no_embedding:
    model = NoEmbedding(base_model, normalize=not opts.no_normalize).cuda()
else:
    model = BaseEmbedding(base_model,
                          output_size=base_model.output_size,
                          embedding_size=opts.embedding_size,
                          normalize=not opts.no_normalize).cuda()


criterion = opts.loss(sampler=opts.sample())
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=opts.lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


def train(net, loader, ep, scheduler=None):
    if scheduler is not None:
        scheduler.step()
    net.train()
    loss_all, norm_all = [], []

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
        loss = criterion(net(images), labels)
        loss_all.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
    print('[Epoch %d] Loss: %.5f\n' % (ep, torch.Tensor(loss_all).mean()))


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
        print('[Epoch %d] Recall@1: [%.6f]\n' % (ep, recall(embeddings_all, labels_all)))


#eval(model, loader_eval, 0)
for epoch in range(1, opts.epochs+1):
    train(model, loader_train, epoch, scheduler=lr_scheduler)
    if epoch % 5 == 0:
        eval(model, loader_eval, epoch)


if opts.save is not None:
    torch.save(model.state_dict(), opts.save)
