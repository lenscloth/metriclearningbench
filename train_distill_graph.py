import os
import random
import argparse
import pickle
import dataset
import torch.nn.functional as F
import model.backbone as backbone

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import metric.loss as loss
import metric.sampler.pair as pair

from tqdm import tqdm
from utils import recall, pdist
from model import LinearEmbedding, GraphEmbedding
from torch.utils.data import DataLoader
from metric.sampler.batch import NPairs


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

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

parser.add_argument('--base_teacher',
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

parser.add_argument('--no_embedding', default=False, action='store_true')
parser.add_argument('--no_normalize', default=False, action='store_true')

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch', default=210, type=int)

parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--teacher_embedding_size', default=128, type=int)

parser.add_argument('--teacher_load', default=None)
parser.add_argument('--save', default=None)

opts = parser.parse_args()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)

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
                                                                     iter_per_epoch=100),
                                 pin_memory=True, num_workers=4)
loader_train_eval = DataLoader(dataset_train_eval, shuffle=True, batch_size=opts.batch, drop_last=False,
                               pin_memory=False, num_workers=4)
loader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=opts.batch, drop_last=False,
                         pin_memory=True, num_workers=4)


model = LinearEmbedding(base_model,
                        output_size=base_model.output_size,
                        embedding_size=opts.embedding_size,
                        normalize=not opts.no_normalize)


teacher = LinearEmbedding(teacher_base,
                          output_size=teacher_base.output_size,
                          embedding_size=opts.teacher_embedding_size,
                          normalize=not opts.no_normalize)

teacher = GraphEmbedding(teacher, in_feature=opts.teacher_embedding_size,
                         out_feature=opts.teacher_embedding_size, n_layer=2,
                         normalize=not opts.no_normalize,
                         n_node=30).cuda()

teacher.load_state_dict(torch.load(opts.teacher_load))

teacher = teacher.cuda()
teacher.eval()
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = opts.loss(sampler=opts.sample(), margin=opts.margin)

def train(net, loader, ep, scheduler=None):
    if scheduler is not None:
        scheduler.step()
    net.train()
    loss_all, norm_all = [], []

    # teacher.normalize = False
    # net.normalize = False

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            teacher_embedding = torch.cat([teacher(img) for img in images.split(30, 0)])
        embedding = net(images)

        loss = F.kl_div(F.log_softmax(embedding*20, dim=1), F.softmax(teacher_embedding*20, dim=1), reduction='none').sum(dim=1).mean()
        loss += criterion(embedding, labels)
        #loss = (teacher_embedding - embedding).abs().sum(dim=1).mean()

        loss_all.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
    print('[Epoch %d] Loss: %.5f\n' % (ep, torch.Tensor(loss_all).mean()))


def eval(net, loader, ep):
    net.eval()

    # teacher.normalize = True
    # net.normalize = True

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


# eval(teacher, loader_train_eval, 0)
# eval(teacher, loader_eval, 0)
# eval(model, loader_train_eval, 0)
# eval(model, loader_eval, 0)

for epoch in range(1, opts.epochs+1):
    train(model, loader_train_sample, epoch, scheduler=lr_scheduler)
    eval(model, loader_train_eval, epoch)
    eval(model, loader_eval, epoch)


if opts.save is not None:
    torch.save(model.state_dict(), opts.save)
