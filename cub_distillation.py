'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


import os
import argparse

from model.cub import CUB200ResNet
from dataset.cub2011 import CUB2011Classification
from utils.etc import progress_bar
from metric.loss import DistillRelativeDistanceV2, DistillAngle, FitNet
from utils import recall


LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', help='resume from checkpoint', default=None)
parser.add_argument('--model',
                    choices=dict(resnet18=lambda p: torchvision.models.resnet18(pretrained=p),
                                 resnet50=lambda p: torchvision.models.resnet50(pretrained=p)),
                    action=LookupChoices,
                    default=None, required=True)
parser.add_argument('--teacher',
                    choices=dict(resnet18=lambda p: torchvision.models.resnet18(pretrained=p),
                                 resnet50=lambda p: torchvision.models.resnet50(pretrained=p)),
                    action=LookupChoices,
                    default=None, required=True)
parser.add_argument('--dist', type=float, default=0)
parser.add_argument('--angle', type=float, default=0)
parser.add_argument('--fitnet', type=float, default=0)
parser.add_argument('--train_fitnet', default=False, action='store_true')

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--save', default='checkpoint')
parser.add_argument('--data', default='data')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
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

trainset = CUB2011Classification(root=args.data, train=True, transform=train_transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

testset = CUB2011Classification(root=args.data, train=False, transform=test_transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

print("Number of Train Examples: %d" % len(trainset))
print("Number of Test Examples: %d" % len(testset))

print('==> Building model..')
student = CUB200ResNet(args.model(False))
student = student.to(device)

teacher = CUB200ResNet(args.teacher(True))
teacher = teacher.to(device)
teacher.eval()

embedding_layer = nn.Linear(512, 128).to(device)

if args.resume is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('%s/ckpt.t7'%args.resume)
    student.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
dist_criterion = DistillRelativeDistanceV2()
angle_criterion = DistillAngle()
fitnet_criterions = nn.ModuleList([FitNet(s, t) for s, t in zip([64, 128, 256, 512], [256, 512, 1024, 2048])]).to(device)

optimizer = optim.Adam(list(student.parameters())+list(embedding_layer.parameters()), lr=args.lr, weight_decay=5e-4)
fitnet_optimizer = optim.Adam(fitnet_criterions.parameters(), lr=0.1*args.lr, weight_decay=5e-4)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
fitnet_scheduler = optim.lr_scheduler.MultiStepLR(fitnet_optimizer, milestones=[60, 120, 160], gamma=0.1)


def avgpool(e):
    return F.adaptive_avg_pool2d(e, (1, 1)).view(e.size(0), -1)

def distill(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    fitnet_scheduler.step()

    student.train()
    train_loss = 0
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        fitnet_optimizer.zero_grad()

        s_out = student(inputs, True)
        with torch.no_grad():
            t_out = teacher(inputs, True)

        dist_loss = args.dist * dist_criterion(embedding_layer(avgpool(s_out[-1])), avgpool(t_out[-1]))
        angle_loss = args.angle * angle_criterion(embedding_layer(avgpool(s_out[-1])), avgpool(t_out[-1]))
        fitnet_loss = args.fitnet * sum([f(e, t_e) for f, e, t_e in zip(fitnet_criterions, s_out[1:], t_out[1:])])
        loss = dist_loss + angle_loss + fitnet_loss

        loss.backward()
        optimizer.step()

        if args.train_fitnet:
            fitnet_optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | FitNet Loss: %.5f | Dist Loss: %.5f | Angle Loss: %.5f'
            % (loss.item(), fitnet_loss.item(), dist_loss.item(), angle_loss.item()))


def eval(net, loader, ep):
    net.eval()
    embeddings_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = avgpool(net(images, True)[-1])
            embeddings_all.append(output.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = recall(embeddings_all, labels_all, K=[1])

        for k, r in enumerate(rec):
            print('[Epoch %d] Recall@%d: [%.4f]\n' % (ep, k+1, 100 * r))

    return rec[0]


for epoch in range(args.epoch):
    distill(epoch)
    eval(student, trainloader, epoch)
    eval(student, testloader, epoch)

    if (epoch+1) % 5 == 0:
        print('Saving..')
        state = {
            'net': student.state_dict(),
            'acc': 0,
            'epoch': 0,
        }
        if not os.path.isdir(args.save):
            os.mkdir(args.save)
        torch.save(state, './%s/ckpt.t7' % args.save)
