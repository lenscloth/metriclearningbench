
'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model.cub import CUB200ResNet
from dataset.cub2011 import CUB2011Classification
from utils.etc import progress_bar
from metric.loss import DistillRelativeDistanceV2, DistillAngle


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
net = CUB200ResNet(args.model(False))
net = net.to(device)

teacher = CUB200ResNet(args.teacher(True))
teacher = teacher.to(device)
teacher.eval()
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('%s/ckpt.t7'%args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
dist_criterion = DistillRelativeDistanceV2()
angle_criterion = DistillAngle()

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)


def distill(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    net.train()
    train_loss = 0
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()

        s_out = net(inputs, True)

        with torch.no_grad():
            t_out = teacher(inputs, True)

        dist_loss = args.dist * sum([dist_criterion(e, t_e) for e, t_e in zip(s_out[:-1], t_out[:-1])])
        angle_loss = args.angle * sum([angle_criterion(e, t_e) for e, t_e in zip(s_out[:-1], t_out[:-1])])
        loss = dist_loss + angle_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | Dist Loss: %.5f | Angle Loss: %.5f'
            % (loss.item(), dist_loss.item(), angle_loss.item()))


for epoch in range(args.epoch):
    distill(epoch)

print('Saving..')
state = {
    'net': net.state_dict(),
    'acc': 0,
    'epoch': 0,
}
if not os.path.isdir(args.save):
    os.mkdir(args.save)
torch.save(state, './%s/ckpt.t7' % args.save)

