
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

from metric.loss import DistillRelativeDistanceV2, DistillAngle
from model.cub import CUB200ResNet, CUBCustomNet
from dataset.cub2011 import CUB2011Classification
from utils.etc import progress_bar

LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', help='resume from checkpoint', default=None)
parser.add_argument('--pretrained', default=False, action='store_true')
parser.add_argument('--model',
                    choices=dict(custom=lambda p: CUBCustomNet(),
                                 resnet18=lambda p: torchvision.models.resnet18(pretrained=p),
                                 resnet50=lambda p: torchvision.models.resnet50(pretrained=p)),
                    action=LookupChoices,
                    default=None, required=True)

parser.add_argument('--dist', type=float, default=0)
parser.add_argument('--angle', type=float, default=0)

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
print("Pretrained: ", args.pretrained)
net = args.model(args.pretrained)
if not isinstance(net, CUBCustomNet):
    net = CUB200ResNet(args.model(args.pretrained))
net = net.to(device)
teacher = CUB200ResNet(torchvision.models.resnet50(True)).to(device)

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

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs, True)
        loss = criterion(outputs[-1], targets)

        if args.dist > 0 or args.angle > 0:
            with torch.no_grad():
                t_outputs = teacher(inputs, True)

            dist_loss = args.dist * dist_criterion(F.normalize(outputs[1]), F.normalize(t_outputs[1]))
            angle_loss = args.angle * angle_criterion(F.normalize(outputs[1]), F.normalize(t_outputs[1]))

        loss = dist_loss + angle_loss + loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs[-1].max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save):
            os.mkdir(args.save)
        torch.save(state, './%s/ckpt.t7' % args.save)
        best_acc = acc
    print("Best Accuracy %.2f" % best_acc)

test(0)
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
