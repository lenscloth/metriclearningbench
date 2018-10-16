import os
import argparse
import random

import dataset
import model.backbone as backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from metric import loss
from utils import recall
from torch.autograd import grad
from model.maml.optim import AdaGradMAML
from torch.utils.data import DataLoader
from metric.sampler.batch import NPairs
from metric.sampler.pair import RandomNegative, AllPairs, HardNegative, SemiHardNegative, DistanceWeighted
from model.maml.functionize import FunctionModule

from tensorboardX import SummaryWriter


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

parser.add_argument('--sample',
                    choices=dict(random=RandomNegative,
                                 hard=HardNegative,
                                 all=AllPairs,
                                 semihard=SemiHardNegative,
                                 distance=DistanceWeighted),
                    default=AllPairs,
                    action=LookupChoices)

parser.add_argument('--loss',
                    choices=dict(l1_triplet=loss.L1Triplet,
                                 l2_triplet=loss.L2Triplet),
                    default=loss.L2Triplet,
                    action=LookupChoices)

parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--no_pretrained', default=False, action='store_true')
parser.add_argument('--no_normalize', default=False, action='store_true')
parser.add_argument('--no_maml', default=False, action='store_true')

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--seed', default=random.randint(1, 1000), type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch', default=30, type=int)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--save_dir', default=None)

parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--iter_per_epoch', type=int, default=100)
parser.add_argument('--tensorboard', type=str, default=None)

opts = parser.parse_args()


if opts.tensorboard is not None:
    writer = SummaryWriter(log_dir='runs/%s'%opts.tensorboard)
else:
    writer = None


for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)

model = FunctionModule(backbone.vgg.VGG16().feat)
model.add_lambda(lambda x: F.adaptive_avg_pool2d(x, output_size=(1, 1)), repr=str(nn.AdaptiveAvgPool2d((1,1))))
model.add_lambda(lambda x: x.view(x.size(0), -1), repr="1D-Flattening")
model.add_module(nn.Linear(512, opts.embedding_size))
model.add_lambda(lambda x: F.normalize(x, dim=1, p=2), repr="L2 Normalize")
model.cuda()
print(model)


class SampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(opts.embedding_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64))

        self.relation = nn.Sequential(nn.Conv2d(128, 64, 1),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 32, 1),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 1, 1))

    def forward(self, x):
        e = self.net(x).t()

        e1 = e.unsqueeze(2).repeat(1, 1, len(x))
        e2 = e.unsqueeze(1).repeat(1, len(x), 1)
        e12 = torch.cat((e1, e2), dim=0)
        e21 = torch.cat((e2, e1), dim=0)

        w1 = self.relation(e12.unsqueeze(0))
        w2 = self.relation(e21.unsqueeze(0))
        w = F.softmax(w1 + w2, dim=1) * len(x)
        return w.squeeze()


sample_net = SampleNet().cuda()
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

print(len(dataset_train))
print(len(dataset_eval))

loader_train_sample = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train, opts.batch, m=5, iter_per_epoch=opts.iter_per_epoch),
                                 pin_memory=True, num_workers=opts.num_workers)
loader_train = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                          pin_memory=False, num_workers=opts.num_workers)

loader_eval_sample = DataLoader(dataset_eval, batch_sampler=NPairs(dataset_train, opts.batch, m=5, iter_per_epoch=opts.iter_per_epoch),
                                pin_memory=True, num_workers=opts.num_workers)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                         pin_memory=True, num_workers=opts.num_workers)

if opts.load is not None:
    params = torch.load(opts.load)
    model.param_dict = params
    print("Loaded Model from %s" % opts.load)


criterion = opts.loss(sampler=opts.sample(), margin=opts.margin)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
maml_optimizer = AdaGradMAML(model.parameters(), lr=1e-3, momentum=0.9)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25, 30], gamma=0.25)

sampler_optimizer = optim.Adam(sample_net.parameters(), lr=1e-3, weight_decay=1e-5)
sampler_lr_scheduler = optim.lr_scheduler.MultiStepLR(sampler_optimizer, milestones=[15, 20, 25, 30], gamma=0.25)

n_iter = 0

if opts.save_dir is not None and not os.path.isdir(opts.save_dir):
    os.mkdir(opts.save_dir)


def train(ep, writer=None):
    global n_iter
    lr_scheduler.step()

    model.train()
    sample_net.train()

    loss_all = []
    train_iter = tqdm(zip(loader_train_sample, loader_eval_sample), total=len(loader_train_sample))

    for (images, labels), (val_images, val_labels) in train_iter:
        n_iter += 1
        images, labels = images.cuda(), labels.cuda()
        val_images, val_labels = val_images.cuda(), val_labels.cuda()
        # weight = torch.ones((len(images), len(labels)), device=images.device, requires_grad=True)
        embedding = model(images)

        if opts.no_maml:
            weight = torch.ones((len(images), len(labels)), device=images.device)
        else:

            weight = sample_net(embedding)
            loss = criterion(embedding, labels, weight)

            optimizer.zero_grad()
            model_gradients = grad(loss, model.parameters(), create_graph=True)

            updated_param = maml_optimizer.update(model.parameters(), model_gradients)
            updated_param = model.format_parameters(updated_param)

            updated_embedding = model(val_images, updated_param)
            sample_loss = criterion(updated_embedding, val_labels)

            sampler_optimizer.zero_grad()
            sample_loss.backward(retain_graph=True)
            sampler_optimizer.step()

            weight = sample_net(embedding)

        loss = criterion(embedding, labels, weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())
        if writer:
            writer.add_scalar('loss/train', loss.item(), n_iter)

        train_iter.set_description("[Train, Emnbedding][Epoch %d] Loss: %.5f" % (ep, loss.item()))
    print('[Epoch %d] Loss: %.5f\n' % (ep, torch.Tensor(loss_all).mean()))


def eval(ep, loader=loader_eval):
    model.eval()
    test_iter = tqdm(loader)
    embeddings_all, labels_all = [], []

    test_iter.set_description("[Eval][Epoch %d]" % ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = model(images)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = recall(embeddings_all, labels_all)
        print('[Epoch %d] Recall@1: [%.4f]\n' % (ep, 100 * rec))
    return rec


if opts.mode == "eval":
    eval(0)
else:
    best_rec = 0
    # train_recall = eval(0, loader_train)
    # val_recall = eval(0, loader_eval)
    #
    # if writer:
    #     writer.add_scalar('recall@1/train', train_recall, 0)
    #     writer.add_scalar('recall@1/val', val_recall, 0)
    #
    # best_rec = val_recall
    for epoch in range(1, opts.epochs+1):
        train(epoch, writer=writer)
        train_recall = eval(epoch, loader=loader_train)
        val_recall = eval(epoch, loader=loader_eval)

        if best_rec < val_recall:
            best_rec = val_recall
            if opts.save_dir is not None:
                torch.save(model.param_dict, "%s/%s"%(opts.save_dir, "best.pth"))
        if opts.save_dir is not None:
            torch.save(model.param_dict, "%s/%s"%(opts.save_dir, "last.pth"))
            with open("%s/result.txt"%opts.save_dir, 'w') as f:
                f.write("Best Recall@1: %.4f\n" % (best_rec * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))

        print("Best Recall@1: %.4f" % best_rec)

        if writer:
            writer.add_scalar('recall@1/train', train_recall, epoch)
            writer.add_scalar('recall@1/val', val_recall, epoch)

if writer:
    writer.close()