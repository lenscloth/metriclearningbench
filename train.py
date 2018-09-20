import os
import argparse
import random

import dataset
import model.backbone as backbone

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from metric import loss
from utils import recall
from model import LinearEmbedding, NoEmbedding
from torch.utils.data import DataLoader
from metric.loss import Noise
from metric.sampler.batch import NPairs
from metric.sampler.pair import RandomNegative, AllPairs, HardNegative, SemiHardNegative, DistanceWeighted
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

parser.add_argument('--base',
                    choices=dict(inception_v1=backbone.InceptionV1,
                                 resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50,
                                 vgg16bn=backbone.VGG16BN,
                                 vgg19bn=backbone.VGG19BN),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--sample',
                    choices=dict(random=RandomNegative,
                                 hard=HardNegative,
                                 all=AllPairs,
                                 semihard=SemiHardNegative,
                                 distance=DistanceWeighted),
                    default=RandomNegative,
                    action=LookupChoices)

parser.add_argument('--loss',
                    choices=dict(l1_triplet=loss.L1Triplet,
                                 l2_triplet=loss.L2Triplet,
                                 cosine_triplet=loss.CosineTriplet),
                    default=loss.NaiveTriplet,
                    action=LookupChoices)


parser.add_argument('--subspace',
                    choices=["none", "subspace", "only_sample"],
                    default="none")

parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--embedding_type', choices=["linear", "none"], default="linear")

parser.add_argument('--no_pretrained', default=False, action='store_true')
parser.add_argument('--no_normalize', default=False, action='store_true')
parser.add_argument('--noise', default=False, action='store_true')

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--seed', default=random.randint(1, 1000), type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch', default=128, type=int)
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

base_model = opts.base(pretrained=not opts.no_pretrained)
if isinstance(base_model, backbone.InceptionV1):
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

print(len(dataset_train))
print(len(dataset_eval))

loader_train_sample = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train, opts.batch, m=5, iter_per_epoch=opts.iter_per_epoch),
                                 pin_memory=True, num_workers=opts.num_workers)
loader_train_eval = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                               pin_memory=False, num_workers=opts.num_workers)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                         pin_memory=True, num_workers=opts.num_workers)

if opts.embedding_type == "none":
    model = NoEmbedding(base_model, normalize=not opts.no_normalize).cuda()
elif opts.embedding_type == "linear":
    model = LinearEmbedding(base_model,
                            output_size=base_model.output_size,
                            embedding_size=opts.embedding_size,
                            normalize=not opts.no_normalize).cuda()

if opts.load is not None:
    model.load_state_dict(torch.load(opts.load))
    print("Loaded Model from %s" % opts.load)


criterion = opts.loss(sampler=opts.sample(), margin=opts.margin)

if opts.subspace == "none":
    pass
elif opts.subspace == "subspace":
    criterion = loss.RandomSubSpaceLoss(criterion).cuda()
elif opts.subspace == "only_sample":
    criterion = loss.RandomSubSpaceOnlySampleLoss(criterion)

if opts.noise:
    criterion = Noise(criterion)

print(type(criterion))

optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 14, 16, 18], gamma=0.5)
n_iter = 0


if opts.save_dir is not None and not os.path.isdir(opts.save_dir):
    os.mkdir(opts.save_dir)


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
    test_iter = tqdm(loader)
    embeddings_all, labels_all = [], []

    test_iter.set_description("[Eval][Epoch %d]" % ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = recall(embeddings_all, labels_all)
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
        train(model, loader_train_sample, epoch, scheduler=lr_scheduler, writer=writer)
        train_recall = eval(model, loader_train_eval, epoch)
        val_recall = eval(model, loader_eval, epoch)

        if best_rec < val_recall:
            best_rec = val_recall
            if opts.save_dir is not None:
                torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "best.pth"))
        if opts.save_dir is not None:
            torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "last.pth"))
            with open("%s/result.txt"%opts.save_dir, 'w') as f:
                f.write("Best Recall@1: %.4f\n" % (best_rec * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))

        print("Best Recall@1: %.4f" % best_rec)

        if writer:
            writer.add_scalar('recall@1/train', train_recall, epoch)
            writer.add_scalar('recall@1/val', val_recall, epoch)

if writer:
    writer.close()
