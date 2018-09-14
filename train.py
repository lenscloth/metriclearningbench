import argparse
import random

import dataset
import model.backbone as backbone

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from utils import recall
from metric import loss
from model import LinearEmbedding, NoEmbedding, ConvEmbedding
from torch.utils.data import DataLoader
from metric.sampler.batch import NPairs
from metric.sampler.pair import RandomNegative, AllPairs, HardNegative, SemiHardNegative, DistanceWeighted


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
                    choices=dict(triplet=lambda *args, **kwargs: loss.NaiveTriplet(*args, squared=False, **kwargs),
                                 triplet_squared=lambda *args, **kwargs: loss.NaiveTriplet(*args, squared=True, **kwargs),
                                 log_triplet=lambda *args, **kwargs: loss.LogTriplet(*args, squared=False, **kwargs),
                                 log_triplet_squared=lambda *args, **kwargs: loss.LogTriplet(*args, squared=True, **kwargs)),
                    default=loss.NaiveTriplet,
                    action=LookupChoices)

parser.add_argument('--embedding_type',
                    choices=["linear", "conv", "none"],
                    default="linear")

parser.add_argument('--no_pretrained', default=False, action='store_true')
parser.add_argument('--no_normalize', default=False, action='store_true')
parser.add_argument('--return_base_embedding', default=False, action='store_true')

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--seed', default=random.randint(1, 1000), type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--save', default=None)
opts = parser.parse_args()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)

base_model = opts.base(pretrained=not opts.no_pretrained)
if isinstance(base_model, backbone.InceptionV1):
    normalize = transforms.Compose([
        transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255),
        transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
    ])
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def make_square(x):
    width, height = x.size
    big_axis = max(width, height)
    padder = transforms.Pad((int((big_axis - width) / 2.0), int((big_axis - height) / 2.0)))
    return padder(x)


train_transform = transforms.Compose([
    transforms.Lambda(lambda x: make_square(x)),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


test_transform = transforms.Compose([
    transforms.Lambda(lambda x: make_square(x)),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=True)
dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=True)

print(len(dataset_train))
print(len(dataset_eval))

loader_train_sample = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train, opts.batch, m=5), pin_memory=True, num_workers=4)
loader_train_eval = DataLoader(dataset_train, shuffle=False, batch_size=opts.batch, drop_last=False, pin_memory=False, num_workers=4)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False, pin_memory=True, num_workers=4)

if opts.embedding_type == "none":
    model = NoEmbedding(base_model, normalize=not opts.no_normalize).cuda()
elif opts.embedding_type == "linear":
    model = LinearEmbedding(base_model,
                            output_size=base_model.output_size,
                            embedding_size=opts.embedding_size,
                            normalize=not opts.no_normalize,
                            return_base_embedding=opts.return_base_embedding).cuda()
elif opts.embedding_type == "conv":
    model = ConvEmbedding(base_model,
                          output_size=base_model.output_size,
                          embedding_size=opts.embedding_size,
                          normalize=not opts.no_normalize,
                          return_base_embedding=opts.return_base_embedding).cuda()


if opts.load is not None:
    model.load_state_dict(torch.load(opts.load))
    print("Loaded Model from %s" % opts.load)

criterion = opts.loss(sampler=opts.sample())
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=opts.lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train(net, loader, ep, scheduler=None):
    if scheduler is not None:
        scheduler.step()
    net.train()
    loss_all, norm_all = [], []

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
        e = net(images)
        loss = criterion(e, labels)
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
        print('[Epoch %d] Recall@1: [%.4f]\n' % (ep, 100 * recall(embeddings_all, labels_all)))


if opts.mode == "eval":
    eval(model, loader_eval, 0)
else:
    eval(model, loader_eval, 0)
    for epoch in range(1, opts.epochs+1):
        train(model, loader_train_sample, epoch, scheduler=lr_scheduler)
        if epoch % 5 == 0:
            eval(model, loader_train_eval, epoch)
            eval(model, loader_eval, epoch)

    if opts.save is not None:
        torch.save(model.state_dict(), opts.save)
