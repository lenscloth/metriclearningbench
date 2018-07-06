import os
import time
import random
import argparse
import itertools
import pickle
import torch
import torch.utils.data
import torchvision.transforms as transforms

import dataset
import model
import loss
import sampler

from sklearn.decomposition import PCA
from torch.utils.data.sampler import Sampler as BaseSampler
from utils import pdist


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset',
                    choices=dict(cub2011=dataset.CUB2011MetricLearning,
                                 cars196=dataset.Cars196MetricLearning,
                                 stanford_online_products=dataset.StanfordOnlineProducts),
                    default=dataset.CUB2011MetricLearning,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(inception_v1_googlenet=model.inception_v1_googlenet,
                                 resnet18=model.resnet18,
                                 resnet50=model.resnet50,
                                 vgg16bn=model.vgg16bn,
                                 vgg19bn=model.vgg19bn),
                    default=model.resnet50,
                    action=LookupChoices)

parser.add_argument('--loss',
                    choices = dict(liftedstruct=loss.LiftedStruct,
                                   triplet=loss.Triplet,
                                   tripletratio=loss.TripletRatio,
                                   pddm=loss.Pddm,
                                   untrained=loss.Untrained,
                                   margin=loss.Margin),
                    default=loss.Margin,
                    action=LookupChoices)

parser.add_argument('--sampler',
                    choices = dict(simple = sampler.simple,
                                   triplet = sampler.triplet,
                                   npairs = sampler.npairs),
                    default = sampler.npairs,
                    action = LookupChoices)

parser.add_argument('--data', default = 'data')
parser.add_argument('--log', default = 'data/log.txt')
parser.add_argument('--seed', default = 1, type = int)
parser.add_argument('--threads', default = 16, type = int)
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--batch', default = 128, type = int)
parser.add_argument('--embedding_size', default=128, type=int)
opts = parser.parse_args()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    set_random_seed(opts.seed)


def recall(embeddings, labels, K = 1):
    pca = PCA(n_components=64)
    embeddings = torch.from_numpy(pca.fit_transform(embeddings.numpy()))
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + K, dim = 1, largest = False)[1][:, 1:]
    return (labels.unsqueeze(-1).expand_as(knn_inds) == labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)).max(1)[0].float().mean()

base_model = opts.base()
base_model_weights_path = os.path.join(opts.data, opts.base.__name__ + '.pkl')
if os.path.exists(base_model_weights_path):
    f = open(base_model_weights_path, 'rb')
    p = pickle.load(f, encoding='bytes')
    d = {k.decode('utf-8'): torch.from_numpy(p[k]) for k in p.keys()}
    base_model.load_state_dict(d)

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * base_model.rescale),
    transforms.Normalize(mean = base_model.rgb_mean, std = base_model.rgb_std),
    transforms.Lambda(lambda x: x[[2, 1, 0], ...])
])

dataset_train = opts.dataset(opts.data, train=True, transform=transforms.Compose([
    transforms.RandomSizedCrop(base_model.input_side),
    transforms.RandomHorizontalFlip(),
    normalize
]), download = True)

dataset_eval = opts.dataset(opts.data, train = False, transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(base_model.input_side),
    normalize
]), download = True)

adapt_sampler = lambda batch, dataset, sampler, **kwargs: type('', (BaseSampler,), dict(__len__ = dataset.__len__, __iter__ = lambda _: itertools.chain.from_iterable(sampler(batch, dataset, **kwargs))))(None)
loader_train = torch.utils.data.DataLoader(dataset_train, sampler=adapt_sampler(opts.batch, dataset_train, opts.sampler), num_workers = opts.threads, batch_size = opts.batch, drop_last = True, pin_memory = True)
loader_eval = torch.utils.data.DataLoader(dataset_eval, shuffle=False, num_workers = opts.threads, batch_size = opts.batch, pin_memory = True)

model = opts.loss(base_model, dataset_train.num_training_classes, embedding_size=opts.embedding_size).cuda()
model_weights, model_biases, base_model_weights, base_model_biases = [[p for k, p in model.named_parameters() if p.requires_grad and ('bias' in k) == is_bias and ('base' in k) == is_base] for is_base in [False, True] for is_bias in [False, True]]

base_model_lr_mult = model.optimizer_params.pop('base_model_lr_mult', 1.0)
optimizer = model.optimizer([dict(params = base_model_weights, lr = base_model_lr_mult * model.optimizer_params['lr']), dict(params = base_model_biases, lr = base_model_lr_mult * model.optimizer_params['lr'], weight_decay = 0.0), dict(params = model_biases, weight_decay = 0.0)], **model.optimizer_params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **model.lr_scheduler_params)

log = open(opts.log, 'w')
for epoch in range(opts.epochs):
    # scheduler.step()
    # model.train()
    # loss_all, norm_all = [], []
    # for batch_idx, batch in enumerate(loader_train if model.criterion is not None else []):
    #     tic = time.time()
    #     images, labels = [torch.autograd.Variable(tensor.cuda()) for tensor in batch]
    #     loss = model.criterion(model(images), labels)
    #     loss_all.append(loss.data[0])
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print('train {:>3}.{:05}  loss  {:.04f}   hz {:.02f}'.format(epoch, batch_idx, loss_all[-1], len(images) / (time.time() - tic)))
    # log.write('loss epoch {}: {:.04f}\n'.format(epoch, torch.Tensor(loss_all or [0.0]).mean()))

    if epoch < 10 or epoch % 5 == 0 or epoch == opts.epochs - 1:
        model.eval()
        embeddings_all, labels_all = [], []
        for batch_idx, batch in enumerate(loader_eval):
            tic = time.time()
            images, labels = [torch.autograd.Variable(tensor.cuda()) for tensor in batch]
            with torch.no_grad():
                output = model(images)
            embeddings_all.append(output.data.cpu())
            labels_all.append(labels.data.cpu())
            print('eval  {:>3}.{:05}  hz {:.02f}'.format(epoch, batch_idx, len(images) / (time.time() - tic)))
        log.write('recall@1 epoch {}: {:.06f}\n'.format(epoch, recall(torch.cat(embeddings_all), torch.cat(labels_all))))
