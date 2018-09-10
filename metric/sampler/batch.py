import math
import random

from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import Sampler


def index_dataset(dataset: ImageFolder):
    kv = [(cls_ind, idx) for idx, (_, cls_ind) in enumerate(dataset.imgs)]
    cls_to_ind = {}

    for k, v in kv:
        if k in cls_to_ind:
            cls_to_ind[k].append(v)
        else:
            cls_to_ind[k] = [v]

    return cls_to_ind


class NPairs(Sampler):
    def __init__(self, data_source: ImageFolder, batch_size, m=5):
        super(Sampler, self).__init__()
        self.m = m
        self.batch_size = batch_size
        self.n_batch = int(math.floor(len(data_source) / float(batch_size)))

        self.class_idx = list(data_source.class_to_idx.values())
        self.images_by_class = index_dataset(data_source)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            selected_class = random.sample(self.class_idx, k=len(self.class_idx))
            example_indices = []

            for c in selected_class:
                img_ind_of_cls = self.images_by_class[c]
                new_ind = random.sample(img_ind_of_cls, k=min(self.m, len(img_ind_of_cls)))
                example_indices += new_ind

                if len(example_indices) >= self.batch_size:
                    break

            yield example_indices[:self.batch_size]

#
#
# def sample_from_class(images_by_class, class_label_ind):
#     return images_by_class[class_label_ind][random.randrange(len(images_by_class[class_label_ind]))]
#
#
# def simple(batch_size, dataset, prob_other = 0.5):
#     '''lazy sampling, not like in lifted_struct.
#     they add to the pool all postiive combinations,
#     then compute the average number of positive pairs per image,
#     then sample for every image the same number of negative pairs'''
#     images_by_class = index_dataset(dataset)
#     for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
#         example_indices = []
#         for i in range(0, batch_size, 2):
#             perm = random.sample(images_by_class.keys(), 2)
#             example_indices += [sample_from_class(images_by_class, perm[0]), sample_from_class(images_by_class, perm[0 if i == 0 or random.random() > prob_other else 1])]
#         yield example_indices[:batch_size]
#
#
# def triplet(batch_size, dataset):
#     images_by_class = index_dataset(dataset)
#     for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
#         example_indices = []
#         for i in range(0, batch_size, 3):
#             perm = random.sample(images_by_class.keys(), 2)
#             example_indices += [sample_from_class(images_by_class, perm[0]), sample_from_class(images_by_class, perm[0]), sample_from_class(images_by_class, perm[1])]
#         yield example_indices[:batch_size]
#
#
# def npairs(batch_size, dataset, K=4):
#     images_by_class = index_dataset(dataset)
#     for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
#         example_indices = [sample_from_class(images_by_class, class_label_ind) for k in range(int(math.ceil(batch_size * 1.0 / K))) for class_label_ind in [random.choice(list(images_by_class.keys()))] for i in range(K)]
#         yield example_indices[:batch_size]
#
