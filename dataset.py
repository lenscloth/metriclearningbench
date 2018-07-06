import os
import scipy.io
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url


class StanfordOnlineProducts(ImageFolder, CIFAR10):
    base_folder = 'Stanford_Online_Products'
    url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    filename = 'Stanford_Online_Products.zip'
    zip_md5 = '7f73d41a2f44250d4779881525aea32e'

    train_list = [
        ['bicycle_final/111265328556_0.JPG', '77420a4db9dd9284378d7287a0729edb'],
        ['chair_final/111182689872_0.JPG', 'ce78d10ed68560f4ea5fa1bec90206ba']
    ]
    test_list = [
        ['table_final/111194782300_0.JPG', '8203e079b5c134161bbfa7ee2a43a0a1'],
        ['toaster_final/111157129195_0.JPG', 'd6c24ee8c05d986cafffa6af82ae224e']
    ]
    num_training_classes = 11318

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        ImageFolder.__init__(self, os.path.join(root, self.base_folder),
                             transform=transform, target_transform=target_transform, **kwargs)
        self.samples = [(os.path.join(root, self.base_folder, path), int(class_id) - 1) for i, (image_id, class_id, super_class_id, path) in enumerate(map(str.split, open(os.path.join(root, self.base_folder, 'Ebay_{}.txt'.format('train' if train else 'test'))))) if i > 1]
        self.imgs = self.samples

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.zip_md5)

        # extract file
        cwd = os.getcwd()
        os.chdir(root)
        with zipfile.ZipFile(self.filename, "r") as zip:
            zip.extractall()
        os.chdir(cwd)


class CUB2011(ImageFolder, CIFAR10):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    train_list = [
        ['001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg', '4c84da568f89519f84640c54b7fba7c2'],
        ['002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg', 'e7db63424d0e384dba02aacaf298cdc0'],
    ]
    test_list = [
        ['198.Rock_Wren/Rock_Wren_0001_189289.jpg', '487d082f1fbd58faa7b08aa5ede3cc00'],
        ['200.Common_Yellowthroat/Common_Yellowthroat_0003_190521.jpg', '96fd60ce4b4805e64368efc32bf5c6fe']
    ]

    def __init__(self, root, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        ImageFolder.__init__(self, os.path.join(root, self.base_folder),
            transform=transform, target_transform=target_transform, **kwargs)


class CUB2011MetricLearning(CUB2011):
    num_training_classes = 100

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        CUB2011.__init__(self, root, transform=transform, target_transform=target_transform, download=download, **kwargs)
        self.classes = self.classes[:self.num_training_classes] if train else self.classes[self.num_training_classes:]
        self.class_to_idx = {class_label : class_label_ind for class_label, class_label_ind in self.class_to_idx.items() if class_label in self.classes}
        self.samples = [(image_file_path, class_label_ind) for image_file_path, class_label_ind in self.imgs if class_label_ind in self.class_to_idx.values()]
        self.imgs = self.samples


class Cars196MetricLearning(ImageFolder, CIFAR10):
    base_folder_devkit = 'devkit'
    url_devkit = 'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
    filename_devkit = 'cars_devkit.tgz'
    tgz_md5_devkit = 'c3b158d763b6e2245038c8ad08e45376'

    base_folder_trainims = 'cars_train'
    url_trainims = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    filename_trainims = 'cars_ims_train.tgz'
    tgz_md5_trainims = '065e5b463ae28d29e77c1b4b166cfe61'

    base_folder_testims = 'cars_test'
    url_testims = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    filename_testims = 'cars_ims_test.tgz'
    tgz_md5_testims = '4ce7ebf6a94d07f1952d94dd34c4d501'

    url_testanno = 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'
    filename_testanno = 'cars_test_annos_withlabels.mat'
    mat_md5_testanno = 'b0a2b23655a3edd16d84508592a98d10'

    filename_trainanno = 'cars_train_annos.mat'

    base_folder = 'cars_train'
    train_list = [
        ['00001.jpg', '8df595812fee3ca9a215e1ad4b0fb0c4'],
        ['00002.jpg', '4b9e5efcc3612378ec63a22f618b5028']
    ]
    test_list = []
    num_training_classes = 98

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        if download:
            self.url, self.filename, self.tgz_md5 = self.url_devkit, self.filename_devkit, self.tgz_md5_devkit
            self.download()

            self.url, self.filename, self.tgz_md5 = self.url_trainims, self.filename_trainims, self.tgz_md5_trainims
            self.download()

            self.url, self.filename, self.tgz_md5 = self.url_testims, self.filename_testims, self.tgz_md5_testims
            self.download()

            download_url(self.url_testanno, os.path.join(root, self.base_folder_devkit), self.filename_testanno, self.mat_md5_testanno)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        ImageFolder.__init__(self, os.path.join(root, self.base_folder),
                             transform=transform, target_transform=target_transform, **kwargs)
        self.samples = [(os.path.join(root, self.base_folder_trainims, a[-1][0]), int(a[-2][0]) - 1) for filename in [self.filename_trainanno] for a in scipy.io.loadmat(os.path.join(root, self.base_folder_devkit, filename))['annotations'][0] if (int(a[-2][0]) - 1 < self.num_training_classes) == train] + [(os.path.join(root, self.base_folder_testims, a[-1][0]), int(a[-2][0]) - 1) for filename in [self.filename_testanno] for a in scipy.io.loadmat(os.path.join(root, self.base_folder_devkit, filename))['annotations'][0] if (int(a[-2][0]) - 1 < self.num_training_classes) == train]
        self.imgs = self.samples