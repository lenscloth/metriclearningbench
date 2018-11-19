import torch.nn as nn
import torch.nn.functional as F


class CUB200ResNet(nn.Module):
    def __init__(self, base_resnet):
        super().__init__()
        self.base = nn.Sequential(base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool)
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool

        self.fc = nn.Linear(base_resnet.fc.in_features, 200)

    def forward(self, x, get_pool=False):
        l1 = self.layer1(self.base(x))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        pool = self.avgpool(l4).view(x.size(0), -1)

        if get_pool:
            return F.adaptive_avg_pool2d(l3, (1, 1)).view(x.size(0), -1), pool,\
                   self.fc(pool)

        return self.fc(pool)


class CUBCustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(self.get_block(3, 64), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(self.get_block(64, 128), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(self.get_block(128, 128), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(self.get_block(128, 256), nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(self.get_block(256, 256), nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(256, 200)


    def get_block(self, in_hidden, out_hidden):
        return nn.Sequential(nn.Conv2d(in_hidden, out_hidden, 3, 1, 1),
                             nn.BatchNorm2d(out_hidden),
                             nn.ReLU(),
                             nn.Conv2d(out_hidden, out_hidden, 3, 1, 1),
                             nn.BatchNorm2d(out_hidden),
                             nn.ReLU())

    def forward(self, x, return_avg=False):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)

        pool = F.adaptive_avg_pool2d(b5, (1, 1)).view(x.size(0), -1)

        if return_avg:
            return F.adaptive_avg_pool2d(b4, (1, 1)).view(x.size(0), -1), pool, self.fc(pool)

        return self.fc(pool)