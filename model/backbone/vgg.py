import torchvision
import torch.nn as nn

__all__ = ['VGG16BN', 'VGG19BN']


class VGG16(nn.Module):
    output_size = 512

    def __init__(self):
        super(VGG16, self).__init__()
        m = torchvision.models.vgg16(pretrained=True)
        self.feat = m.features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.avg_pool(self.feat(x))


class VGG16BN(nn.Module):
    output_size = 512

    def __init__(self):
        super(VGG16BN, self).__init__()
        m = torchvision.models.vgg16_bn(pretrained=True)
        self.feat = m.features
        self.embedding = m.classifier[:4]

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x


class VGG19BN(nn.Sequential):
    output_size = 512

    def __init__(self, pretrained=True):
        super(VGG19BN, self).__init__()
        self.add_module('vgg19bn', torchvision.models.vgg19_bn(pretrained=pretrained).features)
        self.add_module('avgpool', nn.AdaptiveMaxPool2d((1, 1)))
