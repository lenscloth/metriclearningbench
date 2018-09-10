import torchvision
import torch.nn as nn

__all__ = ['vgg16bn', 'vgg19bn']


class vgg16bn(nn.Module):
    output_size = 512

    def __init__(self):
        super(vgg16bn, self).__init__()
        m = torchvision.models.vgg16_bn(pretrained=True)
        self.feat = m.features
        self.embedding = m.classifier[:4]

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x


class vgg19bn(nn.Sequential):
    output_size = 512

    def __init__(self, pretrained=True):
        super(vgg19bn, self).__init__()
        self.add_module('vgg19bn', torchvision.models.vgg19_bn(pretrained=pretrained).features)
        self.add_module('avgpool', nn.AdaptiveMaxPool2d((1, 1)))
