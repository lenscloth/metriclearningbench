from collections import OrderedDict
import torch.nn as nn
import torchvision
import torch

torchvision.models.inception_v3()

class vgg16bn(nn.Module):
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    input_side = 224
    output_size = 512
    rescale = 1

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
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    input_side = 224
    output_size = 512
    rescale = 1

    def __init__(self):
        super(vgg19bn, self).__init__()
        self.add_module('vgg19bn', torchvision.models.vgg19_bn(pretrained=True).features)
        self.add_module('avgpool', nn.AdaptiveMaxPool2d((1, 1)))


class resnet18(nn.Sequential):
    output_size = 512
    input_side = 224
    rescale = 1
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]

    def __init__(self, dilation = False):
        super(resnet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained = True)
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, pretrained.modules()):
            module.eval()
            module.train = lambda _: None

        if dilation:
            pretrained.layer4[0].conv1.dilation = (2, 2)
            pretrained.layer4[0].conv1.padding = (2, 2)
            pretrained.layer4[0].conv1.stride = (1, 1)
            pretrained.layer4[0].downsample[0].stride = (1, 1)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))


class resnet50(nn.Sequential):
    output_size = 2048
    input_side = 224
    rescale = 1
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]

    def __init__(self, dilation = False):
        super(resnet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained = True)
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, pretrained.modules()):
            module.eval()
            module.train = lambda _: None

        if dilation:
            pretrained.layer4[0].conv1.dilation = (2, 2)
            pretrained.layer4[0].conv1.padding = (2, 2)
            pretrained.layer4[0].conv1.stride = (1, 1)
            pretrained.layer4[0].downsample[0].stride = (1, 1)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))


class inception_v1_googlenet(nn.Sequential):
    output_size = 1024
    input_side = 227
    rescale = 255.0
    rgb_mean = [122.7717, 115.9465, 102.9801]
    rgb_std = [1, 1, 1]

    def __init__(self):
        super(inception_v1_googlenet, self).__init__(OrderedDict([
            ('conv1', nn.Sequential(OrderedDict([
                ('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3))),
                ('relu1', nn.ReLU(True)),
                ('pool1', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),
                ('lrn1', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1))
            ]))),

            ('conv2', nn.Sequential(OrderedDict([
                ('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))),
                ('relu1', nn.ReLU(True)),
                ('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1))),
                ('relu2', nn.ReLU(True)),
                ('lrn2', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)),
                ('pool2', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True))
            ]))),

            ('inception_3a', _InceptionModule(192, 64, 96, 128, 16, 32, 32)),
            ('inception_3b', _InceptionModule(256, 128, 128, 192, 32, 96, 64)),

            ('pool3', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),

            ('inception_4a', _InceptionModule(480, 192, 96, 208, 16, 48, 64)),
            ('inception_4b', _InceptionModule(512, 160, 112, 224, 24, 64, 64)),
            ('inception_4c', _InceptionModule(512, 128, 128, 256, 24, 64, 64)),
            ('inception_4d', _InceptionModule(512, 112, 144, 288, 32, 64, 64)),
            ('inception_4e', _InceptionModule(528, 256, 160, 320, 32, 128, 128)),

            ('pool4', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),

            ('inception_5a', _InceptionModule(832, 256, 160, 320, 32, 128, 128)),
            ('inception_5b', _InceptionModule(832, 384, 192, 384, 48, 128, 128)),

            ('pool5', nn.AvgPool2d((7, 7), (1, 1), ceil_mode=True)),

            # ('drop5', nn.Dropout(0.4))
        ]))


class _InceptionModule(nn.Module):
    def __init__(self, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5,
                 outplane_pool_proj):
        super(_InceptionModule, self).__init__()
        a = nn.Sequential(OrderedDict([
            ('1x1', nn.Conv2d(inplane, outplane_a1x1, (1, 1), (1, 1), (0, 0))),
            ('1x1_relu', nn.ReLU(True))
        ]))

        b = nn.Sequential(OrderedDict([
            ('3x3_reduce', nn.Conv2d(inplane, outplane_b3x3_reduce, (1, 1), (1, 1), (0, 0))),
            ('3x3_relu1', nn.ReLU(True)),
            ('3x3', nn.Conv2d(outplane_b3x3_reduce, outplane_b3x3, (3, 3), (1, 1), (1, 1))),
            ('3x3_relu2', nn.ReLU(True))
        ]))

        c = nn.Sequential(OrderedDict([
            ('5x5_reduce', nn.Conv2d(inplane, outplane_c5x5_reduce, (1, 1), (1, 1), (0, 0))),
            ('5x5_relu1', nn.ReLU(True)),
            ('5x5', nn.Conv2d(outplane_c5x5_reduce, outplane_c5x5, (5, 5), (1, 1), (2, 2))),
            ('5x5_relu2', nn.ReLU(True))
        ]))

        d = nn.Sequential(OrderedDict([
            ('pool_pool', nn.MaxPool2d((3, 3), (1, 1), (1, 1))),
            ('pool_proj', nn.Conv2d(inplane, outplane_pool_proj, (1, 1), (1, 1), (0, 0))),
            ('pool_relu', nn.ReLU(True))
        ]))

        for container in [a, b, c, d]:
            for name, module in container.named_children():
                self.add_module(name, module)

        self.branches = [a, b, c, d]

    def forward(self, input):
        return torch.cat([branch(input) for branch in self.branches], 1)


