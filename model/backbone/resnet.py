import torchvision
import torch.nn as nn

__all__ = ['resnet18', 'resnet50']


class resnet18(nn.Sequential):
    output_size = 512

    def __init__(self):
        super(resnet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=True)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))


class resnet50(nn.Sequential):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))
