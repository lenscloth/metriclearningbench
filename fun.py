import torch
import torch.nn as nn
from model.maml.functionize import FunctionModule
from torchvision.models.vgg import vgg16
from torch.autograd import grad

model = vgg16(pretrained=True)

net = FunctionModule(model.features)
net.add_lambda(lambda x: x.view(x.size(0), -1), repr="Flattening")
net.add_module(nn.Linear(512, 128))
params = net.param_dict()

print(net)
x = torch.randn(1, 3, 32, 32)
out = net(x)
out = net(x)
out = net(x)

loss = out.sum()

gradients = grad(loss, params, create_graph=True)

new_params = [p - g for p, g in zip(params, gradients)]
new_param_dict = net.format_parameters(new_params)
new_out = net(x, param=new_param_dict)
new_loss = new_out.sum()

second_order = grad(new_loss, new_params)

print([p == g for p, g in zip(gradients, second_order)])
print('hello')