import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict


def functionize(modules: nn.Sequential):
    def linear(x, p, n, **kwargs):
        return F.linear(x, p[n][0], bias=p[n][1])

    def conv2d(x, p, n, stride=None, padding=None, dilation=None, groups=None, **kwargs):
        return F.conv2d(x, p[n][0], bias=p[n][1], stride=stride, padding=padding, dilation=dilation, groups=groups)

    def relu(x, p, n, **kwargs):
        return F.relu(x, inplace=False)

    def maxpool_2d(x, p, n, kernel_size=None, stride=None, padding=0, dilation=1,
                   ceil_mode=False, return_indices=False, **kwargs):
        return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                            ceil_mode=ceil_mode, return_indices=return_indices)

    def dropout(x, p, n, prob=0.5, inplace=False, parent=None):
        return F.dropout(x, training=parent.training, p=prob, inplace=inplace)

    parameters = OrderedDict()
    functions = []
    modules = copy.deepcopy(modules)
    for i, m in enumerate(modules):
        if isinstance(m, nn.Linear):
            parameters[i] = [m.weight.data, m.bias.data]
            functions.append(linear)

        elif isinstance(m, nn.Conv2d):
            parameters[i] = [m.weight.data, m.bias.data]
            s, p, d, g = m.stride, m.padding, m.dilation, m.groups
            fun = partial(conv2d, stride=s, padding=p, dilation=d, groups=g)
            functions.append(fun)

        elif isinstance(m, nn.ReLU):
            parameters[i] = []
            functions.append(relu)

        elif isinstance(m, nn.MaxPool2d):
            parameters[i] = []
            fun = partial(maxpool_2d, kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                          dilation=m.dilation, ceil_mode=m.ceil_mode, return_indicies=m.return_indices)
            functions.append(fun)

        elif isinstance(m, nn.Dropout):
            parameters[i] = []
            fun = partial(dropout, prob=m.p, inplace=m.inplace)
            functions.append(fun)

        else:
            raise Exception("The module (%s) is not supported" % m)

    wrapped_functions = [ReprWrapper(f, str(m), len(parameters[i])) for i, (f, m) in enumerate(zip(functions, modules))]
    for k, v in parameters.items():
        for w in v:
            w.requires_grad_()

    return wrapped_functions, parameters


class ReprWrapper(object):
    def __init__(self, func, repr, n_param=0):
        self._repr = repr
        self._func = func
        self._n_param = n_param

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __repr__(self):
        return self._repr

    def __len__(self):
        return self._n_param


class FunctionModule(object):
    def __init__(self, modules):
        functions, param = functionize(modules)
        self.functions = functions
        self.param_dict = param
        self.training = True

    def __call__(self, x, param=None):
        param = param if param else self.param_dict

        for n, f in enumerate(self.functions):
            x = f(x, param, n, parent=self)
        return x

    def __len__(self):
        return len(self.functions)

    def __repr__(self):
        r = 'Functional(\n'
        for i, f in enumerate(self.functions):
            r += '\t(%d): %s\n' % (i, f)
        r += ')'
        return r

    def parameters(self):
        for k, v in self.param_dict.items():
            for w in v:
                yield w

    def format_parameters(self, param_list):
        count = 0
        params = OrderedDict()
        for i, f in enumerate(self.functions):
            if len(f) > 0:
                params[i] = param_list[count:count+len(f)]
                count += len(f)
        return params

    def add_lambda(self, f, repr=""):
        def fun(x, p, n, lam=None, **kwargs):
            return lam(x)
        fun = ReprWrapper(partial(fun, lam=f), repr)
        self.functions.append(fun)

    def add_module(self, m):
        m = FunctionModule(nn.Sequential(m))
        self.merge(m)

    def merge(self, other):
        count = len(self)
        other_param = {k+count:v for k, v in other.param_dict.items()}
        self.param_dict = {**self.param_dict, **other_param}
        self.functions = self.functions + other.functions

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def cuda(self):
        for k, v in self.param_dict.items():
            for w in v:
                w.data = w.data.cuda()
                if w.grad is not None:
                    w.grad.data = w.grad.data.cuda()

    def cpu(self):
        for k, v in self.param_dict.items():
            for w in v:
                w.data = w.data.cpu()
                if w.grad is not None:
                    w.grad.data = w.grad.data.cpu()

    def detach(self):
        for k, v in self.param_dict.items():
            for w in v:
                w.detach_()
