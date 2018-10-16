import torch
import torch.nn as nn


class AdaGradMAML:
    def __init__(self, parameters, momentum=0.9, lr=1e-4):
        self.lr = lr
        self.momentum = momentum
        self.squared_gradients = [torch.zeros_like(p) for p in parameters]

    def update(self, parameters, gradients, eps=1e-12):
        self.squared_gradients = [(self.momentum * old_g) + ((1 - self.momentum) * new_g.data.pow(2)) for old_g, new_g in zip(self.squared_gradients, gradients)]
        new_params = []

        for p, g, acc_g in zip(parameters, gradients, self.squared_gradients):
            new_p = p - self.lr * (g / (acc_g + eps).sqrt())
            new_params.append(new_p)

        return new_params