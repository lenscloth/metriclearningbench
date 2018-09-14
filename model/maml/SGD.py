import torch
import copy
from torch.optim.sgd import SGD


__all__ = ["MamlSGD"]


class MamlSGD(SGD):
    def maml_step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            pairs = []
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p += weight_decay * p
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf = buf * momentum + d_p
                    else:
                        buf = param_state['momentum_buffer']
                        buf = buf * momentum + (1- dampening) * d_p
                    if nesterov:
                        d_p = d_p + (momentum * buf)
                    else:
                        d_p = buf
                new_p = p - (group['lr'] * d_p)
                pairs.append((p, new_p))
        return dict(pairs)

    def maml_replace(self, model, pairs):
        original = []
        for name, param_replaced in model.named_parameters():
            param_replacing = torch.nn.Parameter(pairs[param_replaced])
            words = name.split(".")
            m = model
            for w in words[:-1]:
                m = getattr(m, w)
            setattr(m, words[-1], param_replacing)
            original.append((param_replacing, param_replaced))
        return dict(original)

    def maml_detach(self, model):
        for _, p in model._parameters:
            p.detach_()
