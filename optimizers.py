import torch
from ranger import Ranger

"""
Code imported from https://github.com/KaiyangZhou/deep-person-reid
"""

__all__ = ['init_optim']

def init_optim(optim, params, lr, weight_decay, momentum):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'ranger':
        return Ranger(params, lr=lr, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))