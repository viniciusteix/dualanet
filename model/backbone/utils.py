import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop

"""
code reference:
https://github.com/jfhealthcare/Chexpert.git
"""

def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps,
                                 affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))


def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))
