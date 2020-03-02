import torch

__all__ = ['init_scheduler']

def init_scheduler(scheduler, optim, patience, gamma):
    if scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=gamma, patience=patience, mode = 'min')
    elif scheduler == 'LambdaLR':
        return torch.optim.lr_scheduler.LambdaLR(optim)
    elif scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optim, step_size=patience, gamma=gamma)
    elif scheduler == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optim)
    else:
        raise KeyError("Unsupported scheduler: {}".format(scheduler))