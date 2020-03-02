from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['Fusion']

class Fusion(nn.Module):
    def __init__(self, input_size, output_size):
        super(Fusion, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()
        self.output = output_size

    def forward(self, global_pool, local_pool):
        global_pool = global_pool.squeeze(-1).squeeze(-1)
        local_pool = local_pool.squeeze(-1).squeeze(-1)
        fusion = torch.cat((global_pool,local_pool), 1).cuda()
        fusion_var = torch.autograd.Variable(fusion)
        
        x = self.fc(fusion_var)
        x = self.Sigmoid(x)
        x = x.unsqueeze(-1)
        
        out = []
        for i in range(self.output):
            out.append(x[:,i])
        return out