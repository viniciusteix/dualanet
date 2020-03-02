from __future__ import print_function, absolute_import

import torch
from torch.autograd import Variable

from utils import AverageMeter, print_scores, attention_gen_patchs
from eval_metrics import compute_roc_auc

import math
import numpy as np
import torch.nn.functional as F

import ttach as tta

def get_loss(output, target, index, device, cfg):
    
    """
    code reference:
    https://github.com/jfhealthcare/Chexpert.git
    """
    
    if cfg.criterion == 'BCE':
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target[:, index].view(-1)
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                weight = (target.size()[0] - target.sum()) / target.sum()
                loss = F.binary_cross_entropy_with_logits(
                    output[index].view(-1), target, pos_weight=weight)
        else:
            loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1), target, pos_weight=pos_weight[index])

        label = torch.sigmoid(output[index].view(-1)).ge(0.5).float()
        acc = (target == label).float().sum() / len(label)
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return (loss, acc)
            
def train(branch, model_dense, model_res, model_fusion, dataloader, optimizer_dense, optimizer_res, optimizer_fusion, criterion, print_freq, epoch, max_epoch, cfg, data_transforms):
    torch.set_grad_enabled(True)
    
    if branch == 'step1':
        model_dense.train()
        model_res.eval()
    elif branch == 'step2':
        model_res.train()
        model_dense.eval()
    else:
        model_dense.train()
        model_res.train()
        
    model_fusion.train()
    
    device = torch.device('cuda:0')
    num_tasks = len(cfg.num_classes)
    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    
    loss_total = 0
    i_batch = 0
    
    for batch_idx, tuple_i in enumerate(dataloader):
        i_batch += 1
        data, target = tuple_i
        
        data = data.to(device)
        target = target.to(device)
        
        output_dense, pool_dense = model_dense(data)
        output_res, pool_res = model_res(data)
        output_fusion = model_fusion(pool_dense, pool_res)
        
        output = []
        
        if branch == 'step1':
            for i in range(num_tasks):
                aux = (output_dense[i] * 0.3) + (output_res[i] * 0.0) + (output_fusion[i] * 1.0)
                output.append(aux)
        elif branch == 'step2':
            for i in range(num_tasks):
                aux = (output_dense[i] * 0.0) + (output_res[i] * 0.3) + (output_fusion[i] * 1.0)
                output.append(aux)
        else:
            for i in range(num_tasks):
                aux = (output_dense[i] * 0.3) + (output_res[i] * 0.3) + (output_fusion[i] * 1.0)
                output.append(aux)
        
        loss = 0
        for t in range(num_tasks):
            loss_t, acc_t = get_loss(output, target, t, device, cfg)
            loss += loss_t
            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
        
        loss_total += loss
        
        if branch == 'step1':
            optimizer_dense.zero_grad()
            optimizer_fusion.zero_grad()
        elif branch == 'step2':
            optimizer_res.zero_grad()
            optimizer_fusion.zero_grad()
        else:
            optimizer_dense.zero_grad()
            optimizer_res.zero_grad()
            optimizer_fusion.zero_grad()
        
        loss.backward()
        
        if branch == 'step1':
            optimizer_dense.step()
            optimizer_fusion.step()
        elif branch == 'step2':
            optimizer_res.step()
            optimizer_fusion.step()
        else:
            optimizer_dense.step()
            optimizer_res.step()
            optimizer_fusion.step()
        
        if (batch_idx+1) % print_freq == 0:
            print("Epoch {}/{}\t Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                epoch+1, max_epoch, batch_idx+1, len(dataloader), loss, loss_total / i_batch))

def valid(branch, model_dense, model_res, model_fusion, dataloader, criterion, print_freq, classes, cfg, data_transforms):
    
    torch.set_grad_enabled(False)
    model_dense.eval()
    model_res.eval()
    model_fusion.eval()
    
    device = torch.device('cuda:0')
    num_tasks = len(cfg.num_classes)
    steps = len(dataloader)
    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)

    predlist = list(x for x in range(len(cfg.num_classes)))
    true_list = list(x for x in range(len(cfg.num_classes)))
    
    loss_total = 0
    i_batch = 0
    
    with torch.no_grad():
        for batch_idx, tuple_i in enumerate(dataloader):
            i_batch += 1
            data, target = tuple_i
        
            data = data.to(device)
            target = target.to(device)

            output_dense, pool_dense = model_dense(data)
            output_res, pool_res = model_res(data)
            output_fusion = model_fusion(pool_dense, pool_res)

            output = []
            
            if branch == 'step1':
                for i in range(num_tasks):
                    aux = (output_dense[i] * 0.3) + (output_res[i] * 0.0) + (output_fusion[i] * 1.0)
                    output.append(aux)
            elif branch == 'step2':
                for i in range(num_tasks):
                    aux = (output_dense[i] * 0.0) + (output_res[i] * 0.3) + (output_fusion[i] * 1.0)
                    output.append(aux)
            else:
                for i in range(num_tasks):
                    aux = (output_dense[i] * 0.3) + (output_res[i] * 0.3) + (output_fusion[i] * 1.0)
                    output.append(aux)
            
            loss = 0
            
            for t in range(len(cfg.num_classes)):
                loss_t, acc_t = get_loss(output, target, t, device, cfg)
                # AUC
                
                loss += loss_t
                
                output_tensor = torch.sigmoid(
                    output[t].view(-1)).cpu().detach().numpy()
                target_tensor = target[:, t].view(-1).cpu().detach().numpy()
                if batch_idx == 0:
                    predlist[t] = output_tensor
                    true_list[t] = target_tensor
                else:
                    predlist[t] = np.append(predlist[t], output_tensor)
                    true_list[t] = np.append(true_list[t], target_tensor)
                    
                loss_sum[t] += loss_t.item()
                acc_sum[t] += acc_t.item()
            
            loss = loss / 14.0
            
            loss_total += loss
            
            if (batch_idx+1) % print_freq == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                        batch_idx+1, len(dataloader), loss, loss_total / i_batch))
        roc_classes, roc_mean = compute_roc_auc(true_list, predlist, len(classes))
        
        loss_total = loss_total / steps
        
        print_scores('auroc', classes, roc_classes, roc_mean, loss_total)
        
    return loss_total