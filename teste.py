from __future__ import print_function, absolute_import

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import cv2
from skimage.measure import label
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.autograd import Variable

import models
from utils import Logger, save_checkpoint
from dataset_loader import DatasetGenerator
from optimizers import init_optim
from scheduler import init_scheduler
from train_test import train, valid
from utils import AverageMeter, print_scores
from eval_metrics import compute_roc_auc

# datasets
base_dir = '/home/viniciusteixeira/datasets'
dataset = 'ChestXray-NIHCC'
workers = 4
height = 224
width = 224
size = 224
no_fiding = False

# optimization options
optim = 'adam'
start_epoch = 0
max_epoch = 50
train_batch = 16
valid_batch = 16
train_shuffle = True
valid_shuffle = False
learning_rate = 0.0001
stepsize = 10
momentum = 0.9
gamma = 0.1
weight_decay = 0.00001
scheduler = 'StepLR'

# architecture
arch = 'agnn'
trained = True

# feature extraction
feature_extraction = False
feature_model = ''
feature_file = ''

# miscs
print_freq = 50
seed = 42
evaluate = False
resume = ''

eval_step = 1
save_dir = 'log'
use_cpu = False
gpu_devices = '0'

if dataset == 'ChestXray-NIHCC':
    if no_fiding:
        classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Fiding']
    else:
        classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
elif dataset == 'CheXpert-v1.0-small':
    classes = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
               'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
               'Pleural Effusion','Pleural Other','Fracture','Support Devices']
else:
    print('--dataset incorrect')
    
torch.manual_seed(seed)
use_gpu = torch.cuda.is_available()
print(use_gpu)

print("Currently using GPU {}".format(gpu_devices))
cudnn.benchmark = True
torch.cuda.manual_seed_all(seed)

pin_memory = True if use_gpu else False

print("Initializing dataset: {}".format(dataset))
    
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

#datasetTrain = DatasetGenerator(path_base=base_dir, 
#                            dataset_file='train', 
#                            transform=data_transforms['train'],
#                            dataset_=dataset,
#                            no_fiding=no_fiding)

datasetVal = DatasetGenerator(path_base=base_dir, 
                            dataset_file='valid', 
                            transform=data_transforms['valid'],
                            dataset_=dataset,
                            no_fiding=no_fiding)

#train_loader = DataLoader(dataset=datasetTrain, 
#                          batch_size=train_batch, 
#                          shuffle=train_shuffle, 
#                          num_workers=workers, 
#                          pin_memory=pin_memory)

valid_loader = DataLoader(dataset=datasetVal, 
                          batch_size=valid_batch, 
                          shuffle=valid_shuffle, 
                          num_workers=workers, 
                          pin_memory=pin_memory)

print("Initializing model: {}".format(arch))
model = models.init_model(name=arch, num_classes = len(classes), is_trained = trained)
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

# print(model)

print("Initializing optimizer: {}".format(optim))
optimizer = init_optim(optim, model.parameters(), learning_rate, weight_decay, momentum)

if use_gpu:
    model = nn.DataParallel(model).cuda()
    
model.train()
losses = AverageMeter()

for batch_idx, tuple_i in enumerate(valid_loader):
    data, target = tuple_i

    data = Variable(torch.FloatTensor(data).cuda(),requires_grad=True)
    target = Variable(torch.FloatTensor(target).cuda())

    output = model(data)
    print(output.shape)
    break
