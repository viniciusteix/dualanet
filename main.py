from __future__ import print_function, absolute_import

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

from utils import Logger, save_checkpoint
from dataset_loader import DatasetGenerator
from optimizers import init_optim
from scheduler import init_scheduler
from train_test import train, valid
from Fusion import Fusion

from model.classifier import Classifier  # noqa
from model.utils import get_optimizer  # noqa

parser = argparse.ArgumentParser(description='Train image model')

# datasets
parser.add_argument('--base-dir', type=str, default='/home/viniciusteixeira/datasets')
parser.add_argument('--dataset', type=str, default='ChestXray-NIHCC')
parser.add_argument('--workers', type=int, default=4, help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224, help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=224, help="width of an image (default: 224)")
parser.add_argument('--size', type=int, default=224, help="size of an image (default: (224 x 224))")
parser.add_argument('--no-fiding', action='store_true', help="add no finding class")

# optimzation options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--start-epoch', type=int, default=0, help="manual epoch number (useful on restarts)")
parser.add_argument('--max-epoch', type=int, default=10, help="maximum epochs to run (default: 10)")
parser.add_argument('--train-batch', type=int, default=4, help="train batch size (default: 8)")
parser.add_argument('--valid-batch', type=int, default=4, help="valid batch size (default: 8)")
parser.add_argument('--train-shuffle', type=bool, default=True, help='shuffle train dataloader (default: false)')
parser.add_argument('--valid-shuffle', type=bool, default=False, help='shuffle valid dataloader (default: false)')
parser.add_argument('--learning-rate', type=float, default=0.0001, help="initial learning rate (default: 0.0001)")
parser.add_argument('--stepsize', default=10, type=int, help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum value (default: 0.9)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay")
parser.add_argument('--weight-decay', type=float, default=0.00001, help="weight decay (default: 0.00001)")
parser.add_argument('--scheduler', type=str, default='StepLR', help="allows dynamic learning rate reducing based on some validation measurements")

# feature extraction
parser.add_argument('--feature-extraction', action='store_true', help="generate feature vector and save in file")
parser.add_argument('--feature-model', type=str, default='', help="path to model generate features")
parser.add_argument('--feature-file', type=str, default='', help="file for save the feature vectors")

# miscs
parser.add_argument('--print-freq', type=int, default=400, help="print frequency (default: 50)")
parser.add_argument('--seed', type=int, default=42, help="manual seed")
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--only-fusion', action='store_true', help="output is fusion branch only")
parser.add_argument('--eval-step', type=int, default=-1, help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--step', type=int, default=1, help="step of code")

# paths
parser.add_argument('--resume-densenet', type=str, default='', help="resume densenet model checkpoint")
parser.add_argument('--resume-rednet', type=str, default='', help="resume resnet model checkpoint")
parser.add_argument('--resume-fusion', type=str, default='', help="resume fusion model checkpoint")
parser.add_argument('--infos-densenet', type=str, default='densenet.json', help="resume fusion model checkpoint")
parser.add_argument('--infos-resnet', type=str, default='resnet.json', help="resume fusion model checkpoint")

args = parser.parse_args()

def main():
    if args.dataset == 'ChestXray-NIHCC':
        if args.no_fiding:
            classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Fiding']
        else:
            classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    elif args.dataset == 'CheXpert-v1.0-small':
        classes = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
                   'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
                   'Pleural Effusion','Pleural Other','Fracture','Support Devices']
    else:
        print('--dataset incorrect')
        return
    
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
        
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    pin_memory = True if use_gpu else False

    print("Initializing dataset: {}".format(args.dataset))
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(556),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(556),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    
    datasetTrain = DatasetGenerator(path_base=args.base_dir, 
                            dataset_file='train', 
                            transform=data_transforms['train'],
                            dataset_=args.dataset,
                            no_fiding=args.no_fiding)

    datasetVal =   DatasetGenerator(path_base=args.base_dir, 
                                dataset_file='valid', 
                                transform=data_transforms['valid'],
                                dataset_=args.dataset,
                                no_fiding=args.no_fiding)
    
    train_loader = DataLoader(dataset=datasetTrain, batch_size=args.train_batch, shuffle=args.train_shuffle, num_workers=args.workers, pin_memory=pin_memory)
    valid_loader = DataLoader(dataset=datasetVal, batch_size=args.valid_batch, shuffle=args.valid_shuffle, num_workers=args.workers, pin_memory=pin_memory)
    
    with open(args.infos_densenet) as f:
        cfg = edict(json.load(f))
    
    print('Initializing densenet branch')
    model_dense = Classifier(cfg)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model_dense.parameters())/1000000.0))
    
    with open(args.infos_resnet) as f:
        cfg = edict(json.load(f))
    
    print('Initializing resnet branch')
    model_res = Classifier(cfg)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model_res.parameters())/1000000.0))

    print('Initializing fusion branch')
    model_fusion = Fusion(input_size = 7424, output_size = len(classes))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model_fusion.parameters())/1000000.0))
    
    print("Initializing optimizers")
    optimizer_dense = init_optim(args.optim, model_dense.parameters(), args.learning_rate, args.weight_decay, args.momentum)
    optimizer_res = init_optim(args.optim, model_res.parameters(), args.learning_rate, args.weight_decay, args.momentum)
    optimizer_fusion = init_optim(args.optim, model_fusion.parameters(), args.learning_rate, args.weight_decay, args.momentum)
    
    criterion = nn.BCELoss()
    
    print("Initializing scheduler: {}".format(args.scheduler))
    if args.stepsize > 0:
        scheduler_dense = init_scheduler(args.scheduler, optimizer_dense, args.stepsize, args.gamma)
        scheduler_res = init_scheduler(args.scheduler, optimizer_res, args.stepsize, args.gamma)
        scheduler_fusion = init_scheduler(args.scheduler, optimizer_fusion, args.stepsize, args.gamma)
        
    start_epoch = args.start_epoch
    best_loss = np.inf
    
    if args.resume_densenet:
        checkpoint_dense = torch.load(args.resume_densenet)
        model_dense.load_state_dict(checkpoint_dense['state_dict'])
        epoch_dense = checkpoint_dense['epoch']
        print("Resuming densenet from epoch {}".format(epoch_dense + 1))
        
    if args.resume_resnet:
        checkpoint_res = torch.load(args.resume_resnet)
        model_res.load_state_dict(checkpoint_res['state_dict'])
        epoch_res = checkpoint_res['epoch']
        print("Resuming resnet from epoch {}".format(epoch_res + 1))
        
    if args.resume_fusion:
        checkpoint_fusion = torch.load(args.resume_fusion)
        model_fusion.load_state_dict(checkpoint_fusion['state_dict'])
        epoch_fusion = checkpoint_fusion['epoch']
        print("Resuming fusion from epoch {}".format(epoch_fusion + 1))
        
    if use_gpu:
        model_dense = nn.DataParallel(model_dense).cuda()
        model_res = nn.DataParallel(model_res).cuda()
        model_fusion = nn.DataParallel(model_fusion).cuda()
        
    if args.evaluate:
        print("Evaluate only")
        if args.step == 1:
            valid('step1', model_dense, model_res, model_fusion, valid_loader, criterion, args.print_freq, classes, cfg, data_transforms['valid'])
        elif args.step == 2:
            valid('step2', model_dense, model_res, model_fusion, valid_loader, criterion, args.print_freq, classes, cfg, data_transforms['valid'])
        elif args.step == 3:
            valid('step3', model_dense, model_res, model_fusion, valid_loader, criterion, args.print_freq, classes, cfg, data_transforms['valid'])
        else:
            print('args.step not found')
        return
    
    if args.step == 1:
        #################################### DENSENET BRANCH INIT ##########################################
        start_time = time.time()
        train_time = 0
        best_epoch = 0
        print("==> Start training of densenet branch")

        for p in model_dense.parameters():
            p.requires_grad = True

        for p in model_res.parameters():
            p.requires_grad = False

        for p in model_fusion.parameters():
            p.requires_grad = True

        for epoch in range(start_epoch, args.max_epoch):
            start_train_time = time.time()
            train('step1', model_dense, model_res, model_fusion, train_loader, optimizer_dense, optimizer_res, optimizer_fusion, criterion, args.print_freq, epoch, args.max_epoch, cfg, data_transforms['train'])
            train_time += round(time.time() - start_train_time)
            if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
                print("==> Validation")
                loss_val = valid('step1', model_dense, model_res, model_fusion, valid_loader, criterion, args.print_freq, classes, cfg, data_transforms['valid'])

                if args.stepsize > 0: 
                    if args.scheduler == 'ReduceLROnPlateau':
                        scheduler_dense.step(loss_val)
                        scheduler_fusion.step(loss_val)
                    else:
                        scheduler_dense.step()
                        scheduler_fusion.step()

                is_best = loss_val < best_loss
                if is_best:
                    best_loss = loss_val
                    best_epoch = epoch + 1

                if use_gpu:
                    state_dict_dense = model_dense.module.state_dict()
                    state_dict_fusion = model_fusion.module.state_dict()
                else:
                    state_dict_dense = model_dense.state_dict()
                    state_dict_fusion = model_fusion.state_dict()

                save_checkpoint({
                    'state_dict': state_dict_dense,
                    'loss': best_loss,
                    'epoch': epoch,
                }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar', 'dense')
                save_checkpoint({
                    'state_dict': state_dict_fusion,
                    'loss': best_loss,
                    'epoch': epoch,
                }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar', 'fusion')



        print("==> Best Validation Loss {:.4%}, achieved at epoch {}".format(best_loss, best_epoch))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        train_time = str(datetime.timedelta(seconds=train_time))
        print("Dense branch finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        #################################### DENSENET BRANCH END ##########################################
    
    elif args.step == 2:
        #################################### RESNET BRANCH INIT ##########################################
        start_time = time.time()
        train_time = 0
        best_epoch = 0
        print("==> Start training of local branch")

        for p in model_dense.parameters():
            p.requires_grad = False

        for p in model_res.parameters():
            p.requires_grad = True

        for p in model_fusion.parameters():
            p.requires_grad = True

        for epoch in range(start_epoch, args.max_epoch):
            start_train_time = time.time()
            train('step2', model_dense, model_res, model_fusion, train_loader, optimizer_dense, optimizer_res, optimizer_fusion, criterion, args.print_freq, epoch, args.max_epoch, cfg, data_transforms['train'])
            train_time += round(time.time() - start_train_time)
            if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
                print("==> Validation")
                loss_val = valid('step2', model_dense, model_res, model_fusion, valid_loader, criterion, args.print_freq, classes, cfg, data_transforms['valid'])

                if args.stepsize > 0: 
                    if args.scheduler == 'ReduceLROnPlateau':
                        scheduler_res.step(loss_val)
                        scheduler_fusion.step(loss_val)
                    else:
                        scheduler_res.step()
                        scheduler_fusion.step()

                is_best = loss_val < best_loss
                if is_best:
                    best_loss = loss_val
                    best_epoch = epoch + 1

                if use_gpu:
                    state_dict_res = model_res.module.state_dict()
                    state_dict_fusion = model_fusion.module.state_dict()
                else:
                    state_dict_res = model_res.state_dict()
                    state_dict_fusion = model_fusion.state_dict()

                save_checkpoint({
                    'state_dict': state_dict_res,
                    'loss': best_loss,
                    'epoch': epoch,
                }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar', 'res')
                save_checkpoint({
                    'state_dict': state_dict_fusion,
                    'loss': best_loss,
                    'epoch': epoch,
                }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar', 'fusion')



        print("==> Best Validation Loss {:.4%}, achieved at epoch {}".format(best_loss, best_epoch))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        train_time = str(datetime.timedelta(seconds=train_time))
        print("Resnet branch finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        #################################### RESNET BRANCH END ##########################################
    
    elif args.step == 3:
        #################################### FUSION BRANCH INIT ##########################################
        start_time = time.time()
        train_time = 0
        best_epoch = 0
        print("==> Start training of fusion branch")

        for p in model_dense.parameters():
            p.requires_grad = True

        for p in model_res.parameters():
            p.requires_grad = True

        for p in model_fusion.parameters():
            p.requires_grad = True

        for epoch in range(start_epoch, args.max_epoch):
            start_train_time = time.time()
            train('step3', model_dense, model_res, model_fusion, train_loader, optimizer_dense, optimizer_res, optimizer_fusion, criterion, args.print_freq, epoch, args.max_epoch, cfg, data_transforms['train'])
            train_time += round(time.time() - start_train_time)
            if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
                print("==> Validation")
                loss_val = valid('step3', model_dense, model_res, model_fusion, valid_loader, criterion, args.print_freq, classes, cfg, data_transforms['valid'])

                if args.stepsize > 0: 
                    if args.scheduler == 'ReduceLROnPlateau':
                        scheduler_dense.step(loss_val)
                        scheduler_res.step(loss_val)
                        scheduler_fusion.step(loss_val)
                    else:
                        scheduler_dense.step()
                        scheduler_res.step()
                        scheduler_fusion.step()

                is_best = loss_val < best_loss
                if is_best:
                    best_loss = loss_val
                    best_epoch = epoch + 1

                if use_gpu:
                    state_dict_dense = model_dense.module.state_dict()
                    state_dict_res = model_res.module.state_dict()
                    state_dict_fusion = model_fusion.module.state_dict()
                else:
                    state_dict_dense = model_dense.state_dict()
                    state_dict_res = model_res.state_dict()
                    state_dict_fusion = model_fusion.state_dict()

                save_checkpoint({
                    'state_dict': state_dict_dense,
                    'loss': best_loss,
                    'epoch': epoch,
                }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar', 'dense')
                save_checkpoint({
                    'state_dict': state_dict_res,
                    'loss': best_loss,
                    'epoch': epoch,
                }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar', 'res')
                save_checkpoint({
                    'state_dict': state_dict_fusion,
                    'loss': best_loss,
                    'epoch': epoch,
                }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar', 'fusion')



        print("==> Best Validation Loss {:.4%}, achieved at epoch {}".format(best_loss, best_epoch))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        train_time = str(datetime.timedelta(seconds=train_time))
        print("Fusion branch finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        #################################### FUSION BRANCH END ##########################################
    
    else:
        print('args.step not found')
    
if __name__ == '__main__':
    main()
