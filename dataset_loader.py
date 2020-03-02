from __future__ import print_function, absolute_import

import os
from PIL import Image
import numpy as np
import os.path as osp
import cv2

import torch
from torch.utils.data import Dataset

class DatasetGenerator (Dataset):
    def __init__ (self, path_base, dataset_file, transform, dataset_, no_fiding):
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        fileDescriptor = open(osp.join(path_base, dataset_, dataset_file + '.csv'), "r")
        line = True        
        while line:
                
            line = fileDescriptor.readline()
            if line:
          
                lineItems = line.split()
                
                if dataset_ == 'ChestXray-NIHCC':
                    imagePath = osp.join(path_base, dataset_, 'images', lineItems[0])
                elif dataset_ == 'CheXpert-v1.0-small':
                    imagePath = osp.join(path_base, lineItems[0])
                    
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                imageLabel = np.array(imageLabel)
                imageLabel[imageLabel == 2] = 0
                imageLabel = list(imageLabel)
                
                if no_fiding:
                    if np.sum(imageLabel) == 0:
                        imageLabel.append(1)
                    else:
                        imageLabel.append(0)
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
#         imageLabel= torch.FloatTensor(self.listImageLabels[index])
        imageLabel = np.array(self.listImageLabels[index]).astype(np.float32)
        
        if self.transform != None: imageData = self.transform(imageData)
        
#         image = cv2.imread(self.listImagePaths[index], 0)
#         image = Image.fromarray(image)
#         image = np.array(image)
#         if self.transform != None: image = self.transform(image)
            
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
#         image = image.transpose((2, 0, 1))
    
        return imageData, imageLabel
    
    def __len__(self):
        return len(self.listImagePaths)