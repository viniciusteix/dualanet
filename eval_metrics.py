from __future__ import print_function, absolute_import

import numpy as np
import sys
import os.path as osp
import os
from sklearn import metrics

def compute_roc_auc(data_gt, data_pd, num_classes):
    roc_auc = []
    for i in range(num_classes):
        roc_auc.append(metrics.roc_auc_score(data_gt[i], data_pd[i]))
    roc_auc = np.array(roc_auc)
    return roc_auc, np.mean(roc_auc)