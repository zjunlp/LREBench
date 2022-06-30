import math
from types import prepare_class
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, datasetpath, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        num_per_cls_dict = {}
        with open(datasetpath,'r') as f:
            for line in f.readlines():
                data = eval(line)
                if data['relation'] not in num_per_cls_dict:
                    num_per_cls_dict[data['relation']] = 1
                else:
                    num_per_cls_dict[data['relation']] += 1
        self.cls_num_list = []
        for value in num_per_cls_dict.values():
            self.cls_num_list.append(value)
        prefix = '/'.join(datasetpath.split('/')[:-1])
        relpath = os.path.join(prefix,"rel2id.json")
        rel2id = json.load(open(relpath,"r"))
        if len(rel2id)>len(self.cls_num_list):
            diff = len(rel2id)-len(self.cls_num_list)
            for _ in range(diff):
                self.cls_num_list.append(0)
        m_list = 1.0 / np.sqrt(np.sqrt(self.cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        train_sampler = None
        idx = 0
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(0)
        self.weight = per_cls_weights
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)