"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        """Compute the focal loss between `logits` and the ground truth `labels`.

        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

        Args:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.

        Returns:
        focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = self.alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss


class CBLoss(nn.Module):
    def __init__(self,datasetpath,loss_type="focal", beta=0.9999, gamma=2.0):
        super(CBLoss,self).__init__()
        datasetpath = '/'.join(datasetpath.split('/')[:-1])
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.rel2id = json.load(open(os.path.join(datasetpath,"rel2id.json"),"r"))
        self.no_of_classes = len(self.rel2id)
        self.samples_per_cls = []
        num_per_cls_dict = {}
        with open(os.path.join(datasetpath,"train.json"),'r') as f:
            for line in f.readlines():
                data = eval(line)
                if data['relation'] not in num_per_cls_dict:
                    num_per_cls_dict[data['relation']] = 1
                else:
                    num_per_cls_dict[data['relation']] += 1
        for value in num_per_cls_dict.values():
            self.samples_per_cls.append(value)
        if len(self.rel2id)>len(self.samples_per_cls):
            diff = len(self.rel2id)-len(self.samples_per_cls)
            for _ in range(diff):
                self.samples_per_cls.append(0)

    def forward(self, logits, labels):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(weights).float().cuda(0)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.no_of_classes)

        if self.loss_type == "focal":
            loss_fn = FocalLoss(weights, self.gamma)
            cb_loss = loss_fn(logits, labels_one_hot)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss






# def CB_loss(logits, labels, samples_per_cls, no_of_classes, loss_type="focal", beta=0.9999, gamma=2.0):
#     """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

#     Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
#     where Loss is one of the standard losses used for Neural Networks.

#     Args:
#       labels: A int tensor of size [batch].
#       logits: A float tensor of size [batch, no_of_classes].
#       samples_per_cls: A python list of size [no_of_classes].
#       no_of_classes: total number of classes. int
#       loss_type: string. One of "sigmoid", "focal", "softmax".
#       beta: float. Hyperparameter for Class balanced loss.
#       gamma: float. Hyperparameter for Focal loss.

#     Returns:
#       cb_loss: A float tensor representing class balanced loss
#     """
#     effective_num = 1.0 - np.power(beta, samples_per_cls)
#     weights = (1.0 - beta) / np.array(effective_num)
#     weights = weights / np.sum(weights) * no_of_classes

#     labels_one_hot = F.one_hot(labels, no_of_classes).float()

#     weights = torch.tensor(weights).float()
#     weights = weights.unsqueeze(0)
#     weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
#     weights = weights.sum(1)
#     weights = weights.unsqueeze(1)
#     weights = weights.repeat(1,no_of_classes)

#     if loss_type == "focal":
#         cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
#     elif loss_type == "sigmoid":
#         cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
#     elif loss_type == "softmax":
#         pred = logits.softmax(dim = 1)
#         cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
#     return cb_loss



if __name__ == '__main__':
    no_of_classes = 19
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "focal"
    loss_fn = CBLoss(datasetpath="/data/xxu/OpenNRE/benchmark/semeval")
    cb_loss = loss_fn(logits, labels)
    print(cb_loss)