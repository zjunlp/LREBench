import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
from .dice_loss import DiceLoss, make_one_hot
from .dice_loss_nlp import MultiDSCLoss
from .focal_loss import MultiFocalLoss
from .GHM_loss import GHMC_Loss
from .LDAM_loss import LDAMLoss
from .CB_loss import CBLoss
from torch.nn import BCEWithLogitsLoss
import json
import numpy as np

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd',
                 stutrain="false",
                 lambda_u=0.5,
                 use_loss = "nn.CrossEntropyLoss"):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.rel2id = model.rel2id
        self.test_path = test_path
        self.stutrain = stutrain
        self.lambda_u = lambda_u
        self.use_loss = use_loss
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                stutrain=self.stutrain,
                iftrain=True)        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                stutrain=self.stutrain,
                iftrain=False)
        # Model
        self.model = model
        self.parallel_model = self.model
        # self.parallel_model = nn.DataParallel(self.model)
       
        # Criterion
        if use_loss=="MultiFocalLoss":
            self.criterion = eval(use_loss+"(num_class="+str(len(self.rel2id))+")")
        elif use_loss=="LDAMLoss":
            self.criterion = LDAMLoss(datasetpath=train_path, max_m=0.5, s=30)
        elif use_loss=="CBLoss":
            self.criterion = CBLoss(datasetpath=train_path)
        else:
            self.criterion = eval(use_loss+"()")
        # self.criterion = MultiDSCLoss()
        
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                if self.stutrain=="False":
                    label = data[0]
                    args = data[1:]
                    logits = self.parallel_model(*args)
                    if self.use_loss=="DiceLoss" or \
                       self.use_loss=="BCEWithLogitsLoss" or \
                       self.use_loss=="GHMC_Loss" :
                        target = make_one_hot(label.unsqueeze(1),len(self.rel2id))
                        loss = self.criterion(logits.cpu(), target)
                    else:
                        loss = self.criterion(logits, label)
                    score, pred = logits.max(-1) # (B)
                    acc = float((pred == label).long().sum()) / label.size(0)
                    # Log
                    avg_loss.update(loss.item(), 1)
                    avg_acc.update(acc, 1)
                    t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                    # Optimize
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                else:
                    label = data[0]
                    guids = data[1]
                    goldidx = []
                    softidx = []
                    for i,x in enumerate(guids):
                        if x=="soft":
                            softidx.append(i)
                        elif x=="gold":
                            goldidx.append(i)
                    assert len(softidx)+len(goldidx) == len(label)
                    args = data[2:]
                    logits = self.parallel_model(*args)
                    goldlogits = logits[goldidx]
                    softlogits = logits[softidx]
                    goldlabels = label[goldidx]
                    softlabels = label[softidx]
                    if self.use_loss=="DiceLoss" or \
                       self.use_loss=="BCEWithLogitsLoss" or \
                       self.use_loss=="GHMC_Loss" :
                        goldtarget = make_one_hot(goldlabels.unsqueeze(1),len(self.rel2id))
                        softtarget = make_one_hot(softlabels.unsqueeze(1),len(self.rel2id))
                        goldloss = self.criterion(goldlogits.cpu(), goldtarget)
                        softloss = self.criterion(softlogits.cpu(), softtarget)
                    else:
                        goldloss = self.criterion(goldlogits, goldlabels)
                        softloss = self.criterion(softlogits, softlabels)
                    # if len(goldidx)==0:
                    #     loss = self.lambda_u * softloss / float(len(softidx))
                    # else:
                    loss = goldloss / float(len(goldidx)) + \
                               self.lambda_u * softloss / float(len(softidx))
                    score, pred = logits.max(-1) # (B)
                    acc = float((pred == label).long().sum()) / label.size(0)
                    # Log
                    avg_loss.update(loss.item(), 1)
                    avg_acc.update(acc, 1)
                    t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                    # Optimize
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1



            # Val 
            if epoch==self.max_epoch-1:
                # logging.info("=== Epoch %d val ===" % epoch)
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
            # logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            # if result[metric] > best_metric:
            #     if result[metric] - best_metric < 0.0001: continue
            #     logging.info("Best ckpt and saved.")
            #     folder_path = '/'.join(self.ckpt.split('/')[:-1])
            #     if not os.path.exists(folder_path):
            #         os.mkdir(folder_path)
            #     torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
            #     best_metric = result[metric]
        # logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result
    
    def test_model(self, test_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(test_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        id2rel = {}
        for k,v in self.rel2id.items():
            id2rel[v] = k
        with open(self.test_path,'r') as f:
            lines = f.readlines()
        testdata = []
        for line in lines:
            testdata.append(json.loads(line))
        assert len(testdata) == len(pred_result)
        for i in range(len(testdata)):
            testdata[i]['relation'] = id2rel[pred_result[i]]
            testdata[i]['guid'] = 'soft'
        with open(self.test_path,'w') as f:
            for data in testdata:
                f.writelines(json.dumps(data))
                f.write('\n')
        # result = eval_loader.dataset.eval(pred_result)
        # return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
