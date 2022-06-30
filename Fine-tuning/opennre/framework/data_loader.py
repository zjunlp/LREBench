import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from typing import Tuple, List
import copy
import json

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer, stutrain, iftrain, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        self.stutrain = stutrain
        self.iftrain = iftrain

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(json.loads(line))
        f.close()
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data), len(self.rel2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        if self.stutrain=="True" and self.iftrain:
            res = [[self.rel2id[item['relation']],item['guid']]] + seq
        else:
            res = [self.rel2id[item['relation']]] + seq
        return res
        # return [self.rel2id[item['relation']]] + seq # label, seq1, seq2, ...
    
    def collate_fn(data):
        data = list(zip(*data))
        if isinstance(data[0][0],List):
            labels = []
            guids = []
            for da in data[0]:
                labels.append(da[0])
                guids.append(da[1])
            labels = tuple(labels)
            guids = tuple(guids)
            seqs = data[1:]
            batch_labels = torch.tensor(labels).long() # (B)
            # batch_guids = torch.tensor(guids).long()
            batch_seqs = []
            for seq in seqs:
                batch_seqs.append(torch.cat(seq, 0)) # (B, L)
            ans = [batch_labels] + [guids] + batch_seqs
        else:
            labels = data[0]
            seqs = data[1:]
            batch_labels = torch.tensor(labels).long() # (B)
            batch_seqs = []
            for seq in seqs:
                batch_seqs.append(torch.cat(seq, 0)) # (B, L)
            ans = [batch_labels] + batch_seqs
        return ans
    
    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others','false',"unanswerable"]:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]
            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive +=1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        # try:
        #     micro_p = float(correct_positive) / float(pred_positive)
        # except:
        #     micro_p = 0
        # try:
        #     micro_r = float(correct_positive) / float(gold_positive)
        # except:
        #     micro_r = 0
        # try:
        #     micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        # except:
        #     micro_f1 = 0
        
        # f1 per relation
        # rel2id2 = sorted(self.rel2id.items(), key = lambda kv:(kv[1], kv[0]))
        
        y_true = []
        for i in range(total):
            y_true.append(self.rel2id[self.data[i]['relation']])
        y_pred = pred_result
        # target_names = []
        # for (k,v) in rel2id2:
        #     target_names.append(k)
        pred_labels = list(set(y_true))
        needlabels = []
        if neg in y_true:
            needlabels = copy.deepcopy(pred_labels)
            needlabels.remove(neg)
        else:
            needlabels = copy.deepcopy(pred_labels)
        needlabels.sort()
        alllabels = list(self.rel2id.values()).sort()
        micro_pre = precision_score(y_true, y_pred, labels=needlabels, average='micro')
        micro_recall = recall_score(y_true, y_pred, labels=needlabels, average='micro')
        micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=needlabels, average='micro', zero_division=0)
        # micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=alllabels, average='micro', zero_division=0)
        macro_pre = precision_score(y_true, y_pred, labels=needlabels, average='macro')
        macro_recall = recall_score(y_true, y_pred, labels=needlabels, average='macro')
        macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=needlabels, average='macro', zero_division=0)
        f1_per_relation = list(f1_score(y_true=y_true, y_pred=y_pred, labels=alllabels, average=None, zero_division=0))
        report = classification_report(y_true, y_pred)
        
        result = {'acc': acc, \
                  'micro_p': micro_pre, \
                  'micro_r': micro_recall, \
                  'micro_f1': micro_f1, \
                  'macro_p': macro_pre, \
                  'macro_r': macro_recall, \
                  'macro_f1': macro_f1, \
                  'f1_per_relation': f1_per_relation, \
                  'report': '\n'+ report, \
                  'pred_labels': pred_labels}
 
      
        logging.info('Evaluation result: {}.'.format(result))
        return result
    
def SentenceRELoader(path, rel2id, tokenizer, batch_size, shuffle, stutrain, iftrain, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = SentenceREDataset(path = path, rel2id = rel2id, tokenizer = tokenizer, stutrain=stutrain, iftrain=iftrain, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader
