from argparse import ArgumentParser
import enum
from json import decoder
from lib2to3.pgen2 import token
from logging import debug
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
# import wandb
import numpy as np
# Hide lines above until Lab 5

from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup

from functools import partial

import random

from .dice_loss import DiceLoss, make_one_hot
from .dice_loss_nlp import MultiDSCLoss
from .focal_loss import MultiFocalLoss
from .GHM_loss import GHMC_Loss
from .LDAM_loss import LDAMLoss
from .CB_loss import CBLoss
from torch.nn import BCEWithLogitsLoss
import logging
import os

def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0  # 只优化id为1～8的token
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        self.args = args
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        self.rel2id = rel2id
        if args.labeling=="True":
            self.unlabeleddataset = []
            with open(f"{self.args.data_dir}/label.json","r") as f:
                for line in f.readlines():
                    self.unlabeleddataset.append(json.loads(line))
        
        Na_num = -1
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other" or k=="false" or k=="unanswerable":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        if args.useloss=="nn.CrossEntropyLoss":
            self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        elif args.useloss=="MultiFocalLoss":
            self.loss_fn = MultiFocalLoss(num_class=len(self.rel2id))
        elif args.useloss=="LDAMLoss":
            self.loss_fn = LDAMLoss(datasetpath=self.args.data_dir, max_m=0.5, s=30)
        elif args.useloss=="CBLoss":
            self.loss_fn = CBLoss(datasetpath=self.args.data_dir)
        else:
            self.loss_fn = eval(args.useloss+"()")
        # self.loss_fn = AMSoftmax(self.model.config.hidden_size, num_relation)
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel2id=self.rel2id, rel_num=num_relation, na_num=Na_num, datasetname=self.args.data_dir.split('/')[1])
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.tokenizer = tokenizer
    
        self._init_label_word()
        
        # with torch.no_grad():
        #     self.loss_fn.fc.weight = nn.Parameter(self.model.get_output_embeddings().weight[self.label_st_id:self.label_st_id+num_relation])
            # self.loss_fn.fc.bias = nn.Parameter(self.model.get_output_embeddings().bias[self.label_st_id:self.label_st_id+num_relation])

    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        
        num_labels = len(label_word_idx)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]
            
            # for abaltion study
            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]
            
            if self.args.init_type_words:
                datasetname = self.args.data_dir.split('/')[1]
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                if datasetname == "SciERC":
                    meaning_word = [a[0] for a in self.tokenizer(["generic","material", "method", "metric", "task","term"], add_special_tokens=False)['input_ids']]
                elif datasetname == "DuIE2":
                    meaning_word = [a[0] for a in self.tokenizer(["语言", "电视综艺", "企业", "学科专业", "景点", "人物", "娱乐人物", "国家", "图书作品", "地点", "学校", "机构", "行政区", "企业/品牌", "奖项", "城市", "影视作品", "数字", "音乐专辑", "作品", "文本", "歌曲", "历史人物", "文学作品", "日期", "气候"], add_special_tokens=False)['input_ids']]
                elif datasetname == "dialog":
                    meaning_word = [a[0] for a in self.tokenizer(["geo-political","organization", "person", "string", "value"], add_special_tokens=False)['input_ids']]
                elif datasetname == "CMeIE":
                    meaning_word = [a[0] for a in self.tokenizer(['其他','其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后'], add_special_tokens=False)['input_ids']]
                elif datasetname == "ChemProt":
                    meaning_word = [a[0] for a in self.tokenizer(["chemical","gene"], add_special_tokens=False)['input_ids']]
                else:
                    meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]

            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        
        self.word2label = continous_label_word # a continous list
            
                
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # logging.info("=== Epoch %d train ===" % batch_idx)
        if self.args.stutrain=="False":
            if len(batch)==5:
                input_ids, attention_mask, token_type_ids, labels, so = batch
                result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
                logits = result.logits
            else:
                input_ids, attention_mask, labels, so = batch
                result = self.model(input_ids, attention_mask,return_dict=True, output_hidden_states=True)
                logits = result.logits
        else:
            if len(batch)==6:
                input_ids, attention_mask, token_type_ids, labels, so, guids = batch
                result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
                logits = result.logits
            else:
                input_ids, attention_mask, labels, so, guids = batch
                result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
                logits = result.logits
        if self.args.stutrain=="False":
            output_embedding = result.hidden_states[-1]
            logits = self.pvp(logits, input_ids)
            # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
            # loss = self.get_loss(logits, input_ids, labels)

            # ke_loss = self.ke_loss(output_embedding, labels, so, input_ids)
            # loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss
            if self.args.useloss=="DiceLoss" or \
                self.args.useloss=="BCEWithLogitsLoss" or \
                self.args.useloss=="GHMC_Loss" :
                target = make_one_hot(labels.unsqueeze(1),len(self.rel2id))
                loss = self.loss_fn(logits.cpu(), target)
            elif self.args.useloss=="CB_Loss":
                loss = self.loss_fn(logits, labels)
            else:
                loss = self.loss_fn(logits, labels)

            self.log("Train/loss", loss)
            self.log("Train/ke_loss", loss)
        else:
            output_embedding = result.hidden_states[-1]
            logits = self.pvp(logits, input_ids)
            # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
            # loss = self.get_loss(logits, input_ids, labels)
            # ke_loss = self.ke_loss(output_embedding, labels, so, input_ids)
            goldidx = []
            softidx = []
            for i, x in enumerate(guids):
                if x==0:
                    goldidx.append(i)
                else:
                    softidx.append(i)
            goldlabel = labels[goldidx]
            softlabel = labels[softidx]
            goldlogit = logits[goldidx]
            softlogit = logits[softidx]
            if self.args.useloss=="DiceLoss" or \
               self.args.useloss=="BCEWithLogitsLoss" or \
               self.args.useloss=="GHMC_Loss" :
                goldtarget = make_one_hot(goldlabel.unsqueeze(1),len(self.rel2id))
                softtarget = make_one_hot(softlabel.unsqueeze(1),len(self.rel2id))
                goldloss = self.loss_fn(goldlogit.cpu(),goldtarget)
                softloss = self.loss_fn(softlogit.cpu(),softtarget)
            else:
                goldloss = self.loss_fn(goldlogit,goldlabel)
                softloss = self.loss_fn(softlogit,softlabel)
            loss = goldloss / float(len(goldidx)) + self.args.lambda_u * softloss / float(len(softidx))
            self.log("Train/loss", loss)
            self.log("Train/ke_loss", loss)

        return loss
    
    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss


    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # logging.info("=== Epoch %d val ===" % batch_idx)
        if  len(batch)==5:
            input_ids, attention_mask, token_type_ids, labels, _ = batch
            logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        else:
            input_ids, attention_mask, labels, _ = batch
            logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        if self.args.useloss=="DiceLoss" or \
            self.args.useloss=="BCEWithLogitsLoss" or \
            self.args.useloss=="GHMC_Loss" :
            target = make_one_hot(labels.unsqueeze(1),len(self.rel2id))
            loss = self.loss_fn(logits.cpu(), target)
        else:
            loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        result = self.eval_fn(logits, labels)
        f1 = result['micro_f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        logging.info('Evaluation result: {}.'.format(result))

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if len(batch)==4:
            input_ids, attention_mask, labels, _ = batch
            logits = self.model(input_ids, attention_mask, return_dict=True).logits
        else:
            input_ids, attention_mask, token_type_ids, labels, _ = batch
            logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        if self.args.labeling=="True":
            predictedlabel = np.argmax(logits.cpu(), axis=-1).numpy()
            if batch_idx * self.args.batch_size + self.args.batch_size <= len(self.unlabeleddataset):
                unlabeled = self.unlabeleddataset[batch_idx * self.args.batch_size : batch_idx * self.args.batch_size + self.args.batch_size]
            else:
                unlabeled = self.unlabeleddataset[batch_idx * self.args.batch_size:]
            assert len(unlabeled)==len(predictedlabel)
            id2rel = {}
            for k,v in self.rel2id.items():
                id2rel[v] = k
            for i in range(len(unlabeled)):
                unlabeled[i]['relation'] = id2rel[predictedlabel[i]]
                unlabeled[i]['guid'] = 'soft'
            if batch_idx==0:
                with open(f"{self.args.data_dir}/label2.json","w") as f:
                    for data in unlabeled:
                        f.writelines(json.dumps(data,ensure_ascii=False))
                        f.write('\n')
            else:
                with open(f"{self.args.data_dir}/label2.json","a") as f:
                    for data in unlabeled:
                        f.writelines(json.dumps(data,ensure_ascii=False))
                        f.write('\n')
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
#         if self.args.labeling=="False":
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])
        result = self.eval_fn(logits, labels)
        logging.info('Test set results:')
        logging.info('Dataset: ' + self.args.data_dir)
        logging.info('Loss: ' + self.args.useloss)
        logging.info('Accuracy: {}'.format(result['acc']))
        logging.info('Micro precision: {}'.format(result['micro_p']))
        logging.info('Micro recall: {}'.format(result['micro_r']))
        logging.info('Micro F1: {}'.format(result['micro_f1']))
        logging.info('Macro precision: {}'.format(result['macro_p']))
        logging.info('Macro recall: {}'.format(result['macro_r']))
        logging.info('Macro F1: {}'.format(result['macro_f1']))
        logging.info("Predicted Labels: {}".format(result['pred_labels']))
        logging.info('F1 per Relation: {}'.format(result['f1_per_relation']))
        if self.args.data_dir.split('/')[1] in ['semeval','tacrev', 'wiki80','SciERC']:
            logging.info('Micro F1 of Few-level Relations: {}'.format(result['few_mif1']))
            logging.info('Macro F1 of Few-level Relations: {}'.format(result['few_maf1']))
            logging.info('Micro F1 of Medium-level Relations: {}'.format(result['med_mif1']))
            logging.info('Macro F1 of Medium-level Relations: {}'.format(result['med_maf1']))
            logging.info('Micro F1 of Many-level Relations: {}'.format(result['many_mif1']))
            logging.info('Macro F1 of Many-level Relations: {}'.format(result['many_maf1']))
        logging.info('Classification Report: {}'.format(result['report']))
            # f1 = result['micro_f1']
            # self.log("Test/f1", f1)
        


    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.00, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
        
    def ke_loss(self, logits, labels, so, input_ids):
        subject_embedding = []
        object_embedding = []
        neg_subject_embedding = []
        neg_object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))

            # random select the neg samples
            st_sub = random.randint(1, logits[i].shape[0] - 6)
            span_sub = random.randint(1, 5)
            st_obj = random.randint(1, logits[i].shape[0] - 6)
            span_obj = random.randint(1, 5)
            neg_subject_embedding.append(torch.mean(logits[i, st_sub:st_sub+span_sub], dim=0))
            neg_object_embedding.append(torch.mean(logits[i, st_obj:st_obj+span_obj], dim=0))
            
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        neg_subject_embedding = torch.stack(neg_subject_embedding)
        neg_object_embedding = torch.stack(neg_object_embedding)
        # trick , the relation ids is concated, 


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = logits[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        
        d_1 = torch.norm(subject_embedding + mask_relation_embedding - object_embedding, p=2) / bsz
        d_2 = torch.norm(neg_subject_embedding + real_relation_embedding - neg_object_embedding, p=2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1.*f(self.args.t_gamma - d_1) - f(d_2 - self.args.t_gamma)
        
        return loss

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

class TransformerLitModelTwoSteps(BertLitModel):
    def configure_optimizers(self):
        no_decay_param = ["bais", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.args.lr_2, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
