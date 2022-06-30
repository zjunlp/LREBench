# coding:utf-8
from cmath import log
from curses import A_REVERSE
from re import L
import sys
sys.path.append('../')
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import os
import argparse
import logging
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='roberta-large', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--pooler', default='cls', choices=['cls', 'entity'], 
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true', 
        help='Mask entity mentions')
parser.add_argument('--labeling', default="False", choices=['True','False'], 
        help='Generate soft labels')
parser.add_argument("--stutrain",default="False",choices=['True','False'],
        help='Teacher training or student training')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc', 'macro_f1'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='semeval', required=True,
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
# parser.add_argument('--val_file', default='', type=str,
#         help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data  file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=10, type=int,
        help='Max number of training epochs')
parser.add_argument('--lambda_u',default=0.2,type=float,
        help="weight of soft loss")
parser.add_argument('--use_loss',default="nn.CrossEntropyLoss",type=str, 
        choices=["nn.CrossEntropyLoss","DiceLoss","MultiDSCLoss","MultiFocalLoss","GHMC_Loss","LDAMLoss","CBLoss"],
        help='Loss function')

# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path.split('/')[-1], args.pooler)
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)


opennre.download(args.dataset, root_path=root_path)
args.train_file = os.path.join(root_path, 'benchmark', args.dataset, 'train.json')
if args.labeling=="True":
        args.test_file = os.path.join(root_path, 'benchmark', args.dataset, 'label.json')
else:
        args.test_file = os.path.join(root_path, 'benchmark', args.dataset, 'test.json')
args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel2id.json')


logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))


sentence_encoder = opennre.encoder.BERTEncoder(
max_length=args.max_length, 
pretrain_path=args.pretrain_path,
mask_entity=args.mask_entity
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt='adamw',
    stutrain = args.stutrain,
    lambda_u=args.lambda_u,
    use_loss=args.use_loss
)

# Train the model
if args.labeling=="False" and not args.only_test:
    framework.train_model(args.metric)

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
if args.labeling=="False":
        result = framework.eval_model(framework.test_loader)
        # Print the result
        logging.info('Test set results:')
        logging.info('Accuracy: {}'.format(result['acc']))
        logging.info('Micro precision: {}'.format(result['micro_p']))
        logging.info('Micro recall: {}'.format(result['micro_r']))
        logging.info('Micro F1: {}'.format(result['micro_f1']))
        logging.info('Macro precision: {}'.format(result['macro_p']))
        logging.info('Macro recall: {}'.format(result['macro_r']))
        logging.info('Macro F1: {}'.format(result['macro_f1']))
        # logging.info("Predicted Labels: {}".format(result['pred_labels']))
        logging.info('F1 per Relation: {}'.format(result['f1_per_relation']))
        logging.info('Classification Report: {}'.format(result['report']))
else:
    result = framework.test_model(framework.test_loader)