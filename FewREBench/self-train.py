import json
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gold_file", "-g", type=str, required=True,
                    help="The path of the gold-labeled dataset.")
parser.add_argument("--pseudo_file",'-p', type=str, required=True,
                    help="The path of the pseudo-labeled dataset.")
parser.add_argument("--output_dir", "-o", type=str, required=True,
                    help="The directory of the combined labeled dataset.")
args = parser.parse_args()
os.makedirs(args.output_dir,exist_ok=True)
label = []
gold = []
dataset = []

with open(args.pseudo_file,'r') as f:
    for data in f.readlines():
        label.append(json.loads(data))
with open(args.gold_file,'r') as f:
    for data in f.readlines():
        gold.append(json.loads(data))
for data in gold:
    data['guid'] = 'gold'
dataset = label+gold

# print(len(label),len(gold),len(dataset))

np.random.seed(42)
np.random.shuffle(dataset)
with open(os.path.join(args.output_dir, 'train.json'),'w') as f:
    for data in dataset:
        f.writelines(json.dumps(data,ensure_ascii=False))
        f.write('\n')