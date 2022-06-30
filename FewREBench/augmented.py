import json
import numpy as np
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Input file containing the augmented dataset')
parser.add_argument("--output_dir", "-o", type=str, required=True,
                    help="The directory of the augmented files.")
args = parser.parse_args()
os.makedirs(args.output_dir,exist_ok=True)

origin = []
DA = []
with open(args.input_file,'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        if 'aug' in line:
            DA.append(line)
        else:
            origin.append(line)
np.random.seed(42)
DA30 = random.sample(DA,int(len(origin)*0.3)) + origin
DA100 = random.sample(DA,len(origin)) + origin
print(len(origin))
print(len(DA30))
print(len(DA100))
np.random.shuffle(DA30)
np.random.shuffle(DA100)
with open(os.path.join(args.output_dir, "aug_30.json"),'w') as f:
    for line in DA30:
        f.writelines(json.dumps(line))
        f.write('\n')
with open(os.path.join(args.output_dir, "aug_100.json"),'w') as f:
    for line in DA100:
        f.writelines(json.dumps(line))
        f.write('\n')