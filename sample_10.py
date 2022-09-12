import json
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="The path of the training file.")
parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help="The directory of the sampled files.")
args = parser.parse_args()


seeds = [1, 2, 3, 4, 5]
with open(os.path.join(args.input_dir, "train.json"),'r') as f:
    lines = f.readlines()

dataset = []
for line in lines:
    dataset.append(json.loads(line))

num = int(float(len(dataset)) * 0.1)

for i, seed in enumerate(seeds):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,'train10per_'+str(seed)+'.json'),'w') as f:
        for data in dataset[:num+1]:
            f.writelines(json.dumps(data,ensure_ascii=False))
            f.write("\n")
    
    with open(os.path.join(args.output_dir, 'unlabel10per_'+str(seed)+'.json'),'w') as f:
        for line in dataset[num+1:]:
            f.writelines(json.dumps(line,ensure_ascii=False))
            f.write("\n")

