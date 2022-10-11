import argparse
import os
import numpy as np
import json
import copy
import argparse

# read files
def get_labels(path):
    with open(path, "r") as f:
        features = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                features.append(json.loads(line))
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="The directory of the training file.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="The directory of the sampled files.")
    args = parser.parse_args()
    k_seed = [(8,1), (8,2), (8,3), (8,4), (8,5)]
    dataset = get_labels(os.path.join(args.input_dir, 'train.json'))
    os.makedirs(args.output_dir, exist_ok=True)
    for (k,seed) in k_seed:
        np.random.seed(seed)
        np.random.shuffle(dataset)
        label_list = {}
        for line in dataset:
            label = copy.deepcopy(line['relation'])
            if label not in label_list:
                label_list[label] = [copy.deepcopy(line)]
            else:
                label_list[label].append(copy.deepcopy(line))

#         unlabeled = []
        lessrel = []
        with open(os.path.join(args.output_dir, "train_" + str(k) + "_" + str(seed) + ".json"), "w") as f:
            for label in label_list:
                if len(label_list[label])>=k: # K-shot, remove classes with less than k instances
                    for line in label_list[label][:k]:
                        f.writelines(json.dumps(line, ensure_ascii=False))
                        f.write('\n')
#                     unlabeled.extend(label_list[label][k:])
                else:
                    lessrel.append(label)

#         with open(os.path.join(args.output_dir, "unlabel_" + str(k) + "_" + str(seed) + ".json"), "w") as f:
#             for line in unlabeled:
#                 f.writelines(json.dumps(line, ensure_ascii=False))
#                 f.write('\n')
    
    if len(lessrel)!=0:
        test = get_labels(os.path.join(args.input_dir, "test.json"))
        with open(os.path.join(args.output_dir, "new_test.json"), "w") as f:
            for line in test:
                if line['relation'] not in lessrel:
                    f.writelines(json.dumps(line, ensure_ascii=False))
                    f.write('\n')
        
        rel2id = {}
        idx = 0
        for label in label_list:
            if label not in lessrel:
                rel2id[label] = idx
                idx += 1
        json.dump(rel2id, open(os.path.join(args.output_dir, "new_rel2id.json"), "w"), ensure_ascii=False)





if __name__=="__main__":
    main()
