import argparse
import os
import numpy as np
import json
import copy
import argparse

# read train_file
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
    parser.add_argument("--input_file", "-i", type=str, required=True,
                        help="The path of the training file.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="The directory of the sampled files.")
    args = parser.parse_args()
    k_seed = [(8,1),(8,2),(8,3),(8,4),(8,5)]
    dataset = get_labels(args.input_file)
    os.makedirs(args.output_dir,exist_ok=True)
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

        # unlabeled = []
        with open(os.path.join(args.output_dir, "train_"+str(k)+"_"+str(seed)+".json"), "w") as f:
            for label in label_list:
                if len(label_list[label])>=k:
                    for line in label_list[label][:k]:  # K-shot
                        f.writelines(json.dumps(line,ensure_ascii=False))
                        f.write('\n')
                    # unlabeled.extend(label_list[label][k:])
        # with open(os.path.join(args.output_dir, "unlabel_"+str(k)+"_"+str(seed)+".json"), "w") as f:
        #     for line in unlabeled:
        #         f.writelines(json.dumps(line,ensure_ascii=False))
        #         f.write("\n")

if __name__=="__main__":
    main()