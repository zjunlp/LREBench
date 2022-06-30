from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json
import argparse
import logging

def split_label_words(tokenizer, label_list):
    label_word_list = []
    for label in label_list:
        if label == 'no_relation' or label == "NA" or label=="Other" or label=="false" or label=="unanswerable":
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            tmps = label
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("-"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    # print(padded_label_word_list[0])
    return padded_label_word_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', '-m', type=str, required=True,
                        choices=["roberta-large", "hfl/chinese-roberta-wwm-ext-large", "dmis-lab/biobert-large-cased-v1.1"],
                        help='the path of the pretrained model or the model name from HuggingFace')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=["semeval","tacrev","wiki80","SciERC","ChemProt","dialog","DuIE2","CMeIE"],
                        help='the dataset name, the same as the dataset path name')
    
    args = parser.parse_args()
    logging.info(str(args))

    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
    if args.dataset == "SciERC":
        label_list = ["term's hyponym is term",
                      "metric is evaluated for object",
                      "method is compared with method",
                      "method is used for task",
                      "term is feature of term",
                      "term is incorporated with term",
                      "term is part of system"]
    else:
        with open(f"dataset/{args.dataset}/rel2id.json", "r") as file:
            t = json.load(file)
            label_list = list(t)

    t = split_label_words(tokenizer, label_list)

    # with open(f"./dataset/biobert-large-cased-v1.1_ChemProt.pt", "wb") as file:
    #     torch.save(t, file)

    with open(f"./dataset/{args.modelpath.split('/')[-1]}_{args.dataset}.pt", "wb") as file:
        torch.save(t, file)