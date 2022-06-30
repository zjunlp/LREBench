import os
import json
from collections import deque
from tqdm import tqdm
import argparse
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action


def read_dataset(fname):
    dataset = []
    with open(fname, 'r') as f:
        for line in tqdm(f.readlines(), desc='reading dataset'):
            dataset.append(json.loads(line))
    return dataset

def merge(sent_dict):
    # Merge sentence parts in order
    sent_order = ['sent1', 'ent1', 'sent2', 'ent2', 'sent3']
    q = deque([[sent] for sent in sent_dict[sent_order[0]]])
    curr_idx = 1
    while curr_idx < len(sent_order):
        curr_len = len(q)
        for _ in range(curr_len):
            prev_sents = q.pop()
            for postfix in sent_dict[sent_order[curr_idx]]:
                q.appendleft(prev_sents + [postfix])
        curr_idx += 1
    return list(q)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True,
                        help='Input file containing dataset')
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="The directory of the sampled files.")
    parser.add_argument('--DAmethod', '-d', type=str, required=True,
                        choices=["word2vec","TF-IDF","word_embedding_bert","word_embedding_distilbert","word_embedding_roberta","random_swap","synonym"],
                        help='Data augmentation method')
    parser.add_argument("--model_name", "-mn", type=str, default="roberta-large",
                        help="model from huggingface")
    # parser.add_argument('--model_dir','-m', type=str, required=True,
    #                     help="the path of pretrained models used in DA methods")
    parser.add_argument('--locations', '-l', nargs='+',
                        choices=['sent1', 'sent2', 'sent3', 'ent1', 'ent2'],
                        default=['sent1', 'sent2', 'sent3', 'ent1', 'ent2'],
                        help='List of positions that you want to manipulate')
    args = parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    
    DAmethods = {
        "TF-IDF": '''naw.TfIdfAug(
                    model_path=args.model_dir,
                    action="substitute"
                    )''',
        "word_embedding_bert": '''naw.ContextualWordEmbsAug(
                            model_path=args.model_name, 
                            action="substitute",device='cuda')''',
        "word_embedding_roberta": '''naw.ContextualWordEmbsAug(
                            model_path="roberta-base",
                            action="substitute",device='cuda')''',
        "synonym": '''naw.SynonymAug(aug_src='wordnet')'''
    }
    origin_data = read_dataset(args.input_file)
    DA_data = []
    replaced_samples = []
    perturb_func = eval(DAmethods[args.DAmethod])
    for i in range(1,6):
        for example in tqdm(origin_data,desc="Augment the dataset"):
            tokens = example['token']
            relation = example['relation']
            head_pos, tail_pos = example['h']['pos'], example['t']['pos']
            rev = head_pos[0] > tail_pos[0]
            # Split the tokens
            sent1, ent1, sent2, ent2, sent3 = (' '.join(tokens[:head_pos[0]]),
                                            ' '.join(
                                                tokens[head_pos[0]:head_pos[1]]),
                                            ' '.join(
                                                tokens[head_pos[1]:tail_pos[0]]),
                                            ' '.join(
                                                tokens[tail_pos[0]:tail_pos[1]]),
                                            ' '.join(tokens[tail_pos[1]:]))
            if rev:
                # Reversed order: tail appears before head
                ent1, ent2 = ent2, ent1
            # Pack all parts into a dict and modify by names
            sent_dict = {'sent1': [sent1], 'ent1': [ent1], 'sent2': [sent2],
                        'ent2': [ent2], 'sent3': [sent3]}
            sent_dict_copy = sent_dict.copy()
            # Diverge
            for loc in args.locations:
                origin = sent_dict[loc][0]
                if not origin:
                    # No tokens given
                    continue
                ret = perturb_func.augment(origin)

                # Process result
                if not ret:
                    # Returned nothing
                    ret = [sent_dict[loc][0]]
                if isinstance(ret, str):
                    # Wrap single sentence
                    ret = [ret]
                sent_dict_copy[loc] = ret

                # Merge all parts of perturbed sentences and filter out original sentence
                for merged_sent in filter(lambda perturbed_tokens: perturbed_tokens != tokens,
                                        merge(sent_dict_copy)):
                    tokens = ' '.join(merged_sent).split(' ')
                    sent1, ent1, sent2, ent2, sent3 = merged_sent
                    head_pos = [len(sent1.split(' '))]
                    head_pos.append(head_pos[0] + len(ent1.split(' ')))
                    tail_pos = [head_pos[1] + len(sent2.split(' '))]
                    tail_pos.append(tail_pos[0] + len(ent2.split(' ')))
                    if rev:
                        head_pos, tail_pos = tail_pos, head_pos
                    replaced_samples.append({
                        'token': tokens,
                        'h': {'name':ent1, 'pos': head_pos},
                        't': {'name':ent2, 'pos': tail_pos},
                        'relation': relation,
                        'aug': args.DAmethod
                    })
        with open(os.path.join(args.output_dir, "aug_"+str(i)+".json"),'w') as f:
            for line in replaced_samples:
                f.writelines(json.dumps(line))
                f.write('\n')