import numpy as np
import sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score,classification_report
import copy


def f1_eval(logits, labels):
    def getpred(result, T1 = 0.5, T2 = 0.4) :
        # 使用阈值得到preds, result = logits
        # T2 表示如果都低于T2 那么就是 no relation, 否则选取一个最大的
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret.append(r)
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0
        
        for i in range(len(data)):
            # 每一个样本 都是[1,4,...,20] 表示有1,4,20 是1， 如果没有就是[36]
            for id in data[i]:
                if id != 36:
                    # 标签中 1 的个数
                    correct_gt += 1
                    if id in devp[i]:
                        # 预测正确
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys/all_sys
        recall = 0 if correct_gt == 0 else correct_sys/correct_gt
        f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return f_1

    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    temp_labels = []
    for l in labels:
        t = []
        for i in range(36):
            if l[i] == 1:
                t += [i]
        if len(t) == 0:
            t = [36]
        temp_labels.append(t)
    assert(len(labels) == len(logits))
    labels = temp_labels
    
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2/100.)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

    return dict(f1=bestf_1, T2=bestT2)



def compute_f1(logits, labels):
    n_gold = n_pred = n_correct = 0
    preds = np.argmax(logits, axis=-1)
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if pred != 0 and label != 0 and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


def acc(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return (preds == labels).mean()

from collections import Counter
def f1_score(output, label, rel2id, rel_num=42, na_num=13, datasetname=None):
    # correct_by_relation = Counter()
    # guess_by_relation = Counter()
    # gold_by_relation = Counter()
    if output.shape != label.shape:
        output = np.argmax(output, axis=-1)

    # for i in range(len(output)):
    #     guess = output[i]
    #     gold = label[i]

    #     if guess == na_num:
    #         guess = 0
    #     elif guess < na_num:
    #         guess += 1

    #     if gold == na_num:
    #         gold = 0
    #     elif gold < na_num:
    #         gold += 1

    #     if gold == 0 and guess == 0:
    #         continue
    #     if gold == 0 and guess != 0:
    #         guess_by_relation[guess] += 1
    #     if gold != 0 and guess == 0:
    #         gold_by_relation[gold] += 1
    #     if gold != 0 and guess != 0:
    #         guess_by_relation[guess] += 1
    #         gold_by_relation[gold] += 1
    #         if gold == guess:
    #             correct_by_relation[gold] += 1
    
    # f1_by_relation = Counter()
    # recall_by_relation = Counter()
    # prec_by_relation = Counter()
    # for i in range(1, rel_num):
    #     recall = 0
    #     if gold_by_relation[i] > 0:
    #         recall = correct_by_relation[i] / gold_by_relation[i]
    #     precision = 0
    #     if guess_by_relation[i] > 0:
    #         precision = correct_by_relation[i] / guess_by_relation[i]
    #     if recall + precision > 0 :
    #         f1_by_relation[i] = 2 * recall * precision / (recall + precision)
    #     recall_by_relation[i] = recall
    #     prec_by_relation[i] = precision

    # micro_f1 = 0.0
    # recall = 0.0
    # prec = 0.0
    # micro_f1 = 0.0
    # if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
    #     recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
    #     prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
    #     micro_f1 = 2 * recall * prec / (recall+prec)

    y_true = label
    y_pred = output
    pred_labels = list(set(y_true))
    needlabels = []
    neg=-1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others','false',"unanswerable"]:
        if name in rel2id:
            neg = rel2id[name]
    if neg in y_true:
        needlabels = copy.deepcopy(pred_labels)
        needlabels.remove(neg)
    else:
        needlabels = copy.deepcopy(pred_labels)
    needlabels.sort()
    alllabels = list(rel2id.values())
    micro_pre = precision_score(y_true, y_pred, labels=needlabels, average='micro')
    micro_recall = recall_score(y_true, y_pred, labels=needlabels, average='micro')
    micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=needlabels, average='micro', zero_division=0)
    macro_pre = precision_score(y_true, y_pred, labels=needlabels, average='macro')
    macro_recall = recall_score(y_true, y_pred, labels=needlabels, average='macro')
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=needlabels, average='macro', zero_division=0)
    f1_per_relation = list(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, labels=alllabels, average=None, zero_division=0))
    if datasetname=="semeval":
        few_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[3, 18, 17], average='micro', zero_division=0)
        few_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[3, 18, 17], average='macro', zero_division=0)
        med_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[13, 15,16,14], average='micro', zero_division=0)
        med_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[13, 15,16,14], average='macro', zero_division=0)
        many_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[11, 6, 8, 2, 12, 1, 7, 10, 9, 4, 5], average='micro', zero_division=0)
        many_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[11, 6, 8, 2, 12, 1, 7, 10, 9, 4, 5], average='macro', zero_division=0)
    elif datasetname=="tacrev":
        few_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[18, 32, 7, 5, 40, 24, 2, 14, 29, 31, 28, 39, 0, 34], average='micro', zero_division=0)
        few_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[18, 32, 7, 5, 40, 24, 2, 14, 29, 31, 28, 39, 0, 34], average='macro', zero_division=0)
        med_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[16, 9, 3, 25, 41, 21, 17, 11, 35, 22, 38, 19, 36], average='micro', zero_division=0)
        med_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[16, 9, 3, 25, 41, 21, 17, 11, 35, 22, 38, 19, 36], average='macro', zero_division=0)
        many_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[37, 15, 1, 27, 8, 10, 26, 4, 6, 33, 23, 12, 20, 30], average='micro', zero_division=0)
        many_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[37, 15, 1, 27, 8, 10, 26, 4, 6, 33, 23, 12, 20, 30], average='macro', zero_division=0)
    elif datasetname=="wiki80":
        few_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[61, 29, 15, 78, 42, 66, 51, 69, 4, 35, 6, 16, 64, 20, 18], average='micro', zero_division=0)
        few_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[61, 29, 15, 78, 42, 66, 51, 69, 4, 35, 6, 16, 64, 20, 18], average='macro', zero_division=0)
        med_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[54, 19, 59, 26, 73, 44, 12, 5, 32, 8, 28, 34, 60, 52, 31, 71, 63, 14, 58, 13, 79, 27, 77, 30, 21, 75, 76, 1, 22, 24, 23, 62, 43, 36, 45, 41, 65, 7, 46, 67, 40, 10, 3], average='micro', zero_division=0)
        med_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[54, 19, 59, 26, 73, 44, 12, 5, 32, 8, 28, 34, 60, 52, 31, 71, 63, 14, 58, 13, 79, 27, 77, 30, 21, 75, 76, 1, 22, 24, 23, 62, 43, 36, 45, 41, 65, 7, 46, 67, 40, 10, 3], average='macro', zero_division=0)
        many_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[9, 0, 47, 11, 70, 17, 53, 2, 37, 50, 55, 33, 68, 56, 74, 39, 25, 57, 48, 72, 49, 38], average='micro', zero_division=0)
        many_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[9, 0, 47, 11, 70, 17, 53, 2, 37, 50, 55, 33, 68, 56, 74, 39, 25, 57, 48, 72, 49, 38], average='macro', zero_division=0)
    elif datasetname=="SciERC":
        few_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[2,4,6], average='micro', zero_division=0)
        few_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[2,4,6], average='macro', zero_division=0)
        med_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[0,1], average='micro', zero_division=0)
        med_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[0,1], average='macro', zero_division=0)
        many_mif1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[3,5], average='micro', zero_division=0)
        many_maf1 = sklearn.metrics.f1_score(y_true, y_pred, labels=[3,5], average='macro', zero_division=0)
    report = classification_report(y_true, y_pred)

    if datasetname in ['semeval','tacrev','wiki80','SciERC']:
        return {'acc': acc, \
                'micro_p': micro_pre, \
                'micro_r': micro_recall, \
                'micro_f1': micro_f1, \
                'macro_p': macro_pre, \
                'macro_r': macro_recall, \
                'macro_f1': macro_f1, \
                'f1_per_relation': f1_per_relation, \
                'few_mif1': few_mif1, \
                'few_maf1': few_maf1, \
                'med_mif1': med_mif1, \
                'med_maf1': med_maf1, \
                'many_mif1': many_mif1, \
                'many_maf1': many_maf1, \
                'report': '\n'+ report, \
                'pred_labels': pred_labels}
    else:
        return {'acc': acc, \
                'micro_p': micro_pre, \
                'micro_r': micro_recall, \
                'micro_f1': micro_f1, \
                'macro_p': macro_pre, \
                'macro_r': macro_recall, \
                'macro_f1': macro_f1, \
                'f1_per_relation': f1_per_relation, \
                'report': '\n'+ report, \
                'pred_labels': pred_labels}