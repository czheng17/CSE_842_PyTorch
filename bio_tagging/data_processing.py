### Chen Zheng 05/19/2022

import torch
from torch import nn
import time
import torchtext
import numpy as np
import random
from collections import defaultdict, Counter

def read_data(corpus_file, datafields):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                examples.append(torchtext.legacy.data.Example.fromlist([words, labels], datafields))
                words = []
                labels = []
            else:
                columns = line.split()
                words.append(columns[0])
                labels.append(columns[-1])
        print(len(examples))
        return torchtext.legacy.data.Dataset(examples[0:len(examples)//10*10], datafields)


def to_spans(l_ids, voc):
    spans = {}
    current_lbl = None
    current_start = None
    for i, l_id in enumerate(l_ids):
        l = voc[l_id]

        if l[0] == 'B': 
            if current_lbl:
                spans[current_start] = (current_lbl, i)
            current_lbl = l[2:]
            current_start = i
        elif l[0] == 'I':
            if current_lbl:
                if current_lbl != l[2:]:
                    spans[current_start] = (current_lbl, i)
                    current_lbl = l[2:]
                    current_start = i
            else:
                current_lbl = l[2:]
                current_start = i
        else:
            if current_lbl:
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
    return spans

# Compares two sets of spans and records the results for future aggregation.
def compare(gold, pred, stats):
    for start, (lbl, end) in gold.items():
        stats['total']['gold'] += 1
        stats[lbl]['gold'] += 1
    for start, (lbl, end) in pred.items():
        stats['total']['pred'] += 1
        stats[lbl]['pred'] += 1
    for start, (glbl, gend) in gold.items():
        if start in pred:
            plbl, pend = pred[start]
            if glbl == plbl and gend == pend:
                stats['total']['corr'] += 1
                stats[glbl]['corr'] += 1

def evaluate_iob(predicted, gold, label_field, stats):
    gold_cpu = gold.t().cpu().numpy()
    gold_cpu = list(gold_cpu.reshape(-1))

    pred_cpu = [l for sen in predicted for l in sen]
    
    gold_spans = to_spans(gold_cpu, label_field.vocab.itos)
    pred_spans = to_spans(pred_cpu, label_field.vocab.itos)

    compare(gold_spans, pred_spans, stats)

def prf(stats):
    if stats['pred'] == 0:
        return 0, 0, 0
    p = stats['corr']/stats['pred']
    r = stats['corr']/stats['gold']
    if p > 0 and r > 0:
        f = 2*p*r/(p+r)
    else:
        f = 0
    return p, r, f