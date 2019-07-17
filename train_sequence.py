import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pysam
import time
import math
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from scipy.stats.mstats import zscore
import itertools as itls
import pycm
from re import sub
from torchsummary import summary

def build_training_tensors(featurefile, seqfile, groseq_positive, groseq_negative, isEnhancer=False):
    encoder = {
        'A' : [1, 0, 0, 0],
        'C' : [0, 1, 0, 0], 
        'G' : [0, 0, 1, 0],
        'T' : [0, 0, 0, 1],
        'N' : [0, 0, 0, 0]
    }
    features = list(pysam.TabixFile(featurefile).fetch(parser=pysam.asTuple()))
    seqs = pysam.FastaFile(seqfile)
    finished_sequences = []
    for i, feat in enumerate(features):
        if isEnhancer:
            if (not groseq_positive[i]) and (not groseq_negative[i]):
                continue
        chrom, start, end = feat[0], int(feat[1]), int(feat[2])
        center = int((start+end) /2)
        center_start = int(math.ceil(center / 100.0)) * 100
        s = seqs.fetch(chrom, center_start-150, center_start+150)
        s = sub('[a-z]', 'N', s)
        encoded_seq = np.array([encoder[base] for base in s])
        finished_sequences.append(encoded_seq)
    stacked_data = np.stack(finished_sequences)
    return torch.from_numpy(stacked_data)

def build_pos_labels(file, groseq_positive, groseq_negative):
    l = []
    f = list(pysam.TabixFile(file).fetch(parser=pysam.asTuple()))
    for i, feat in enumerate(f):
        if groseq_positive[i] or groseq_negative[i]:
            l.append(0)
    return torch.from_numpy(np.array(l)).long()

def make_sequence_tensors(groseq_positive, groseq_negative):
    enhancer_sequence_tensor = build_training_tensors('strict_enhancers.bed.gz','hg38.fa', groseq_positive, groseq_negative, isEnhancer=True)
    print(enhancer_sequence_tensor.shape)
    print('Made enhancer sequence dataset')
    
    tss_sequence_tensor = build_training_tensors('true_tss.bed.gz','hg38.fa', groseq_positive, groseq_negative)
    print(tss_sequence_tensor.shape)
    print('Made tss sequence dataset')
    
    bg_sequence_tensor = build_training_tensors('used_bg.sorted.bed.gz','hg38.fa', groseq_positive, groseq_negative)
    print(bg_sequence_tensor.shape)
    print('Made background sequence dataset')
    
    pos_labels = build_pos_labels('strict_enhancers.bed.gz', groseq_positive, groseq_negative)
    
    tss_labels, bg_labels = torch.full((8015,), 1).long(), torch.full((32000,), 2).long()
    
    posDataset = torch.utils.data.TensorDataset(enhancer_sequence_tensor.float(), pos_labels)
    tssDataset = torch.utils.data.TensorDataset(tss_sequence_tensor.float(), tss_labels)
    bgDataset = torch.utils.data.TensorDataset(bg_sequence_tensor.float(), bg_labels)
    
    sequence_training_dataset = torch.utils.data.ConcatDataset((posDataset, tssDataset, bgDataset))
    
    torch.save(sequence_training_dataset, "sequence_train_dataset.pt" )