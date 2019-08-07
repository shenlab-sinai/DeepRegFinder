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

def build_training_tensors(featurefile, seqfile, groseq_positive, groseq_negative, sequence_window_width, isEnhancer=False):
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
        half = sequence_window_width / 2
        s = seqs.fetch(chrom, center_start-half, center_start+half)
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

    
def make_sequence_tensors(groseq_positive, groseq_negative, enhancer, tss, background, histone, genome_fasta_file, sequence_window_width, output_folder):
    
    enhancer_sequence_tensor = build_training_tensors(enhancer, genome_fasta_file, groseq_positive, groseq_negative, sequence_window_width, isEnhancer=True)
    print('Made enhancer sequence dataset')
    
    tss_sequence_tensor = build_training_tensors(tss, genome_fasta_file, groseq_positive, groseq_negative, sequence_window_width)
    print('Made tss sequence dataset')
    
    bg_sequence_tensor = build_training_tensors(background, genome_fasta_file, groseq_positive, groseq_negative, sequence_window_width)
    print('Made background sequence dataset')
    
    pos_labels = build_pos_labels(enhancer, groseq_positive, groseq_negative)
    
    tss_labels = torch.full((len(tss_sequence_tensor),), 1).long()
    bg_labels = torch.full((len(bg_sequence_tensor),), 2).long()
    
    posDataset = torch.utils.data.TensorDataset(enhancer_sequence_tensor.float(), pos_labels)
    tssDataset = torch.utils.data.TensorDataset(tss_sequence_tensor.float(), tss_labels)
    bgDataset = torch.utils.data.TensorDataset(bg_sequence_tensor.float(), bg_labels)
    
    sequence_training_dataset = torch.utils.data.ConcatDataset((posDataset, tssDataset, bgDataset))
    
    torch.save(sequence_training_dataset, output_folder + "sequence_train_dataset.pt" )