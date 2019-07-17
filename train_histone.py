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
from torchsummary import summary
import subprocess
from pybedtools import BedTool

# sense_file = '/home/kims/work/enhancer-prediction-data/GRO-seq/strict_no_overlap_srr15assigns.txt'
# sense_bam_file = 'srr1745515.bam'
# antisense_file = '/home/kims/work/enhancer-prediction-data/GRO-seq/strict_no_overlap_srr16assigns.txt'
# antisense_bam_file = 'srr1745516.bam'

"""
Read GRO-seq reads aligned to an enhancer list, normalize the reads to RPKM values and log transform them, use K-means to cluster the log RPKM values into active and poised classes. This procedure is done for sense and antisense reads separately. An enhancer is classified as active or poised only when both sense and antisense derived class labels agree.
"""
def rpkm_normalize_rnaseq_reads(total_reads, feature_reads, feature_length):
    rpkm = ((10**9) * feature_reads) / (feature_length * total_reads).astype(float)
    return rpkm 

def active_poised_clustering(enhancer_rnaseq_file, sense, seed=12345):
    ap_df = pd.read_table(enhancer_rnaseq_file, skiprows = 1)
    ap_df['Chr'] = ap_df.Chr.astype('category')
    total_reads = ap_df[sense].sum()
    ap_df['rpkm'] = rpkm_normalize_rnaseq_reads(total_reads, ap_df[sense], ap_df['Length'])
    ap_df['logrpkm'] = np.log(ap_df['rpkm'] + 1)
    assert(not ap_df.isnull().values.any())
    clusters = KMeans(n_clusters=2, random_state=seed).fit(
        ap_df['logrpkm'].values.reshape(-1, 1))
    return clusters.labels_, clusters.cluster_centers_

def clustering(sense_file, sense_bam_file, antisense_file, antisense_bam_file):
    sense_labels, sense_centers = active_poised_clustering(sense_file, sense_bam_file)
    antisense_labels, antisense_centers = active_poised_clustering(antisense_file, antisense_bam_file)
    groseq_positive = np.logical_and(sense_labels, np.logical_not(antisense_labels))
    groseq_negative = np.logical_and(np.logical_not(sense_labels), antisense_labels)
    return groseq_positive, groseq_negative

def build_training_tensors(featurefile, rpkmsfile, isEnhancer, groseq_positive, groseq_negative, nb_mark=7):
    """Build a pytorch tensor at designated regions for training
    Args:
        featurefile (str): file for regions.
        rpkmsfile (str): file for genome-wide 100bp bin RPKMs.
        isEnhancer (bool): is enhancer region?
        nb_mark ([int]): number of histone marks. Default is 7.
    """
    features = list(pysam.TabixFile(featurefile).fetch(parser=pysam.asTuple()))
    rpkms = pysam.TabixFile(rpkmsfile)
    elist = []  # list for all regions.
    for i, feat in enumerate(features):
        if isEnhancer:  # ignore ambiguous enhancers.
            if (not groseq_positive[i]) and (not groseq_negative[i]):
                continue
        # Get up and down 2Kb of region centers.
        chrom, start, end = feat[0], int(feat[1]), int(feat[2])
        center = int((start + end)/2)
        center_start = int(math.ceil(center/100.0))*100
        rows = list(rpkms.fetch(chrom, center_start - 1000, center_start + 1000, 
                                parser=pysam.asTuple()))
        assert(len(rows) == 20)
        # Assemble RPKMs for different histone marks.
        forward_tmp = []
        for row in rows:
            e = [ float(item.strip()) for item in row[3:3+nb_mark]]
#             e = [float(row[3].strip()), float(row[4].strip()), float(row[5].strip()), float(row[6].strip()), 
#                  float(row[7].strip()), float(row[8].strip()), float(row[9].strip())]
            forward_tmp.append(e)
        forward_data = np.array(forward_tmp).T
        elist.append(forward_data)
    out = np.stack(elist)
    return torch.from_numpy(out)

def build_background_data(bgfile, rpkmsfile, trimmed_bg_file = None):
    """
    Helper to create background dataset
    """
    if trimmed_bg_file:
        #we already have the ~32000 sites we want to use
        return build_training_tensors(trimmed_bg_file, rpkmsfile, isEnhancer=False)
    bg = list(pysam.TabixFile(bgfile).fetch(parser = pysam.asTuple()))
    rpkms = pysam.TabixFile(rpkmsfile)
    elist = []
    print("Gathering all non-zero potential background sites!")
    count = 0
    for i, pb in enumerate(bg):
        chrom, start, end = pb[0], int(pb[1]), int(pb[2])
        rows = np.array(list(rpkms.fetch(chrom, start, end, parser=pysam.asTuple())))
        
        if rows.shape[0] != 20: 
            #We will have fewer than 20 bins if we are at a chromosome end; 
            #we can safely exclude this from our training set
            continue
            
        #discard chr, start, end so we just have histone marks (columns 3:10), and convert to int dtype
        #take transpose so we have channels x rows (histone marks x bins; 7 x 20)
        data = rows[:, 3:].astype(float).T
        
        #if the sample is not all 0, include it in our list of potential samples
        if not np.all(np.isclose(data, 0)):  
            elist.append({'data': data, 'location': (chrom, str(start), str(end))})
            count += 1
            
        if i % 100000 == 0:
            print(str(i) + " regions checked so far")
            print("Gathered " + str(count) + " potential regions so far")
    print("Done gathering potential sites. Selecting 32000 randomly")
    if len(elist) > 32000:
        elist = random.sample(elist, 32000)
    
    out = np.stack([x['data'] for x in elist])
    print("Writing BED file of background sites used in used_bg.bed")
    with open("used_bg.bed", "w+") as bed:
        for x in elist:
            c, s, e = x['location'][0], x['location'][1], x['location'][2]
            bed.write(c + "\t" + s + "\t" + e + "\n")
    print("Wrote succesfully")
    return torch.from_numpy(out)

def gen_pos_lab(gp, gn):
    if gp:
        return 1
    elif gn:
        return 0
    else:
        return -1
    
def get_statistics(data):
    mean, std = 0., 0.
    total_observed = 0.
    for sample in data:
        x = sample[0]
        x_mean = x.mean(1)
        x_std = x.std(1)
        current_observations = x.shape[1]
        old_mean = mean
        old_std = std
        old_constant = (total_observed/(total_observed + current_observations))
        new_constant = (current_observations/(total_observed + current_observations))
        std_update_term = (total_observed*current_observations) \
            / (np.power((total_observed + current_observations), 2)) \
            * np.power(np.subtract(old_mean, x_mean), 2)
        mean = (old_constant*old_mean) + (new_constant*x_mean)
        std = np.sqrt((old_constant*np.power(old_std, 2)) + (new_constant*np.power(x_std, 2)) + std_update_term)
        total_observed += current_observations
    return mean, std

class NormDataset(Dataset):
    def __init__(self, data, mean, std, norm=True, nb_mark=7):
        self.data = data
        self.mean = mean.view(nb_mark, -1)
        self.std = std.view(nb_mark, -1)
        self.norm = norm
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, label = self.data[idx]
        if self.norm:
            sample = torch.div(sample.sub(self.mean), self.std)
        return sample, label

def make_histone_tensors(groseq_positive, groseq_negative, nb_mark):
    enhancer_tensor = build_training_tensors('strict_enhancers.bed.gz', 'alltogether_notnormed_noheader.txt.gz', True, groseq_positive, groseq_negative, nb_mark)
    print("Made enhancer histone tensor")
    
    print(enhancer_tensor.shape)
    
    tss_tensor = build_training_tensors('true_tss.bed.gz', 'alltogether_notnormed_noheader.txt.gz', False, groseq_positive, groseq_negative, nb_mark)
    print(tss_tensor.shape)
    print("Made tss histone tensor")
      
#     bg_tensor = build_background_data('/home/kims/work/enhancer-prediction-data/final_bg.bed.gz',
#                                  '/home/kims/work/enhancer-prediction-data/alltogether_notnormed_noheader.txt.gz')
    
#     used_bg = BedTool('used_bg.bed')
#     used_bg_sorted = used_bg.sort()
#     used_bg_sorted.saveas('used_bg.sorted.bed')
#     subprocess.call(['./index_file.sh', 'used_bg.sorted.bed', 'used_bg.sorted.bed.gz'])
 
    bg_tensor = build_training_tensors('used_bg.sorted.bed.gz', 'alltogether_notnormed_noheader.txt.gz', False, groseq_positive, groseq_negative,nb_mark)
    print("Made background histone tensor")
    print(bg_tensor.shape)
    
    pos_labels = [ gen_pos_lab(gp, gn) for gp, gn in zip(groseq_positive, groseq_negative)]
    pos_labels = np.array(pos_labels)
    pos_labels = pos_labels[pos_labels != -1]
    pos_labels = torch.from_numpy(pos_labels).long()
    
    tss_labels = torch.full((len(tss_tensor),), 2).long()
    bg_labels = torch.full((len(bg_tensor),), 3).long()
    
    posDataset = torch.utils.data.TensorDataset(enhancer_tensor.float(), pos_labels)
    tssDataset = torch.utils.data.TensorDataset(tss_tensor.float(), tss_labels)
    bgDataset = torch.utils.data.TensorDataset(bg_tensor.float(), bg_labels)
    
    data = torch.utils.data.ConcatDataset((posDataset, tssDataset, bgDataset))
    mean, std = get_statistics(data)
    
    final_training_dataset = NormDataset(data, mean=mean, std=std, norm=True, nb_mark=nb_mark)
    torch.save(final_training_dataset, "sevenmark_train_dataset.pt")

