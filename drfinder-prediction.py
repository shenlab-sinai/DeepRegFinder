#!/usr/bin/env python3
from DeepRegFinder.prediction_functions import *
from DeepRegFinder.traineval_functions import prediction_loop
from DeepRegFinder.nn_models import ConvNet
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
import yaml
import os
from time import time
# from operator import itemgetter 
# import pysam 
from pybedtools import BedTool
# import subprocess


"""
Takes in yaml file as first input
Takes in name of output folder as second input
"""
params = sys.argv[1]
with open(params) as f:
    # use safe_load instead load
    dataMap = yaml.safe_load(f)

output_folder = sys.argv[2]
output_folder = os.path.join(output_folder, 'predictions')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Read channel-wise mean and std.
chann_stats_file = dataMap['chann_stats_file']
chann_stats = pd.read_csv(chann_stats_file, index_col=0)
chann_mean = chann_stats['mean'].values
chann_std = chann_stats['std'].values

# Whole genome dataset and dataloader.
whole_genome_bincnt_file = dataMap['whole_genome_bincnt_file']
number_of_windows = dataMap['number_of_windows']
window_width = dataMap['window_width']
batch_size = dataMap['batch_size']
cpu_threads = dataMap['cpu_threads']
wgd = WholeGenomeDataset(wgbc_tsv=whole_genome_bincnt_file, 
                         mean=chann_mean, std=chann_std, norm=True, 
                         half_size=number_of_windows//2)
wg_loader = DataLoader(wgd, batch_size=batch_size, shuffle=False, 
                       num_workers=cpu_threads, drop_last=False)
num_marks = wgd[0][0].shape[0]  # 1st sample->bincnt->#chann.

# Load saved model.
num_classes = dataMap['num_classes']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_state_dict = dataMap['model_state_dict']
model = ConvNet(marks=num_marks, nb_cls=num_classes, use_leakyrelu=False)
model.load_state_dict(torch.load(model_state_dict, map_location=device))
model = model.to(device)

# Make predictions.
data_augment = dataMap['data_augment']
time_begin = time()
wg_preds, wg_maxprobs, wg_info = prediction_loop(
    model, device, wg_loader, pred_only=True, dat_augment=data_augment, 
    nb_batch=None)
elapsed = time() - time_begin
print('Prediction finished. Elapsed time: {:.1f}s.'.format(elapsed))

# Post-process predictions and write the results.
prob_conf_cutoff = dataMap['prob_conf_cutoff']
output_bed = os.path.join(output_folder, dataMap['output_bed'])
wg_blocks = process_genome_preds(
    wg_preds, wg_info[0], wg_info[1], wg_maxprobs, ignore_labels=[4], 
    maxprob_cutoff=prob_conf_cutoff, nb_block=None)
bed_dict = post_merge_blocks(wg_blocks, window_width, number_of_windows)

# TPMs validation rate.
tpms_file = dataMap['tpms_file']
tpm_bps_added = dataMap['tpm_bps_added']
tpms = BedTool(tpms_file)
tvr_list = []
for name in bed_dict:
    # TPMs by creation are away from TSS. We only consider enhancers here.
    if name.endswith('Enh'):
        bed = bed_dict[name]
        tvr = len(bed.window(b=tpms, w=tpm_bps_added, u=True))/len(bed.saveas())
        print('Validation rate for {}={:.3f}'.format(name, tvr))
        tvr_list.append(tvr)
print('Average validation rate={:.3f}'.format(np.mean(tvr_list)))

# merge different types together without merging the intervals.
for i, bed in enumerate(bed_dict.values()):
    if i == 0:
        merged_bed = bed
    else:
        merged_bed = merged_bed.cat(bed, postmerge=False)
merged_bed = merged_bed.sort().saveas(output_bed)
print('{:d} records written.'.format(len(merged_bed)))



