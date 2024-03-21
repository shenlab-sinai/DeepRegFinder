#!/usr/bin/env python3
from DeepRegFinder.preprocessing_functions import _bed_to_saf, process_histones, ChannNormDataset
from DeepRegFinder.prediction_functions import *
from DeepRegFinder.traineval_functions import prediction_loop
from DeepRegFinder.nn_models import create_model
from torch.utils.data import Dataset, TensorDataset
import numpy as np 
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import pysam
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
parser=argparse.ArgumentParser(
        description='''This module runs the trained models on the genomic read coverage to perform classifications for each 2 Kb window of the genome. This module also has the ability to generate predictions on a BED file of the user's choice. See https://github.com/shenlab-sinai/DeepRegFinder for details.''')
parser.add_argument('wg_prediction_data.yaml', help='Name of the wg_prediction_data.yaml file')
parser.add_argument('output', help='Name of the output folder (same as what was used for drfinder-preprocessing.py and drfinder-training.py)')
args=parser.parse_args()

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

# whether to perform whole genome prediction
bed_file_prediction = dataMap['bed_file_prediction']
whole_genome_prediction = dataMap['whole_genome_prediction']
data_augment = dataMap['data_augment']
cpu_threads = dataMap['cpu_threads']
number_of_windows = dataMap['number_of_windows']
window_width = dataMap['window_width']
batch_size = dataMap['batch_size']
# Load model
num_classes = dataMap['num_classes']
net_choice = dataMap['net_choice']
conv_rnn = dataMap['conv_rnn']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_marks = dataMap['num_marks']
model = create_model(net_choice, num_marks, num_classes, 
                     num_bins=number_of_windows, conv_rnn=conv_rnn, 
                     device=device)
model_state_dict = dataMap['model_state_dict']
model.load_state_dict(torch.load(model_state_dict, map_location=device))

# Whole genome dataset and dataloader.
def run_whole_genome_prediction():
    whole_genome_bincnt_file = dataMap['whole_genome_bincnt_file']
    batch_size = dataMap['batch_size']
    wgd = FastWholeGenomeDataset(
        wgbc_tsv=whole_genome_bincnt_file, 
        mean=chann_mean, std=chann_std, norm=True, 
        half_size=number_of_windows//2
    )
    # If set num_workers > 0, the dataloader may throw an "ancdata zero item" error.
    wg_loader = DataLoader(wgd, batch_size=batch_size, shuffle=False, 
                           num_workers=0, drop_last=False)
    num_marks = wgd[0][0].shape[0]  # 1st sample->bincnt->#chann.

    # Make predictions.
    time_begin = time()
    wg_preds, wg_probs, wg_info = prediction_loop(
        model, num_classes, device, wg_loader, pred_only=True, dat_augment=data_augment, 
        nb_batch=None, show_status=True)

    wg_maxprobs = np.max(wg_probs, axis=1)

    elapsed = time() - time_begin
    print('Prediction finished. Elapsed time: {:.1f}s.'.format(elapsed))


    # Post-process predictions and write the results.
    prob_conf_cutoff = dataMap['prob_conf_cutoff']
    output_bed = os.path.join(output_folder, dataMap['output_bed'])
    known_tss_file = dataMap['known_tss_file']

    if num_classes == 5:
    	ignore_labels_val = 4
    elif num_classes == 2 or num_classes == 3:
    	ignore_labels_val = 0

    wg_blocks = process_genome_preds(
        wg_preds, wg_info[0], wg_info[1], wg_maxprobs, ignore_labels=[ignore_labels_val], 
        maxprob_cutoff=prob_conf_cutoff, nb_block=None)

    bed_dict = post_merge_blocks(wg_blocks, window_width, number_of_windows, 
                                 num_classes=num_classes, 
                                 known_tss_file=known_tss_file)

    # TPMs validation rate.
    tpms_file = dataMap['tpms_file']
    tpm_bps_added = dataMap['tpm_bps_added']
    output_txt = os.path.join(output_folder, dataMap['output_txt'])
    fh = open(output_txt, 'w')
    tpms = BedTool(tpms_file)
    tvr_list = []

    for name in bed_dict:
        bed = bed_dict[name]
        tvr = len(bed.window(b=tpms, w=tpm_bps_added, u=True))/len(bed)
        # TPMs by creation are away from TSS. We only consider enhancers here.
        if num_classes == 5 and name.endswith('Enh'):
            print('Validation rate for {}={:.3f}'.format(name, tvr))
            print('Validation rate for {}={:.3f}'.format(name, tvr), file=fh)
            tvr_list.append(tvr)
            print('Average validation rate={:.3f}'.format(np.mean(tvr_list)))
            print('Average validation rate={:.3f}'.format(np.mean(tvr_list)), file=fh)

        elif (num_classes == 2 or num_classes == 3) and name.startswith('Enh'):
            print('Validation rate for {}={:.3f}'.format(name, tvr))
            print('Validation rate for {}={:.3f}'.format(name, tvr), file=fh)

    fh.close()

    # merge different types together without merging the intervals.
    for i, bed in enumerate(bed_dict.values()):
        if i == 0:
            merged_bed = bed
        else:
            merged_bed = merged_bed.cat(bed, postmerge=False)
    merged_bed = merged_bed.sort().saveas(output_bed)
    print('{:d} records written.'.format(len(merged_bed)))


def run_custom_bed_prediction():
    global chann_mean, chann_std

    histone_out_folder_custom = os.path.join(output_folder, 'histone_data_custom_bed')
    
    if not os.path.exists(histone_out_folder_custom):
        os.mkdir(histone_out_folder_custom)
    
    bed_file_for_prediction = dataMap['bed_file_for_prediction']
    histone_folder = dataMap['histone_folder']
    predictions_filename = dataMap['predictions_filename']

    def convert_bedfile_to_100bp_bins(infilename, outfilename):
            with open(infilename, "r") as f, \
                    open(outfilename, "w") as outfile:
                    for lines in f:
                            lines = lines.strip().split()
                            chromosome = lines[0]
                            start = lines[1]
                            end = lines[2]

                            for i in range(int(start), int(end), 100):
                                    outfile.write(chromosome+"\t"+str(i)+"\t"+str(i+100)+"\n")

    # Convert BED file coordinates to 100bp bins
    new_100bp_filename = bed_file_for_prediction.split(".bed")[0] + ".100bp.bed"

    convert_bedfile_to_100bp_bins(bed_file_for_prediction, new_100bp_filename)

    # Convert BED to saf
    saf_100bp_filename = new_100bp_filename.split(".bed")[0] + ".saf"
    _bed_to_saf(new_100bp_filename, saf_100bp_filename)

    # Get coverage of histones for coordinates in BED file
    process_histones(saf_100bp_filename, 
                    histone_folder,
                    histone_out_folder_custom, 
                    mode="prediction_custom",
                    outfilename="alltogether_notnormed_custom.txt", 
                    hist_logtrans=False, 
                    cpu_threads=cpu_threads)

    # Convert counts matrix to tabix
    bed = BedTool(histone_out_folder_custom + "/histone_data/alltogether_notnormed_custom.txt")
    bed.tabix(force=True, is_sorted=True)

    print('Finished compressing and indexing files for custom BED file')

    regions = BedTool(bed_file_for_prediction)
    region_size = 100

    hist_cnts = pysam.TabixFile(histone_out_folder_custom + "/histone_data/alltogether_notnormed_custom.txt.gz")

    dlist, rlist, ylist = [], [], []

    for i, r in enumerate(regions):
            label = 0
            r_start = r.start + 1
            r_end = r.end//region_size*region_size  # remove incomplete region.
            if r_end - r_start + 1 < region_size:
                    continue
            rows = np.array(
                    list(hist_cnts.fetch(r.chrom, r_start, r_end,
                                     parser=pysam.asTuple()))
            )
            # Fetches all rows from alltogether file that fall b/w start and end
            d = rows[:, 3:].astype(float).T
            dlist.append(d)
            rlist.append((r.chrom, r_start - 1, r_end))
            ylist.append(label)

    hist_t = np.stack(dlist)
    regions_t = np.stack(rlist)
    label_t = np.array(ylist)

    X_pred, y_pred = hist_t, label_t

    # pytorch datasets
    X_pred_t, y_pred_t = torch.from_numpy(X_pred).float(), torch.from_numpy(y_pred).long()
    prediction_dataset = TensorDataset(X_pred_t, y_pred_t)
    chann_mean = torch.tensor(chann_mean).view(-1,1)

    chann_std = torch.tensor(chann_std).view(-1,1)
    prediction_dataset = ChannNormDataset(prediction_dataset, chann_mean, chann_std)


    prediction_loader = DataLoader(prediction_dataset, batch_size=10000000,
                             num_workers=cpu_threads, drop_last=False)


    # Make predictions
    bed_preds, bed_probs, bed_info = prediction_loop(
        model, num_classes, device, prediction_loader, pred_only=True, dat_augment=data_augment,
        nb_batch=None, show_status=True)

    if num_classes == 2:
        class_lookup = {0: 'Background', 1: 'Enhancer'}

    elif num_classes == 3:
        class_lookup = {0: 'Background', 1: 'TSS', 2: 'Enhancer'}

    elif num_classes == 5:
        class_lookup = {0: 'Poised_Enh', 1: 'Active_Enh', 2: 'Poised_TSS', 
                        3: 'Active_TSS', 4: 'Background'}


    with open(predictions_filename, "w") as outfile:
            for i in range(0, len(list(bed_preds))):
                    coordinates = ("\t".join(list(regions_t)[i]))
                    outfile.write(class_lookup[list(bed_preds)[i]]+"\t"+coordinates+"\n")


def main():
	if whole_genome_prediction == True:
    		run_whole_genome_prediction()

	if bed_file_prediction == True:
    		run_custom_bed_prediction() 


if __name__ == "__main__":
	main()


