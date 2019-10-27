import itertools as itls
import pycm
from torchsummary import summary
from scipy import interp 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
np.seterr(divide='ignore', invalid='ignore')
from sklearn.preprocessing import label_binarize
import subprocess
import os
from pybedtools import BedTool

def normalize_dat_loader(sequence_loader, histone_loader):
    '''Choose dataloader to be histone, sequence or both
    '''
    if sequence_loader is not None and histone_loader is not None:
        dat_loader = zip(sequence_loader, histone_loader)
    elif sequence_loader is not None:
        dat_loader = sequence_loader
    elif histone_loader is not None:
        dat_loader = histone_loader
    return dat_loader
    
def normalize_dat_dict(batch_data, use_sequence, use_histone, 
                       dat_augment, device, histone_list=None,
                       has_label=True):
    '''Returned correct batch data for model training and evaluation
    '''
    # forward samples.
    if use_sequence and use_histone:
        sequence_data, histone_data = batch_data
        if has_label:
            forward_histone_sample, label = histone_data[0], histone_data[1]
        else:
            forward_histone_sample = histone_data
        if isinstance(forward_histone_sample, np.ndarray):
            forward_histone_sample = torch.from_numpy(forward_histone_sample)
        forward_histone_sample = (forward_histone_sample 
                                  if histone_list is None else 
                                  forward_histone_sample[:, histone_list, :])
        forward_sequence_sample = sequence_data[0]
        if isinstance(forward_sequence_sample, np.ndarray):
            forward_sequence_sample = torch.from_numpy(forward_sequence_sample)
        dat_dict = {'histone_forward': forward_histone_sample.float().to(device), 
                    'sequence_forward': forward_sequence_sample.float().to(device)}
    elif use_sequence:
        if has_label:
            forward_sequence_sample, label = batch_data[0], batch_data[1]
        else:
            forward_sequence_sample = batch_data
        if isinstance(forward_sequence_sample, np.ndarray):
            forward_sequence_sample = torch.from_numpy(forward_sequence_sample)
        dat_dict = {'sequence_forward': forward_sequence_sample.float().to(device)}
    elif use_histone:
        if has_label:
            forward_histone_sample, label = batch_data[0], batch_data[1]
        else:
            forward_histone_sample = batch_data
        if isinstance(forward_histone_sample, np.ndarray):
            forward_histone_sample = torch.from_numpy(forward_histone_sample)
        forward_histone_sample = (forward_histone_sample 
                                  if histone_list is None else 
                                  forward_histone_sample[:, histone_list, :])
        dat_dict = {'histone_forward': forward_histone_sample.float().to(device)}
    # reverse samples for histone.
    if dat_augment and use_histone:
        reverse_histone_sample = forward_histone_sample.flip(dims=[2])
        dat_dict.update({
            'histone_reverse': reverse_histone_sample.float().to(device)
        })
    # reverse samples for sequence.
    if dat_augment and use_sequence:
        reverse_sequence_sample = forward_sequence_sample.flip(dims=[2])
        # sequence reverse complement.
        rev_sequence_comp = forward_sequence_sample.flip(dims=[1])
        rev_rev_sequence = rev_sequence_comp.flip(dims=[2])
        dat_dict.update({
            'sequence_reverse': reverse_sequence_sample.float().to(device),
            'sequence_complement': rev_sequence_comp.float().to(device), 
            'sequence_complement_reverse': rev_rev_sequence.float().to(device)
        })
    if has_label:
        label = label.long().to(device)
        return dat_dict, label
    return dat_dict

def prediction_loop(model, device, sequence_loader=None, 
                    histone_loader=None, histone_list=None, 
                    dat_augment=False, return_scores=False,
                    nb_batch=None):
    '''Make predictions on an entire dataset
    Args:
        nb_batch ([int]): #batches to predict. Default is None.
            This is useful for debug.
    '''
    assert(sequence_loader is not None or histone_loader is not None)
    dat_loader = normalize_dat_loader(sequence_loader, histone_loader)
    model.eval()  # set evaluation state.
    with torch.no_grad():
        p, s, others = [], [], []
        for i, batch in enumerate(dat_loader):
            if isinstance(batch, list):
                batch_dat, batch_info = batch[0], batch[1:]
            dat_dict = normalize_dat_dict(
                batch_dat, 
                use_sequence=(sequence_loader is not None), 
                use_histone=(histone_loader is not None), 
                dat_augment=dat_augment, device=device, 
                histone_list=histone_list, 
                has_label=False)
            # scoring.
            pscores = model(**dat_dict)
            _, preds = torch.max(pscores.data, 1)
            # accumulate results.
            p.append(preds.cpu().numpy())
            s.append(pscores.cpu().numpy())
            others.append(batch_info)
            if nb_batch is not None and (i+1) >= nb_batch:
                break
        predictions = np.concatenate(p)
        scores = np.concatenate(s)
        # transpose: batches x items -> items x batches.
        others = list(map(list, list(zip(*others))))
        info_list = [ np.concatenate(item_list) for item_list in others]
        if return_scores:
            return predictions, info_list, scores
        return predictions, info_list
"""
Creating TPMs file by getting rid of TPMs that are not distal to tss
Merge TPMs and save as final_tpms.bed
"""
def process_tpms(slopped_tss, TPMs, final_tfbs, output_folder):
    tpms_out_folder = './' + output_folder + '/tpms_data/'
    if not os.path.exists(tpms_out_folder):
        os.mkdir(tpms_out_folder)
        
    tpms_sub_tss = []
    #Getting rid of TFBS that aren't distal to tss
    #tss file is slopped so range of each interval is 2kb, making sure anything that is 
    #overlapping is at least 1kb away

    for tpms in TPMs:
        tpms = BedTool(tpms)
        tpms_sub_tss.append(tpms.subtract(slopped_tss, A=True))

    tpms_sub_tss.append(final_tfbs)
    final_tpms = tpms_sub_tss[0]

    #Merging the features using cat
    for i in range(1, len(TPMs)):
        final_tpms = final_tpms.cat(TPMs[i])
        
    #Sorting the features
    file = tpms_out_folder + 'final_tpms.bed'
    file_compress_name = file + '.gz'
    final_tpms = final_tpms.sort()
    final_tpms.saveas(file)
    subprocess.call(['./index_file.sh', file, file_compress_name])

