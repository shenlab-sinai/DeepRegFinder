import numpy as np 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import yaml
import os
from time import time
from operator import itemgetter 
import pysam 
from pybedtools import BedTool
import subprocess
from wg_prediction_functions import *


"""
Takes in yaml file as first input
Takes in name of output folder as second input
"""
params = sys.argv[1]
with open(params) as f:
    # use safe_load instead load
    dataMap = yaml.safe_load(f)

output_folder = sys.argv[2]
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

num_marks = dataMap['num_marks']
histone_train_loader = torch.load(dataMap['histone_train_loader'])
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

header_names = ['Chr', 'Start', 'End']
for i in range(num_marks):
    header_names.append('Hist' + str(i+1))

whole_genome_histone_data = dataMap['whole_genome_histone_data']
whole_genome = pd.read_table(whole_genome_histone_data, header=None, names=header_names)



def calculate_img_stats_avg(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for imgs,_ in loader:
      
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    return mean,std

bin_means, bin_stds = calculate_img_stats_avg(histone_train_loader)

bin_means = bin_means.cpu().numpy()
bin_stds = bin_stds.cpu().numpy()

class WholeGenomeDataset(Dataset):
    def __init__(self, genome_tsv, mean, std, 
                 norm=True, half_size=10):
        genome_frame = genome_tsv
        #genome_frame = pd.read_table(genome_tsv)
        #genome_frame = pd.read_csv(genome_tsv)
        genome_frame['Chr'] = genome_frame.Chr.astype('category')
        self.chroms = np.array(genome_frame.iloc[:, 0])
        self.starts = np.array(genome_frame.iloc[:, 1])
        self.histone = np.array(genome_frame.iloc[:, 3:], dtype='float32')
        assert len(self.chroms) == len(self.starts)
        self.mean, self.std, self.norm = mean, std, norm
        self.half_size = half_size
        
    def __len__(self):
        return len(self.starts) - self.half_size*2
  
    def __getitem__(self, idx):
        idx = idx + self.half_size  # shift to the central bin.
        togo = np.transpose(
            self.histone[idx-self.half_size:idx+self.half_size, :]
        )
        if self.norm:
            mean_expanded = np.expand_dims(self.mean, axis=1)
            std_expanded = np.expand_dims(self.std, axis=1)
            
            mean_broadcast = np.broadcast_to(mean_expanded, togo.shape)
            std_broadcast = np.broadcast_to(std_expanded, togo.shape)
            
            togo = (togo - mean_broadcast)/std_broadcast
        chrom = self.chroms[idx]
        start = self.starts[idx]
        sample = (togo, chrom, start)
        return sample



wgd = WholeGenomeDataset(genome_tsv=whole_genome, mean=bin_means, std=bin_stds, norm=True)
loader = DataLoader(wgd, batch_size=500, shuffle=False, num_workers=0, drop_last=False)

torch.save(loader, output_folder + '/wg_loader.pt')
#loader = torch.load(output_folder + '/wg_loader.pt')

model_file = __import__(dataMap['model_file'].replace('.py', ''))
model_name = dataMap['model_name']
model_state_dict = dataMap['model_state_dict']

classify = getattr(model_file, model_name)().float().to(device)
classify.load_state_dict(torch.load(model_state_dict))

#Code runs on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

time_begin = time()
wg_preds, wg_info = prediction_loop(
    classify, device, 
    histone_loader=loader, 
    dat_augment=True)
elapsed = time() - time_begin
torch.save(wg_preds, output_folder + '/wg_preds.pt')
torch.save(wg_info, output_folder + '/wg_info.pt')



pred = torch.load(output_folder + '/wg_preds.pt') 
pred_info = torch.load(output_folder + '/wg_info.pt')

slopped_tss = dataMap['slopped_tss']
TPMs = dataMap['TPMs']
final_tfbs = dataMap['TFBS']

process_tpms(slopped_tss, TPMs, final_tfbs, output_folder)


class validator():
    def __init__(self, tpmsfile):
        self.tpms = pysam.TabixFile(tpmsfile)
    #A prediction is valid if it any part of it intersects with any part of a TPM
    def validate(self, chrom, start):
        end = start + 100
        vals = self.tpms.fetch(chrom, start, end)
        if self.isempty(vals):
            return 0
        return 1
    def isempty(self, iterator):
        try:
            next(iterator)                                                               
        except StopIteration:
            return True
        return False 

final_tpms = output_folder + '/tpms_data/final_tpms.bed.gz'

v = validator(final_tpms)
#Removal of elements that aren't enhancers
active_idx = np.where(pred == 1)[0]
poised_idx = np.where(pred == 0)[0]
np_pred_info = np.array(pred_info)


if len(active_idx) > 0 and len(poised_idx) > 0:
    active_chr_list = list(itemgetter(*active_idx)(pred_info[0])) 
    active_start_list = list(itemgetter(*active_idx)(pred_info[1])) 
    poised_chr_list = list(itemgetter(*poised_idx)(pred_info[0])) 
    poised_start_list = list(itemgetter(*poised_idx)(pred_info[1])) 
    enhancer_chr_list = active_chr_list + poised_chr_list
    enhancer_start_list = active_start_list + poised_start_list
elif len(active_idx) > 0 and not len(poised_idx) > 0:
    active_chr_list = list(itemgetter(*active_idx)(pred_info[0])) 
    active_start_list = list(itemgetter(*active_idx)(pred_info[1])) 
    enhancer_chr_list = active_chr_list 
    enhancer_start_list = active_start_list 
else:
    poised_chr_list = list(itemgetter(*poised_idx)(pred_info[0])) 
    poised_start_list = list(itemgetter(*poised_idx)(pred_info[1])) 
    enhancer_chr_list = poised_chr_list
    enhancer_start_list = poised_start_list

time_begin = time()
counter = 0
for i in range(len(enhancer_chr_list)):
    chrom = enhancer_chr_list[i]
    start = enhancer_start_list[i]
    counter += v.validate(chrom, start)
    
print("Time elapsed: "time() - time_begin)
# print(counter)

#Calculating number of correctly predicted enhancers over number of total enhancers
print("Prediction accuracy: " + str(100*counter/len(enhancer_chr_list)))