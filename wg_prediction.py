import numpy as np 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import yaml
import os

"""
Takes in yaml file as first input
Takes in name of output folder as second input where best mAP model saved as well as confusion matrix
"""
params = sys.argv[1]
with open(params) as f:
    # use safe_load instead load
    dataMap = yaml.safe_load(f)

output_folder = sys.argv[2]
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

whole_genome = pd.read_table(dataMap['whole_genome_histone_data'])

print(whole_genome.head())