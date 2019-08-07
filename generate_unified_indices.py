import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pycm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from torchsummary import summary
from helper_funcs import *



def check_ordering(hist_dset, seq_dset):
    '''Make sure the labels of histone mark and sequence 
    datasets represent the same classes
    '''
    assert len(hist_dset) == len(seq_dset), "Unordered: Datasets have different lengths!"
    for hist_sample, seq_sample in zip(hist_dset, seq_dset):
        hist_l, seq_l = hist_sample[1].item(), seq_sample[1].item()
        if (hist_l == 0) or (hist_l == 1):
            assert seq_l == 0, "Unordered: Enhancers don't match"
        elif (hist_l == 2) or (hist_l == 3):
            assert seq_l == 1, "Unordered, TSS don't match"
        elif (hist_l == 4):
            assert seq_l == 2, "Unordered, BG don't match"
    return True 

def make_training_testing_dataloaders(his_dataset, seq_dataset, batch_sz, tensor_out_folder):
    histone_dataset = torch.load(his_dataset)
    sequence_dataset = torch.load(seq_dataset)
    check_ordering(histone_dataset, sequence_dataset)
    
    rng = np.random.RandomState(12345)
    eval_size = .2
    hist_train_X, hist_eval_X = [], []
    hist_train_y, hist_eval_y = [], []
    seq_train_X, seq_eval_X = [], []
    seq_train_y, seq_eval_y = [], []
    for hist_sample, seq_sample in zip(histone_dataset, sequence_dataset):
        hist_d, seq_d = hist_sample[0], seq_sample[0]
        hist_l, seq_l = hist_sample[1], seq_sample[1]
        if rng.rand() > eval_size:
            #if hist_l != 3:
            hist_train_X.append(hist_d)
            hist_train_y.append(hist_l)
            seq_train_X.append(seq_d)
            seq_train_y.append(seq_l)
        else:
            #if hist_l != 3:
            hist_eval_X.append(hist_d)
            hist_eval_y.append(hist_l)
            seq_eval_X.append(seq_d)
            seq_eval_y.append(seq_l)
            
    hist_train_X, hist_train_y = torch.stack(hist_train_X), torch.stack(hist_train_y)
    hist_eval_X, hist_eval_y = torch.stack(hist_eval_X), torch.stack(hist_eval_y)
    seq_train_X, seq_train_y = torch.stack(seq_train_X), torch.stack(seq_train_y)
    seq_eval_X, seq_eval_y = torch.stack(seq_eval_X), torch.stack(seq_eval_y)
    
    #Swap last two dimensions for sequence tensors
    seq_train_X = seq_train_X.permute(0, 2, 1)
    seq_eval_X = seq_eval_X.permute(0, 2, 1)
    
    h_train_loader = DataLoader(
    torch.utils.data.TensorDataset(hist_train_X, hist_train_y), 
    batch_size=batch_sz, shuffle=True, num_workers=4) 
    
    h_eval_loader = DataLoader(
    torch.utils.data.TensorDataset(hist_eval_X, hist_eval_y), 
    batch_size=batch_sz, shuffle=False, num_workers=4)
    
    h_train_loader_3cls = DataLoader(
    torch.utils.data.TensorDataset(hist_train_X, seq_train_y), 
    batch_size=batch_sz, shuffle=True, num_workers=4)
    
    h_eval_loader_3cls = DataLoader(
    torch.utils.data.TensorDataset(hist_eval_X, seq_eval_y), 
    batch_size=batch_sz, shuffle=False, num_workers=4)
    
    s_train_loader = DataLoader(
    torch.utils.data.TensorDataset(seq_train_X, seq_train_y), 
    batch_size=batch_sz, shuffle=True, num_workers=4)
    
    s_eval_loader = DataLoader(
    torch.utils.data.TensorDataset(seq_eval_X, seq_eval_y), 
    batch_size=batch_sz, shuffle=False, num_workers=4)
    
    torch.save(h_train_loader, tensor_out_folder + "histone_train_dataloader.pt")
    torch.save(h_eval_loader, tensor_out_folder + "histone_eval_dataloader.pt")
    torch.save(h_train_loader_3cls, tensor_out_folder + "histone_train_dataloader_3cls.pt")
    torch.save(h_eval_loader_3cls, tensor_out_folder + "histone_eval_dataloader_3cls.pt")
    torch.save(s_train_loader, tensor_out_folder + "sequence_train_dataloader.pt")
    torch.save(s_eval_loader, tensor_out_folder + "sequence_eval_dataloader.pt")
    
    