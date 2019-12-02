import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from net_functions import *
import pycm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, average_precision_score
from torchsummary import summary
import sys
import yaml
import os
import math

"""
Takes in yaml file as first input
Takes in name of output folder as second input
"""
params = sys.argv[1]
with open(params) as f:
    # use safe_load instead load
    dataMap = yaml.safe_load(f)

histone_train_loader = torch.load(dataMap['histone_train_loader'])
histone_test_loader = torch.load(dataMap['histone_eval_loader'])

class_weights_file = dataMap['class_weights']
class_weights = torch.from_numpy(1/np.loadtxt(class_weights_file, delimiter='\n')).float()

num_classes = dataMap['num_classes']
num_marks = dataMap['num_marks']
fig_name = dataMap['fig_name']
dat_aug = dataMap['data_augment']
best_mAP_file = dataMap['best_mAP_filename']

# Make device agnostic code.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from nn_models import ConvNet

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        if m.weight.dim() >= 2:
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='leaky_relu', a=.01)
        if m.bias.dim() >= 2:
            nn.init.kaiming_uniform_(m.bias.data, nonlinearity='leaky_relu', a=.01)


classify = ConvNet(marks = num_marks, nb_cls = num_classes).float().to(device)
#CLASS_WEIGHTS = torch.from_numpy(np.array([23.39, 26.67, 5.42, 1.36])).float()
criterion = nn.NLLLoss(weight=class_weights).to(device)

output_folder = sys.argv[2]
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
start_epoch = 0
nb_epoch = dataMap['num_epochs']
best_model_name = output_folder + '/' + best_mAP_file
# ==== initialization === #
best_mAP = -math.inf
classify.apply(init_weights)
if dataMap['continue_training']:
    classify.load_state_dict(torch.load(dataMap['prev_best_mAP_file']))
#classify.load_state_dict(torch.load(best_model_name))
optimizer = torch.optim.Adam(classify.parameters(), lr=.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# ======================= #
for epoch in range(start_epoch, nb_epoch):
    print('Epoch {}'.format(epoch + 1))
    scheduler.step()
    main_train_loop(classify, criterion, optimizer, device, 
                    sequence_loader=None, 
                    histone_loader=histone_train_loader, 
                    dat_augment=dat_aug, report_iters=300)
    tv, preds, bv, scores = validation_loop(
        classify, criterion, device, 
        sequence_loader=None, 
        histone_loader=histone_test_loader, 
        dat_augment=dat_aug, nb_cls=num_classes)
    all_cls_ap = average_precision_score(bv, scores, average=None)
    mAP = np.mean(all_cls_ap[:-1])  # ignore background class.
    if mAP > best_mAP:
        best_mAP = mAP
        print('--> Best mAP updated to: {:.3f}, enhancer poised: {:.3f}, enhancer active: {:.3f}, TSS poised: {:.3f}, TSS active: {:.3f}'.format(mAP, all_cls_ap[0], all_cls_ap[1], all_cls_ap[2], all_cls_ap[3]))
        state = classify.state_dict()
        torch.save(state, best_model_name)
        
        
classify.load_state_dict(torch.load(best_model_name))
truevals, predictions, binvals, scores = validation_loop(
    classify, criterion, device, 
    sequence_loader=None, 
    histone_loader=histone_test_loader, 
    dat_augment=dat_aug, nb_cls=num_classes)
all_cls_ap = average_precision_score(binvals, scores, average=None)
mAP = np.mean(all_cls_ap[:-1])  # ignore background class.
print('mAP: {:.3f}, enhancer poised: {:.3f}, enhancer active: {:.3f}, TSS poised: {:.3f}, TSS active: {:.3f}'.format(mAP, all_cls_ap[0], all_cls_ap[1], all_cls_ap[2], all_cls_ap[3]))

m = confusion_matrix(truevals, predictions)
plot_confusion_matrix(m, norm=True, n_classes=num_classes)
plt.savefig(output_folder + '/' + fig_name)

pm = pycm.ConfusionMatrix(truevals, predictions)
my_file = open(output_folder + '/' + dataMap['fig_data'], "w")
print(pm, file = my_file)
