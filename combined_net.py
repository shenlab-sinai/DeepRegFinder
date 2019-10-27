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
This can only be run after obtaining a state_dict from histone_net 
and sequence_net run with data_augment = True
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

histone_train_loader = torch.load(dataMap['histone_train_loader'])
histone_test_loader = torch.load(dataMap['histone_eval_loader'])
sequence_train_loader = torch.load(dataMap['seq_train_loader'])
sequence_test_loader = torch.load(dataMap['seq_eval_loader'])

class_weights_file = dataMap['class_weights']
class_weights = torch.from_numpy(1/np.loadtxt(class_weights_file, delimiter='\n')).float()


histone_state_dict = dataMap['histone_state_dict']
sequence_state_dict = dataMap['sequence_state_dict']

num_classes = dataMap['num_classes']
num_marks = dataMap['num_marks']
fig_name = dataMap['fig_name']
best_mAP_file = dataMap['best_mAP_filename']


# Make device agnostic code.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from nn_models import ConvNet, SeqNet, get_last_pooling_layer, Combined_ConvNet

hmodel = ConvNet(marks = num_marks, nb_cls = num_classes).to(device)
smodel = SeqNet().to(device)
hmodel.load_state_dict(torch.load(histone_state_dict))
smodel.load_state_dict(torch.load(sequence_state_dict))
    
new_hmodel = get_last_pooling_layer(hmodel).to(device)
new_smodel = get_last_pooling_layer(smodel).to(device)

classify = Combined_ConvNet(new_hmodel, new_smodel, fc_size=(400, 100), nb_cls = num_classes).to(device)
criterion = nn.NLLLoss(weight=class_weights).to(device)


start_epoch = 0
nb_epoch = dataMap['num_epochs']
best_model_name = output_folder + '/' + best_mAP_file

# ==== initialization === #
best_mAP = -math.inf
if dataMap['continue_training']:
    classify.load_state_dict(torch.load(dataMap['prev_best_mAP_file']))
#classify.load_state_dict(torch.load(best_model_name))
optimizer = torch.optim.SGD(
    [{'params': classify.histone_model.parameters(), 'lr': 1e-4}, 
     {'params': classify.sequence_model.parameters(), 'lr': 1e-4},
     {'params': classify.fc1_layer.parameters()}, 
     {'params': classify.fc2_layer.parameters()}, 
     {'params': classify.final_layer.parameters()}], 
    lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# ======================= #
for epoch in range(start_epoch, nb_epoch):
    print('Epoch {}'.format(epoch + 1))
    scheduler.step()
    main_train_loop(classify, criterion, optimizer, device, 
                    sequence_loader=sequence_train_loader, 
                    histone_loader=histone_train_loader, 
                    dat_augment=True, report_iters=300)
    tv, preds, bv, scores = validation_loop(
        classify, criterion, device, 
        sequence_loader=sequence_test_loader, 
        histone_loader=histone_test_loader, 
        dat_augment=True, nb_cls=num_classes)
    all_cls_ap = average_precision_score(bv, scores, average=None)
    mAP = np.mean(all_cls_ap[:-1])
    if mAP > best_mAP:
        best_mAP = mAP
        print('--> Best mAP updated to: {:.3f}, enhancer poised: {:.3f}, enhancer active: {:.3f}, TSS poised: {:.3f}, TSS active: {:.3f}'.format(mAP, all_cls_ap[0], all_cls_ap[1], all_cls_ap[2], all_cls_ap[3]))
        state = classify.state_dict()
        torch.save(state, best_model_name)

classify.load_state_dict(torch.load(best_model_name))
truevals, predictions, binvals, scores = validation_loop(
    classify, criterion, device, 
    sequence_loader=sequence_test_loader, 
    histone_loader=histone_test_loader, 
    dat_augment=True, nb_cls=num_classes)
all_cls_ap = average_precision_score(binvals, scores, average=None)
mAP = np.mean(all_cls_ap[:-1])  # ignore background class.
print('mAP: {:.3f}, enhancer poised: {:.3f}, enhancer active: {:.3f}, TSS poised: {:.3f}, TSS active: {:.3f}'.format(mAP, all_cls_ap[0], all_cls_ap[1], all_cls_ap[2], all_cls_ap[3]))

m = confusion_matrix(truevals, predictions)
plot_confusion_matrix(m, norm=True, n_classes=num_classes)
plt.savefig(output_folder + '/' + fig_name)

pm = pycm.ConfusionMatrix(truevals, predictions)
my_file = open(output_folder + '/' + dataMap['fig_data'], "w")
print(pm, file = my_file)
