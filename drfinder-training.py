#!/usr/bin/env python3
from DeepRegFinder.traineval_functions import *
from DeepRegFinder.nn_models import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import sys
import yaml
import os

"""
Takes in yaml file as first input
Takes in name of output folder as second input
"""
params = sys.argv[1]
with open(params) as f:
    # use safe_load instead load
    dataMap = yaml.safe_load(f)

output_folder = sys.argv[2]
output_folder = os.path.join(output_folder, 'model')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Load datasets.
d = torch.load(dataMap['all_datasets'])
train_dataset = d['train']
val_dataset = d['val']
test_dataset = d['test']
# Construct dataloaders using weighted sampler.
batch_size = dataMap['batch_size']
cpu_threads = dataMap['cpu_threads']
ys = np.array([ y.item() for _, y in train_dataset])
yu, yc = np.unique(ys, return_counts=True)
assert yu[-1] - yu[0] + 1 == len(yu), \
       'Expect the unique train labels to be a sequence \
        of [0..{}] but got {}'.format(yu[-1], yu)
print('Train unique labels: {}'.format(yu))
print('Train label counts: {}'.format(yc))
weights = np.zeros_like(ys, dtype='float')
for i, f in enumerate(yc):
    weights[ys==i] = 1/f
weighted_sampler = WeightedRandomSampler(
    weights, len(ys)//batch_size*batch_size, 
    replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          sampler=weighted_sampler, num_workers=cpu_threads)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                        num_workers=cpu_threads, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                         num_workers=cpu_threads, drop_last=False)
num_marks, num_bins = train_dataset[0][0].shape
num_classes = len(yu)

# Other training related parameters.
net_choice = dataMap['net_choice']
conv_rnn = dataMap['conv_rnn']
init_lr = dataMap['init_lr']
weight_decay = dataMap['weight_decay']
dat_aug = dataMap['data_augment']
best_model_name = dataMap['best_model_name']
best_model_path = os.path.join(output_folder, best_model_name)
nb_epoch = dataMap['num_epochs']
check_iters = dataMap['check_iters']
train_logs = os.path.join(output_folder, 'train_logs')
confus_mat_name = dataMap['confus_mat_name']
pred_out_name = dataMap['pred_out_name']
summary_out_name = dataMap['summary_out_name']

# model, criterion, optimizer, etc.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if net_choice == 'ConvNet':
    model = ConvNet(marks=num_marks, nb_cls=num_classes, 
                    use_leakyrelu=False).to(device)
elif net_choice == 'KimNet':
    model = KimNet(bins=num_bins, marks=num_marks, 
                   nb_cls=num_classes).to(device)
elif net_choice == 'RecurNet':
    model = RecurNet(marks=num_marks, nb_cls=num_classes, add_conv=conv_rnn, 
                     bidirectional=False).to(device)
else:
    raise Exception('Undefined neural net name:', net_choice)
model.apply(init_weights)
criterion = nn.NLLLoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, 
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# ==== initialization === #
start_epoch = 0
best_mAP = 0
train_loss = 0
writer = SummaryWriter(train_logs)
# ======================= #
for epoch in range(start_epoch, nb_epoch):
    # print('Epoch {}'.format(epoch + 1))
    train_loss, train_iter, best_mAP = train_loop(
        model, criterion, optimizer, device, train_loss, best_mAP, epoch, 
        check_iters, train_loader, val_loader, best_model_path, 
        histone_list=None, dat_augment=dat_aug, writer=writer)
    scheduler.step(best_mAP)

# Evaluate the final model performance.
if train_iter > 0:  # remaining iters not yet checked.
    avg_val_loss, val_ap = prediction_loop(
        model, device, val_loader, criterion=criterion, 
        histone_list=None, dat_augment=dat_aug)
    val_mAP = np.mean(val_ap[:-1])
    print('Finally, avg train loss: {:.3f}; val loss: {:.3f}, val mAP: '
          '{:.3f}'.format(train_loss/train_iter, avg_val_loss, val_mAP), 
          end='')
    if val_mAP > best_mAP:
        best_mAP = val_mAP
        torch.save(model.state_dict(), best_model_path)
        print(' --> best mAP updated; model saved.')
    else:
        print()

# Evaluate on the test set.
model.load_state_dict(torch.load(best_model_path))
avg_test_loss, test_ap, test_preds = prediction_loop(
    model, device, test_loader, criterion=criterion, 
    histone_list=None, dat_augment=dat_aug, 
    return_preds=True)
test_mAP = np.mean(test_ap[:-1])
truevals, predictions, probs = test_preds
def _test_set_summary(fh):
    '''Print summary info on the test set
    '''
    print('='*10, 'On test set', '='*10, file=fh)
    print('avg test loss={:.3f}, mAP={:.3f}'.format(avg_test_loss, test_mAP), 
          file=fh)
    print('AP for each class: poised enh={:.3f}, active enh={:.3f}, '
          'poised tss={:.3f}, active tss={:.3f}'.format(
            test_ap[0], test_ap[1], test_ap[2], test_ap[3]), 
          file=fh
         )
_test_set_summary(sys.stdout)
with open(os.path.join(output_folder, summary_out_name), 'w') as fh:
    _test_set_summary(fh)

# Output figures and other stats.
# confusion matrix.
m = confusion_matrix(truevals, predictions)
cm = plot_confusion_matrix(m, norm=True, n_classes=num_classes)
plt.savefig(os.path.join(output_folder, confus_mat_name + '.png'))
cm.to_csv(os.path.join(output_folder, confus_mat_name + '.csv'))
# test set predictions.
df = np.stack([truevals, predictions], axis=1)
df = np.concatenate([df, probs], axis=1)
col_names = ['label', 'pred', 'poised_enh', 'active_enh', 
             'poised_tss', 'active_tss', 'background']
df = pd.DataFrame(df, columns=col_names).round(3)
df = df.astype({'label': 'int', 'pred': 'int'})
df.to_csv(os.path.join(output_folder, pred_out_name), index=False)


