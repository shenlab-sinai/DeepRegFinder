#!/usr/bin/env python3
from DeepRegFinder.traineval_functions import *
from sklearn.metrics import precision_recall_fscore_support
from DeepRegFinder.nn_models import create_model
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import LabelBinarizer
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
num_classes = dataMap['num_classes']
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

# collapse the non-background classes into one for sampling.
if dataMap['keep_cls_props']:
    if num_classes == 5:
    	bkg_lab = yu[-1]
    else:
        bkg_lab = yu[0]
    ys_ = ys.copy()
    ys_[ys==bkg_lab] = 0
    ys_[ys!=bkg_lab] = 1
    yu_, yc_ = np.unique(ys_, return_counts=True)
    #!!! Exp !!!#
    # yc_[1] //= 2  # basically, up-sample bkg class.
    # Obs: had little effect on APs.
    #!!!!!!!!!!!#
else:
    ys_, yu_, yc_ = ys, yu, yc

weights = np.zeros_like(ys_, dtype='float')
for i, f in enumerate(yc_):
    weights[ys_==i] = 1/f
weighted_sampler = WeightedRandomSampler(
    weights, len(ys_)//batch_size*batch_size, 
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
checkpoint_path = os.path.join(output_folder, 'model_checkpoint.pth.tar')
nb_epoch = dataMap['num_epochs']
check_iters = dataMap['check_iters']
train_logs = os.path.join(output_folder, 'train_logs')
confus_mat_name = dataMap['confus_mat_name']
precision_recall_curve_name = dataMap['precision_recall_curve_name']
pred_out_name = dataMap['pred_out_name']
summary_out_name = dataMap['summary_out_name']

# model, criterion, optimizer, etc.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = create_model(net_choice, num_marks, num_classes, num_bins, 
                     conv_rnn, device)
criterion = nn.NLLLoss(reduction='mean').to(device)
if net_choice == 'KimNet':
    # Use momentum=0.9 will make KimNet more likely to blow. 
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, 
                                weight_decay=weight_decay, momentum=0)
else:
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
        model, num_classes, criterion, optimizer, scheduler, device, train_loss, best_mAP, epoch, 
        check_iters, train_loader, val_loader, best_model_path, checkpoint_path, 
        histone_list=None, dat_augment=dat_aug, writer=writer)
    scheduler.step(best_mAP)

# Evaluate the final model performance.
if train_iter > 0:  # remaining iters not yet checked.
    try:
        avg_val_loss, val_ap = prediction_loop(
            model, num_classes, device, val_loader, criterion=criterion, 
            histone_list=None, dat_augment=dat_aug)
  
        if num_classes == 2 or num_classes == 3:
            val_mAP = np.mean(val_ap[1:])
        elif num_classes == 5:
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
    except ValueError:
        print('Model evaluation failed. Skip.')

# Evaluate on the test set.
model.load_state_dict(torch.load(best_model_path))

avg_test_loss, test_ap, test_preds = prediction_loop(
    model, num_classes, device, test_loader, criterion=criterion, 
    histone_list=None, dat_augment=dat_aug, 
    return_preds=True)

truevals, predictions, probs = test_preds

test_mAP = mAP_conf_interval(truevals, probs, num_classes=num_classes, bs_samples=3000)


if num_classes == 2:
        lb = LabelBinarizer()
        binvals = lb.fit_transform(truevals)
        binvals = np.hstack((binvals, 1 - binvals))	

elif num_classes == 3 or num_classes == 5:
	binvals = label_binarize(truevals, classes=list(range(num_classes)))

fpr, tpr, roc_auc, precision, recall, average_precision = get_statistics(binvals, probs, n_classes=num_classes)

precision_recall = compute_precision(truevals, predictions)
precision_val, recall_val  = precision_recall['precision'], precision_recall['recall']


def _test_set_summary(fh):
    '''Print summary info on the test set
    '''
    print('='*10, 'On test set', '='*10, file=fh)
    print('avg test loss={:.3f} and mAP={:.3f}, 95% CI [{:.3f},{:.3f}]'.format(
        avg_test_loss, test_mAP[0], test_mAP[1], test_mAP[2]), file=fh)


    if num_classes == 2:
        print('AP for each class: Background={:.3f},  Enhancer={:.3f}, '.format(
                test_ap[0], test_ap[1]), 
              file=fh
             )   

        print('Precision for each class: Background={:.3f}, Enhancer={:.3f}'.format(
                precision_val[0], precision_val[1]), 
              file=fh)

        print('Recall for each class: Background={:.3f}, Enhancer={:.3f}'.format(
                recall_val[0], recall_val[1]), 
              file=fh)


    elif num_classes == 3:

        print('AP for each class: Background={:.3f}, TSS={:.3f}, Enhancer={:.3f}, '.format(
                test_ap[0], test_ap[1], test_ap[2]), 
              file=fh
             )   

        print('Precision for each class: Background={:.3f}, TSS={:.3f}, Enhancer={:.3f}'.format(
                precision_val[0], precision_val[1], precision_val[2]), 
              file=fh)

        print('Recall for each class: Background={:.3f}, TSS={:.3f}, Enhancer={:.3f}'.format(
                recall_val[0], recall_val[1], recall_val[2]), 
              file=fh)


    elif num_classes == 5:

        print('AP for each class: poised enh={:.3f}, active enh={:.3f}, '
              'poised tss={:.3f}, active tss={:.3f}'.format(
                test_ap[0], test_ap[1], test_ap[2], test_ap[3]),

              file=fh
             )

        print('Precision for each class: PE={:.3f}, AE={:.3f}, PT={:.3f}, AT={:.3f}, Bgd={:.3f}'.format(
                precision_val[0], precision_val[1], precision_val[2], precision_val[3], precision_val[4]), 
              file=fh)

        print('Recall for each class: PE={:.3f}, AE={:.3f}, PT={:.3f}, AT={:.3f}, Bgd={:.3f}'.format(
                recall_val[0], recall_val[1], recall_val[2], recall_val[3], recall_val[4]), 
              file=fh)


_test_set_summary(sys.stdout)
with open(os.path.join(output_folder, summary_out_name), 'w') as fh:
    _test_set_summary(fh)

# Output figures and other stats.
# confusion matrix.
m = confusion_matrix(truevals, predictions)
cm = plot_confusion_matrix(m, norm=False, n_classes=num_classes)
plt.savefig(os.path.join(output_folder, confus_mat_name + '.png'))
cm.to_csv(os.path.join(output_folder, confus_mat_name + '.csv'))


# precision-recall curve
pr_curve = plot_pr(precision, recall, average_precision, num_classes)
plt.savefig(os.path.join(output_folder, precision_recall_curve_name + '.png'))

# test set predictions.
df = np.stack([truevals, predictions], axis=1)
df = np.concatenate([df, probs], axis=1)

if num_classes == 2:
    col_names = ['label', 'pred', 'Background', 'Enhancer']

elif num_classes == 3:
    col_names = ['label', 'pred', 'Background', 'TSS', 'Enhancer']

elif num_classes == 5:
    col_names = ['label', 'pred', 'poised_enh', 'active_enh', 
             'poised_tss', 'active_tss', 'background']

df = pd.DataFrame(df, columns=col_names).round(3)
df = df.astype({'label': 'int', 'pred': 'int'})
df.to_csv(os.path.join(output_folder, pred_out_name), index=False)


