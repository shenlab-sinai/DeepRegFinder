from sklearn.metrics import precision_recall_fscore_support
import itertools as itls
from scipy import interp 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_curve, auc
)
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
# np.seterr(divide='ignore', invalid='ignore')
"""
Collection of helper functions for validation and accuracy
"""
__all__ = ['plot_confusion_matrix', 'normalize_dat_dict', 'train_loop', 'compute_precision',
           'prediction_loop', 'mAP_conf_interval', 'get_statistics', 'plot_pr', 'plot_rocs']

def plot_pr(precision, recall, average_precision, n_classes):

    """ 
    Given a dictionary of preciscion 
    and recall values for each class, uses 
    pyplot to plot the PR curve for each class
    """

    plt.figure(figsize=(7, 8)) 
    lines = []
    labels = []

    if n_classes == 2:
        class_lookup = {0: "Background", 1: "Enhancer"}
        colors = itls.cycle(['navy', 'turquoise', 'darkorange'])

    elif n_classes == 3:
        class_lookup = {0: "Background", 1: "TSS", 2: "Enhancer"}
        colors = itls.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue'])    

    elif n_classes == 5:
        class_lookup = {0: "PE", 1: "AE", 2: "PT", 3: "AT", 4: "Bgd"}
        colors = itls.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue',
                         'teal', 'olive'])

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for {0} (area = {1:0.2f})'
                      ''.format(class_lookup[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


def plot_rocs(fpr, tpr, roc_auc, n_classes):
    """
    Given dictionaries of true positive rate and false positive rates for each 
    class & roc_auc values for each class, plots micro and macro averaged ROC 
    curves as well as the curve for each class. 
    """
    plt.figure()
    lw=2

    if n_classes == 2:
        class_lookup = {0: "Background", 1: "Enhancer"}
        colors = itls.cycle(['navy', 'turquoise', 'darkorange'])

    elif n_classes == 3:
        class_lookup = {0: "Background", 1: "TSS", 2: "Enhancer"}
        colors = itls.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue'])    

    elif n_classes == 5:
        class_lookup = {0: "PE", 1: "AE", 2: "PT", 3: "AT", 4: "Bgd"}
        colors = itls.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue',
                         'teal', 'olive'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_lookup[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for different classes')
    plt.legend(loc="lower right")


def plot_confusion_matrix(cm, norm=True, n_classes=5):
    """Plot confusion matrix
    """
    cm_ = cm.copy()
    if norm:  # normalize each target class.
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    if n_classes == 3 :
        classes = ['Background', 'TSS', 'Enhancer']
    elif n_classes == 5:
        classes = ['Poised Enhancer', 'Active Enhancer', 
                   'Poised TSS', 'Active TSS', 'Background']
    else:
        classes = [ str(c + 1) for c in range(n_classes)]
    cmap = plt.cm.Blues
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 16})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ticks = np.arange(n_classes)
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    plt.ylim([n_classes - .5, -.5])
    fmt = '.2f' if norm else 'd'
    for i, j in itls.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', 
                 color='white' if cm[i,j] > (cm.max()/2.) else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    df = pd.DataFrame(cm_, columns=classes, index=classes)
    return df

def compute_precision(label, score):
        """Calculate the precision and recall values for each class
        """
        precision, recall, fscore, support = precision_recall_fscore_support(label, score)
        results = {'precision': precision, 'recall': recall}
        return(results)

def get_statistics(binvals, scores, n_classes=4):
    """
    Given the true labels in binarized form ("binvals") and the model outputs 
    (probability distribution over the classes, "scores") over a dataset, calculates:
        -FPR, TPR, precision, recall, average precision, and area under the 
         ROC for each class, as well as micro-averaged over the classes
        -Macro averaged TPR, FPR, and roc_auc
    Returns the above as dictionaries for each statistic:
    fpr, tpr, roc_auc, precision, recall, avg_precision 
    """
    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, average_precision = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binvals[:, i], scores[:, i])
        precision[i], recall[i], _ = precision_recall_curve(binvals[:,i], scores[:,i])
        average_precision[i] = average_precision_score(binvals[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    return fpr, tpr, roc_auc, precision, recall, average_precision

"""
Model training functions
"""
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


def train_loop(model, num_classes, criterion, optimizer, scheduler, device, train_loss, 
               best_mAP, epoch, check_iters, train_loader, val_loader, 
               best_model_path, checkpoint_path, histone_list=None, 
               dat_augment=False, writer=None):
    '''Model training for an epoch
    '''
    start_iter = epoch*len(train_loader)
    model.train()  # set training state.
    for i, batch_dat in enumerate(train_loader):
        optimizer.zero_grad()
        dat_dict, label = normalize_dat_dict(
            batch_dat, use_sequence=False, use_histone=True, 
            dat_augment=dat_augment, device=device, 
            histone_list=histone_list)
        # forward-backward propagation.
        outputs = model(**dat_dict)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        nb_iter = start_iter + i + 1
        if nb_iter % check_iters == 0:
            try: 
                avg_val_loss, val_ap = prediction_loop(
                    model, num_classes, device, val_loader, criterion=criterion, 
                    histone_list=histone_list, dat_augment=dat_augment)

                if num_classes == 5:
                    val_mAP = np.mean(val_ap[:-1])  # ignore background class.
                elif num_classes == 2 or num_classes == 3:
                    val_mAP = np.mean(val_ap[1:])

                avg_train_loss = train_loss/check_iters
                print('Iter={}, avg train loss: {:.3f}; val loss: {:.3f}, '
                      'val mAP: {:.3f}'.format(nb_iter, avg_train_loss, 
                                               avg_val_loss, val_mAP), end='')
                if writer is not None:  # tensorboard logging.
                    writer.add_scalar('Loss/train', avg_train_loss, nb_iter)
                    writer.add_scalar('Loss/val', avg_val_loss, nb_iter)
                    writer.add_scalar('mAP/val', val_mAP, nb_iter)
                torch.save({'model_state_dict': model.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(), 
                            'scheduler_state_dict': scheduler.state_dict()}, 
                           checkpoint_path)  # save model checkpoint.
                if val_mAP > best_mAP:
                    best_mAP = val_mAP
                    torch.save(model.state_dict(), best_model_path)
                    print(' --> best mAP updated; model saved.')
                else:
                    print()
            except ValueError:  # exploding gradients.
                print('Trying to recover model state from a previously '
                      'saved state...', end='')
                try:
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print('loaded.', flush=True)
                except FileNotFoundError as err:
                    print('no previously saved state to load.', flush=True)
                    print('Try to manually reduce learning rate and rerun '
                          'training. Exit now.')
                    raise err
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= .1
                    print('Reduced learning rate to: {}.'.format(
                        param_group['lr']))
            train_loss = .0
            model.train()  # set training state.
    return train_loss, (i + 1) % check_iters, best_mAP


def prediction_loop(model, num_classes, device, dat_loader, pred_only=False, criterion=None, 
                    histone_list=None, dat_augment=False, return_preds=False, 
                    nb_batch=None, show_status=False):
    '''Model validation on an entire val set
    '''
    # assert(sequence_loader is not None or histone_loader is not None)
    # dat_loader = normalize_dat_loader(sequence_loader, histone_loader)
    if not pred_only:
        assert criterion is not None
    model.eval()  # set evaluation state.
    with torch.no_grad():
        t, s, p = [], [], []  # target, score, pred lists.
        others = []  # info other than bin counts.
        total_loss = 0.0
        if show_status:
            print('Start prediction (update for every 100 batches)', flush=True)
        for i, batch in enumerate(dat_loader):
            if pred_only:
                batch_dat, batch_info = batch[0], batch[1:]
                dat_dict = normalize_dat_dict(
                    batch_dat, use_sequence=False, use_histone=True, 
                    dat_augment=dat_augment, device=device, 
                    histone_list=histone_list, has_label=False)
            else:
                dat_dict, label = normalize_dat_dict(
                    batch, use_sequence=False, use_histone=True, 
                    dat_augment=dat_augment, device=device, 
                    histone_list=histone_list, has_label=True)
            # scoring.
            pscores = model(**dat_dict)
            _, preds = torch.max(pscores, 1)
            if not pred_only:
                loss = criterion(pscores, label)
                total_loss += loss.item()
                t.append(label.cpu().numpy())
            else:
                others.append(batch_info)
            # accumulate results.
            p.append(preds.cpu().numpy())
            s.append(pscores.cpu().numpy())
            if show_status and (i + 1) % 100 == 0:
                print('.', end='', flush=True)
            if nb_batch is not None and (i+1) >= nb_batch:
                break
        if show_status:
            print('Done', flush=True)
        predictions = np.concatenate(p)
        scores = np.concatenate(s)
        probs = np.exp(scores) # log(softmax)->prob.
        if not pred_only:
            truevals = np.concatenate(t)

            if num_classes == 2:
                binvals = np.array([[1,0] if l==0 else [0, 1] for l in truevals]) 
            else:
                binvals = label_binarize(truevals, classes=list(range(num_classes)))
            
            try:
                all_cls_ap = average_precision_score(binvals, probs, average=None)
            except ValueError as err:
                print('='*40)
                print('NaN% in model outputs: {:.1f}'.format(
                    np.isnan(probs).mean()*100), file=sys.stderr)
                print('Model blow-up may have happened. Try to reduce '
                      'learning rate or increase weight decay.', 
                      file=sys.stderr)
                # print('='*10, 'System error below', '='*10)
                raise err
            if return_preds:
                return total_loss/len(dat_loader), all_cls_ap, \
                    (truevals, predictions, probs)
            return total_loss/len(dat_loader), all_cls_ap
        else:
            assert len(others) == len(p)
            # transpose: batches x items -> items x batches.
            others = list(map(list, list(zip(*others))))
            info_list = [ np.concatenate(item_list) for item_list in others]
            return predictions, probs, info_list


def mAP_conf_interval(label, score, num_classes=5, bs_samples=1000, 
                      qs=[.025, .975], seed=12345):
    """Calculate the confidence interval of mAP based on bootstrapping.
    bs_samples ([int]): bootstrap sample size.
    qs ([list of ints]): quantiles used for the bootstrap sample.
    seed ([int]): random seed used for bootstrap.
    """
    if num_classes == 2:
        binvals = np.array([[1, 0] if l==0 else [0, 1] for l in label])
        apr = average_precision_score(binvals, score, average=None)
        mAP = np.mean(apr[1:])

    elif num_classes == 3:
        binvals = label_binarize(label, classes=list(range(num_classes)))
        apr = average_precision_score(binvals, score, average=None)
        mAP = np.mean(apr[1:])

    elif num_classes == 5:
        binvals = label_binarize(label, classes=list(range(num_classes)))
        apr = average_precision_score(binvals, score, average=None)
        mAP = np.mean(apr[:-1])
    
    rng = np.random.RandomState(seed)
    mAP_pool = []
    for _ in range(bs_samples):
        ix = rng.choice(range(len(label)), len(label), replace=True)

        if num_classes == 5:
            label_s = label_binarize(label[ix], classes=list(range(num_classes)))
            score_s = score[ix]
            apr_s = average_precision_score(label_s, score_s, average=None)
            mAP_s = np.mean(apr_s[:-1])

        elif num_classes == 3:
            label_s = label_binarize(label[ix], classes=list(range(num_classes)))
            score_s = score[ix]
            apr_s = average_precision_score(label_s, score_s, average=None)
            mAP_s = np.mean(apr_s[1:])

        elif num_classes == 2:
            label_s = np.array([[1,0] if l==0 else [0, 1] for l in label[ix]]) 
            score_s = score[ix]
            apr_s = average_precision_score(label_s, score_s, average=None)
            mAP_s = np.mean(apr_s[1:])

        mAP_pool.append(mAP_s)
    mAP_lower, mAP_upper = np.quantile(mAP_pool, q=qs)
        
    return [mAP, mAP_lower, mAP_upper]



