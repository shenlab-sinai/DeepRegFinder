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

"""
Collection of helper functions for validation and accuracy
"""
class NormDataset(Dataset):
    def __init__(self, data, mean, std, norm ):
        self.data = data
        self.mean = mean.view(7, -1)
        self.std = std.view(7, -1)
        self.norm = norm
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample, label = self.data[idx]
        if self.norm:
            sample = torch.div(sample.sub(self.mean), self.std)
        return sample, label


def plot_pr(precision, recall, average_precision):
    """
    Given a dictionary of preciscion and recall values for each class, uses pyplot to plot the PR
    curve for each class as well as the micro-averaged PR curve and iso-f1 curves. 
    """
    colors = itls.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'olive'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    class_lookup = {0: "Poised Enhancer", 1: "Active Enhancer", 2: "TSS", 3: "Background"}
    for i, color in zip(range(4), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(class_lookup[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

def plot_rocs(fpr, tpr, roc_auc):
    """
    Given dictionaries of true positive rate and falce positive rates for each class & roc_auc values for each class,
    plots micro and macro averaged ROC curves as well as the curve for each class. 
    """
    plt.figure()
    lw=2
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['micro']),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['macro']),
             color='navy', linestyle=':', linewidth=4)
    class_lookup = {0: "Poised Enhancer", 1: "Active Enhancer", 2: "TSS", 3: "Background"}

    colors = itls.cycle(['aqua', 'darkorange', 'cornflowerblue', 'olive'])
    for i, color in zip(range(4), colors):
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
    plt.show()

def plot_confusion_matrix(cm, norm, n_classes=4):
    """
    Given a scikit confusion matrix, plots it in a graphically understandable manner. Norm is bool to indicate if the CM
    should be normalized (class-wise) before plotting.
    """
    if norm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes=[]
    if n_classes ==3 :
        classes = ['enhancer', 'tss', 'background']
    else:
        classes = ['enhancer poised', 'enhancer active', 'tss poised', 'tss active', 'background']
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ticks = []
    if n_classes == 3:
        ticks = np.arange(3)
    else:
        ticks = np.arange(n_classes)
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    fmt = '.2f' if norm else 'd'
    for i, j in itls.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i,j] > (cm.max()/2.) else 'black')
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')


def get_statistics(binvals, scores, n_classes=4):
    """
    Given the true labels in binarized form ("binvals") and the model outputs (probability distribution over
    the classes, "scores") over a dataset, calculates:
        -FPR, TPR, precision, recall, average precision, and area under the ROC for each class, as well as micro-averaged
        over the classes
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
    fpr['micro'], tpr['micro'], _ = roc_curve(binvals.ravel(), scores.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    precision['micro'], recall['micro'], _ = precision_recall_curve(binvals.ravel(), scores.ravel())
    average_precision['micro'] = average_precision_score(binvals, scores, average='micro')
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
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

def main_train_loop(model, criterion, optimizer, device,
                    sequence_loader=None, 
                    histone_loader=None, histone_list=None,
                    dat_augment=False,
                    report_iters=300):
    assert(sequence_loader is not None or histone_loader is not None)
    dat_loader = normalize_dat_loader(sequence_loader, histone_loader)
    total_loss = 0.0
    model.train()  # set training state.
    for i, batch_dat in enumerate(dat_loader):
        optimizer.zero_grad()
        dat_dict, label = normalize_dat_dict(
            batch_dat, 
            use_sequence=(sequence_loader is not None), 
            use_histone=(histone_loader is not None), 
            dat_augment=dat_augment, device=device, 
            histone_list=histone_list)

        # forward-backward propagation.
        #outputs = model(**dat_dict)[0]
        outputs = model(**dat_dict)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss
        #scheduler.step()
        if (i > 0) and (i % report_iters == 0): #about every half-epoch
            print("The average training loss for each batch was {0} after {1} batches".format(
            total_loss/i, i))
#         print("The average training loss for each batch was {0}".format(
#             total_loss))

def validation_loop(model, criterion, device, sequence_loader=None, 
                    histone_loader=None, histone_list=None, 
                    dat_augment=False, nb_cls=4):
    assert(sequence_loader is not None or histone_loader is not None)
    dat_loader = normalize_dat_loader(sequence_loader, histone_loader)
    model.eval()  # set evaluation state.
    with torch.no_grad():
        p, t, s = [], [], []
        total_loss = 0.0
        for batch_dat in dat_loader:
            dat_dict, label = normalize_dat_dict(
                batch_dat, 
                use_sequence=(sequence_loader is not None), 
                use_histone=(histone_loader is not None), 
                dat_augment=dat_augment, device=device, 
                histone_list=histone_list)
            # scoring.
            #pscores, transform = model(**dat_dict)
            pscores = model(**dat_dict)
            loss = criterion(pscores, label)
            total_loss += loss
            _, preds = torch.max(pscores.data, 1)
            ground_truth = label.cpu().numpy()
            # accumulate results.
            p.append(preds.cpu().numpy())
            s.append(pscores.cpu().numpy())
            t.append(ground_truth)
        print("The average validation loss for each batch was {}".format(total_loss/len(p)))
        predictions = np.concatenate(p)
        truevals = np.concatenate(t)
        binvals = label_binarize(truevals, classes=list(range(nb_cls)))
        scores = np.concatenate(s)
        return truevals, predictions, binvals, scores
        #return truevals, predictions, binvals, scores, transform[0], transform[1], dat_dict

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
