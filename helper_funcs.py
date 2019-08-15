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


