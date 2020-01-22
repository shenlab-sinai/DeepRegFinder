import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

__all__ = ['KimNet', 'ConvNet', 'RecurNet', 'init_weights']

"""
Neural network models are defined here
"""
class KimNet(nn.Module):
    """
    An implementation of the histone-mark based NN model from Kim: EP-DNN (2016)
    Network is MLP, with a input-600-500-400-output architecture.
    """
    def __init__(self, bins=20, marks=3, nb_cls=5):
        super(KimNet, self).__init__()
        self.bins = bins
        self.marks = marks
        self.model = nn.Sequential(
            nn.Linear(bins*marks, 600),
            nn.Softplus(),
            nn.Linear(600, 500),
            nn.Softplus(),
            nn.Linear(500, 400),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(400, nb_cls),
            nn.Softmax(dim=1)
        )

    def forward(self, histone_forward, histone_reverse=None):
        o = self.model(histone_forward.view((-1, self.bins*self.marks)))
        if histone_reverse is not None:
            o2 = self.model(histone_reverse.view((-1, self.bins*self.marks)))
            o = torch.add(o, o2)
            o = torch.div(o, 2)
        return torch.log(o)


class ConvNet(nn.Module):
    """
    DeepRegFinder - 1D convolutional neural net
    """
    def __init__(self, marks=3, nb_cls=5, use_leakyrelu=False):
        assert nb_cls > 1, 'output layer size must be at least 2.'
        super(ConvNet, self).__init__()

        self.layer_one = nn.Sequential(
            #in channels, out channels, kernel size
            nn.Conv1d(marks, 32, 7, padding=3), 
            nn.BatchNorm1d(32),
            nn.LeakyReLU() if use_leakyrelu else nn.ReLU(),
        )
        self.layer_two = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU() if use_leakyrelu else nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer_three = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU() if use_leakyrelu else nn.ReLU(),
        )
        self.layer_four = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU() if use_leakyrelu else nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        self.final_layer = nn.Sequential( 
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(64, nb_cls, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, histone_forward, histone_reverse=None):
        def _forward_prop(x):
            '''Forward an input through all layers and return 
            an output
            '''
            o = self.layer_one(x)
            o = self.layer_two(o)
            o = self.layer_three(o)
            o = self.layer_four(o)
            o = self.final_layer(o)
            return o
        
        # forward histone data.
        o = _forward_prop(histone_forward)
        # reverse histone data.
        if histone_reverse is not None:
            o2 = _forward_prop(histone_reverse)
            o = torch.add(o, o2)
            o = torch.div(o, 2)

        return torch.squeeze(torch.log(o))


class RecurNet(nn.Module):
    """
    DeepRegFinder - recurrent neural net
    """
    def __init__(self, marks=3, nb_cls=5, add_conv=False, bidirectional=False):
        super(RecurNet, self).__init__()
        if add_conv:
            self.conv_layer = nn.Sequential(
                #in channels, out channels, kernel size
                nn.Conv1d(marks, 32, 7, padding=3), 
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            lstm_in_size = 32
        else:
            lstm_in_size = marks
        self.add_conv = add_conv
        self.rnn = nn.LSTM(lstm_in_size, 32, num_layers=2, dropout=0.2, 
                           batch_first=True, bidirectional=bidirectional)
        self.hidden2clf = nn.Sequential(
            nn.Linear(32*(2 if bidirectional else 1), nb_cls),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, histone_forward):
        if self.add_conv:
            histone_forward = self.conv_layer(histone_forward)
        # histone_forward: (batch, mark, bin)
        # input of shape (batch, seq, feature).
        histone_forward = histone_forward.transpose(1, 2)
        # output of shape (batch, seq_len, num_directions * hidden_size)
        hidden, _ = self.rnn(histone_forward)
        o = self.hidden2clf(hidden[:, -1, :])
        return o


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.zeros_(m.bias.data)


def create_model(net_choice, num_marks=3, num_classes=5, num_bins=20, 
                 conv_rnn=False, device=None):
    device = torch.device('cpu') if device is None else device
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
    return model


class EMAModelWeights():
    def __init__(self, model, decay=0.99):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay)*param.data + \
                    self.decay*self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def load_shadow(self, shadow):
        self.shadow = deepcopy(shadow)






