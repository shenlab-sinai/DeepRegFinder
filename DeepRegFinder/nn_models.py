import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
"""
Neural network models are defined here
"""

class KimNet(nn.Module):
    """
    An implementation of the histone-mark based NN model from Kim: EP-DNN (2016)
    Architecture is entirely feed-forward, with a 80-600-500-400-1 architecture.
    """
    def __init__(self, batchSz=100):
        super(KimNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(20, 600),
            nn.Softplus(),
            nn.Linear(600, 500),
            nn.Softplus(),
            #unclear in paper between which layers they do dropout so I 
            # just did it here
            nn.Dropout(),
            nn.Linear(500, 400),
            nn.Softplus(),
            nn.Linear(400, 4)
        )

    #def forward(self, x):
    def forward(self, histone_forward, histone_reverse=None):
        #print(histone_forward.shape)
        o = self.model(histone_forward)
        return torch.squeeze(o)


class ConvNet(nn.Module):
    def __init__(self, marks=3, nb_cls=5):
        super(ConvNet, self).__init__()
        self.layer_one = nn.Sequential(
            nn.Conv1d(marks, 32, 3, padding=1), #in channels, out channels, kernel size
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            #nn.MaxPool1d(2, stride=2) #kernel size
            #20-->9
        )
        self.layer_two = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
            #9-->6
        )
        self.layer_three = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
           # nn.MaxPool1d(2, stride=1)
            #6-->3
        )
        self.layer_four = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
            #3-->1
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


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        if m.weight.dim() >= 2:
            nn.init.kaiming_uniform_(
                m.weight.data, nonlinearity='leaky_relu', a=.01)
        if m.bias.dim() >= 2:
            nn.init.kaiming_uniform_(
                m.bias.data, nonlinearity='leaky_relu', a=.01)





