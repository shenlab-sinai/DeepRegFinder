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
            #unclear in paper between which layers they do dropout so I just did it here
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
    
class SeqNet(nn.Module):
    def __init__(self, ):
        super(SeqNet, self).__init__()
        self.layer_one = nn.Sequential(
            nn.Conv1d(4, 32, 8, stride=1), #in channels, out channels, kernel size
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2) #kernel size
            #2000-->1998
        )
        self.layer_two = nn.Sequential(
            nn.Conv1d(32, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
            #9-->6
        )

        self.layer_three = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
            #6-->3
        )
        self.layer_four = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
            #3-->1
        )

        self.final_layer = nn.Sequential( 
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(64, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, sequence_forward, sequence_reverse=None, 
                sequence_complement=None, sequence_complement_reverse=None):
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

        # forward sequence data.
        o = _forward_prop(sequence_forward)
        # reverse and reverse-complement sequence data.
        if sequence_reverse is not None:
            o2 = _forward_prop(sequence_reverse)
            o3 = _forward_prop(sequence_complement)
            o4 = _forward_prop(sequence_complement_reverse)
            o = torch.add(o, o2)
            o = torch.add(o, o3)
            o = torch.add(o, o4)
            o = torch.div(o, 4)
        return torch.squeeze(torch.log(o))


class ConvNet(nn.Module):
    def __init__(self, marks=7, nb_cls=5):
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

class STN(nn.Module):
    def __init__(self, device, isz=20, tsz=None, marks=7, nb_cls=4, use_bn=True):
        '''Create a spatial transformer network
        Args:
            isz ([2-tuple]): input image size.
            tsz ([2-tuple]): target grid size. Default is 
                input image size (None).
            reduce_img ([bool]): reduce image size by half?
            use_bn ([bool]): use batch normalization?
        '''
        super(STN, self).__init__()
        self.device = device
        self.tsz = tsz
        self.use_bn = use_bn
        
        # classification network.
        self.classification = nn.Sequential(
            # conv 1.
            nn.Conv1d(marks, 32, 3, padding=1), #in cha, out cha, kernel size
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            # conv 2.
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2),
            # conv 3.
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            # conv 4.
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2),
            # global avg pool.
            nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(64, nb_cls, 1),
#             nn.Softmax(dim=1),            
        )
        
        self.clf_out = nn.Sequential(
            nn.Linear(64, nb_cls),
            nn.Softmax(dim=1),
        )
        self.clf_out[0].weight.data.zero_()
        self.clf_out[0].bias.data.copy_(
            torch.tensor([np.log(1/23.39), np.log(1/26.67), 
                          np.log(1/5.42), np.log(1/1.36)], 
                         dtype=torch.float))
       

        # Spatial transformer localization-network
        # localization net output size.
        self.localization = nn.Sequential(
            nn.Conv1d(marks, 15, kernel_size = 5),
            nn.BatchNorm1d(15),
            nn.MaxPool1d(2, stride = 2),
            nn.LeakyReLU(),
            nn.Conv1d(15, 30, kernel_size = 3),
            nn.BatchNorm1d(30),
            #nn.MaxPool1d(2, stride = 2),
            nn.LeakyReLU()
        )
        losz = (isz - 4)//2 - 2
        self.locnet_out_size = losz*30

        # Regressor for the 3 * 2 affine matrix
        fc_loc_layers = []
        fc_loc_layers.append(nn.Linear(self.locnet_out_size, 30))        
        if self.use_bn:
            fc_loc_layers.append(nn.BatchNorm1d(30))
        fc_loc_layers.append(nn.LeakyReLU())
#         fc_loc_layers.append(nn.Linear(30, 30))
#         if self.use_bn:
#             fc_loc_layers.append(nn.BatchNorm1d(30))
#         fc_loc_layers.append(nn.ReLU(True))
#         fc_loc_layers.append(nn.Linear(20, 3*2))
        fc_loc_layers.append(nn.Linear(30, 2))
        self.fc_loc = nn.Sequential(*fc_loc_layers)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        # scale and shift.
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn_theta(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.locnet_out_size)
        two_theta = self.fc_loc(xs)
        
#         flip_prob = two_theta[:, 2]
#         flip_prob_indices = flip_prob >= 0.5
#         flip_val = torch.ones([two_theta.shape[0]], dtype=torch.float, device=self.device)
#         flip_val[flip_prob_indices] = -1

        
        theta = torch.zeros([two_theta.shape[0], 2, 3], dtype=torch.float, device=self.device)
        theta[:, 0, 0] = 1
        #theta[:, 1, 1] = flip_val * two_theta[:, 0]  # scale.
        theta[:, 1, 1] = two_theta[:, 0]  # scale.
        theta[:, 1, 2] = two_theta[:, 1]  # shift.
        return theta
    
    def stn_grid(self, x, theta):
        tsz = x.size() if self.tsz is None else (self.tsz, 1)
        grid = F.affine_grid(theta, tsz)
        return grid
    
    def stn_sample(self, x, grid):
        x = F.grid_sample(x, grid)
        return x

    def forward(self, histone_forward, histone_reverse=None):
        # transform the input
        theta = self.stn_theta(histone_forward)
        histone_forward = histone_forward.unsqueeze(-1)
        grid = self.stn_grid(histone_forward, theta)
        transformed = self.stn_sample(histone_forward, grid)
        transformed = torch.squeeze(transformed, -1)        
        # forward histone data.
        o = self.classification(transformed)
        o = o.view(-1, 64)
        o = self.clf_out(o)
        # reverse histone data.
#         if histone_reverse is not None:
#             o2 = self.classification(histone_reverse)
#             o = torch.add(o, o2)
#             o = torch.div(o, 2)

        return torch.log(o), (theta, transformed, histone_forward)

class Combined_ConvNet(nn.Module):
    '''Histone-Sequence combined model
    '''
    def __init__(self, histone_model, sequence_model, 
                 fc_size=(400, 400), nb_cls=5):
        '''
        Args:
            histone_model (model): pretrained histone mark model.
            sequence_model (model): pretrained DNA sequence model.
            fc_size ([tuple]): size of the two FC layers.
        '''
        super(Combined_ConvNet, self).__init__()
        self.histone_model = histone_model
        self.sequence_model = sequence_model
        self.fc1_layer = nn.Sequential(
            nn.Linear(384, fc_size[0]),
            nn.BatchNorm1d(fc_size[0]),
            nn.LeakyReLU()
        )
        self.fc2_layer = nn.Sequential(
            nn.Linear(fc_size[0], fc_size[1]),
            nn.BatchNorm1d(fc_size[1]),
            nn.LeakyReLU()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(fc_size[1], nb_cls),
            #nn.Linear(384, 4),
            nn.LogSoftmax(dim=1)        
        )
        
    def forward(self, histone_forward, histone_reverse, 
                sequence_forward, sequence_reverse, 
                sequence_complement, sequence_complement_reverse):
        '''Forward propagation using 2 histone inputs and 
        4 sequence inputs
        '''
        hfo = torch.squeeze(self.histone_model(histone_forward))
        sfo = torch.squeeze(self.sequence_model(sequence_forward))
        hro = torch.squeeze(self.histone_model(histone_reverse))
        sro = torch.squeeze(self.sequence_model(sequence_reverse))
        scf = torch.squeeze(self.sequence_model(sequence_complement))
        scr = torch.squeeze(self.sequence_model(sequence_complement_reverse))
        out = torch.cat((hfo, hro, sfo, sro, scf, scr), dim=1)
        out = out.view(out.shape[0], -1) # flatten for FC layer
        out = self.fc1_layer(out)
        out = self.fc2_layer(out)
        out = self.final_layer(out)
        return out

def get_last_pooling_layer(model):
    return nn.Sequential(*list(model.children())[:-1], 
                         *list(model.children())[-1][:-2])








