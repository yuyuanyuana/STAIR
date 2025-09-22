import torch
import torch.nn.functional as F
from torch import nn



# ============================================================================ #
# activation function

exp_act = lambda x: torch.exp(x)

def acti_fun(activate):
    if activate == 'relu':
        return F.relu
    elif activate == 'sigmoid':
        return torch.sigmoid
    elif activate == 'exp':
        return exp_act
    elif activate == 'softplus':
        return F.softplus
    elif activate =='tanh':
        return F.tanh


# ============================================================================ #
# layer

class FC_Layer(nn.Module):
    
    def __init__(self, in_features, out_features, bn=False, activate='relu', dropout=0.0):
        
        super(FC_Layer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.bn = bn
        self.activate = activate
        self.dropout = dropout
        
        self.layer = nn.Linear(in_features, out_features)
        self.bn_layer = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        
        x = self.layer(x)
        if self.bn:
            x = self.bn_layer(x)
        
        if self.dropout!=0:
            return F.dropout(acti_fun(self.activate)(x), p=self.dropout, training=self.training)
        return acti_fun(self.activate)(x)       
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# ============================================================================ #
# NB & ZINB model


class New_NB_AE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_batch=None, dropout=0.2):
        
        super(New_NB_AE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.dropout = dropout
        self.n_batch = n_batch
        
        self.input_dim_all = input_dim + n_batch if n_batch is not None else input_dim
        self.latent_dim_all = latent_dim + n_batch if n_batch is not None else latent_dim
        
        self.layer1 = FC_Layer(self.input_dim_all, self.hidden_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer2 = FC_Layer(self.hidden_dim, self.latent_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer3 = FC_Layer(self.latent_dim_all, self.hidden_dim, bn=False, activate='relu')
        self.layer4 = FC_Layer(self.hidden_dim, self.input_dim, bn=False, activate='exp')
        
        if self.n_batch is not None:
            self.layer_logi = torch.nn.Parameter(torch.randn(self.input_dim, self.n_batch))
        else:
            self.layer_logi = torch.nn.Parameter(torch.randn(self.input_dim))
    
    def forward(self, x, batch_tensor=None): 
        
        if self.n_batch is not None:
            assert batch_tensor.shape[1] == self.n_batch
            x = torch.cat((x, batch_tensor), dim=-1)
        z = self.layer2(self.layer1(x))
        
        if self.n_batch is not None:
            z = torch.cat((z, batch_tensor), dim=-1)  
        rate_scaled = self.layer4(self.layer3(z))
        
        if self.n_batch is not None:
            self.logi = F.linear(batch_tensor, self.layer_logi)
        else:
            self.logi = self.layer_logi
        
        return rate_scaled, self.logi.exp(), None, z
    


class New_ZINB_AE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_batch=None, dropout=0.2):
        
        super(New_ZINB_AE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.dropout = dropout
        self.n_batch = n_batch
        
        self.input_dim_all = input_dim + n_batch if n_batch is not None else input_dim
        self.latent_dim_all = latent_dim + n_batch if n_batch is not None else latent_dim
        
        self.layer1 = FC_Layer(self.input_dim_all, self.hidden_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer2 = FC_Layer(self.hidden_dim, self.latent_dim, bn=True, activate='relu', dropout=self.dropout)
        self.layer3 = FC_Layer(self.latent_dim_all, self.hidden_dim, bn=False, activate='relu')
        self.layer_disp = FC_Layer(self.hidden_dim, self.input_dim, bn=False, activate='exp')
        self.layer_drop = FC_Layer(self.hidden_dim, self.input_dim, bn=False, activate='sigmoid')
        
        if self.n_batch is not None:
            self.layer_logi = torch.nn.Parameter(torch.randn(self.input_dim, self.n_batch))
        else:
            self.layer_logi = torch.nn.Parameter(torch.randn(self.input_dim))
    
    def forward(self, x, batch_tensor=None): 
        
        if self.n_batch is not None:
            assert batch_tensor.shape[1] == self.n_batch
            x = torch.cat((x, batch_tensor), dim=-1)
        z = self.layer2(self.layer1(x))
        
        if self.n_batch is not None:
            z = torch.cat((z, batch_tensor), dim=-1)       
        x3 = self.layer3(z)
        
        rate_scaled = self.layer_disp(x3)
        dropout = self.layer_drop(x3)
        if self.n_batch is not None:
            self.logi = F.linear(batch_tensor, self.layer_logi)
        else:
            self.logi = self.layer_logi
        
        return rate_scaled, self.logi.exp(), dropout, z
    

ae_dict = {'zinb':New_ZINB_AE, 'nb':New_NB_AE}


