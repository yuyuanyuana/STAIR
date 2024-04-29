import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


from STAIR.embedding.module_ae import ae_dict
from STAIR.embedding.module_hgat import HGAT
from STAIR.embedding.dataset_ae import MyDataset
from STAIR.embedding.dataset_hgat import hgat_data
from STAIR.embedding.loss import nll_loss

from STAIR.utils import MakeLogClass


class Emb_Align(object):

    """
    
    Spatial embedding alignment of multiplt ST slices, including CE alignment and SE alignment. 
    They perform embedding alignment using AE module and HGAT module, respectively.

    Parameters
    ----------
    adata
        AnnData object object of scanpy package
    batch_key
        The key containing slice information in .obs
    hvg
        Slice order used to perform alignment. Align according to the default order of elements in batch_key if None
    n_hidden
        The number of hidden dimension in the AE module
    n_latent
        The number of latent dimension in the AE module
    dropout_rate
        Dropout rate
    likelihood
        Distribution assumptions for expression matrices, 'nb' or 'zinb'
    
    Examples
    --------
    >>> adata = sc.read_h5ad(path_to_anndata)
    >>> emb_align = Emb_Align(adata, batch_key='batch', result_path=result_path)
    >>> emb_align.prepare()
    >>> emb_align.preprocess()
    >>> emb_align.latent()
    >>> emb_align.prepare_hgat(batch_order=keys_use)
    >>> emb_align.train_hgat()
    >>> adata, atte = emb_align.predict_hgat()

    """
    
    def __init__(
        self,
        adata,
        batch_key = None,
        hvg = False,
        n_hidden: int = 128,
        n_latent: int = 32,
        dropout_rate: float = 0.2,
        likelihood: str = "nb",
        device: str = None, 
        num_workers: int = 4,
        result_path = None, 
        make_log: bool = True
    ):
        super(Emb_Align, self).__init__() 

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.adata = adata
        self.n_batch = None if batch_key is None else len(set(adata.obs[batch_key]))
        self.batch_key = batch_key
        
        if hvg:
            self.n_input = hvg
        else:
            self.n_input = adata.shape[1]
        
        self.hvg = hvg
        self.n_latent = n_latent
        self.likelihood = likelihood
        
        self.ae = ae_dict[self.likelihood]( input_dim = self.n_input, 
                                            hidden_dim = n_hidden, 
                                            latent_dim = n_latent, 
                                            dropout = dropout_rate,
                                            n_batch = self.n_batch 
                                            ).to(self.device)        
        
        self.num_workers = num_workers
        self.result_path = result_path
        self.make_log = make_log
        
        if self.make_log:
            self.makeLog = MakeLogClass(f"{self.result_path}/log_emb.tsv").make
            self.makeLog(f"Module parameter set of AE")
            self.makeLog(f"  Likelihood: {self.likelihood}")
            self.makeLog(f"  Input dim: {self.n_input}")
            self.makeLog(f"  Latent Dim: {self.n_latent}")
            self.makeLog(f"  Dropout: {dropout_rate}")


    def prepare(self, count_key=None, lib_size='explog', normalize=True, scale=False):
       
        '''
        Prepare data for analysis.
        
        Parameters
        ----------
        count_key
            Layer to normalize instead of X. If None, X is normalized.
        lib_size
            The way to calculate the library size for each cell.
        normalize
            Whether to normalize the expression, default is True.
        scale
            Whether to scale the expression, default is False.
        
        '''
        self.data_ae = MyDataset(self.adata, 
                                 count_key = count_key,
                                 size = lib_size, 
                                 normalize = normalize, 
                                 hvg = self.hvg, 
                                 scale = scale, 
                                 batch_key = self.batch_key)
        
        if self.hvg:
            self.adata = self.adata[:,:self.hvg]
        
        if self.make_log:
            self.makeLog(f"  Library size: {lib_size}")
            self.makeLog(f"  Input normalize: {normalize}")
            self.makeLog(f"  Input scale: {scale}")
            self.makeLog(f"  Hvg: {self.hvg}")

    def preprocess( self,
                    lr = 0.001, 
                    weight_decay = 0, 
                    epoch_ae = 100, 
                    batch_size = 64,
                    plot = False):
        '''
        Preprocessed training part using AE module
        
        Parameters
        ----------
        weight_decay
            Weight_decay of training, default is 0.
        epoch_ae
            Total epoch of training, default is 40.
        batch_size
            The number of samples used for each gradient update in mini-batch training, default is 128.
        plot
            Whether to plot the loss in each epoch, default is False.
        '''
        if self.make_log:
            self.makeLog(f"Training parameter set of AE")
            self.makeLog(f"  Learning rate: {lr}")
            self.makeLog(f"  Weight decay: {weight_decay}")
            self.makeLog(f"  Training epoch: {epoch_ae}")
            self.makeLog(f"  Batch size: {batch_size}")
            
        data_loader = DataLoader(self.data_ae, shuffle=True, batch_size=batch_size, num_workers=self.num_workers)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=weight_decay)   

        train_loss = []
        for epoch in tqdm(range(0, epoch_ae)):
            loss_tmp = 0
            if self.batch_key is not None:
                for i, (feat_tmp, count_tmp, size_tmp, batch_tmp) in enumerate(data_loader):
                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    size_tmp = size_tmp.to(self.device)
                    batch_tmp = batch_tmp.to(self.device)
                    self.ae.train()
                    rate_scaled_tmp, logits_tmp, drop_tmp, _ = self.ae(feat_tmp, batch_tmp)
                    rate_tmp = rate_scaled_tmp*size_tmp
                    mean_tmp = rate_tmp*logits_tmp
                    optimizer.zero_grad()
                    loss_train = nll_loss(count_tmp, mean_tmp, rate_tmp, drop_tmp, dist=self.likelihood).mean()
                    loss_train.backward()
                    optimizer.step()
                    # if i%5 == 0:
                        # print("AE Epoch:{},  loss {}".format(epoch,loss_train.item()))
                    loss_tmp += loss_train.item()
                train_loss.append(loss_tmp/len(data_loader))
            else:
                for i, (feat_tmp, count_tmp, size_tmp) in enumerate(data_loader):
                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    size_tmp = size_tmp.to(self.device)
                    self.ae.train()
                    rate_scaled_tmp, logits_tmp, drop_tmp, _ = self.ae(feat_tmp)
                    rate_tmp = rate_scaled_tmp*size_tmp
                    mean_tmp = rate_tmp*logits_tmp
                    optimizer.zero_grad()
                    loss_train = nll_loss(count_tmp, mean_tmp, rate_tmp, drop_tmp, dist=self.likelihood).mean()
                    loss_train.backward()
                    optimizer.step()
                    # if i%5 == 0:
                        # print("AE Epoch:{},  loss {}".format(epoch,loss_train.item()))
                    loss_tmp += loss_train.item()
                train_loss.append(loss_tmp/len(data_loader))

        self.ae.eval()

        # torch.save(self.ae.state_dict(),f'{self.result_path}/embedding/train/model_pre.pth')

        # if plot:
        #     plt.plot(train_loss)
        #     plt.savefig(f'{self.result_path}/embedding/train/loss_pre.png')
        #     plt.close()

    def latent( self,
                batch_size = None,
                return_data = False
              ):
        '''
        Preprocessed predicting part using AE module
        
        Parameters
        ----------
        batch_size
            Batch size in predicting part.
        return_data
            Whether to return adata, default is False.
        '''
        self.ae.eval()
        
        if batch_size is None:
            batch_size = len(self.data_ae)
        
        dataloader = DataLoader(self.data_ae, shuffle=False, batch_size=batch_size, num_workers=self.num_workers)

        z = torch.empty(size=[0,self.n_latent])
        mean = torch.empty(size=[0,self.n_input])

        with torch.no_grad():
            
            if self.batch_key is not None:                
                for _, (feat_tmp, count_tmp, lib_tmp, batch_tmp) in enumerate(dataloader):
                    # print(i)
                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    lib_tmp = lib_tmp.to(self.device)
                    batch_tmp = batch_tmp.to(self.device)
                    rate_scaled_tmp, logits_tmp, _, z_tmp = self.ae(feat_tmp, batch_tmp)
                    z = torch.cat([z, z_tmp.cpu()[:,:self.n_latent]])
                    rate_tmp = rate_scaled_tmp*lib_tmp
                    mean_tmp = rate_tmp*logits_tmp
                    mean = torch.cat([mean, mean_tmp.cpu()])
            else:                
                for _, (feat_tmp, count_tmp, lib_tmp) in enumerate(dataloader):
                    # print(i)
                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    lib_tmp = lib_tmp.to(self.device)
                    rate_scaled_tmp, logits_tmp, _, z_tmp = self.ae(feat_tmp)
                    z = torch.cat([z, z_tmp.cpu()[:,:self.n_latent]])
                    rate_tmp = rate_scaled_tmp*lib_tmp
                    mean_tmp = rate_tmp*logits_tmp
                    mean = torch.cat([mean, mean_tmp.cpu()])

        self.adata.obsm['latent'] = z.detach().cpu().numpy()
        self.adata.layers['Denoise'] = mean.detach().cpu().numpy()

        if return_data:
            return self.adata
    
    def prepare_hgat(self,
                     slice_key = None,
                     slice_order = None,
                     spatial_key = 'spatial',
                     n_neigh_hom = 10, 
                     c_neigh_het = 0.9,
                     kernal_thresh = 0.):
        '''
        Construct heterogeneous graph for HAT module.

        Parameters
        ----------
        slice_key
            Key of slice information in .obs.
        slice_order
            List with slice names ordered by the physical location.
        spatial_key
            Key of raw spatial location of spots in .obsm.
        n_neigh_hom
            Number of neighbors based on location in the same slice, default is 10.
        c_neigh_het
            Similarity cutoff based on expression latent in the defferent slice, default is 0.9.
        '''
        if slice_key is None:
            slice_key = self.batch_key
        
        self.data_hgat, self.kernals, self.index_dict = hgat_data(  self.adata, 
                                                                    batch_key = slice_key, 
                                                                    batch_order = slice_order, 
                                                                    spatial_key = spatial_key, 
                                                                    n_neigh_hom = n_neigh_hom, 
                                                                    n_radius_het = 1-c_neigh_het,
                                                                    kernal_thresh = kernal_thresh)
        
        if self.make_log:
            self.makeLog(f"Module parameter set of HGAT")
            self.makeLog(f"  Spatial key: {spatial_key}")
            self.makeLog(f"  Neighbor number of intra-slice: {n_neigh_hom}")
            self.makeLog(f"  Similarity cutoff of inter-slice: {c_neigh_het}")
            
    def train_hgat( self,
                    gamma: float = 0.8,
                    epoch_hgat: int = 150,
                    re_weight: float = 1.,
                    si_weight: float = 0.,
                    lr: float = 0.001, 
                    weight_decay: float = 0.,
                    negative_slope: float = 0.2,
                    dropout_hom: float = 0.5,
                    dropout_het: float = 0.5,
                    plot = False):
        '''
        Training step HAT module.

        Parameters
        ----------
        gamma
            Weight of homogeneous representation in SE. U=λ∙U^hom+(1-λ)∙U^het
        epoch_hgat
            Total epoch of training, default is 100.
        dropout_hom
            Dropout rate in aggregating intra-slice information.
        dropout_het
            Dropout rate in aggregating inter-slice information.
        plot
            Whether to plot the loss in each epoch, default is False.
        
        '''
        if self.make_log:
            self.makeLog(f"  Dropout_hom: {dropout_hom}")
            self.makeLog(f"  Dropout_het: {dropout_het}")
            self.makeLog(f"Training parameter set of HGAT")
            self.makeLog(f"  Weight of intra-slice: {gamma}")
            self.makeLog(f"  Reconstruction weight: {re_weight}")
            self.makeLog(f"  Similarity weight: {si_weight}")
            self.makeLog(f"  Training epoch: {epoch_hgat}")
            self.makeLog(f"  Learning rate: {lr}")
            self.makeLog(f"  Weight decay: {weight_decay}")
            self.makeLog(f"  Negative slope: {negative_slope}")
 
        self.hgat = HGAT( num_channels = self.n_latent, 
                          metadata = self.data_hgat.metadata(), 
                          negative_slope = negative_slope,
                          dropout_hom = dropout_hom,
                          dropout_het = dropout_het,
                          gamma = gamma
                         ).to(self.device)  

        self.data_hgat = self.data_hgat.to(self.device)  
        optimizer_hgat = torch.optim.Adam(self.hgat.parameters(), lr=lr, weight_decay=weight_decay)

        loss_list = []
        for epoch in tqdm(range(epoch_hgat)):
            
            self.hgat.train()
            
            out = self.hgat(self.data_hgat.x_dict, 
                            self.data_hgat.edge_index_dict, 
                            return_semantic_attention_weights=False)
            
            re_loss = 0
            si_loss = 0
            
            for node_type, x in out.items():
                re_loss += F.mse_loss(x, self.data_hgat[node_type].x)
                if si_weight != 0:
                    si_loss += F.mse_loss(self.kernals[node_type].to(self.device), torch.mm(x,x.T))
            
            if si_weight != 0:
                loss = re_weight * re_loss + si_weight * si_loss  
                loss_list.append([loss.item(), re_loss.item(), si_loss.item()])
            else:
                loss = re_weight * re_loss
                loss_list.append([loss.item(), re_loss.item()])
            
            # print("HGAT Epoch:{}  loss:{} ".format(epoch, loss.item()))
            
            loss.backward()
            optimizer_hgat.step()
            optimizer_hgat.zero_grad()

        self.hgat.eval()
        
        # if not os.path.exists(self.result_path + '/embedding/train'):
        #     os.makedirs(self.result_path + '/embedding/train')
            
        # torch.save(self.hgat.state_dict(),f'{self.result_path}/embedding/train/model_hgat.pth')

        if plot:
            if si_weight != 0:
                ls = plt.plot(loss_list)
                plt.legend(handles=ls, labels=['loss', 're_loss', 'si_loss'], loc='best')
                plt.savefig(f'{self.result_path}/embedding/train/loss_hgat.png')
                plt.close()   
            else:
                ls = plt.plot(np.array(loss_list)[:,:2].tolist())
                plt.legend(handles=ls, labels=['loss', 're_loss'], loc='best')
                plt.savefig(f'{self.result_path}/embedding/train/loss_hgat.png')
                plt.close()   
           

    def predict_hgat(self, get_attention=False):
        '''
        Predicting step HAT module.

        '''
        with torch.no_grad():
    
            self.hgat.eval()
            if get_attention:
                out, atte, atte_node = self.hgat(self.data_hgat.x_dict,
                                                self.data_hgat.edge_index_dict, 
                                                return_semantic_attention_weights=True,
                                                get_attention=True)
            else:
                out, atte = self.hgat(self.data_hgat.x_dict,
                                        self.data_hgat.edge_index_dict, 
                                        return_semantic_attention_weights=True)
                    
        # HAN_SE
        out = [pd.DataFrame(values.detach().cpu().numpy(), 
                            index=self.index_dict[ii]) for ii, values in out.items()]
        
        out = pd.concat(out)
        self.adata.obsm['STAIR'] = out.loc[self.adata.obs_names].values

        # Attention between slices
        # i = 0
        # atte_ = []
        # for _, value_ in atte.items():
        #     value_ = value_.cpu().detach()
        #     value_ = value_.unsqueeze(0)
        #     i += 1
        #     atte_.append(value_)
        
        i = 0
        atte_ = []
        for _, value_ in atte.items():
            value_ = value_.cpu().detach()
            value_ = torch.concat([value_[:i], torch.tensor([1.0]), value_[i:]])
            value_ = value_.unsqueeze(0)
            i += 1
            atte_.append(value_)

        atte_ = torch.concat(atte_)
        atte_ = pd.DataFrame(atte_, index=atte.keys(), columns=atte.keys())
    
        if get_attention:
            return self.adata, atte_, atte_node
        return self.adata, atte_
    
    
    