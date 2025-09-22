import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

def coo_row_sum(data, row, col):
    n_rows = max(row) + 1
    row_sum = [0] * n_rows
    for val, r in zip(data, row):
        row_sum[r] += val
    return row_sum

class MyDataset(Dataset):
    '''
    construct dataset of model1
    input: adata
    output: Dataset with feature, count and library size.
    '''
    def __init__(self, adata, count_key=None, size='explog', normalize=False, hvg=False, scale=False, batch_key=None):
        super(MyDataset,self).__init__()
        
        if count_key is None:
            count_key = 'counts'
            adata.layers['counts'] = adata.X.copy()
            
        # set count matrix
        count = adata.layers[count_key].copy()
        
        if sp.issparse(count):
            data = count.tocoo()
            sum_use = coo_row_sum(data.data, data.row, data.col)
        else:
            sum_use = count.sum(1)

        ### library size vector
        if size == 'explog':
            self.size = torch.from_numpy(np.exp(np.log10(np.array(sum_use)))).unsqueeze(1)  
        elif size == 'sum':
            self.size = torch.from_numpy(np.array(sum_use)).unsqueeze(1)  
        elif size == 'median':
            self.size = torch.from_numpy(sum_use / np.median(sum_use)).unsqueeze(1) 
        adata.obs['size'] = self.size.squeeze(1).numpy()
        
        # if sp.issparse(count):
        #     self.count = torch.from_numpy(count.toarray()).float()
        # else:
        #     self.count = torch.from_numpy(count).float() 
        
        # ### library size vector
        # if size == 'explog':
        #     self.size = torch.exp(torch.log10(self.count.sum(axis=1))).unsqueeze(1)  
        # elif size == 'sum':
        #     self.size = self.count.sum(axis=1).unsqueeze(1)  
        # elif size == 'median':
        #     self.size = (self.count.sum(1) / np.median(self.count.sum(1))).unsqueeze(1) 
        # adata.obs['size'] = self.size.squeeze(1).numpy()
        
        # normalize 
        if normalize:
            sc.pp.normalize_total(adata)
        
        # log1p
        sc.pp.log1p(adata)
        if hvg:
            sc.pp.highly_variable_genes(
                                        adata,
                                        n_top_genes=hvg,
                                        subset=True,
                                        layer=count_key,
                                        flavor="seurat_v3",
                                    )
            adata = adata[:,:hvg]
        
        # scale
        if scale:
            sc.pp.scale(adata)
        
        ### count matrix
        if sp.issparse(adata.layers[count_key]):
            self.count = torch.from_numpy(adata.layers[count_key].toarray()).float()
        else:
            self.count = torch.from_numpy(adata.layers[count_key]).float() 
        
        ### input feature matrix
        if sp.issparse(adata.X):
            self.feature = torch.from_numpy(adata.X.toarray()).float()
        else:
            self.feature = torch.from_numpy(adata.X).float()    
        
        ### batch labels
        if batch_key is not None:
            # self.batch = torch.from_numpy(adata.obs[batch_key].values)
            self.batch = torch.from_numpy(pd.get_dummies(adata.obs[batch_key]).values).float()    
            self.all_data = [(self.feature[i], self.count[i], self.size[i], self.batch[i]) for i in range(self.feature.shape[0])]
        
        else:
            self.all_data = [(self.feature[i], self.count[i], self.size[i]) for i in range(self.feature.shape[0])]
    
    def __getitem__(self,idx):
        return self.all_data[idx]      
    
    def __len__(self):
        return len(self.all_data)