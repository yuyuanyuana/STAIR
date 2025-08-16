import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.data import HeteroData

def calcu_adj(cord, cord2=None, neigh_cal ='knn', n_neigh = 8, n_radius=None, metric='minkowski'):
    '''
    Construct adjacency matrix with coordinates.
    input: cord, np.array
    '''
    
    if cord2 is None:
        cord2 = cord
        n_neigh += 1
    
    if neigh_cal == 'knn':
        neigh = NearestNeighbors(n_neighbors = n_neigh, metric = metric).fit(cord2)
        neigh_index = neigh.kneighbors(cord,return_distance=False)
        index = torch.LongTensor(np.vstack((np.repeat(range(cord.shape[0]),n_neigh),neigh_index.ravel())) ) 
    
    if neigh_cal == 'radius':
        neigh = NearestNeighbors(radius=n_radius, metric = metric).fit(cord2)
        neigh_index = neigh.radius_neighbors(cord, return_distance=False)
        index = np.array([[],[]], dtype=int)
        
        for it in range(cord.shape[0]):
            index = np.hstack(((index, np.vstack((np.array([it]*neigh_index[it].shape[0]), neigh_index[it])))))  
        index = torch.LongTensor(index)
    
    return index 


import torch

def get_high_sim_indices_blocked(x, y, threshold=0.9, block_size=10000):
    """
    分块计算两个矩阵之间的余弦相似度,并返回相似度大于阈值的索引对
    
    Args:
        x (torch.Tensor): 形状为(n, d)的张量
        y (torch.Tensor): 形状为(m, d)的张量
        threshold (float): 相似度阈值
        block_size (int): 分块大小
        
    Returns:
        indices (torch.Tensor): 形状为(k, 2)的张量,每行表示一对高相似索引
    """
    n, d = x.shape
    m, _ = y.shape
    
    # 计算x和y的L2范数
    x_norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
    y_norm = y.pow(2).sum(dim=1, keepdim=True).sqrt()
    
    # 对x和y进行归一化
    x_normalized = x / x_norm
    y_normalized = y / y_norm
    
    indices = []
    
    # 分块计算余弦相似度
    for i in range(0, n, block_size):
        start_i = i
        end_i = min(i + block_size, n)
        
        for j in range(0, m, block_size):
            start_j = j
            end_j = min(j + block_size, m)
            
            x_block = x_normalized[start_i:end_i]
            y_block = y_normalized[start_j:end_j]
            
            sim_block = torch.mm(x_block, y_block.t())
            high_sim_mask = sim_block > threshold
            
            i_indices, j_indices = high_sim_mask.nonzero().unbind(dim=1)
            i_indices += start_i
            j_indices += start_j
            
            indices.append(torch.stack([i_indices, j_indices], dim=1))
    
    indices = torch.cat(indices, dim=0)
    return indices


def hgat_data(adata, 
              batch_key, 
              batch_order = None,
              spatial_key ='spatial', 
              n_neigh_hom = 10, 
              c_neigh_het = 0.9,
              kernal_thresh = 0.):
    
    if batch_order is None:
        batch_order = list(adata.obs[batch_key].value_counts().sort_index().index)

    # adjacency matrix within a slice
    feat_dict = {}
    adj_dict = {}
    coord_dict = {}
    index_dict = {}

    for batch_tmp in batch_order:
        # print('hom: ', batch_tmp)
        adata_tmp = adata[adata.obs[batch_key]==batch_tmp].copy()
        feat_tmp = adata_tmp.obsm['latent']
        coord_tmp = adata_tmp.obsm[spatial_key]
        adj_tmp = calcu_adj(coord_tmp, 
                            neigh_cal = 'knn', 
                            n_neigh = n_neigh_hom, 
                            metric ='minkowski')
        feat_dict[batch_tmp] = feat_tmp
        coord_dict[batch_tmp] = coord_tmp
        adj_dict[batch_tmp, '0', batch_tmp] = adj_tmp
        index_dict[batch_tmp] = list(adata_tmp.obs_names)

    # adjacency matrix between slices
    cross_adj_dict = {}

    for target_tmp in batch_order:
        
        for source_tmp in batch_order:
            
            if target_tmp != source_tmp:
                # print('het: ', target_tmp, source_tmp)
                
                if (source_tmp, '1', target_tmp) in list(feat_dict.keys()):

                    cross_adj_dict[target_tmp, '1', source_tmp] = cross_adj_dict[source_tmp, '1', target_tmp][[1,0],:]

                else:    
                    
                    indices =  get_high_sim_indices_blocked(torch.from_numpy(feat_dict[target_tmp]), 
                                                            torch.from_numpy(feat_dict[source_tmp]), 
                                                            threshold = c_neigh_het, 
                                                            block_size=10000
                                                            ).T
                    num_cols = indices.size(1)
                    if num_cols > 1000000:
                        cols_to_remove = torch.randperm(num_cols)[:num_cols-1000000]
                        indices = torch.index_select(indices, 1, torch.from_numpy(np.setdiff1d(np.arange(num_cols), cols_to_remove)))

                    cross_adj_dict[target_tmp, '1', source_tmp] = indices 

    adj_dict.update(cross_adj_dict)

    # construction of HeteroData
    data = HeteroData()
    for ii in list(feat_dict.keys()):
        data[ii].x = torch.from_numpy(feat_dict[ii]).float()

    for jj in list(adj_dict.keys()):
        data[jj].edge_index = adj_dict[jj]

    # construction of kernal matrix
    if kernal_thresh != 0:
        from scipy.spatial.distance import cdist
        kernals = dict()
        for node_key, node_coord in coord_dict.items():
            dist_tmp = np.sqrt(cdist(node_coord, node_coord, metric='euclidean'))
            dist_tmp = dist_tmp/np.percentile(dist_tmp, kernal_thresh, axis=0)
            kernal_tmp = np.exp(-dist_tmp)
            kernal_tmp = torch.from_numpy(kernal_tmp).float()    
            kernal_tmp = 0.5 * (kernal_tmp + kernal_tmp.T)
            kernals[node_key] = kernal_tmp
    
    else:
        kernals = dict()

    return data, kernals, index_dict
