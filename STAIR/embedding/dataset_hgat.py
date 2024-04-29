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
        dist, neigh_index = neigh.radius_neighbors(cord, return_distance=True)
        index = np.array([[],[]], dtype=int)
        
        for it in range(cord.shape[0]):
            index = np.hstack(((index, np.vstack((np.array([it]*neigh_index[it].shape[0]), neigh_index[it])))))  
        index = torch.LongTensor(index)
    
    value = torch.ones(index.shape[1])
    return torch.sparse.FloatTensor(index, value, torch.Size((cord.shape[0], cord2.shape[0])))


def hgat_data(adata, 
              batch_key, 
              batch_order = None,
              spatial_key ='spatial', 
              n_neigh_hom = 10, 
              n_radius_het = 0.1,
              kernal_thresh = None):
    
    if batch_order is None:
        batch_order = list(adata.obs[batch_key].value_counts().sort_index().index)

    # adjacency matrix within a slice
    feat_dict = {}
    adj_dict = {}
    coord_dict = {}
    index_dict = {}

    for batch_tmp in batch_order:
        adata_tmp = adata[adata.obs[batch_key]==batch_tmp].copy()
        feat_tmp = adata_tmp.obsm['latent']
        coord_tmp = adata_tmp.obsm[spatial_key]
        adj_tmp = calcu_adj(coord_tmp, 
                            neigh_cal = 'knn', 
                            n_neigh = n_neigh_hom, 
                            metric ='minkowski')._indices()
        feat_dict[batch_tmp] = feat_tmp
        coord_dict[batch_tmp] = coord_tmp
        adj_dict[batch_tmp, '0', batch_tmp] = adj_tmp
        index_dict[batch_tmp] = list(adata_tmp.obs_names)

    # adjacency matrix between slices
    cross_adj_dict = {}
    for target_tmp in batch_order:
        for source_tmp in batch_order:
            if target_tmp != source_tmp:
                cross_adj_dict[target_tmp, '1', source_tmp] = calcu_adj(feat_dict[target_tmp], 
                                                                        feat_dict[source_tmp], 
                                                                        neigh_cal ='radius', 
                                                                        n_radius = n_radius_het,
                                                                        metric ='cosine')._indices()

    adj_dict.update(cross_adj_dict)

    # construction of HeteroData
    data = HeteroData()
    for ii in list(feat_dict.keys()):
        data[ii].x = torch.from_numpy(feat_dict[ii]).float()

    for jj in list(adj_dict.keys()):
        data[jj].edge_index = adj_dict[jj]

    # construction of kernal matrix
    if kernal_thresh is not None:
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
        kernals = None
    
    return data, kernals, index_dict
