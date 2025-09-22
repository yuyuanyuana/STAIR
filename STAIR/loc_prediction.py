import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors

from STAIR.location.transformation import best_fit_transform, transform
from STAIR.location.edge_detection import alpha_shape
from STAIR.location.align_fine import fine_alignment
from STAIR.utils import MakeLogClass


def sort_slices(atte, start=None, return_tree=False):
    
    import networkx as nx
    
    matrix = 1 - (atte + atte.T)/2
    keys_use = atte.index.tolist()

    G = nx.Graph()
    for i in range(len(keys_use)):
        for j in range(len(keys_use)):
            if i == j:
                G.add_edge(keys_use[i], keys_use[j], weight=0)
            if i < j :
                G.add_edge(keys_use[i], keys_use[j], weight=matrix.iloc[i,j])

    # 计算最小生成树
    minimum_spanning_tree = nx.minimum_spanning_tree(G)

    # 输出最小生成树的边
#     for edge in minimum_spanning_tree.edges(data=True):
#         print(edge)

    result = list(minimum_spanning_tree.edges(data=True))

    if start:
        dists = {start:0.} 
    else:
        dists = {result[0][0]:0.} 

    un_calcu = []

    for edge in minimum_spanning_tree.edges(data=True):
        vert1, vert2, weight = edge
        weight = weight['weight']
        if vert1 in dists.keys() and vert2 in dists.keys():
            pass
        elif vert1 in dists.keys():
            dists[vert2] = dists[vert1] + weight
        elif vert2 in dists.keys():
            dists[vert1] = dists[vert2] - weight
        else:
            un_calcu.append(edge)

    while len(un_calcu)!=0:
        edge = un_calcu.pop(0)
        vert1, vert2, weight = edge
        weight = weight['weight']
        if vert1 in dists.keys() and vert2 in dists.keys():
            pass
        elif vert1 in dists.keys():
            dists[vert2] = dists[vert1] + weight
        elif vert2 in dists.keys():
            dists[vert1] = dists[vert2] - weight
        else:
            un_calcu.append(edge)
    
    for key in keys_use:
        if dists[key] < 0:
            dists[key] = - dists[key]

    if return_tree:
        return dists, minimum_spanning_tree
    return dists


def loc_predict_z(adata, 
                  atte, 
                  querys, 
                  loc_key, 
                  batch_key, 
                  knowns = None, 
                  num_mnn = 3): 
    use = adata.obs[[batch_key, loc_key]].drop_duplicates()
    use.index = use[batch_key]
    use = use[loc_key]
    if knowns is None:
        knowns = list(set(adata.obs[batch_key].value_counts().index.tolist()) - set(querys))
        knowns.sort()
    loc_knowns = use.loc[knowns] 
    preds = []
    for query_tmp in querys:
        atte_in = atte.loc[query_tmp, knowns]
        atte_out = atte.loc[knowns, query_tmp]
        neigh_in = atte_in.sort_values(ascending=False)[:num_mnn]
        neigh_out = atte_out.sort_values(ascending=False)[:num_mnn]
        neigh_index = list(set(neigh_in.index).intersection(set(neigh_out.index)))       
        pred = (atte_in[neigh_index]*loc_knowns[neigh_index]).sum() / (atte_in[neigh_index]).sum()
        preds.append(pred)
    nearest_slices = [loc_knowns.index[abs(loc_knowns - preds[i]).argmin()] for i in range(len(preds))]
    return preds, nearest_slices



def init_align_with_scale(  adata_ref,
                            adata_query, 
                            emb_key = 'STAIR',
                            num_mnn = 1,
                            spatial_key1 = 'spatial',
                            spatial_key2 = None,
                            use_scale = True, 
                            key_added = 'init_scale',
                            return_scale = False
                        ):
    emb1_tmp = adata_ref.obsm[emb_key]
    emb2_tmp = adata_query.obsm[emb_key]
    # print(emb1_tmp)
    # print(emb2_tmp)

    # Search for mutual nearest neighbors of two slices
    neigh1 = NearestNeighbors(n_neighbors=num_mnn, metric='cosine')
    neigh1.fit(emb2_tmp)
    indices1 = neigh1.kneighbors(emb1_tmp, return_distance=False)
    neigh2 = NearestNeighbors(n_neighbors=num_mnn, metric='cosine')
    neigh2.fit(emb1_tmp)
    indices2 = neigh2.kneighbors(emb2_tmp, return_distance=False)
    set1 = {(i, indices1[i,j]) for i in range(indices1.shape[0]) for j in range(indices1.shape[1])}
    set2 = {(indices2[j,i], j) for j in range(indices2.shape[0]) for i in range(indices2.shape[1])}
    pair = set1.intersection(set2)
    pair = np.array(list(pair))
    if spatial_key2 is None:
        spatial_key2 = spatial_key1
    B = adata_ref[pair[:,0],:].obsm[spatial_key1].copy()
    A = adata_query[pair[:,1],:].obsm[spatial_key2].copy()
    scales = np.array([np.sqrt(((B[i] - B[j])**2).sum()) / np.sqrt(((A[i] - A[j])**2).sum()) for i in range(B.shape[0]) for j in range(B.shape[0]) if i!=j])
    scale_use = np.median(scales)
    A_scaled = A * scale_use
    T,_,_ = best_fit_transform(A_scaled, B)
    # transform the coordinates of adata2
    adata_query.obsm[key_added] = transform(adata_query.obsm[spatial_key2] * scale_use, T)   
    if  key_added != spatial_key1:
        adata_ref.obsm[key_added] = adata_ref.obsm[spatial_key1].copy()
    if return_scale:
        return adata_ref, adata_query, scale_use
    return adata_ref, adata_query


def plot_edge( adata_a, adata_b, edge_tmp1, edge_tmp2, query_index, 
               alpha_query, alpha_ref, spatial_key='spatial', 
               figsize=(5,5), s_query=10, s_ref=50, result_path='.'):
    
    spatial_tmp1 = adata_a.obsm[spatial_key]
    spatial_tmp2 = adata_b.obsm[spatial_key]

    if not os.path.exists(result_path + '/loc_pred/edge'):
        os.makedirs(result_path + '/loc_pred/edge')

    xx,yy = np.median(spatial_tmp1, 0)
    plt.figure(figsize=figsize)
    plt.scatter(spatial_tmp1[:, 0], spatial_tmp1[:, 1], s = s_ref)
    for i, j in edge_tmp1:
        plt.plot(spatial_tmp1[[i, j], 0], spatial_tmp1[[i, j], 1], c='#E24A33')

    plt.text(xx, yy, f"alpha_ref={alpha_ref}", size=18)
    plt.savefig(f'{result_path}/loc_pred/edge/spatial_edge_{str(query_index)}_a.png', bbox_inches='tight')
    plt.close()

    xx,yy = np.median(spatial_tmp2, 0)
    plt.figure(figsize=figsize)
    plt.scatter(spatial_tmp2[:, 0], spatial_tmp2[:, 1], s = s_query)
    for i, j in edge_tmp2:
        plt.plot(spatial_tmp2[[i, j], 0], spatial_tmp2[[i, j], 1], c='#8EBA42')

    plt.text(xx, yy, f"alpha_query={alpha_query}", size=18)
    plt.savefig(f'{result_path}/loc_pred/edge/spatial_edge_{str(query_index)}_b.png', bbox_inches='tight')
    plt.close()



class Loc_Pred(object):
    
    """
    
    Location prediction of new slices based on known reference, including location prediction and alignment. 
    They perform location prediction and alignment based on attention and spatial embedding in HAT module.

    Parameters
    ----------
    adata
        AnnData object of scanpy package including query and reference data
    atte
        A pandas dataframe with attention as value and slice names as index and columns
    batch_key
        The key containing slice information in .obs
    querys 
        list of names of slices to predict

    Examples
    --------
    >>> adata = sc.read_h5ad(path_to_anndata)
    >>> loc_pred = Loc_Pred(adata, atte, batch_key = 'section_index', querys = ['10X_1'])
    >>> loc_pred.pred_z(loc_key = 'stereo_AP', num_mnn = 20)
    >>> ladata_query = loc_pred.pred_xy(spatial_key_query = 'spatial',
                                        spatial_key_ref = 'spatial_ccf_2d',
                                        spatial_key_3d = 'spatial_ccf_3d',
                                        emb_key = 'HAN_SE',
                                        )

    """
   
    def __init__(
        self,
        adata,
        atte, 
        batch_key,
        querys, 
        make_log = True, 
        result_path = '.'
    ):
        super(Loc_Pred, self).__init__()

        self.adata = adata
        self.atte = atte
        self.batch_key = batch_key
        self.querys = querys
        self.make_log = make_log
        self.result_path = result_path

        if self.make_log:
            self.makeLog = MakeLogClass(f"{self.result_path}/log_loc_pred.tsv").make
            self.makeLog(f"Location prediction")
            self.makeLog(f"  Slice key: {self.batch_key}")
            self.makeLog(f"  Query slices: {self.querys}")


    def pred_z( self, 
                loc_key = 'stereo_AP', 
                knowns = None, 
                num_mnn = 20, 
                return_result = True
               ):
        """
        Predict the coordinate position parallel to the slice in the reference, and find the nearest slice with the new one.
        
        Parameters
        ----------
        loc_key
            The key containing coordinates of paralle direction in .obs
        knowns
            List of slice names with known coordinates in paralle direction
        num_mnn
            Number of neighbor slices in predicting new z
        return_result
            Whether return the predicted locations and nearset slices for each query slice.
        """
        self.preds, self.nearest_slices = loc_predict_z( self.adata, 
                                                         self.atte, 
                                                         querys = self.querys, 
                                                         loc_key = loc_key, 
                                                         batch_key = self.batch_key, 
                                                         knowns = knowns, 
                                                         num_mnn = num_mnn)
        
        if self.make_log:
            self.makeLog(f"Location prediction of z")
            self.makeLog(f"  Location key: {loc_key}")
            self.makeLog(f"  Number of neighbor slices: {num_mnn}")

        if return_result:
            return self.preds, self.nearest_slices

    def pred_xy( self, 
                 spatial_key_query = 'spatial',
                 spatial_key_ref = 'spatial_ccf_2d',
                 spatial_key_init = 'spatial_init',
                 spatial_key_3d = 'spatial_ccf_3d',
                 emb_key = 'CASTLE',
                 num_mnn_init = 1,
                 return_scale = True,
                 alpha_query = 1,
                 alpha_ref = 1,
                 add_3d = True,
                 plot_init = True,
                 figsize = (5,5),
                 s_query = 10, 
                 s_ref = 50
                 ):
        """
        Get the coordinates of the new slice consistent with the reference, including initial alignment and fine alignment.

        Parameters
        ----------
        spatial_key_query
            The key of 2D coordinates of query datasets in .obsm.
        spatial_key_ref
            The key of 2D coordinates of reference in .obsm.
        spatial_key_init
            Added key of initial aligned 2D coordinates in .obsm.
        spatial_key_3d
            Added key of 3D coordinates consistent with reference in .obsm.
        emb_key
            The key containing the spatial embedding, default is 'CASTLE'.
        """

        adata_querys = []
        for i in range(len(self.querys)):
            query_slice = self.querys[i]
            nearest_slice = self.nearest_slices[i]
            adata_a = self.adata[self.adata.obs['section_index'] == nearest_slice].copy()
            adata_b = self.adata[self.adata.obs['section_index'] == query_slice].copy()

            adata_a, adata_b, scale = init_align_with_scale(adata_ref = adata_a, 
                                                            adata_query = adata_b, 
                                                            emb_key = emb_key,
                                                            num_mnn = num_mnn_init,
                                                            spatial_key1 = spatial_key_ref,
                                                            spatial_key2 = spatial_key_query,
                                                            key_added = spatial_key_init,
                                                            return_scale = return_scale)

            boundary_tmp1, edge_tmp1, _ = alpha_shape(adata_a.obsm[spatial_key_ref], alpha=alpha_ref, only_outer=True)
            boundary_tmp2, edge_tmp2, _ = alpha_shape(adata_b.obsm[spatial_key_init], alpha=alpha_query, only_outer=True)

            if plot_init:
                
                plot_edge( adata_a, adata_b, edge_tmp1, edge_tmp2, i, alpha_query, alpha_ref, 
                           spatial_key = spatial_key_init, result_path = self.result_path, 
                           figsize = figsize, s_query = s_query, s_ref = s_ref)


            fine_adatas, Ts_fine = fine_alignment(  [adata_a, adata_b], 
                                                    [(boundary_tmp1, boundary_tmp2)], 
                                                    spatial_key = spatial_key_init, 
                                                    key_added = spatial_key_ref, 
                                                    init_pose = None, 
                                                    max_iterations = 40, 
                                                    tolerance = 1e-8)

            adata_b = fine_adatas[1]
            
            if add_3d:
                adata_b.obsm[spatial_key_3d] = np.hstack((adata_b.obsm[spatial_key_ref], np.array([self.preds[i]]*adata_b.shape[0])[:,None]))
            
            adata_querys.append(adata_b)
            adata_query = sc.concat(adata_querys)
        
        if self.make_log:
            self.makeLog(f"Location prediction of x and y")
            self.makeLog(f"  2D coordinates of query: {spatial_key_query}")
            self.makeLog(f"  2D coordinates of reference: {spatial_key_ref}")
            self.makeLog(f"  2D coordinates of initial alignment: {spatial_key_init}")
            self.makeLog(f"  3D coordinates of final alignment: {spatial_key_3d}")
            self.makeLog(f"  Number of neighbor spots in initial alignment: {num_mnn_init}")
            self.makeLog(f"  Alpha of edge detection in query data: {alpha_query}")
            self.makeLog(f"  Alpha of edge detection in reference: {alpha_ref}")

        return adata_query








