import numpy as np 
from sklearn.neighbors import NearestNeighbors

from .transformation import best_fit_transform, nearest_neighbor, transform


def align_init_pair(adata1, 
                    adata2, 
                    spatial_key1,
                    spatial_key2 = None,
                    num_mnn = 1,
                    emb_key = 'STAIR',
                    key_added = 'transform_init',
                    use_scale = False
                    ):
    '''
    Align spatial coordinates of adata2 to match adata1 according to the similarity of emb_key.
    Input:
      adata1
      adata2
      spatial_key1: Key of spatial coordinates in .obsm 
      spatial_key2: Key of spatial coordinates in .obsm
      num_mnn:      The number of mutual nearest neighbors calculated according to emb_key
      emb_key:      The key used to calculate spot similarity in .obsm
      key_added:    Key of transformed coordinates added in .obsm
    Returns:
      adata1 with 'transformed' in .obsm
      adata2 with 'transformed' in .obsm
      pair: np.array of indices of nearest neighbor pairs
      T: transformation matrix with 3 x 3
    '''
    
    # print(f'Finding similar pairs using {emb_key}...')
    
    # Extract features used to calculate spot similarity
    emb1_tmp = adata1.obsm[emb_key]
    emb2_tmp = adata2.obsm[emb_key]
    
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
    
    print(f'Aligning slices using {pair.shape[0]} pairs of similar spots!')
    
    # Extract the coordinates of the nearest neighbors
    if spatial_key2 is None:
        spatial_key2 = spatial_key1
    
    B = adata1[pair[:,0],:].obsm[spatial_key1]
    A = adata2[pair[:,1],:].obsm[spatial_key2]
    
    if use_scale:
        scales_ = np.array([np.sqrt(((B[i] - B[j])**2).sum()) / np.sqrt(((A[i] - A[j])**2).sum()) for i in range(B.shape[0]) for j in range(B.shape[0]) if i!=j])
        scale_use = np.median(scales_)
        A = A * scale_use
    else:
        scale_use=None

    # search for best transformation
    T,_,_ = best_fit_transform(A, B)
    
    # transform the coordinates of adata2
    if use_scale:
        adata2.obsm[key_added] = transform(adata2.obsm[spatial_key2] * scale_use, T)    
    else:
        adata2.obsm[key_added] = transform(adata2.obsm[spatial_key2], T)    
    
    if key_added!=spatial_key1:
        adata1.obsm[key_added] = adata1.obsm[spatial_key1]
    
    return adata1, adata2, pair, T, scale_use





def initial_alignment(  adatas, 
                        spatial_key = 'spatial',
                        emb_key = 'HAN_SE',
                        num_mnn = 1, 
                        key_added = 'transform_init',
                        use_scale = False,
                        batch_order = None
                      ):
    
    '''
    Perform initial alignment for adatas according to the similarity of emb_key.
    Input:
      adatas:       Adata list to be aligned
      spatial_key:  Key of spatial coordinates in .obsm 
      emb_key:      The key used to calculate spot similarity in .obsm
      num_mnn:      The number of mutual nearest neighbors calculated according to emb_key
      key_added:    Key of transformed coordinates added in .obsm
    Returns:
      adatas:       Aligned adata list with 'transformed' in .obsm
      anchors:      np.array of indices of nearest neighbor pairs
    '''
    
    aligned_init = []
    anchors = []
    Ts = []
    scales = []
    

    print('Performing initial alignment...')
    print(f'    Aligning slice {batch_order[1]} to {batch_order[0]}...')
    
    # initial alignment of the first two adatas
    adata1, adata2 = adatas[0], adatas[1]
    
    adata1, adata2, anchor_tmp, T_tmp, scale_tmp = align_init_pair( adata1, 
                                                                    adata2, 
                                                                    spatial_key1 = spatial_key,
                                                                    spatial_key2 = None,
                                                                    num_mnn = num_mnn, 
                                                                    emb_key = emb_key,
                                                                    key_added = key_added,
                                                                    use_scale = use_scale
                                                                    )
    
    aligned_init.append(adata1)
    aligned_init.append(adata2)
    anchors.append(anchor_tmp)
    Ts.append(T_tmp)
    scales.append(scale_tmp)
    
    # initial alignment of the rest adatas
    if len(aligned_init)!=len(adatas):
        for i in range(1, len(adatas)-1):
            
            print(f'    Aligning slice {batch_order[i+1]} to {batch_order[i]}...')
            
            adata1_tmp = aligned_init[-1]
            adata2_tmp = adatas[i+1]
            
            for T_ in Ts:
                adata2_tmp.obsm[spatial_key] = transform(adata2_tmp.obsm[spatial_key], T_)
            
            _, adata2_tmp, anchor_tmp, T_tmp, scale_tmp = align_init_pair( adata1_tmp, 
                                                                            adata2_tmp, 
                                                                            spatial_key1 = key_added, 
                                                                            spatial_key2 = spatial_key, 
                                                                            num_mnn = num_mnn, 
                                                                            emb_key = emb_key,
                                                                            key_added = key_added,
                                                                            use_scale = use_scale
                                                                           )
                    
            aligned_init.append(adata2_tmp)
            anchors.append(anchor_tmp)
            Ts.append(T_tmp)
            scales.append(scale_tmp)
    
    return aligned_init, anchors, Ts, scales


# def align_init_pair(adata1, 
#                     adata2, 
#                     spatial_key1,
#                     spatial_key2 = None,
#                     num_mnn = 1,
#                     emb_key = 'HAN_SE',
#                     key_added = 'transform_init',
#                     init_pose = None, 
#                     max_iterations = 50, 
#                     tolerance = 0.001
#                     ):
#     '''
#     Align spatial coordinates of adata2 to match adata1 according to the similarity of emb_key.
#     Input:
#       adata1
#       adata2
#       spatial_key1: Key of spatial coordinates in .obsm 
#       spatial_key2: Key of spatial coordinates in .obsm
#       num_mnn:      The number of mutual nearest neighbors calculated according to emb_key
#       emb_key:      The key used to calculate spot similarity in .obsm
#       key_added:    Key of transformed coordinates added in .obsm
#     Returns:
#       adata1 with 'transformed' in .obsm
#       adata2 with 'transformed' in .obsm
#       pair: np.array of indices of nearest neighbor pairs
#       T: transformation matrix with 3 x 3
#     '''
    
#     print(f'Finding similar pairs using {emb_key}...')
    
#     # Extract features used to calculate spot similarity
#     emb1_tmp = adata1.obsm[emb_key]
#     emb2_tmp = adata2.obsm[emb_key]
    
#     # Search for mutual nearest neighbors of two slices
#     neigh1 = NearestNeighbors(n_neighbors=num_mnn, metric='cosine')
#     neigh1.fit(emb2_tmp)
#     indices1 = neigh1.kneighbors(emb1_tmp, return_distance=False)
#     neigh2 = NearestNeighbors(n_neighbors=num_mnn, metric='cosine')
#     neigh2.fit(emb1_tmp)
#     indices2 = neigh2.kneighbors(emb2_tmp, return_distance=False)
    
#     set1 = {(i, indices1[i,j]) for i in range(indices1.shape[0]) for j in range(indices1.shape[1])}
#     set2 = {(indices2[j,i], j) for j in range(indices2.shape[0]) for i in range(indices2.shape[1])}
    
#     pair = set1.intersection(set2)
#     pair = np.array(list(pair))
    
#     print(f'Aligning slices using {pair.shape[0]} pairs of similar spots!')
    
#     # Extract the coordinates of the nearest neighbors
#     if spatial_key2 is None:
#         spatial_key2 = spatial_key1
    
#     B = adata1[pair[:,0],:].obsm[spatial_key1]
#     A = adata2[pair[:,1],:].obsm[spatial_key2]
    
#    # get number of dimensions
#     m = A.shape[1]

#     # make points homogeneous, copy them to maintain the originals
#     src = np.ones((m+1,A.shape[0]))   # (3, 238)
#     dst = np.ones((m+1,B.shape[0]))   # (3, 248)
#     src[:m,:] = np.copy(A.T)
#     dst[:m,:] = np.copy(B.T)

#     # apply the initial pose estimation
#     if init_pose is not None:
#         src = np.dot(init_pose, src)

#     prev_error = 0
#     for i in range(max_iterations): 
        
#         # find the nearest neighbors between the current source and destination points
#         distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T) 
        
#         # compute the transformation between the current source and nearest destination points
#         T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        
#         # update the current source
#         src = np.dot(T, src)
        
#         # check error
#         mean_error = np.mean(distances)
#         print('iter', i, 'error:', np.abs(prev_error - mean_error))
        
#         if np.abs(prev_error - mean_error) < tolerance:
#             break
        
#         prev_error = mean_error
        
#     # calculate final transformation
#     T,_,_ = best_fit_transform(A, src[:m,:].T)

#     if key_added != spatial_key1:
#         adata1.obsm[key_added] = adata1.obsm[spatial_key1]
    
#     adata2.obsm[key_added] = transform(adata2.obsm[spatial_key2], T)
    
#     return adata1, adata2, pair, T



# def initial_alignment_( adatas, 
#                         spatial_key = 'spatial',
#                         emb_key = 'HAN_SE',
#                         num_mnn = 1, 
#                         key_added = 'transform_init'
#                         ):
#     '''
#     Perform initial alignment for adatas according to the similarity of emb_key.
#     Input:
#       adatas:       Adata list to be aligned
#       spatial_key:  Key of spatial coordinates in .obsm 
#       emb_key:      The key used to calculate spot similarity in .obsm
#       num_mnn:      The number of mutual nearest neighbors calculated according to emb_key
#       key_added:    Key of transformed coordinates added in .obsm
#     Returns:
#       adatas:       Aligned adata list with 'transformed' in .obsm
#       anchors:      np.array of indices of nearest neighbor pairs
#     '''
    
#     Ts_init = []
#     for i in range(len(adatas)-1):
#         adata1 = adatas[i]
#         adata2 = adatas[i+1]
#         _, _, _, T_ = align_init_pair(  adata1, 
#                                         adata2, 
#                                         spatial_key1 = spatial_key,
#                                         spatial_key2 = None,
#                                         num_mnn = num_mnn,
#                                         emb_key = emb_key,
#                                         key_added = key_added
#                                         )
#         Ts_init.append(T_)
#     # Compute initial transformed coordinates
#     for i in range(len(adatas)):          
#         adatas[i].obsm[key_added] = adatas[i].obsm[spatial_key]
#         if i != 0:
#             T_tmp = Ts_init[:i]
#             for T_ in T_tmp:
#                 adatas[i].obsm[key_added] = transform(adatas[i].obsm[key_added], T_)
#     return adatas, Ts_init




