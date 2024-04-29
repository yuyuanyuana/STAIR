import numpy as np
from .transformation import best_fit_transform, nearest_neighbor, transform

def align_fine_pair(adata1, 
                    adata2, 
                    index_use1, 
                    index_use2, 
                    spatial_key1='spatial', 
                    spatial_key2=None, 
                    key_added = 'transformed',
                    init_pose = None, 
                    max_iterations = 20, 
                    tolerance = 0.001):
    '''
    Align spatial coordinates of adata2 to match adata1 according to the specified spatial domains.
    Input:
      adata1
      adata2
      spatial_key1: key of spatial coordinates in .obsm 
      spatial_key2: key of spatial coordinates in .obsm
      domain_key: key of spatial domains in .obs 
      domain_use: domains included in 'domain_key' used to align the slice
    Returns:
      adata1 with 'transformed' in .obsm
      adata2 with 'transformed' in .obsm
      T: transformation matrix with 3 x 3
    '''
    if spatial_key2 is None:
        spatial_key2 = spatial_key1
    
    B = adata1[index_use1,:].obsm[spatial_key1]
    A = adata2[index_use2,:].obsm[spatial_key2]

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))   # (3, 238)
    dst = np.ones((m+1,B.shape[0]))   # (3, 248)
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    for i in range(max_iterations): 
        
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T) 
        
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        
        # update the current source
        src = np.dot(T, src)
        
        # check error
        mean_error = np.mean(distances)
        # print('iter', i, 'error:', np.abs(prev_error - mean_error))
        
        if np.abs(prev_error - mean_error) < tolerance:
            break
        
        prev_error = mean_error
        
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    if key_added != spatial_key1:
        adata1.obsm[key_added] = adata1.obsm[spatial_key1]
    
    adata2.obsm[key_added] = transform(adata2.obsm[spatial_key2], T)

    return adata1, adata2, T



def fine_alignment( adatas, 
                    boundarys, 
                    spatial_key='spatial', 
                    key_added='transformed', 
                    init_pose = None, 
                    max_iterations = 20, 
                    tolerance = 1e-8):
    '''
    Perform fine alignment to match the first adata in adatas according to the edge of slices and most clustered spatial domains.
    Input:
      adatas:      list of adatas to be aligned. The first adata is the fixed slice
      boundarys:   list of bounary indexs of each adata
      spatial_key: Key of spatial coordinates in .obsm
      key_added:   Key of transformed coordinates added in .obsm
      init_pose:   The initial pose estimation
    Returns:
      aligned:     adata list with key_added in .obsm added. default is 'transformed'
    '''
    
    aligned = []
    Ts = []
    
    print('Performing fine alignment of the 1 pair of data...')
    
    # fine alignment of the first two adatas
    adata1 = adatas[0]
    adata2 = adatas[1]
    
    boundary_tmp1, boundary_tmp2 = boundarys[0]

    adata1, adata2, T_tmp = align_fine_pair(adata1 = adatas[0], 
                                            adata2 = adatas[1], 
                                            index_use1=boundary_tmp1, 
                                            index_use2=boundary_tmp2, 
                                            spatial_key1=spatial_key, 
                                            spatial_key2=spatial_key, 
                                            key_added=key_added,
                                            init_pose = init_pose, 
                                            max_iterations = max_iterations, 
                                            tolerance = tolerance)
    
    aligned.append(adata1)
    aligned.append(adata2)
    Ts.append(T_tmp)
    
    # fine alignment of the rest adatas
    if len(aligned)!=len(adatas):
        for i in range(1, len(adatas)-1):
            
            print(f'Performing fine alignment of the {i+1} pair of data...')
            
            adata1_tmp = aligned[-1]
            adata2_tmp = adatas[i+1]           
            boundary_tmp1, boundary_tmp2 = boundarys[i]  
            
            for T_ in Ts:
                adata2_tmp.obsm[spatial_key] = transform(adata2_tmp.obsm[spatial_key], T_)
            
            _, adata2_tmp, T_tmp = align_fine_pair( adata1_tmp, 
                                                    adata2_tmp, 
                                                    index_use1=boundary_tmp1, 
                                                    index_use2=boundary_tmp2, 
                                                    spatial_key1=key_added, 
                                                    spatial_key2=spatial_key,
                                                    key_added = key_added, 
                                                    init_pose = init_pose, 
                                                    max_iterations = max_iterations, 
                                                    tolerance = tolerance)
            
            aligned.append(adata2_tmp)
            Ts.append(T_tmp)
    
    return aligned, Ts



def fine_alignment_(adatas, 
                    boundarys, 
                    spatial_key='spatial', 
                    key_added='transformed', 
                    init_pose = None, 
                    max_iterations = 20, 
                    tolerance = 0.001):
    '''
    Align multiple slices to match the first adata in adatas according to the specified spatial domains.
    Input:
      adatas:      list of adatas to be aligned. The first adata is the fixed slice
      boundarys:   list of bounary indexs of each adata
      spatial_key: Key of spatial coordinates in .obsm
      key_added:   Key of transformed coordinates added in .obsm
      init_pose:   The initial pose estimation
    Returns:
      aligned:     adata list with key_added in .obsm added. default is 'transformed'
    '''
    
    
    Ts_fine = []
    for i in range(len(adatas)-1):
        adata1 = adatas[i]
        adata2 = adatas[i+1]
        boundary_tmp1 = boundarys[i][0]
        boundary_tmp2 = boundarys[i][1]
        _, _, T_ = align_fine_pair( adata1, 
                                    adata2, 
                                    index_use1=boundary_tmp1, 
                                    index_use2=boundary_tmp2, 
                                    spatial_key1=spatial_key, 
                                    spatial_key2=spatial_key,
                                    key_added = key_added, 
                                    init_pose = init_pose, 
                                    max_iterations = max_iterations, 
                                    tolerance = tolerance)
        Ts_fine.append(T_)
    
    # Compute initial transformed coordinates
    for i in range(len(adatas)):          
        adatas[i].obsm[key_added] = adatas[i].obsm[spatial_key]
        if i != 0:
            T_tmp = Ts_fine[:i]
            # print(len(T_tmp))
            for T_ in T_tmp:
                adatas[i].obsm[key_added] = transform(adatas[i].obsm[key_added], T_)
    return adatas


  