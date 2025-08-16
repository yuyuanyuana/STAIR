import numpy as np
from sklearn.neighbors import NearestNeighbors


def transform(point_cloud, T):
    ''' 
    Transform point clouds given transformation matrix
    Input:
        point_cloud: the coordinates of point clouds being transform
        T: (m+1)x(m+1) homogeneous transformation matrix. m:the dimension of coordinates.
    Output:
        point_cloud_align: the coordinates of point clouds
    '''
    
    point_cloud_align = np.ones((point_cloud.shape[0], 3))
    point_cloud_align[:,0:2] = np.copy(point_cloud)
    point_cloud_align = np.dot(T, point_cloud_align.T).T
    
    return point_cloud_align[:, :2]


def nearest_neighbor(src, dst, metric='minkowski'):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


# https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    
    assert A.shape == B.shape
    
    # get number of dimensions
    m = A.shape[1]
   
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    
    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    
    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)
    
    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    
    return T, R, t

