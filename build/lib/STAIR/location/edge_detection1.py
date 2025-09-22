import pandas as pd
import numpy as np


def calcu_lisi(adata, domain_key, spatial_key):
    '''
    Calculate index LISI of given cluster and spatial coordinates;
    Input:  adata, domain_key in .obs, spatial_keyin .obsm;
    Output: list of sorted domains
    '''
    import rpy2.robjects as ro
    ro.r.library("lisi")
    ro.r['set.seed'](1)
    
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    lisi = ro.r['compute_lisi']
    result = adata.obs[[domain_key]]
    result[['x', 'y']] = adata.obsm[spatial_key].copy()
    
    res = lisi(ro.conversion.py2rpy(result[['x', 'y']]), 
                ro.conversion.py2rpy(result[[domain_key]]), 
                ro.StrVector([domain_key]))
    
    result['lisi'] = pd.DataFrame(res).astype(float).values.flatten()
    
    return result


def select_clustered_domains(lisi1, lisi2, domain_key, use_domain_nums=3, sep_sort=True):
    '''
    Sort domains according to the index LISI of each domain;
    Input:  lisi1: DataFrame of data1 with columns: ['lisi', domain_key]
            lisi2: DataFrame of data2 with columns: ['lisi', domain_key]
            domain_key;
            use_domain_num: number of domains used to aligning slices.
            sep_sort:  Boolean value, whether to sort spatial clustered pattern together.
    Output: list of used domains
    '''
    
    if sep_sort:
        
        # select domains with more than 3 points
        domain_ok1 = lisi1[domain_key].value_counts()[lisi1[domain_key].value_counts() > 3].index.tolist()
        domain_ok2 = lisi2[domain_key].value_counts()[lisi2[domain_key].value_counts() > 3].index.tolist()

        # sort spatial clustering pattern of domains according to index LISI
        domains_sorted1 = lisi1.groupby(domain_key).median().loc[domain_ok1,:].sort_values('lisi').index.tolist()
        domains_sorted2 = lisi2.groupby(domain_key).median().loc[domain_ok2,:].sort_values('lisi').index.tolist()
        
        
        anchor1_dict = {domains_sorted1[ii]:ii for ii in range(len(domains_sorted1))}
        anchor2_dict = {domains_sorted2[ii]:ii for ii in range(len(domains_sorted2))}
        
        # common domains exist in two slices
        common_domains = list(set(anchor1_dict.keys()).intersection(set(anchor2_dict.keys())))
        
        # check the number of domains and domains needed
        if len(common_domains) < use_domain_nums:
            print('The number of regions required is greater than the actual number of regions')
        
        # sort domains based on two slices 
        anchor_dict = {ii: anchor1_dict[ii] + anchor2_dict[ii] for ii in common_domains}
        domains_sorted = sorted(anchor_dict.items(), key = lambda kv:(kv[1], kv[0]))
        domains_sorted = [ii[0] for ii in domains_sorted]
        domain_use = domains_sorted[:use_domain_nums]
    
    else:
        
        # common domains exist in two slices
        common_domains = list(set(lisi1[domain_key]).intersection(set(lisi2[domain_key])))
        
        # check the number of domains and domains needed
        if len(common_domains) < use_domain_nums:
            print('The number of regions required is greater than the actual number of regions')
        
        lisi_tmp = pd.concat([lisi1, lisi2])
        lisi_tmp = lisi_tmp[lisi_tmp[domain_key].isin(common_domains)]
        
        # sort spatial clustering pattern of domains according to index LISI
        domains_sorted = lisi_tmp.groupby(domain_key).median().sort_values('lisi').index.tolist()
        domain_use = domains_sorted[:use_domain_nums]
    
    return domain_use



from typing import List
from itertools import chain
from scipy.spatial import Delaunay

def alpha_shape(points, alpha, only_outer=True)->List:
    '''
    Compute the alpha shape (concave hull) of a set of points.  
    Parameters
    ----------
    points
        np.array of shape (n,2) points.
    alpha
        alpha value.
    only_outer
    boolean value to specify if we keep only the outer border or also inner edges.
    Return
    ----------
    Set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.
    Refer
    ----------
    https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
    '''
    assert points.shape[0] > 3, "Need at least four points"
    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))
    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    circum_r_list = []
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        circum_r_list.append(circum_r)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    boundary = list(set(list(chain.from_iterable(list(edges)))))
    return boundary, edges, circum_r_list



from operator import itemgetter

def detect_edge_of_slice_and_domains(adatas, domain_key, domains_use, slice_boundary=True, spatial_key='spatial', alpha=50):
    '''
    Detect edges which were used to aligning the slices, including edges of slices and used domains.
    Parameters
    ----------
    adatas
        List of adata.
    domain_key
        Key of spatial domains in .obs 
    domains_use
        Domains used to align slices calculated 'by select_clustered_domains'. list with length len(adatas)-1.
    spatial_key
        Key of spatial coordinates in .obsm
    alpha
        alpha value.
    plot_edges
        Boolean value of whether to plot the used edges or not.
    result_path
        Path to save the used edges if plot_edges=True.
    Return
    ----------
    boundary_all
        List of boundary indexs in each slices 
    edge_all
        List of edge indexs in each slices 
    '''
    # detect edge of slices
    boundary_slices = []
    edge_slices = []
    
    for ii in range(len(adatas)):
        if slice_boundary:
            adata_tmp = adatas[ii]
            spatial_tmp = adata_tmp.obsm[spatial_key]
            boundary_tmp, edge_tmp, _ = alpha_shape(spatial_tmp, alpha=alpha, only_outer=True)
        else:
            boundary_tmp, edge_tmp = [], set()
        boundary_slices.append(boundary_tmp)
        edge_slices.append(edge_tmp)

    # detect edge of domains and combine them 
    assert len(domains_use) == len(adatas)-1
    boundary_all = []
    edge_all = []
    for i in range(len(domains_use)): 

        domain_use = domains_use[i]
        adata1_tmp = adatas[i]
        adata2_tmp = adatas[i+1]    
        
        # initial boundary and edge using them of slices 
        boundary_use_tmp1 = boundary_slices[i]
        boundary_use_tmp2 = boundary_slices[i+1]
        edge_use_tmp1 = edge_slices[i]
        edge_use_tmp2 = edge_slices[i+1]
        
        # Detect the edges of each region separately
        for domain_use_tmp in domain_use:
            
            points1 = adata1_tmp.obsm[spatial_key]   # coordinates of data1
            index_tmp1 = np.where(adata1_tmp.obs[domain_key]==domain_use_tmp)[0]   # index of target domain in data1
            index_tmp_dict1 = dict(enumerate(index_tmp1))   # dict of index.  new index: raw index
            boundary_tmp1, edge_tmp1, _ = alpha_shape(points1[index_tmp1], alpha)   # detect edge of target domain 
            boundary_use_tmp1 += list(itemgetter(*boundary_tmp1)(index_tmp_dict1))    # map the boundary index back to raw index
            edge_use_tmp1 = edge_use_tmp1.union({(index_tmp_dict1[i], index_tmp_dict1[j]) for i,j in edge_tmp1})   # combine slice edge and domain edge
            
            points2 = adata2_tmp.obsm[spatial_key]
            index_tmp2 = np.where(adata2_tmp.obs[domain_key]==domain_use_tmp)[0]
            index_tmp_dict2 = dict(enumerate(index_tmp2))
            boundary_tmp2, edge_tmp2, _ = alpha_shape(points2[index_tmp2], alpha)
            boundary_use_tmp2 += list(itemgetter(*boundary_tmp2)(index_tmp_dict2))
            edge_use_tmp2 = edge_use_tmp2.union({(index_tmp_dict2[i], index_tmp_dict2[j]) for i,j in edge_tmp2})
        
        boundary_use_tmp1 = list(set(boundary_use_tmp1))
        boundary_use_tmp2 = list(set(boundary_use_tmp2))
        
        boundary_all.append((boundary_use_tmp1, boundary_use_tmp2))
        edge_all.append((edge_use_tmp1, edge_use_tmp2))

    return boundary_all, edge_all
