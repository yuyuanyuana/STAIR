import numpy as np
import scanpy as sc
import os
import datetime
import torch


def set_seed(seed):
    import random 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def make_seeds(n, per_batch, desired_batches):
    total = per_batch * desired_batches
    base = np.arange(n, dtype=np.int64)
    if total <= n:
        # 如果需要的数量不超过 n，直接取前 total 个
        return base[:total]
    else:
        # 至少放一遍全部节点
        seeds = base.tolist()
        # 还需要补充多少
        remain = total - n
        # 随机补齐
        extra = np.random.choice(base, size=remain, replace=True)
        seeds.extend(extra.tolist())
        return np.array(seeds, dtype=np.int64)



def construct_folder(path_name):
    result_path = path_name+'_'+datetime.datetime.now().strftime('%m-%d-%y %H:%M')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(result_path + '/embedding')
        os.makedirs(result_path + '/embedding/train')
        os.makedirs(result_path + '/location')
        os.makedirs(result_path + '/location/edge')
    return result_path


class MakeLogClass:
    def __init__(self, log_file):
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    def make(self, *args):
        # print(*args)
        # Write the message to the file
        with open(self.log_file, "a") as f:
            for arg in args:
                f.write("{}\r\n".format(arg))


def mclust_R(adata, num_cluster=10, modelNames='EEE', used_obsm='latent', random_seed=2022, key_add='clusters'):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    from rpy2.robjects import default_converter
    from rpy2.robjects import numpy2ri
    robjects.conversion.set_conversion(default_converter + numpy2ri.converter)
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[key_add] = mclust_res
    adata.obs[key_add] = adata.obs[key_add].astype('str')
    return adata



def cluster_func(adata, clustering, use_rep, res=1, cluster_num=None, key_add='cluster'):
    if clustering == 'louvain':
        sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_add)
        sc.tl.louvain(adata, resolution=res, neighbors_key=key_add, key_added=key_add)
    if clustering == 'leiden':
        sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_add)
        sc.tl.leiden(adata, resolution=res, neighbors_key=key_add, key_added=key_add)
    if clustering == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=cluster_num, random_state=2022).fit(adata.obsm[use_rep])
        adata.obs[key_add] = km.labels_
    if clustering == 'mclust':
        adata = mclust_R(adata, num_cluster=cluster_num, modelNames='EEE', used_obsm=use_rep, random_seed=2022, key_add=key_add)
    adata.obs[key_add] = adata.obs[key_add].astype('category')
    return adata


