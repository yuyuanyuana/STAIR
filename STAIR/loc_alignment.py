import os
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

from STAIR.location.align_init import initial_alignment
from STAIR.location.align_fine import fine_alignment
from STAIR.location.edge_detection import alpha_shape, calcu_lisi, select_clustered_domains, detect_edge_of_domains
from STAIR.utils import MakeLogClass


class Loc_Align(object):
    
    """
    
    Location alignment of multiplt ST slices, including initial alignment and fine alignment. 
    They perform spatial alignmentusing similarity of spatial embedding and spatial coordinates, respectively.

    Parameters
    ----------
    adata
        AnnData object object of scanpy package
    batch_key
        The key containing slice information in .obs
    batch_order
        Slice order used to perform alignment. Align according to the default order of elements in batch_key if None

    Examples
    --------
    >>> adata = sc.read_h5ad(path_to_anndata)
    >>> loc_align = Loc_Align(adata, batch_key='batch')
    >>> loc_align.init_align(emb_key = 'HAN_SE')
    >>> loc_align.detect_edge_fine_align(domain_key = 'layer_cluster')
    >>> loc_align.plot_edge(spatial_key = 'transform_init')
    >>> adata_aligned = loc_align.fine_align()

    """

   
    def __init__(
        self,
        adata,
        batch_key,
        batch_order = None,
        make_log = True, 
        result_path = '.'
    ):
        super(Loc_Align, self).__init__()

        self.batch_key = batch_key
        
        if not batch_order:
            batch_order = list(adata.obs[batch_key].value_counts().sort_index().index)

        self.batch_order = batch_order
        self.batch_n = len(batch_order)
        self.adata_list = [adata[adata.obs[batch_key]==key].copy() for key in batch_order]

        self.make_log = make_log
        self.result_path = result_path
        if self.make_log:
            self.makeLog = MakeLogClass(f"{self.result_path}/log_loc.tsv").make
            
    
    def init_align(self, 
                   emb_key,
                   spatial_key = 'spatial',
                   num_mnn = 1, 
                   init_align_key = 'transform_init',
                   use_scale = False,
                   return_result = False
                   ):
        '''
        Initial alignment of spatial location.
        
        Parameters
        ----------
        emb_key
            AnnData object object of scanpy package
        spatial_key
            Key of raw spatial coordinates in .obsm 
        num_mnn
            The number of mutual nearest neighbors calculated according to emb_key
        init_align_key
            Key of initial transformed coordinates added in .obsm
        
        '''
        self.init_align_key = init_align_key
        self.init_adatas, anchors, self.Ts_init, scales = initial_alignment(self.adata_list, 
		                                                                    spatial_key = spatial_key,
		                                                                    emb_key = emb_key,
		                                                                    num_mnn = num_mnn,
		                                                                    key_added = init_align_key,
		                                                                    use_scale = use_scale,
                                                                            batch_order = self.batch_order
		                                                                )
        # self.boundary = [(anchor[:,0].tolist(), anchor[:,1].tolist()) for anchor in anchors]
        
        if self.make_log:
            self.makeLog(f"Parameter set for initial alignment")
            self.makeLog(f"  Starting coordinates: {spatial_key}")
            self.makeLog(f"  K of MNN: {num_mnn}")
            self.makeLog(f"  Aligned coordinates: {init_align_key}")
        
        if return_result:
            return anchors, self.Ts_init, scales


    def detect_fine_points( self,
                            slice_boundary = True, 
                            domain_boundary = True, 
                            domain_key = 'layer_cluster',
                            num_domains = 1,
                            sep_sort = True,
                            alpha = 70,
                            return_result = False):
    
        '''
        Prepare for fine alignment. 
        First, the spatial domain with the highest degree of spatial aggregation is selected according to the index LISI, 
        and then the edge of the slice and the aforementioned spatial domain is detected.
        
        Parameters
        ----------
        domain_key
            Key of spatial domains in .obs 
        num_domains
            Number of domains used to aligning slices.
        sep_sort
            Boolean value, whether to sort spatial clustered pattern together.
        alpha
            alpha value to detect the edge of slices and dimains.
        '''
        
        self.alpha = alpha
        self.boundary = []
        self.edge = []
        
        # detect edge of slices
        if slice_boundary:
            boundary_slices, edge_slices = [], []
            for ii in range(len(self.init_adatas)):
                if slice_boundary:
                    adata_tmp = self.init_adatas[ii]
                    spatial_tmp = adata_tmp.obsm[self.init_align_key]
                    boundary_tmp, edge_tmp, _ = alpha_shape(spatial_tmp, alpha=alpha, only_outer=True)
                else:
                    boundary_tmp, edge_tmp = [], set()
                boundary_slices.append(boundary_tmp)
                edge_slices.append(edge_tmp)
                if ii !=0 :
                    self.boundary += [(boundary_slices[ii-1], boundary_slices[ii])]
                    self.edge += [(edge_slices[ii-1], edge_slices[ii])]

        # detect edge of domains
        if domain_boundary:
            
            # calculate lisi of each domains
            lisi_list = [calcu_lisi(adata_tmp, domain_key=domain_key, spatial_key=self.init_align_key) for adata_tmp in self.init_adatas]
            
            # sort the domains according to lisi
            domains_use = [select_clustered_domains(lisi_list[i], 
                                                    lisi_list[i+1], 
                                                    domain_key, 
                                                    use_domain_nums = num_domains, 
                                                    sep_sort=sep_sort) for i in range(self.batch_n-1)]
            
            # detect edge of domains
            boundary_domain, edge_domain = detect_edge_of_domains(  self.init_adatas, 
                                                                    domain_key = domain_key, 
                                                                    domains_use = domains_use, 
                                                                    spatial_key = self.init_align_key, 
                                                                    alpha = alpha)

            for ii in range(len(self.boundary)):
                boundary_tmp = self.boundary[ii]
                boundary_tmp1, boundary_tmp2 = boundary_tmp
                boundary_tmp1 += boundary_domain[ii][0]
                boundary_tmp2 += boundary_domain[ii][1]
                boundary_tmp1 = list(set(boundary_tmp1))
                boundary_tmp2 = list(set(boundary_tmp2))
                self.boundary[ii] = (boundary_tmp1, boundary_tmp2)
                edge_tmp1, edge_tmp2 = self.edge[ii]
                edge_tmp1 = edge_tmp1.union(edge_domain[ii][0])
                edge_tmp2 = edge_tmp2.union(edge_domain[ii][1])
                self.edge[ii] = (edge_tmp1, edge_tmp2)
                
        if self.make_log:
            self.makeLog(f"Parameter set for edge detection")
            self.makeLog(f"  Spatial coordinates: {domain_key}")
            self.makeLog(f"  Number of domains: {num_domains}")
            self.makeLog(f"  Sep sort: {sep_sort}")
            self.makeLog(f"  Alpha: {alpha}")
        
        if return_result:
            if domain_boundary:
                return self.boundary, self.edge, lisi_list, domains_use
            return self.boundary, self.edge        


    def fine_align( self, 
                    fine_align_key = 'transform_fine',
                    max_iterations = 20,
                    tolerance = 1e-10,
                    return_result = False
                   ):
        
        '''
        Fine alignment of spatial location.
        
        Parameters
        ----------
        fine_align_key
            Key of fine transformed coordinates added in .obsm
        max_iterations
            Maximum number of iterations for icp
        tolerance
            Maximum error allowed for early stopping
            
        Return
        ----------
        adata_aligned
            Fine aligned adata with 'init_align_key' and 'fine_align_key' added in .obsm
        '''
        
        self.fine_adatas, Ts_fine = fine_alignment( self.init_adatas, 
                                                    self.boundary, 
                                                    spatial_key=self.init_align_key, 
                                                    key_added=fine_align_key, 
                                                    init_pose = None, 
                                                    max_iterations = max_iterations, 
                                                    tolerance = tolerance)
                    
        adata_aligned = ad.concat(self.fine_adatas)
        
        if self.make_log:
            self.makeLog(f"Parameter set for fine alignment")
            self.makeLog(f"  Starting coordinates: {self.init_align_key}")
            self.makeLog(f"  Aligned coordinates: {fine_align_key}")
            self.makeLog(f"  Max iterations: {max_iterations}")
            self.makeLog(f"  Tolerance: {tolerance}")

        if return_result:
            return adata_aligned, Ts_fine         
        return adata_aligned


    def plot_edge(self,
                  spatial_key,
                  figsize = (6,6),
                  s=1
                  ):
        
        '''
        Plot the detected edges of slices and domains to select an suitable alpha value.
        
        Parameters
        ----------
        spatial_key
            Spatial coordinates used for plot in .obsm.

        '''
        
        if spatial_key in list(self.init_adatas[0].obsm.keys()):
            adatas = self.init_adatas
        else:
            adatas = self.fine_adatas
            
        ### check edges of slices 
        for ii in range(len(self.boundary)):
            
            # get adata
            adata_tmp1 = adatas[ii]
            adata_tmp2 = adatas[ii+1]
            
            # get slices
            slice_tmp1 = list(set(adata_tmp1.obs[self.batch_key]))[0]
            slice_tmp2 = list(set(adata_tmp2.obs[self.batch_key]))[0]
            
            # get boundarys and edges
            # boundary_tmp1, boundary_tmp2 = self.boundary[ii]
            edge_tmp1, edge_tmp2 = self.edge[ii]
            
            # get coordinates of boundarys
            spatial_tmp1 = adata_tmp1.obsm[spatial_key]
            spatial_tmp2 = adata_tmp2.obsm[spatial_key]
            
            if not os.path.exists(self.result_path + '/location/edge'):
                os.makedirs(self.result_path + '/location/edge')
            
            xx,yy = np.median(spatial_tmp1, 0)
            plt.figure(figsize=figsize)
            plt.scatter(spatial_tmp1[:, 0], spatial_tmp1[:, 1], s = s)
            for i, j in edge_tmp1:
                plt.plot(spatial_tmp1[[i, j], 0], spatial_tmp1[[i, j], 1], c='#E24A33')
            plt.text(xx, yy, f"alpha={self.alpha}", size=18)
            plt.savefig(f'{self.result_path}/location/edge/spatial_edge_{slice_tmp1}_{ii}.png', bbox_inches='tight')
            plt.close()
            
            xx,yy = np.median(spatial_tmp2, 0)
            plt.figure(figsize=figsize)
            plt.scatter(spatial_tmp2[:, 0], spatial_tmp2[:, 1], s = s)
            for i, j in edge_tmp2:
                plt.plot(spatial_tmp2[[i, j], 0], spatial_tmp2[[i, j], 1], c='#8EBA42')
                # plt.plot(spatial_tmp2[[i, j], 0], spatial_tmp2[[i, j], 1], c='#988ED5')
            plt.text(xx, yy, f"alpha={self.alpha}", size=18)
            plt.savefig(f'{self.result_path}/location/edge/spatial_edge_{slice_tmp2}_{ii}.png', bbox_inches='tight')
            plt.close()
    


