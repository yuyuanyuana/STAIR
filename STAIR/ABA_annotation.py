import nrrd
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import os
basepath = os.path.abspath(__file__)
folder = os.path.dirname(basepath)


ANO, metaANO = nrrd.read(folder+'/ABAanno/ccfv3/annotation_25.nrrd')
onto1_r = pickle.load(open(folder+'/ABAanno/ontology1_reverse.pkl', 'rb'))
onto2_r = pickle.load(open(folder+'/ABAanno/ontology2_reverse.pkl', 'rb'))
id_name = pd.read_csv(folder+'/ABAanno/ontology.csv', index_col=0)['name'].to_dict()
name_color = pd.read_csv(folder+'/ABAanno/ontology.csv', index_col=0).set_index('name')['color_hex_triplet'].to_dict()


# get index of annotation array from CCF coordinates.
def get_index(coord):
    ML_ref, DV_ref, AP_ref = coord[:,0], coord[:,1], coord[:,2]
    ML = ML_ref*40 + 228
    DV = -DV_ref*40
    AP = 214 - AP_ref*40
    AP [AP > 527] = 527
    DV [DV > 319] = 319
    ML [ML > 455] = 455
    return np.vstack((AP, DV, ML)).T.astype(int).tolist()



def ABA_anno(adata, ML_key, DV_key, AP_key, spatial_key='spatial'):
 
    """
    
    Get ABA annotation based on CCF coordinates
    
    Parameters
    ----------
    adata
        AnnData object of scanpy package
    ML_key
        The key of medial/lateral (ML) coordinates in .obs
    DV_key
        The key of dorsal/ventral (DV) coordinates in .obs
    AP_key
        The key of anterior/posterior (AP) coordinates in .obs
    spatial_key
        The key of 2D coordinates in .obsm

    """
    
    coord = adata.obs[[ML_key, DV_key, AP_key]].values  # get 3D coordinates
    index_ABA = get_index(coord)   # get 3D index in annotation array

    annos = []  # annotation of each spots
    for index_tmp in index_ABA:
        index_ap, index_dv, index_ml = index_tmp
        annos.append(ANO[index_ap, index_dv, index_ml])

    annos = np.array(annos)

    # refine annotation spots with invalid annotaion 0 or 997.
    inval_index = np.array(np.where(annos==0)[0].tolist() + np.where(annos==997)[0].tolist())

    if len(inval_index) != 0:
        coord_2d = adata.obsm[spatial_key]
        cord_query = coord_2d[inval_index]

        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=50)
        neigh.fit(coord_2d)
        indices = neigh.kneighbors(cord_query, return_distance=False)
        indices = indices[:,1:]

        annos_inval_ = annos[indices]
        annos_inval = [np.setdiff1d(annos_inval_[i], [0, 997])[0] for i in range(len(annos_inval_))]
        annos[inval_index] = annos_inval

    # define annotaions and the cooresponding colors in ABA in 2 levels.
    adata.obs['ABA_id'] = annos
    adata.obs['ABA_name_level2'] = adata.obs['ABA_id'].replace(onto2_r).replace(id_name).astype('category')
    adata.obs['ABA_name_level1'] = adata.obs['ABA_id'].replace(onto2_r).replace(onto1_r).replace(id_name).astype('category')

    adata.obs['ABA_color_level2'] = adata.obs['ABA_name_level2'].replace(name_color)
    adata.obs['ABA_color_level1'] = adata.obs['ABA_name_level1'].replace(name_color)
    
    return adata


def plot_spatial_ABA(adatas, spatial_key, level='level2', title_key='section_index', 
                     s=10, ncols=4, figsize=(5.5, 3), legend=False, save='spatial.png'):
    """
    
    Spatial plot with ABA color maps.
    
    Parameters
    ----------
    adatas
        List of AnnData object of scanpy package
    spatial_key
        The key of 2D coordinates in .obsm
    level
        The annotation lavel of ABA
    """
    
    import seaborn as sns
    import matplotlib.patches as mpatches
    if isinstance(s, list):
        assert len(adatas) == len(s)
    else:
        s = [s]*len(adatas)  
    level_use = 'ABA_name_' + level
    color_use = 'ABA_color_' + level 
    if len(adatas) < ncols:
        ncols = len(adatas)
        nrows = 1
    else:
        nrows = int(np.ceil(len(adatas)/ncols))
    if (nrows == 1) & (ncols == 1):
        plt.figure(figsize=figsize)
        sns.scatterplot(data = adatas[0].obs,
                        x = 'stereo_ML',
                        y = 'stereo_DV',
                        label = level_use, 
                        color = adatas[0].obs[color_use], 
                        s = s[0],
                        legend=legend)
        if legend:
            plt.legend( handles=[mpatches.Patch(color=name_color[name_tmp], label=name_tmp) for name_tmp in adata.obs[level_use].cat.categories.tolist()], 
                            bbox_to_anchor=(1.0,1.0),
                            fontsize=5,
                            title_fontsize=5)
        plt.axis('off')
        plt.title(adatas[0].obs[title_key][0])
        plt.savefig(save, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
    else:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize,constrained_layout=True)   
        if nrows == 1:  
            for i in range(len(adatas)):
                axs[i].get_xaxis().set_visible(False)
                axs[i].get_yaxis().set_visible(False)
                axs[i].axis('off')
                axs[i].set_xlim([adatas[i].obsm[spatial_key].min(0)[0], adatas[i].obsm[spatial_key].max(0)[0]])
                axs[i].set_ylim([adatas[i].obsm[spatial_key].min(0)[1], adatas[i].obsm[spatial_key].max(0)[1]])
                sns.scatterplot(data = adatas[i].obs,
                                x = 'stereo_ML',
                                y = 'stereo_DV',
                                label = level_use, 
                                color = adatas[i].obs[color_use], 
                                s = s[i], 
                                ax=axs[i],
                                legend=legend)
                axs[i].set_title(adatas[i].obs[title_key][0])
                if legend:
                    if i == len(adatas)-1:
                        plt.legend( handles=[mpatches.Patch(color=name_color[name_tmp], label=name_tmp) for name_tmp in adata.obs[level_use].cat.categories.tolist()], 
                                    bbox_to_anchor=(1.0,1.0),
                                    fontsize=5,
                                    title_fontsize=5)
            fig.savefig(save, bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()
        else:
            index = 0
            for i in range(nrows):
                for j in range(ncols):
                    axs[i,j].get_xaxis().set_visible(False)
                    axs[i,j].get_yaxis().set_visible(False)
                    axs[i,j].axis('off')
                    axs[i,j].set_xlim([adatas[index].obsm[spatial_key].min(0)[0], adatas[index].obsm[spatial_key].max(0)[0]])
                    axs[i,j].set_ylim([adatas[index].obsm[spatial_key].min(0)[1], adatas[index].obsm[spatial_key].max(0)[1]])
                    axs[i,j].set_title(adatas[index].obs[title_key][0])
                    if index < len(adatas):
                        key = keys_use[index]
                        if j<(n_col-1):
                            sns.scatterplot(data = adatas[index].obs,
                                            x = 'stereo_ML',
                                            y = 'stereo_DV',
                                            label = level_use, 
                                            color = adatas[index].obs[color_use], 
                                            s = s[index], 
                                            ax=axs[i,j],
                                            legend=legend)
                            j += 1 
                        else:
                            sns.scatterplot(data = adatas[index].obs,
                                            x = 'stereo_ML',
                                            y = 'stereo_DV',
                                            label = level_use, 
                                            color = adatas[index].obs[color_use], 
                                            s = s[index], 
                                            ax=axs[i,j],
                                            legend=legend)
                            if legend:
                                plt.legend( handles=[mpatches.Patch(color=name_color[name_tmp], label=name_tmp) for name_tmp in adata.obs[level_use].cat.categories.tolist()], 
                                            bbox_to_anchor=(1.0,1.0),
                                            fontsize=5,
                                            title_fontsize=5)
                            i += 1
                            j = 0       
                    index += 1
            fig.savefig(save, bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()





