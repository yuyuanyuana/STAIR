library(monocle3)


traj_monocle3_r <- function(expression_matrix, cell_metadata, emb_use, cluster_use, start_cluster=None, name=None, result_path='.', seed=100){
    set.seed(seed)
    cds <- new_cell_data_set(t(as.matrix(expression_matrix)),
                             cell_metadata = cell_metadata)
    num <- dim(cell_metadata)[2]-3
    SingleCellExperiment::reducedDims(cds)[[emb_use]] = as.matrix(cell_metadata[,paste0(emb_use,as.character(seq(0, num-1, by=1)))])
    SingleCellExperiment::reducedDims(cds)[["UMAP"]] = as.matrix(cell_metadata[,c('umap1', 'umap2')])

    cds <- cluster_cells(cds, reduction_method = "UMAP")
    pdf(paste0(result_path,'/', name, 'clusters.pdf'), width=4, height=4)
    print(plot_cells(cds, color_cells_by = cluster_use))
    dev.off()
    pdf(paste0(result_path,'/', name, 'partition.pdf'), width=4, height=4)
    print(plot_cells(cds, color_cells_by = 'partition'))
    dev.off()
    cds <- learn_graph(cds)


    get_earliest_principal_node <- function(cds, time_bin=start_cluster){
                                                                            cell_ids <- which(colData(cds)[, cluster_use] == time_bin)
                                                                            closest_vertex <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
                                                                            closest_vertex <- as.matrix(closest_vertex[colnames(cds), ])
                                                                            root_pr_nodes <- igraph::V(principal_graph(cds)[["UMAP"]])$name[as.numeric(names(which.max(table(closest_vertex[cell_ids,]))))]
                                                                            root_pr_nodes
                                                                        }   
    cds <- order_cells(cds, root_pr_nodes=get_earliest_principal_node(cds))


    pseudo <- cds@ principal_graph_aux@ listData$ UMAP$ pseudotime
    pdf(paste0(result_path,'/', name, 'pseudotime.pdf'), width=4, height=4)
    print(plot_cells(cds,
               color_cells_by = "pseudotime",
               label_cell_groups=FALSE,
               label_leaves=FALSE,
               label_branch_points=FALSE,
               graph_label_size=1.5))
    dev.off()
    ciliated_cds_pr_test_res <- graph_test(cds, neighbor_graph="principal_graph", cores=4)
    pr_deg_ids <- row.names(subset(ciliated_cds_pr_test_res, q_value < 0.00001))
    l <- c(pseudo, pr_deg_ids)
    return(l)
}



