import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv

from typing import Dict, List, Optional, Tuple, Union


# ============================================================================ #
# GAT Module

class GAT_pyg(nn.Module):
    
    def __init__(self, latent_dim=32, dropout_gat=0.5):
        super(GAT_pyg, self).__init__()
        
        self.dropout = dropout_gat
        self.gat1 = GATConv(latent_dim, latent_dim, 1)
        self.gat2 = GATConv(latent_dim, latent_dim, 1)
    
    def forward(self, x, edge_index, get_attention=False):
        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        z = F.dropout(x, self.dropout, training=self.training)
        
        if get_attention:
            xbar, atte_tmp = self.gat2(z, edge_index, return_attention_weights=True)
        else:  
            xbar = self.gat2(z, edge_index)
        
        xbar = F.elu(xbar)
        xbar = F.dropout(xbar, self.dropout, training=self.training)
        
        if get_attention:
            return xbar, z, atte_tmp
        return xbar, z


# ============================================================================ #
# corss graph attention

# def group(
#     x_gat_,
#     xs: Dict,
#     k_lin: nn.Module,
# ) -> Tuple[OptTensor, OptTensor]:
#     if len(xs) == 0:
#         return None, None    
#     else:
#         num_edge_types = len(xs)
#         out = torch.stack(list(xs.values()))     
#         if out.numel() == 0:
#             return out.view(0, out.size(-1)), None
#         attn_score = (torch.tanh(k_lin(x_gat_.mean(1)).mean(0)) * torch.tanh(k_lin(out)).mean(1)).sum(-1)
#         attn = F.softmax(attn_score, dim=0)
#         out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
#     return out, attn


def group(
    xs: Dict,
    q: nn.Parameter,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:
    if len(xs) == 0:
        return None, None
    else:
        num_edge_types = len(xs)
        out = torch.stack(list(xs.values()))
        if out.numel() == 0:
            return out.view(0, out.size(-1)), None
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out, attn


# def group(
#     xs: List[Tensor],
#     q: nn.Parameter,
#     k_lin: nn.Module,
# ) -> Tuple[OptTensor, OptTensor]:
#     if len(xs) == 0:
#         return None, None
#     else:
#         num_edge_types = len(xs)
#         out = torch.stack(xs)
#         if out.numel() == 0:
#             return out.view(0, out.size(-1)), None
#         attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
#         attn = F.softmax(attn_score, dim=0)
#         out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
#         return out, attn


class HGAT(MessagePassing):
    
    def __init__(
        self,
        num_channels: Union[int, Dict[str, int]],
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout_hom: float = 0.7,
        dropout_het: float = 0.5,
        gamma: float = 0.9,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.heads = heads
        self.num_channels = num_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout_hom = dropout_hom
        self.dropout_het = dropout_het
        self.gamma = gamma
        self.k_lin = nn.Linear(num_channels, num_channels)
        self.q = nn.Parameter(torch.Tensor(1, num_channels))
        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()  
        for edge_type in metadata[1]:
            if edge_type[0] != edge_type[-1]:            
                edge_type = '__'.join(edge_type)
                self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, num_channels))
                self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, num_channels))
        # print(self.lin_src)
        # print(self.lin_dst)
        self.gat_layer = GAT_pyg(latent_dim=num_channels, dropout_gat=self.dropout_hom)
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)
    
    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        return_semantic_attention_weights: bool = False,
        get_attention: bool = False
    ):
        
        H, D = self.heads, self.num_channels
        x_gat_dict, x_hat_dict = {}, {}      # intra-slice embedding; inter-slice embedding 
        cell_attn_dict = {}

        # Intra-slices aggregation
        for node_type, x in x_dict.items():
            if get_attention:
                xbar_tmp, z_tmp, atte_tmp = self.gat_layer( x, 
                                                            edge_index_dict[node_type, '0', node_type],
                                                            get_attention = True)
                cell_attn_dict[(node_type, '0', node_type)] = atte_tmp

            else:
                xbar_tmp, z_tmp = self.gat_layer( x, edge_index_dict[node_type, '0', node_type])
            
            x_gat_dict[node_type] = xbar_tmp.view(-1, H, D)
            x_hat_dict[node_type] = {}    
        
        # Inter-slices aggregation
        for edge_type, edge_index in edge_index_dict.items():
            
            src_type, _, dst_type = edge_type
            
            if src_type != dst_type:
                
                edge_type = '__'.join(edge_type)
                
                lin_src = self.lin_src[edge_type]
                lin_dst = self.lin_dst[edge_type]
                
                x_src = x_gat_dict[src_type]
                x_dst = x_gat_dict[dst_type]
                
                alpha_src = (x_src * lin_src).sum(dim=-1)
                alpha_dst = (x_dst * lin_dst).sum(dim=-1)
                
                # propagate_type: (x_dst: PairTensor, alpha: PairTensor)
                out = self.propagate(edge_index, 
                                    x=(x_src, x_dst),
                                    alpha=(alpha_src, alpha_dst), size=None)

                x_hat_dict[dst_type][src_type] = F.relu(out)

                if get_attention:
                    # N1, N2 = x_src.shape[0], x_dst.shape[0]
                    # alpha = alpha_src.repeat(N2, 1).T + alpha_dst.repeat(N1, 1)
                    # alpha = F.leaky_relu(alpha, self.negative_slope)
                    # alpha = softmax(alpha, index, ptr, size_i)
                    cell_attn_dict[(dst_type, '1', src_type)] = (x_src, x_dst)

        
        # aggregating from other slices
        semantic_attn_dict = {}
        
        for node_type, x_hat_ in x_hat_dict.items():
            # x_gat_ = x_gat_dict[node_type]
            x_hat_out, attn = group(x_hat_, self.q, self.k_lin)
            # x_hat_out, attn = group(x_gat_, x_hat_, self.k_lin)
            x_hat_dict[node_type] = x_hat_out
            semantic_attn_dict[node_type] = attn
        
        out_gat_dict = {key:value.sum(1) for key, value in x_gat_dict.items()}
        # print(out_gat_dict)
        # print(x_hat_dict)

        # for key in x_hat_dict.keys():
        #     if x_hat_dict[key] is None:
        #         x_hat_dict[key] = torch.zeros_like(out_gat_dict[key], device=out_gat_dict[key].device)

        out = {key:self.gamma*out_gat_dict[key] + (1-self.gamma)*x_hat_dict[key] for key in x_hat_dict.keys()}
        
        if get_attention:
            if return_semantic_attention_weights:
                return out, semantic_attn_dict, cell_attn_dict
            return out, cell_attn_dict
        
        else:
            if return_semantic_attention_weights:
                return out, semantic_attn_dict
            return out
    
    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout_het, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        
        return out.view(-1, self.num_channels)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_channels}, '
                f'heads={self.heads})')




