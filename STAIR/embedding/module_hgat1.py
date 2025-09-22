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
    
    def forward(self, x, edge_index):
        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        z = F.dropout(x, self.dropout, training=self.training)
        
        xbar = self.gat2(z, edge_index)
        xbar = F.elu(xbar)
        xbar = F.dropout(xbar, self.dropout, training=self.training)
        
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
        dropout: float = 0.5,
        gamma: float = 0.9,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.heads = heads
        self.num_channels = num_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.gamma = gamma
        self.k_lin = nn.Linear(num_channels, num_channels)
        self.q = nn.Parameter(torch.Tensor(1, num_channels))
        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()  
        for edge_type in metadata[1]:         
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, num_channels))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, num_channels))
        print(self.lin_src)
        print(self.lin_dst)
        self.gat_layer = GAT_pyg(latent_dim=num_channels, dropout_gat=dropout)
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
    ):
        
        H, D = self.heads, self.num_channels
        x_gat_dict, x_hat_dict = {}, {}      # intra-slice embedding; inter-slice embedding 
        
        # Intra-slices aggregation
        for node_type, x in x_dict.items():
            x_gat_dict[node_type] = self.gat_layer(x, edge_index_dict[node_type, '0', node_type])[0].view(-1, H, D)
            x_hat_dict[node_type] = {}    
        
        # Inter-slices aggregation
        for edge_type, edge_index in edge_index_dict.items():
            
            src_type, _, dst_type = edge_type                
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
            
            out = F.relu(out)
            
            x_hat_dict[dst_type][src_type] = out
        
        # aggregating from other slices
        semantic_attn_dict = {}
        
        for node_type, x_hat_ in x_hat_dict.items():
            # x_gat_ = x_gat_dict[node_type]
            x_hat_out, attn = group(x_hat_, self.q, self.k_lin)
            # x_hat_out, attn = group(x_gat_, x_hat_, self.k_lin)
            x_hat_dict[node_type] = x_hat_out
            semantic_attn_dict[node_type] = attn
               
        if return_semantic_attention_weights:
            return x_hat_dict, semantic_attn_dict
        
        return x_hat_dict
    
    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        
        return out.view(-1, self.num_channels)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_channels}, '
                f'heads={self.heads})')




