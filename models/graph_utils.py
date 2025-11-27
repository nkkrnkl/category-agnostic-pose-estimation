"""
Graph utilities for geometry-only CAPE.

This module contains graph convolution components copied from CapeX for
processing skeleton structure in pose estimation.

Source: capex-code/models/models/utils/encoder_decoder.py
Lines: 507-555
"""

import torch
import torch.nn as nn


def adj_from_skeleton(num_pts, skeleton, mask, device='cuda'):
    """
    Build normalized adjacency matrix from skeleton edges.
    
    Copied from CapeX encoder_decoder.py:507-521
    
    This function creates a dual-channel adjacency matrix for graph convolution:
    - Channel 0: Self-loops (identity matrix with masked points zeroed)
    - Channel 1: Normalized neighbor adjacency (row-normalized edge connections)
    
    Args:
        num_pts (int): Number of keypoints
        skeleton (list): List of edge lists [[[i,j], ...], ...] (one per batch element)
                        Edges are 0-indexed
        mask (torch.Tensor): Boolean mask [bs, num_pts] (True=invalid/invisible keypoint)
        device (str): Device to create tensors on ('cuda', 'cpu', 'mps')
    
    Returns:
        adj (torch.Tensor): [bs, 2, num_pts, num_pts] dual-channel adjacency matrix
            - adj[:, 0, :, :]: Self-loop channel (diagonal)
            - adj[:, 1, :, :]: Neighbor channel (normalized adjacency)
    
    Example:
        >>> skeleton = [[[0,1], [1,2], [2,0]]]  # Triangle
        >>> mask = torch.zeros(1, 3).bool()
        >>> adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
        >>> adj.shape
        torch.Size([1, 2, 3, 3])
        >>> torch.allclose(adj[0,1], adj[0,1].T)  # Symmetric
        True
    """
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    
    # Build adjacency matrix for each batch element
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b], device=device)
        adj = torch.zeros(num_pts, num_pts, device=device)
        if len(edges) > 0:  # Only add edges if skeleton is non-empty
            adj[edges[:, 0], edges[:, 1]] = 1
        adj_mx = torch.concatenate((adj_mx, adj.unsqueeze(0)), dim=0)
    
    # Make symmetric (undirected graph)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    
    # Zero out rows/columns for masked keypoints
    # CRITICAL: Ensure mask is boolean before using ~ operator
    if mask.dtype != torch.bool:
        mask = mask.bool()
    
    adj = adj * ~mask[..., None] * ~mask[:, None]
    
    # Row-normalize (each row sums to 1)
    adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
    
    # Stack dual-channel: [self-loops, neighbors]
    adj = torch.stack((torch.diag_embed((~mask).float()), adj), dim=1)
    
    return adj


class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer with dual-channel adjacency.
    
    Copied from CapeX encoder_decoder.py:524-555
    
    This layer performs graph convolution using a dual-channel adjacency matrix
    (self-loops + neighbor connections). It uses Conv1d + einsum for efficient
    batched graph convolution.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        kernel_size (int): Number of adjacency channels (typically 2: self + neighbors)
        use_bias (bool): Whether to use bias in convolution (default: True)
        activation: Activation function (default: ReLU(inplace=True))
        batch_first (bool): If True, expects input shape [bs, num_pts, feat_dim]
                           If False, expects [num_pts, bs, feat_dim]
    
    Shape:
        - Input (batch_first=True): [bs, num_pts, in_features]
        - Input (batch_first=False): [num_pts, bs, in_features]
        - Adjacency: [bs, kernel_size, num_pts, num_pts]
        - Output (batch_first=True): [bs, num_pts, out_features]
        - Output (batch_first=False): [num_pts, bs, out_features]
    
    Example:
        >>> gcn = GCNLayer(256, 256, kernel_size=2, batch_first=True)
        >>> x = torch.rand(2, 10, 256)  # [bs, num_pts, features]
        >>> adj = torch.rand(2, 2, 10, 10)  # [bs, 2, num_pts, num_pts]
        >>> out = gcn(x, adj)
        >>> out.shape
        torch.Size([2, 10, 256])
    """
    
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=2,
                 use_bias=True,
                 activation=nn.ReLU(inplace=True),
                 batch_first=True):
        super(GCNLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_features, 
            out_features * kernel_size, 
            kernel_size=1,
            padding=0, 
            stride=1, 
            dilation=1, 
            bias=use_bias
        )
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_first = batch_first

    def forward(self, x, adj):
        """
        Forward pass with graph convolution.
        
        Args:
            x (torch.Tensor): Node features
                - If batch_first=True: [bs, num_pts, in_features]
                - If batch_first=False: [num_pts, bs, in_features]
            adj (torch.Tensor): Adjacency matrix [bs, kernel_size, num_pts, num_pts]
        
        Returns:
            out (torch.Tensor): Graph-convolved features
                - If batch_first=True: [bs, num_pts, out_features]
                - If batch_first=False: [num_pts, bs, out_features]
        """
        assert adj.size(1) == self.kernel_size, \
            f"Adjacency channel dim {adj.size(1)} must match kernel_size {self.kernel_size}"
        
        # Reshape for Conv1d: [bs, in_features, num_pts]
        if not self.batch_first:
            x = x.permute(1, 2, 0)  # [num_pts, bs, C] -> [bs, C, num_pts]
        else:
            x = x.transpose(1, 2)   # [bs, num_pts, C] -> [bs, C, num_pts]
        
        # Apply 1D convolution to expand features
        x = self.conv(x)  # [bs, out_features * kernel_size, num_pts]
        
        # Reshape to separate kernel channels
        b, kc, v = x.size()
        x = x.view(b, self.kernel_size, kc // self.kernel_size, v)
        # [bs, kernel_size, out_features, num_pts]
        
        # Graph convolution via einsum
        # Aggregates features from neighbors weighted by adjacency
        x = torch.einsum('bkcv,bkvw->bcw', (x, adj))
        # [bs, out_features, num_pts]
        
        # Apply activation
        if self.activation is not None:
            x = self.activation(x)
        
        # Reshape back to original format
        if not self.batch_first:
            x = x.permute(2, 0, 1)  # [bs, C, num_pts] -> [num_pts, bs, C]
        else:
            x = x.transpose(1, 2)   # [bs, C, num_pts] -> [bs, num_pts, C]
        
        return x

