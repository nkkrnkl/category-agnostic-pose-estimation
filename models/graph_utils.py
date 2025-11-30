"""
Graph utilities for geometry-only CAPE.
"""
import torch
import torch.nn as nn
def adj_from_skeleton(num_pts, skeleton, mask, device='cuda'):
    """
    Build normalized adjacency matrix from skeleton edges.
    
    Args:
        num_pts: Number of keypoints
        skeleton: List of edge lists, one per batch element
        mask: Boolean mask, shape [bs, num_pts]
        device: Device to create tensors on
    
    Returns:
        adj: Dual-channel adjacency matrix, shape [bs, 2, num_pts, num_pts]
    """
    batch_size = len(skeleton)
    adj_mx = torch.zeros(batch_size, num_pts, num_pts, device=device, dtype=torch.float32)
    for b in range(batch_size):
        skeleton_b = skeleton[b]
        if skeleton_b is not None and len(skeleton_b) > 0:
            min_idx = None
            for edge in skeleton_b:
                try:
                    if not isinstance(edge, (list, tuple)):
                        continue
                    if len(edge) != 2:
                        continue
                    src_val = int(edge[0])
                    dst_val = int(edge[1])
                    edge_min = min(src_val, dst_val)
                    if min_idx is None or edge_min < min_idx:
                        min_idx = edge_min
                except (ValueError, TypeError, IndexError):
                    continue
            is_1indexed = (min_idx is not None) and (min_idx >= 1)
            valid_edges = []
            for edge in skeleton_b:
                try:
                    if not isinstance(edge, (list, tuple)):
                        continue
                    if len(edge) != 2:
                        continue
                    src = int(edge[0])
                    dst = int(edge[1])
                    if is_1indexed:
                        src = src - 1 if src > 0 else src
                        dst = dst - 1 if dst > 0 else dst
                    if 0 <= src < num_pts and 0 <= dst < num_pts:
                        valid_edges.append([src, dst])
                except (ValueError, TypeError, IndexError, OverflowError):
                    continue
            if len(valid_edges) > 0:
                try:
                    edges_tensor = torch.tensor(valid_edges, dtype=torch.long, device=device)
                    if edges_tensor.numel() > 0:
                        if (edges_tensor >= 0).all() and (edges_tensor < num_pts).all():
                            adj_mx[b].index_put_(
                                (edges_tensor[:, 0], edges_tensor[:, 1]),
                                torch.ones(len(valid_edges), device=device, dtype=torch.float32)
                            )
                except (RuntimeError, ValueError, IndexError) as e:
                    import warnings
                    warnings.warn(
                        f"Failed to create adjacency matrix for batch element {b}: {e}. "
                        f"Using empty adjacency (no skeleton edges)."
                    )
    adj_mx = torch.maximum(adj_mx, adj_mx.transpose(-2, -1))
    if mask.dtype != torch.bool:
        mask = mask.bool()
    adj_mx = adj_mx * (~mask[..., None]).float() * (~mask[:, None]).float()
    row_sums = adj_mx.sum(dim=-1, keepdim=True)
    adj_mx = torch.nan_to_num(adj_mx / (row_sums + 1e-8))
    self_loops = torch.diag_embed((~mask).float())
    adj = torch.stack((self_loops, adj_mx), dim=1)
    return adj
class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer with dual-channel adjacency.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        kernel_size: Number of adjacency channels
        use_bias: Whether to use bias
        activation: Activation function
        batch_first: Whether batch dimension is first
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
        if not self.batch_first:
            x = x.permute(1, 2, 0)
        else:
            x = x.transpose(1, 2)
        x = self.conv(x)
        b, kc, v = x.size()
        x = x.view(b, self.kernel_size, kc // self.kernel_size, v)
        x = torch.einsum('bkcv,bkvw->bcw', (x, adj))
        if self.activation is not None:
            x = self.activation(x)
        if not self.batch_first:
            x = x.permute(2, 0, 1)
        else:
            x = x.transpose(1, 2)
        return x