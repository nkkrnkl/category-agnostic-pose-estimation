import torch
from torch import nn
from timm.layers import DropPath
from timm.layers import Mlp
class LayerScale(nn.Module):
    """
    Layer scale module.
    """
    def __init__(self, dim, init_values=1e-5, inplace=False):
        """
        Initialize layer scale.
        
        Args:
            dim: Dimension
            init_values: Initial values
            inplace: Whether to do inplace operation
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Scaled tensor
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
class BiXAttn(nn.Module):
    """
    BiXAttn module for bi-attention between latents and patches/tokens.
    """
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Initialize BiXAttn.
        
        Args:
            dim_lat: Latent dimension
            dim_pat: Patch dimension
            dim_attn: Attention dimension
            num_heads: Number of attention heads
            rv_bias: Whether to use bias in rv projections
            attn_drop: Attention dropout
            proj_drop: Projection dropout
        """
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn
        self.rv_latents = nn.Linear(dim_lat, dim_attn * 2, bias=rv_bias)
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_dropT = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)
        self.proj_drop_lat = nn.Dropout(proj_drop)
        self.proj_pat = nn.Linear(dim_attn, dim_pat)
        self.proj_drop_pat = nn.Dropout(proj_drop)
    def forward(self, x_latents, x_patches):
        B_lat, N_lat, _ = x_latents.shape
        B_pat, N_pat, _ = x_patches.shape
        rv_lat = self.rv_latents(x_latents).reshape(B_lat, N_lat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_lat, v_lat = rv_lat.unbind(0)
        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale
        attn_T = attn.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_T = attn_T.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_T = self.attn_dropT(attn_T)
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)
        x_patches = (attn_T @ v_lat).transpose(1, 2).reshape(B_pat, N_pat, self.dim_attn)
        x_patches = self.proj_pat(x_patches)
        x_patches = self.proj_drop_pat(x_patches)
        return x_latents, x_patches
class BiXAttnBlock(nn.Module):
    """
    Block performing Cross-Attention between latents and input tokens.
    """
    def __init__(
            self, dim_lat, dim_pat, dim_attn, num_heads, rv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            lat_mlp_ratio=4., pat_mlp_ratio=4.):
        """
        Initialize BiXAttnBlock.
        
        Args:
            dim_lat: Latent dimension
            dim_pat: Patch dimension
            dim_attn: Attention dimension
            num_heads: Number of attention heads
            rv_bias: Whether to use bias in rv projections
            drop: Dropout rate
            attn_drop: Attention dropout
            init_values: Initial values for layer scale
            drop_path: Drop path rate
            act_layer: Activation layer
            norm_layer: Normalization layer
            lat_mlp_ratio: MLP ratio for latents
            pat_mlp_ratio: MLP ratio for patches
        """
        super().__init__()
        self.norm1_lat = norm_layer(dim_lat)
        self.norm1_pat = norm_layer(dim_pat)
        self.attn = BiXAttn(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn, num_heads=num_heads,
                                   rv_bias=rv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.ls1_pat = LayerScale(dim_pat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1_pat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_lat = norm_layer(dim_lat)
        self.mlp_lat = Mlp(in_features=dim_lat, hidden_features=int(dim_lat * lat_mlp_ratio),
                           act_layer=act_layer, drop=drop)
        self.ls2_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_pat = norm_layer(dim_pat)
        self.mlp_pat = Mlp(in_features=dim_pat, hidden_features=int(dim_pat * pat_mlp_ratio),
                           act_layer=act_layer, drop=drop)
        self.ls2_pat = LayerScale(dim_pat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_pat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x_latents, x_patches):
        x_lat = self.norm1_lat(x_latents)
        x_pat = self.norm1_pat(x_patches)
        x_lat, x_pat = self.attn(x_lat, x_pat)
        x_latents = x_latents + self.drop_path1_lat(self.ls1_lat(x_lat))
        x_latents = x_latents + self.drop_path2_lat(self.ls2_lat(self.mlp_lat(self.norm2_lat(x_latents))))
        x_patches = x_patches + self.drop_path1_pat(self.ls1_pat(x_pat))
        x_patches = x_patches + self.drop_path2_pat(self.ls2_pat(self.mlp_pat(self.norm2_pat(x_patches))))
        return x_latents, x_patches
class CrossAttentionOneSided(nn.Module):
    """
    Cross-Attention between latents and input tokens, only returning refined latents.
    """
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Initialize CrossAttentionOneSided.
        
        Args:
            dim_lat: Latent dimension
            dim_pat: Patch dimension
            dim_attn: Attention dimension
            num_heads: Number of attention heads
            rv_bias: Whether to use bias in rv projections
            attn_drop: Attention dropout
            proj_drop: Projection dropout
        """
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn
        self.r_latents = nn.Linear(dim_lat, dim_attn, bias=rv_bias)
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)
        self.proj_drop_lat = nn.Dropout(proj_drop)
    def forward(self, x_latents, x_patches):
        B_lat, N_lat, _ = x_latents.shape
        B_pat, N_pat, _ = x_patches.shape
        r_lat = self.r_latents(x_latents).reshape(B_lat, N_lat, 1, self.num_heads,
                                                  self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)
        return x_latents
class CAOneSidedBlock(nn.Module):
    """
    Block performing one-sided Cross-Attention between latents and input tokens.
    """
    def __init__(
            self, dim_lat, dim_pat, dim_attn, num_heads, rv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            lat_mlp_ratio=4.):
        """
        Initialize CAOneSidedBlock.
        
        Args:
            dim_lat: Latent dimension
            dim_pat: Patch dimension
            dim_attn: Attention dimension
            num_heads: Number of attention heads
            rv_bias: Whether to use bias in rv projections
            drop: Dropout rate
            attn_drop: Attention dropout
            init_values: Initial values for layer scale
            drop_path: Drop path rate
            act_layer: Activation layer
            norm_layer: Normalization layer
            lat_mlp_ratio: MLP ratio for latents
        """
        super().__init__()
        self.norm1_lat = norm_layer(dim_lat)
        self.norm1_pat = norm_layer(dim_pat)
        self.attn = CrossAttentionOneSided(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn, num_heads=num_heads,
                                           rv_bias=rv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_lat = norm_layer(dim_lat)
        self.mlp_lat = Mlp(in_features=dim_lat, hidden_features=int(dim_lat * lat_mlp_ratio),
                           act_layer=act_layer, drop=drop)
        self.ls2_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x_latents, x_patches):
        """
        Forward pass.
        
        Args:
            x_latents: Latent tokens
            x_patches: Patch tokens
        
        Returns:
            Tuple of (refined_latents, None)
        """
        x_lat = self.norm1_lat(x_latents)
        x_pat = self.norm1_pat(x_patches)
        x_lat = self.attn(x_lat, x_pat)
        x_latents = x_latents + self.drop_path1_lat(self.ls1_lat(x_lat))
        x_latents = x_latents + self.drop_path2_lat(self.ls2_lat(self.mlp_lat(self.norm2_lat(x_latents))))
        return x_latents, None