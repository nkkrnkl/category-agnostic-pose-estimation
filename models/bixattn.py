import torch
from torch import nn

from timm.layers import DropPath
from timm.layers import Mlp

# LayerScale NOT used by default, but might be beneficial for larger / deeper models
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class BiXAttn(nn.Module):
    """BiXAttn module for bi-attention between latents and patches/tokens"""
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.rv_latents = nn.Linear(dim_lat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_dropT = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)             # 'out-projection' for latents
        self.proj_drop_lat = nn.Dropout(proj_drop)
        self.proj_pat = nn.Linear(dim_attn, dim_pat)             # 'out-projection' for patches/tokens
        self.proj_drop_pat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        B_lat, N_lat, _ = x_latents.shape  # Note: need B_lat since 1 at very first pass, then broadcasted/extended to bs
        B_pat, N_pat, _ = x_patches.shape
        rv_lat = self.rv_latents(x_latents).reshape(B_lat, N_lat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_lat, v_lat = rv_lat.unbind(0)
        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale  # query from latent, key from patches
        attn_T = attn.transpose(-2, -1)  # bidirectional attention, associated with the values from the query q

        attn = attn.softmax(dim=-1)  # softmax along patch token dimension
        attn_T = attn_T.softmax(dim=-1)  # softmax along latent token dimension

        attn = self.attn_drop(attn)
        attn_T = self.attn_dropT(attn_T)

        # Retrieve information form the patch tokens via latent query:
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)

        # Likewise, store information from the latents in the patch tokens via transposed attention:
        x_patches = (attn_T @ v_lat).transpose(1, 2).reshape(B_pat, N_pat, self.dim_attn)
        x_patches = self.proj_pat(x_patches)
        x_patches = self.proj_drop_pat(x_patches)

        return x_latents, x_patches


class BiXAttnBlock(nn.Module):
    """Block performing Cross-Attention between the latents and input tokens, bi-directional attention"""
    def __init__(
            self, dim_lat, dim_pat, dim_attn, num_heads, rv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            lat_mlp_ratio=4., pat_mlp_ratio=4.):
        super().__init__()
        self.norm1_lat = norm_layer(dim_lat)
        self.norm1_pat = norm_layer(dim_pat)
        self.attn = BiXAttn(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn, num_heads=num_heads,
                                   rv_bias=rv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.ls1_pat = LayerScale(dim_pat, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, papers/repos indicate that that's better than dropout here
        self.drop_path1_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1_pat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Latents -- components for further refinement after attention:
        self.norm2_lat = norm_layer(dim_lat)
        self.mlp_lat = Mlp(in_features=dim_lat, hidden_features=int(dim_lat * lat_mlp_ratio),
                           act_layer=act_layer, drop=drop)

        self.ls2_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Patches -- components for further refinement after attention:
        self.norm2_pat = norm_layer(dim_pat)
        self.mlp_pat = Mlp(in_features=dim_pat, hidden_features=int(dim_pat * pat_mlp_ratio),
                           act_layer=act_layer, drop=drop)

        self.ls2_pat = LayerScale(dim_pat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_pat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_latents, x_patches):
        # Cross attention forwards
        x_lat = self.norm1_lat(x_latents)
        x_pat = self.norm1_pat(x_patches)
        x_lat, x_pat = self.attn(x_lat, x_pat)

        x_latents = x_latents + self.drop_path1_lat(self.ls1_lat(x_lat))
        x_latents = x_latents + self.drop_path2_lat(self.ls2_lat(self.mlp_lat(self.norm2_lat(x_latents))))

        x_patches = x_patches + self.drop_path1_pat(self.ls1_pat(x_pat))
        x_patches = x_patches + self.drop_path2_pat(self.ls2_pat(self.mlp_pat(self.norm2_pat(x_patches))))

        return x_latents, x_patches


class CrossAttentionOneSided(nn.Module):
    """Cross-Attention between latents and input tokens -- only returning the refined latents here, used at the last
        stage of the BiXT (since we don't use the patch tokens afterwards, we can save the compute) """
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.r_latents = nn.Linear(dim_lat, dim_attn, bias=rv_bias)             # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)        # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)                            # 'out-projection' for latents
        self.proj_drop_lat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        B_lat, N_lat, _ = x_latents.shape  # Note: need B_lat since 1 at very first pass, then broadcasted/extended to bs
        B_pat, N_pat, _ = x_patches.shape
        r_lat = self.r_latents(x_latents).reshape(B_lat, N_lat, 1, self.num_heads,
                                                  self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)

        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)   # softmax along patch token dimension
        attn = self.attn_drop(attn)

        # Retrieve information form the patch tokens via latent query:
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)

        return x_latents


class CAOneSidedBlock(nn.Module):
    """Block performing one-sided Cross-Attention between the latents and input tokens, no information transfer
       to input tokens for this block!"""
    def __init__(
            self, dim_lat, dim_pat, dim_attn, num_heads, rv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            lat_mlp_ratio=4.):
        super().__init__()
        self.norm1_lat = norm_layer(dim_lat)
        self.norm1_pat = norm_layer(dim_pat)
        self.attn = CrossAttentionOneSided(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn, num_heads=num_heads,
                                           rv_bias=rv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Latents -- components for further refinement after attention:
        self.norm2_lat = norm_layer(dim_lat)

        self.mlp_lat = Mlp(in_features=dim_lat, hidden_features=int(dim_lat * lat_mlp_ratio),
                           act_layer=act_layer, drop=drop)
        self.ls2_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_latents, x_patches):
        # Cross attention forwards
        x_lat = self.norm1_lat(x_latents)
        x_pat = self.norm1_pat(x_patches)
        x_lat = self.attn(x_lat, x_pat)

        x_latents = x_latents + self.drop_path1_lat(self.ls1_lat(x_lat))
        x_latents = x_latents + self.drop_path2_lat(self.ls2_lat(self.mlp_lat(self.norm2_lat(x_latents))))

        return x_latents, None