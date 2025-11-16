import torch
from torch import nn
import torch.nn.functional as F

import einops


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class MSDeformablePoints(nn.Module):

    def __init__(
        self, embed_dim, n_levels, n_heads,
        offset_range_factor=-1, 
    ):

        super().__init__()
        self.n_head_channels = embed_dim // n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.nc = self.n_head_channels * n_heads
        self.offset_range_factor = offset_range_factor

        self.kernel_sizes = [(n_levels - 1 - i)*2 + 1 for i in range(n_levels)] # [7, 5, 3, 1]
        self.strides = [2**(n_levels-i) for i in range(n_levels)] # [16, 8, 4, 2]

        self.conv_offset = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.n_head_channels, self.n_head_channels, self.kernel_sizes[i], self.strides[i], self.kernel_sizes[i] // 2, groups=self.n_heads),
            LayerNormProxy(self.n_head_channels),
            nn.GELU(),
            nn.Conv2d(self.n_head_channels, 2, 1, 1, 0, bias=False)
        ) for i in range(n_levels)])

        self.proj_q =  nn.ModuleList([nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        ) for _ in range(n_levels)])

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_heads, -1, -1, -1) # [B*g, H, W, 2]

        return ref
    
    def forward(self, x, spatial_shapes, level_start_index):
        B, C = x.size(0), x.size(2)
        dtype, device = x.dtype, x.device

        x_list = x.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        out = []
        for i in range(len(x_list)):
            cur_x = x_list[i]
            q = self.proj_q[i](einops.rearrange(cur_x, 'b (h w) c -> b c h w', h=spatial_shapes[i][0], w=spatial_shapes[i][1]))
            q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_heads, c=self.n_head_channels)
            offset = self.conv_offset[i](q_off).contiguous()  # [B*g, 2, Hg Wg]
            Hk, Wk = offset.size(2), offset.size(3)
            n_sample = Hk * Wk

            if self.offset_range_factor >= 0:
                offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
                offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

            offset = einops.rearrange(offset, 'b two h w -> b h w two')
            reference = self._get_ref_points(Hk, Wk, B, dtype, device)

            if self.offset_range_factor >= 0:
                pos = offset + reference
            else:
                pos = (offset + reference).clamp(-1., +1.)

            H, W = spatial_shapes[i]
            x_sampled = F.grid_sample(
                input=cur_x.reshape(B * self.n_heads, self.n_head_channels, H, W), # [B*g, Cg, H, W]
                grid=pos[..., (1, 0)], # y, x -> x, y: [B*g, Hg, Wg, 2]
                mode='bilinear', align_corners=True) # [B*g, Cg, Hg, Wg]
                    
            x_sampled = einops.rearrange(x_sampled, '(B g) C H W -> B (H W) (g C)', B=B)
            out.append(x_sampled)
        return torch.cat(out, dim=1)