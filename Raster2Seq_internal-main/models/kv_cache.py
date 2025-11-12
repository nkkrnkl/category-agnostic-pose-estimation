import torch
from torch import nn


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, model_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, model_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, S, H, D]
        index = input_pos[0].long() + 1
        self.k_cache[:, input_pos, ...] = k_val
        self.v_cache[:, input_pos, ...] = v_val

        return self.k_cache[:, :index], self.v_cache[:, :index]


class VCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, num_heads, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, num_heads, head_dim)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, v_val):
        self.v_cache = v_val
    
    def get(self):
        return self.v_cache