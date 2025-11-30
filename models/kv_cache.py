import torch
from torch import nn
class KVCache(nn.Module):
    """
    Key-Value cache for autoregressive generation.
    """
    def __init__(self, max_batch_size, max_seq_length, model_dim, dtype):
        """
        Initialize KV cache.
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
            model_dim: Model dimension
            dtype: Data type
        """
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, model_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
    def update(self, input_pos, k_val, v_val):
        """
        Update cache with new key-value pairs.
        
        Args:
            input_pos: Input positions
            k_val: Key values
            v_val: Value values
        
        Returns:
            Tuple of (k_cache, v_cache)
        """
        index = input_pos[0].long() + 1
        self.k_cache[:, input_pos, ...] = k_val
        self.v_cache[:, input_pos, ...] = v_val
        return self.k_cache[:, :index], self.v_cache[:, :index]
class VCache(nn.Module):
    """
    Value cache for autoregressive generation.
    """
    def __init__(self, max_batch_size, max_seq_length, num_heads, head_dim, dtype):
        """
        Initialize V cache.
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Head dimension
            dtype: Data type
        """
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, num_heads, head_dim)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
    def update(self, v_val):
        """
        Update cache with new value.
        
        Args:
            v_val: Value tensor
        """
        self.v_cache = v_val
    def get(self):
        """
        Get cached values.
        
        Returns:
            Cached value tensor
        """
        return self.v_cache