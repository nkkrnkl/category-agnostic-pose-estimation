import numpy as np
import torch
class DiscreteTokenizer(object):
    """
    Discrete tokenizer for sequence generation.
    """
    def __init__(self, num_bins, seq_len, add_cls=False):
        """
        Initialize discrete tokenizer.
        
        Args:
            num_bins: Number of bins per dimension
            seq_len: Maximum sequence length
            add_cls: Whether to add CLS token
        """
        self.num_bins = num_bins
        vocab_size = num_bins * num_bins
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.bos = vocab_size + 0
        self.eos = vocab_size + 1
        self.sep = vocab_size + 2
        self.pad = vocab_size + 3
        if add_cls:
            self.cls = vocab_size + 4
            self.vocab_size = vocab_size + 5
        else:
            self.vocab_size = vocab_size + 4
    def __len__(self):
        """
        Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return self.vocab_size
    def __call__(self, seq, add_bos, add_eos, dtype):
        """
        Tokenize sequence.
        
        Args:
            seq: Input sequence
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            dtype: Data type for output tensor
        
        Returns:
            Tokenized sequence tensor
        """
        out = []
        if add_bos:
            out = [self.bos]
        num_extra = 1 if not self.add_cls else 2
        for sub in seq:
            cur_len = len(out)
            if cur_len + len(sub) + num_extra <= self.seq_len:
                out.extend(sub)
            else:
                break
            if self.add_cls:
                out.append(self.cls)
            out.append(self.sep)
        if out and out[-1] == self.sep:
            out.pop(-1)
        if self.seq_len > len(out):
            out.extend([self.pad] * (self.seq_len - len(out)))
        if add_eos:
            out[-1] = self.eos
        return torch.tensor(out, dtype=dtype)
    def _padding(self, seq, pad_value, dtype):
        """
        Pad sequence to fixed length.
        
        Args:
            seq: Input sequence
            pad_value: Padding value
            dtype: Data type for output tensor
        
        Returns:
            Padded sequence tensor
        """
        if self.seq_len > len(seq):
            seq.extend([pad_value] * (self.seq_len - len(seq)))
        return torch.tensor(np.array(seq), dtype=dtype)
class DiscreteTokenizerV2(DiscreteTokenizer):
    """
    Discrete tokenizer version 2 with index tracking.
    """
    def __call__(self, seq, add_bos, add_eos, dtype, return_indices=False):
        """
        Tokenize sequence with optional index tracking.
        
        Args:
            seq: Input sequence
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            dtype: Data type for output tensor
            return_indices: Whether to return indices
        
        Returns:
            Tokenized sequence tensor, optionally with indices
        """
        out = []
        if add_bos:
            out = [self.bos]
        num_extra = 1 if not self.add_cls else 2
        indices = []
        for i, sub in enumerate(seq):
            cur_len = len(out)
            if cur_len + len(sub) + num_extra <= self.seq_len:
                out.extend(sub)
                indices.append(i)
            else:
                continue
            if self.add_cls:
                out.append(self.cls)
            out.append(self.sep)
        if out and out[-1] == self.sep:
            out.pop(-1)
        if self.seq_len > len(out):
            out.extend([self.pad] * (self.seq_len - len(out)))
        if add_eos:
            out[-1] = self.eos
        if return_indices:
            return torch.tensor(out, dtype=dtype), indices
        return torch.tensor(out, dtype=dtype)