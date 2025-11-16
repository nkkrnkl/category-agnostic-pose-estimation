import numpy as np
import torch

class DiscreteTokenizer(object):
    def __init__(self, num_bins, seq_len, add_cls=False):
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
        return self.vocab_size

    def __call__(self, seq, add_bos, add_eos, dtype):
        out = []
        if add_bos:
            out = [self.bos]
        num_extra = 1 if not self.add_cls else 2 # cls and sep
        for sub in seq:
            cur_len = len(out)
            # Append sub only if it doesn't exceed seq_len
            if cur_len + len(sub) + num_extra <= self.seq_len:
                out.extend(sub)
            else:
                break
            # Append cls and sep tokens only if it doesn't exceed seq_len
            if self.add_cls:
                out.append(self.cls) # cls token
            out.append(self.sep)
        # Remove last separator token if present
        if out and out[-1] == self.sep:
            out.pop(-1) # remove last separator token

        if self.seq_len > len(out):
            out.extend([self.pad] * (self.seq_len - len(out)))

        if add_eos:
            out[-1] = self.eos

        return torch.tensor(out, dtype=dtype)
    
    def _padding(self, seq, pad_value, dtype):
        if self.seq_len > len(seq):
            seq.extend([pad_value] * (self.seq_len - len(seq)))
        return torch.tensor(np.array(seq), dtype=dtype)


class DiscreteTokenizerV2(DiscreteTokenizer):
    def __call__(self, seq, add_bos, add_eos, dtype, return_indices=False):
        out = []
        if add_bos:
            out = [self.bos]
        num_extra = 1 if not self.add_cls else 2 # cls and sep
        indices = []
        for i, sub in enumerate(seq):
            cur_len = len(out)
            # Append sub only if it doesn't exceed seq_len
            if cur_len + len(sub) + num_extra <= self.seq_len:
                out.extend(sub)
                indices.append(i)
            else:
                continue
            # Append cls and sep tokens only if it doesn't exceed seq_len
            if self.add_cls:
                out.append(self.cls) # cls token
            out.append(self.sep)
        # Remove last separator token if present
        if out and out[-1] == self.sep:
            out.pop(-1) # remove last separator token

        if self.seq_len > len(out):
            out.extend([self.pad] * (self.seq_len - len(out)))

        if add_eos:
            out[-1] = self.eos

        if return_indices:
            return torch.tensor(out, dtype=dtype), indices
        return torch.tensor(out, dtype=dtype)