import torch
import torch.nn.functional as F
from torch import nn

def kl_loss(p, q):
    p_loss = F.kl_div(p, torch.exp(q), reduction='sum')
    q_loss = F.kl_div(q, torch.exp(p), reduction='sum')
    loss = (p_loss + q_loss) / 2
    return loss


def label_smoothed_nll_loss(
        logits, target, epsilon, reduction='sum',
):
    lprobs = F.log_softmax(logits, dim=-1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    ntokens = loss.numel()
    nll_loss = nll_loss.sum()

    loss = loss.sum()
    if reduction == 'mean':
        loss /= ntokens

    return loss # nll_loss, ntokens