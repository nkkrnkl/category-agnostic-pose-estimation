"""
Masking utilities for handling variable-length keypoints and visibility.
"""
import torch
import torch.nn.functional as F


def create_keypoint_mask(num_keypoints_per_sample, max_keypoints):
    """
    Create a mask for valid keypoints in a batch.

    Args:
        num_keypoints_per_sample: list or tensor of shape [B] - number of keypoints per sample
        max_keypoints: int - maximum number of keypoints (T_max)

    Returns:
        mask: torch.Tensor [B, max_keypoints] - 1 for valid keypoints, 0 for padding
    """
    B = len(num_keypoints_per_sample)
    mask = torch.zeros(B, max_keypoints, dtype=torch.bool)

    for b, num_kpts in enumerate(num_keypoints_per_sample):
        mask[b, :num_kpts] = True

    return mask


def create_visibility_mask(visibility, keypoint_mask=None):
    """
    Create a mask for visible keypoints.

    Args:
        visibility: torch.Tensor [B, T] - visibility flags (0: not labeled, >0: visible)
        keypoint_mask: torch.Tensor [B, T] - optional mask for valid keypoints

    Returns:
        mask: torch.Tensor [B, T] - 1 for visible keypoints, 0 otherwise
    """
    vis_mask = (visibility > 0).float()

    if keypoint_mask is not None:
        vis_mask = vis_mask * keypoint_mask.float()

    return vis_mask


def create_attention_mask(keypoint_mask, attention_type='encoder'):
    """
    Create attention mask for transformer layers.

    Args:
        keypoint_mask: torch.Tensor [B, T] - mask for valid keypoints
        attention_type: str - 'encoder' or 'causal' (for decoder self-attention)

    Returns:
        attention_mask: torch.Tensor - attention mask in appropriate format
    """
    B, T = keypoint_mask.shape

    if attention_type == 'encoder':
        # For encoder self-attention: [B, 1, 1, T]
        # Positions with False should not be attended to
        attention_mask = keypoint_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        return attention_mask

    elif attention_type == 'causal':
        # For decoder self-attention: causal mask + keypoint mask
        # Create causal mask [T, T]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        causal_mask = ~causal_mask  # Invert: True means can attend

        # Expand to batch
        causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]

        # Combine with keypoint mask
        # [B, T, T] & [B, 1, T]
        combined_mask = causal_mask & keypoint_mask.unsqueeze(1)

        return combined_mask.unsqueeze(1)  # [B, 1, T, T]

    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")


def create_cross_attention_mask(query_mask, key_mask):
    """
    Create cross-attention mask for decoder attending to encoder.

    Args:
        query_mask: torch.Tensor [B, T_q] - mask for query tokens
        key_mask: torch.Tensor [B, T_k] - mask for key tokens

    Returns:
        attention_mask: torch.Tensor [B, 1, T_q, T_k]
    """
    # [B, T_q, 1] & [B, 1, T_k] -> [B, T_q, T_k]
    attention_mask = query_mask.unsqueeze(2) & key_mask.unsqueeze(1)

    return attention_mask.unsqueeze(1)  # [B, 1, T_q, T_k]


def pad_keypoints(keypoints_list, max_keypoints, pad_value=0.0):
    """
    Pad a list of keypoint tensors to the same length.

    Args:
        keypoints_list: list of tensors, each [N_i, 2]
        max_keypoints: int - target length
        pad_value: float - value to use for padding

    Returns:
        padded: torch.Tensor [B, max_keypoints, 2]
    """
    B = len(keypoints_list)
    padded = torch.full((B, max_keypoints, 2), pad_value, dtype=torch.float32)

    for b, kpts in enumerate(keypoints_list):
        N = min(kpts.shape[0], max_keypoints)
        padded[b, :N] = kpts[:N]

    return padded


def pad_visibility(visibility_list, max_keypoints, pad_value=0):
    """
    Pad a list of visibility tensors to the same length.

    Args:
        visibility_list: list of tensors, each [N_i]
        max_keypoints: int - target length
        pad_value: int - value to use for padding (typically 0)

    Returns:
        padded: torch.Tensor [B, max_keypoints]
    """
    B = len(visibility_list)
    padded = torch.full((B, max_keypoints), pad_value, dtype=torch.float32)

    for b, vis in enumerate(visibility_list):
        N = min(vis.shape[0], max_keypoints)
        padded[b, :N] = vis[:N]

    return padded


def apply_mask_to_loss(loss_tensor, mask):
    """
    Apply mask to loss tensor and compute mean over valid elements.

    Args:
        loss_tensor: torch.Tensor [..., T] or [..., T, D]
        mask: torch.Tensor [..., T] - binary mask (1 for valid, 0 for invalid)

    Returns:
        masked_loss: scalar tensor
    """
    # Expand mask to match loss dimensions if needed
    while mask.ndim < loss_tensor.ndim:
        mask = mask.unsqueeze(-1)

    # Apply mask
    masked_loss = loss_tensor * mask

    # Compute mean over valid elements
    num_valid = mask.sum()

    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=loss_tensor.device)


def create_sequence_mask(sequence_length, max_length):
    """
    Create a mask for sequences with variable lengths.

    Args:
        sequence_length: torch.Tensor [B] - length of each sequence
        max_length: int - maximum sequence length

    Returns:
        mask: torch.Tensor [B, max_length] - boolean mask
    """
    B = sequence_length.shape[0]
    mask = torch.arange(max_length, device=sequence_length.device).unsqueeze(0).expand(B, -1)
    mask = mask < sequence_length.unsqueeze(1)
    return mask
