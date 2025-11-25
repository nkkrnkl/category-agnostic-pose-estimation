"""
Utilities for sequence processing and keypoint extraction.

This module provides functions to extract keypoints from model-generated
sequences, handling token types and coordinate filtering.
"""

import torch
from typing import Optional


def extract_keypoints_from_predictions(
    pred_coords: torch.Tensor,
    pred_logits: torch.Tensor,
    max_keypoints: Optional[int] = None
) -> torch.Tensor:
    """
    Extract keypoints from PREDICTED sequence using PREDICTED token types.
    
    CRITICAL: This function uses the MODEL'S predicted token types, not ground truth.
    This is essential for proper evaluation - we must use what the model actually
    generated, not what we expected it to generate.
    
    Args:
        pred_coords: Predicted coordinates, shape (B, seq_len, 2)
        pred_logits: Predicted token logits, shape (B, seq_len, vocab_size)
                    These are the raw logits from the model's classification head
        max_keypoints: Maximum number of keypoints to extract (or None for all)
    
    Returns:
        keypoints: Extracted keypoints, shape (B, N, 2) where N = num keypoints
                  Padded with zeros if instances have different numbers of keypoints
    
    Example:
        >>> pred_coords = model_output['coordinates']  # (2, 200, 2)
        >>> pred_logits = model_output['logits']  # (2, 200, 5)
        >>> keypoints = extract_keypoints_from_predictions(pred_coords, pred_logits)
        >>> # keypoints shape: (2, N, 2) where N is the actual number of keypoints
    """
    from datasets.token_types import TokenType
    
    # Get predicted token types by taking argmax over vocabulary
    # This tells us what token type the model predicted at each position
    pred_token_types = pred_logits.argmax(dim=-1)  # (B, seq_len)
    
    batch_size = pred_coords.shape[0]
    all_keypoints = []
    
    for i in range(batch_size):
        # Extract coordinate tokens based on PREDICTED types
        # TokenType.coord.value is 0, which indicates a coordinate token
        coord_mask = pred_token_types[i] == TokenType.coord.value
        kpts = pred_coords[i][coord_mask]
        
        # Limit to max_keypoints if specified
        if max_keypoints is not None and len(kpts) > max_keypoints:
            kpts = kpts[:max_keypoints]
        
        all_keypoints.append(kpts)
    
    # Pad to same length across batch
    if len(all_keypoints) > 0:
        max_len = max(len(kpts) for kpts in all_keypoints)
        padded_keypoints = []
        
        for kpts in all_keypoints:
            if len(kpts) < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - len(kpts), 2, device=kpts.device)
                kpts = torch.cat([kpts, padding], dim=0)
            padded_keypoints.append(kpts)
        
        return torch.stack(padded_keypoints)
    else:
        # No keypoints found
        return torch.zeros(batch_size, 0, 2, device=pred_coords.device)


def extract_keypoints_from_gt_sequence(
    gt_coords: torch.Tensor,
    token_labels: torch.Tensor,
    mask: torch.Tensor,
    max_keypoints: Optional[int] = None
) -> torch.Tensor:
    """
    Extract keypoints from GROUND TRUTH sequence using GT token types.
    
    This is a wrapper around extract_keypoints_from_sequence for clarity.
    Use this for ground truth, and extract_keypoints_from_predictions for model outputs.
    
    Args:
        gt_coords: Ground truth coordinates, shape (B, seq_len, 2)
        token_labels: GT token type labels, shape (B, seq_len)
        mask: Valid token mask, shape (B, seq_len)
        max_keypoints: Maximum number of keypoints to extract
    
    Returns:
        keypoints: Extracted keypoints, shape (B, N, 2)
    """
    # Import here to avoid circular dependency
    from models.engine_cape import extract_keypoints_from_sequence
    
    return extract_keypoints_from_sequence(
        gt_coords, token_labels, mask, max_keypoints
    )


def compare_pred_gt_keypoints(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> dict:
    """
    Compare predicted and ground truth keypoints to detect data leakage.
    
    Args:
        pred_kpts: Predicted keypoints, shape (B, N, 2)
        gt_kpts: Ground truth keypoints, shape (B, N, 2)
        tolerance: Tolerance for considering values identical
        verbose: Whether to print detailed comparison
    
    Returns:
        dict with:
            - 'all_identical': bool, whether all instances have identical pred/gt
            - 'num_identical': int, number of instances with identical pred/gt
            - 'identical_indices': list of indices where pred == gt
    """
    batch_size = pred_kpts.shape[0]
    identical_indices = []
    
    for i in range(batch_size):
        if torch.allclose(pred_kpts[i], gt_kpts[i], atol=tolerance):
            identical_indices.append(i)
            if verbose:
                print(f"  Instance {i}: Predictions IDENTICAL to GT (data leakage!)")
        elif verbose:
            # Compute average distance
            dist = torch.norm(pred_kpts[i] - gt_kpts[i], dim=-1).mean()
            print(f"  Instance {i}: Avg distance = {dist:.6f}")
    
    return {
        'all_identical': len(identical_indices) == batch_size,
        'num_identical': len(identical_indices),
        'identical_indices': identical_indices
    }

