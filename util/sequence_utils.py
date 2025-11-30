"""
Utilities for sequence processing and keypoint extraction.
"""
import os
import torch
from typing import Optional
from datasets.token_types import TokenType
def extract_keypoints_from_predictions(
    pred_coords: torch.Tensor,
    pred_logits: torch.Tensor,
    max_keypoints: Optional[int] = None
) -> torch.Tensor:
    """
    Extract keypoints from predicted sequence using predicted token types.
    
    Args:
        pred_coords: Predicted coordinates, shape (B, seq_len, 2)
        pred_logits: Predicted token logits, shape (B, seq_len, vocab_size)
        max_keypoints: Maximum number of keypoints to extract
    
    Returns:
        keypoints: Extracted keypoints, shape (B, N, 2)
    """
    from datasets.token_types import TokenType
    pred_token_types = pred_logits.argmax(dim=-1)
    batch_size = pred_coords.shape[0]
    all_keypoints = []
    for i in range(batch_size):
        coord_mask = pred_token_types[i] == TokenType.coord.value
        DEBUG_EXTRACT = os.environ.get('DEBUG_EXTRACT', '0') == '1'
        if DEBUG_EXTRACT and i < 2:
            print(f"\n[DEBUG extract_keypoints_from_predictions] Sample {i}:")
            print(f"  pred_coords[i] shape: {pred_coords[i].shape}")
            print(f"  coord_mask shape: {coord_mask.shape}")
            print(f"  coord_mask sum: {coord_mask.sum().item()}")
        try:
            kpts = pred_coords[i][coord_mask]
        except Exception as e:
            print(f"\n❌ ERROR in extract_keypoints_from_predictions, sample {i}:")
            print(f"   pred_coords[i].shape = {pred_coords[i].shape}")
            print(f"   coord_mask.shape = {coord_mask.shape}")
            print(f"   coord_mask.dtype = {coord_mask.dtype}")
            print(f"   Error: {e}")
            raise
        if DEBUG_EXTRACT and i < 2:
            print(f"  kpts shape: {kpts.shape}")
        if os.environ.get('DEBUG_KEYPOINT_COUNT', '0') == '1' and i == 0:
            print(f"[DIAG sequence_utils] Extracted {len(kpts)} keypoints from predicted sequence")
            print(f"  Total tokens: {len(pred_token_types[i])}")
            print(f"  COORD tokens: {(pred_token_types[i] == TokenType.coord.value).sum()}")
            print(f"  max_keypoints param: {max_keypoints}")
        if max_keypoints is not None and len(kpts) > max_keypoints:
            kpts = kpts[:max_keypoints]
        all_keypoints.append(kpts)
    if len(all_keypoints) > 0:
        max_len = max(len(kpts) for kpts in all_keypoints)
        padded_keypoints = []
        for kpts in all_keypoints:
            if len(kpts) < max_len:
                padding = torch.zeros(max_len - len(kpts), 2, device=kpts.device)
                kpts = torch.cat([kpts, padding], dim=0)
            padded_keypoints.append(kpts)
        return torch.stack(padded_keypoints)
    else:
        return torch.zeros(batch_size, 0, 2, device=pred_coords.device)
def extract_keypoints_from_gt_sequence(
    gt_coords: torch.Tensor,
    token_labels: torch.Tensor,
    mask: torch.Tensor,
    max_keypoints: Optional[int] = None
) -> torch.Tensor:
    """
    Extract keypoints from ground truth sequence using GT token types.
    
    Args:
        gt_coords: Ground truth coordinates, shape (B, seq_len, 2)
        token_labels: GT token type labels, shape (B, seq_len)
        mask: Valid token mask, shape (B, seq_len)
        max_keypoints: Maximum number of keypoints to extract
    
    Returns:
        keypoints: Extracted keypoints, shape (B, N, 2)
    """
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
        dict with all_identical, num_identical, identical_indices
    """
    batch_size = pred_kpts.shape[0]
    identical_indices = []
    for i in range(batch_size):
        if torch.allclose(pred_kpts[i], gt_kpts[i], atol=tolerance):
            identical_indices.append(i)
            if verbose:
                print(f"  Instance {i}: Predictions IDENTICAL to GT (data leakage!)")
        elif verbose:
            dist = torch.norm(pred_kpts[i] - gt_kpts[i], dim=-1).mean()
            print(f"  Instance {i}: Avg distance = {dist:.6f}")
    return {
        'all_identical': len(identical_indices) == batch_size,
        'num_identical': len(identical_indices),
        'identical_indices': identical_indices
    }