"""
Geometry utilities for coordinate transformations and PCK computation.
"""
import torch
import numpy as np


def normalize_keypoints(keypoints, bbox, img_size=(512, 512)):
    """
    Normalize keypoints to [0, 1]^2 based on bbox.

    Args:
        keypoints: np.array of shape [N, 3] where columns are (x, y, visibility)
        bbox: [x, y, w, h]
        img_size: target image size (height, width)

    Returns:
        normalized_coords: np.array of shape [N, 2] in [0, 1]^2
        visibility: np.array of shape [N] with visibility flags
    """
    x, y, w, h = bbox

    # Extract coordinates and visibility
    coords = keypoints[:, :2].astype(np.float32).copy()
    visibility = keypoints[:, 2].astype(np.float32).copy()

    # Translate to bbox origin
    coords[:, 0] -= x
    coords[:, 1] -= y

    # Normalize by bbox size
    coords[:, 0] /= w
    coords[:, 1] /= h

    return coords, visibility


def denormalize_keypoints(normalized_coords, bbox):
    """
    Convert normalized keypoints back to original image coordinates.

    Args:
        normalized_coords: np.array or torch.Tensor of shape [N, 2] in [0, 1]^2
        bbox: [x, y, w, h]

    Returns:
        coords: np.array of shape [N, 2] in original image coordinates
    """
    x, y, w, h = bbox

    if isinstance(normalized_coords, torch.Tensor):
        coords = normalized_coords.detach().cpu().numpy()
    else:
        coords = normalized_coords.copy()

    # Scale by bbox size
    coords[:, 0] *= w
    coords[:, 1] *= h

    # Translate to bbox origin
    coords[:, 0] += x
    coords[:, 1] += y

    return coords


def compute_pck(pred_coords, gt_coords, bbox, visibility, threshold=0.2):
    """
    Compute Percentage of Correct Keypoints (PCK).

    Args:
        pred_coords: np.array or torch.Tensor [N, 2] - predicted keypoints (normalized)
        gt_coords: np.array or torch.Tensor [N, 2] - ground truth keypoints (normalized)
        bbox: [x, y, w, h] - bounding box
        visibility: np.array or torch.Tensor [N] - visibility flags (0: not labeled, >0: visible)
        threshold: PCK threshold (default 0.2 for PCK@0.2)

    Returns:
        pck: float - percentage of correct keypoints
        correct: np.array [N] - binary correctness per keypoint
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()
    if isinstance(visibility, torch.Tensor):
        visibility = visibility.detach().cpu().numpy()

    x, y, w, h = bbox
    bbox_size = max(w, h)

    # Denormalize coordinates
    pred = denormalize_keypoints(pred_coords, bbox)
    gt = denormalize_keypoints(gt_coords, bbox)

    # Compute Euclidean distances
    distances = np.linalg.norm(pred - gt, axis=1)

    # Normalize by bbox size
    normalized_distances = distances / bbox_size

    # Check if distance is below threshold
    correct = (normalized_distances < threshold).astype(np.float32)

    # Only consider visible keypoints
    valid_mask = visibility > 0

    if valid_mask.sum() == 0:
        return 0.0, correct

    # Compute PCK only on visible keypoints
    pck = (correct * valid_mask).sum() / valid_mask.sum()

    return float(pck), correct


def apply_affine_to_keypoints(keypoints, affine_matrix):
    """
    Apply affine transformation to keypoints.

    Args:
        keypoints: np.array [N, 2]
        affine_matrix: 2x3 affine transformation matrix

    Returns:
        transformed_keypoints: np.array [N, 2]
    """
    N = keypoints.shape[0]

    # Add homogeneous coordinate
    keypoints_homo = np.concatenate([keypoints, np.ones((N, 1))], axis=1)

    # Apply transformation
    transformed = keypoints_homo @ affine_matrix.T

    return transformed


def create_bbox_from_keypoints(keypoints, visibility, padding_ratio=0.1):
    """
    Create a bounding box from visible keypoints with padding.

    Args:
        keypoints: np.array [N, 2]
        visibility: np.array [N]
        padding_ratio: ratio of padding to add

    Returns:
        bbox: [x, y, w, h]
    """
    valid_mask = visibility > 0
    valid_kpts = keypoints[valid_mask]

    if len(valid_kpts) == 0:
        return [0, 0, 1, 1]

    x_min, y_min = valid_kpts.min(axis=0)
    x_max, y_max = valid_kpts.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min

    # Add padding
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio

    x = x_min - pad_w
    y = y_min - pad_h
    w = w + 2 * pad_w
    h = h + 2 * pad_h

    return [x, y, w, h]
