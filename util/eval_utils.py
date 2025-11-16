import numpy as np
import torch


def compute_f1(quant_result_dict, metric_category):
    for metric in metric_category:
        prec = quant_result_dict[metric+'_prec']
        rec = quant_result_dict[metric+'_rec']
        f1 = 2*prec*rec/(prec+rec+1e-5)
        quant_result_dict[metric+'_f1'] = f1
    return quant_result_dict


def compute_pck(pred_keypoints, gt_keypoints, image_size, thresholds=None, normalize_by='image'):
    """
    Compute PCK (Percentage of Correct Keypoints) metric for pose estimation.
    
    Args:
        pred_keypoints: List of predicted keypoints, each as list of [x, y] in normalized [0, 1] coordinates
                       or numpy array of shape (N, 2) where N is number of keypoints
        gt_keypoints: List of ground truth keypoints, each as list of [x, y] in normalized [0, 1] coordinates
                     or numpy array of shape (N, 2)
        image_size: Image size (assumes square images, or tuple (h, w))
        thresholds: List of threshold values. If None, uses default thresholds.
                   For 'image': thresholds as fraction of image diagonal (default: [0.05, 0.1, 0.2])
                   For 'bbox': thresholds as fraction of bounding box diagonal (default: [0.05, 0.1, 0.2])
                   For 'pixel': thresholds in pixels (default: [5, 10, 20])
        normalize_by: How to normalize thresholds. Options: 'image', 'bbox', 'pixel'
    
    Returns:
        dict: Dictionary with PCK scores at different thresholds
              Keys: 'pck_0.05', 'pck_0.1', 'pck_0.2', etc.
              Also includes 'mean_pck' (average across all thresholds)
    """
    if thresholds is None:
        if normalize_by == 'pixel':
            thresholds = [5, 10, 20]
        else:
            thresholds = [0.05, 0.1, 0.2]
    
    # Convert to numpy arrays if needed
    if isinstance(pred_keypoints, list):
        if len(pred_keypoints) == 0:
            return {f'pck_{t}': 0.0 for t in thresholds}
        # Handle list of lists or list of arrays
        if isinstance(pred_keypoints[0], list):
            pred_kpts = np.array(pred_keypoints, dtype=np.float32)
        else:
            pred_kpts = np.array([np.array(kpt) for kpt in pred_keypoints if isinstance(kpt, (list, np.ndarray))], dtype=np.float32)
    else:
        pred_kpts = np.array(pred_keypoints, dtype=np.float32)
    
    if isinstance(gt_keypoints, list):
        if len(gt_keypoints) == 0:
            return {f'pck_{t}': 0.0 for t in thresholds}
        if isinstance(gt_keypoints[0], list):
            gt_kpts = np.array(gt_keypoints, dtype=np.float32)
        else:
            gt_kpts = np.array([np.array(kpt) for kpt in gt_keypoints if isinstance(kpt, (list, np.ndarray))], dtype=np.float32)
    else:
        gt_kpts = np.array(gt_keypoints, dtype=np.float32)
    
    # Ensure same number of keypoints
    min_len = min(len(pred_kpts), len(gt_kpts))
    if min_len == 0:
        return {f'pck_{t}': 0.0 for t in thresholds}
    
    pred_kpts = pred_kpts[:min_len]
    gt_kpts = gt_kpts[:min_len]
    
    # Convert normalized coordinates to pixel coordinates
    if isinstance(image_size, (int, float)):
        img_h, img_w = int(image_size), int(image_size)
    else:
        img_h, img_w = int(image_size[0]), int(image_size[1])
    
    pred_pixels = pred_kpts * np.array([img_w, img_h])
    gt_pixels = gt_kpts * np.array([img_w, img_h])
    
    # Compute Euclidean distances
    distances = np.sqrt(np.sum((pred_pixels - gt_pixels) ** 2, axis=1))
    
    # Determine normalization factor
    if normalize_by == 'image':
        # Normalize by image diagonal
        norm_factor = np.sqrt(img_h ** 2 + img_w ** 2)
    elif normalize_by == 'bbox':
        # Normalize by bounding box diagonal (bounding box of all keypoints)
        if len(gt_pixels) > 0:
            bbox_min = gt_pixels.min(axis=0)
            bbox_max = gt_pixels.max(axis=0)
            bbox_size = np.max(bbox_max - bbox_min)
            norm_factor = bbox_size * np.sqrt(2) if bbox_size > 0 else np.sqrt(img_h ** 2 + img_w ** 2)
        else:
            norm_factor = np.sqrt(img_h ** 2 + img_w ** 2)
    else:  # 'pixel'
        norm_factor = 1.0
    
    # Compute PCK at each threshold
    pck_scores = {}
    for threshold in thresholds:
        if normalize_by == 'pixel':
            threshold_pixels = threshold
        else:
            threshold_pixels = threshold * norm_factor
        
        # Count correct keypoints (distance < threshold)
        correct = np.sum(distances < threshold_pixels)
        pck = correct / len(distances) if len(distances) > 0 else 0.0
        pck_scores[f'pck_{threshold}'] = pck
    
    # Compute mean PCK across all thresholds
    pck_scores['mean_pck'] = np.mean(list(pck_scores.values()))
    
    return pck_scores


def compute_pck_batch(pred_keypoints_list, gt_keypoints_list, image_sizes, thresholds=None, normalize_by='image'):
    """
    Compute PCK for a batch of samples.
    
    Args:
        pred_keypoints_list: List of predicted keypoint lists/arrays (one per sample)
        gt_keypoints_list: List of ground truth keypoint lists/arrays (one per sample)
        image_sizes: List of image sizes or single image size (assumed same for all)
        thresholds: Same as compute_pck
        normalize_by: Same as compute_pck
    
    Returns:
        dict: Averaged PCK scores across the batch
    """
    if isinstance(image_sizes, (int, float, tuple)):
        image_sizes = [image_sizes] * len(pred_keypoints_list)
    
    all_pck_scores = []
    for pred_kpts, gt_kpts, img_size in zip(pred_keypoints_list, gt_keypoints_list, image_sizes):
        pck_scores = compute_pck(pred_kpts, gt_kpts, img_size, thresholds, normalize_by)
        all_pck_scores.append(pck_scores)
    
    # Average across batch
    if len(all_pck_scores) == 0:
        return {}
    
    avg_pck = {}
    for key in all_pck_scores[0].keys():
        avg_pck[key] = np.mean([scores[key] for scores in all_pck_scores])
    
    return avg_pck