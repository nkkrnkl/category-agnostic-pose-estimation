#!/usr/bin/env python3
"""
Aggregate K-Fold Cross-Validation Results

This script collects evaluation results from all 5 splits of MP-100 k-fold
cross-validation and computes mean ± standard deviation of metrics.

Usage:
    python scripts/aggregate_kfold_results.py \
        --input_base outputs/kfold_20251126_120000 \
        --output_file outputs/kfold_20251126_120000/kfold_summary.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys


def load_eval_results(split_dir: Path, split_num: int) -> Dict[str, Any]:
    """
    Load evaluation results for a single split.
    
    Args:
        split_dir: Path to split directory (e.g., outputs/kfold/split1)
        split_num: Split number (1-5)
        
    Returns:
        Dictionary with test and validation metrics
    """
    result = {
        'split': split_num,
        'split_dir': str(split_dir),
        'test_metrics': None,
        'val_metrics': None,
        'checkpoint': None
    }
    
    # Load test evaluation results
    test_metrics_file = split_dir / 'test_eval' / 'metrics.json'
    if test_metrics_file.exists():
        with open(test_metrics_file) as f:
            result['test_metrics'] = json.load(f)
    else:
        print(f"⚠️  Test metrics not found for split {split_num}: {test_metrics_file}")
    
    # Load validation evaluation results
    val_metrics_file = split_dir / 'val_eval' / 'metrics.json'
    if val_metrics_file.exists():
        with open(val_metrics_file) as f:
            result['val_metrics'] = json.load(f)
    else:
        print(f"⚠️  Validation metrics not found for split {split_num}: {val_metrics_file}")
    
    # Find checkpoint path
    checkpoint_best = split_dir / 'checkpoint_best.pth'
    if checkpoint_best.exists():
        result['checkpoint'] = str(checkpoint_best)
    else:
        # Look for any checkpoint
        checkpoints = list(split_dir.glob('checkpoint_*.pth'))
        if checkpoints:
            result['checkpoint'] = str(checkpoints[0])
    
    return result


def extract_metric(metrics_dict: Dict, key_path: str, default=None):
    """
    Extract a metric from nested dictionary using dot-separated path.
    
    Args:
        metrics_dict: Dictionary with metrics
        key_path: Dot-separated path (e.g., 'pck.overall')
        default: Default value if key not found
        
    Returns:
        Metric value or default
    """
    if metrics_dict is None:
        return default
    
    keys = key_path.split('.')
    value = metrics_dict
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute mean, std, min, max of a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics
    """
    if not values:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }
    
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array, ddof=1 if len(values) > 1 else 0)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'count': len(values)
    }


def aggregate_results(split_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across all splits.
    
    Args:
        split_results: List of results from each split
        
    Returns:
        Aggregated results dictionary
    """
    aggregated = {
        'method': 'CAPE - 5-Fold Cross-Validation on MP-100',
        'num_folds': len(split_results),
        'test': {},
        'val': {},
        'per_fold_results': []
    }
    
    # Collect metrics from all splits
    test_pck_overall = []
    test_pck_mean_cats = []
    val_pck_overall = []
    val_pck_mean_cats = []
    
    for result in split_results:
        fold_summary = {
            'split': result['split'],
            'checkpoint': result['checkpoint']
        }
        
        # Extract test metrics
        if result['test_metrics']:
            test_pck = extract_metric(result['test_metrics'], 'pck_overall', 0.0)
            test_pck_cats = extract_metric(result['test_metrics'], 'pck_mean_categories', 0.0)
            
            fold_summary['test_pck_overall'] = test_pck
            fold_summary['test_pck_mean_categories'] = test_pck_cats
            
            test_pck_overall.append(test_pck)
            test_pck_mean_cats.append(test_pck_cats)
        else:
            fold_summary['test_pck_overall'] = None
            fold_summary['test_pck_mean_categories'] = None
        
        # Extract validation metrics
        if result['val_metrics']:
            val_pck = extract_metric(result['val_metrics'], 'pck_overall', 0.0)
            val_pck_cats = extract_metric(result['val_metrics'], 'pck_mean_categories', 0.0)
            
            fold_summary['val_pck_overall'] = val_pck
            fold_summary['val_pck_mean_categories'] = val_pck_cats
            
            val_pck_overall.append(val_pck)
            val_pck_mean_cats.append(val_pck_cats)
        else:
            fold_summary['val_pck_overall'] = None
            fold_summary['val_pck_mean_categories'] = None
        
        aggregated['per_fold_results'].append(fold_summary)
    
    # Compute aggregated statistics
    aggregated['test']['pck_overall'] = compute_statistics(test_pck_overall)
    aggregated['test']['pck_mean_categories'] = compute_statistics(test_pck_mean_cats)
    aggregated['val']['pck_overall'] = compute_statistics(val_pck_overall)
    aggregated['val']['pck_mean_categories'] = compute_statistics(val_pck_mean_cats)
    
    # Add per-fold values for easy reference
    aggregated['test']['pck_overall_per_fold'] = test_pck_overall
    aggregated['test']['pck_mean_categories_per_fold'] = test_pck_mean_cats
    aggregated['val']['pck_overall_per_fold'] = val_pck_overall
    aggregated['val']['pck_mean_categories_per_fold'] = val_pck_mean_cats
    
    return aggregated


def format_report(aggregated: Dict[str, Any]) -> str:
    """
    Format aggregated results as a human-readable report.
    
    Args:
        aggregated: Aggregated results dictionary
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("K-FOLD CROSS-VALIDATION RESULTS")
    report.append("=" * 80)
    report.append("")
    report.append(f"Method: {aggregated['method']}")
    report.append(f"Number of folds: {aggregated['num_folds']}")
    report.append("")
    
    # Test results
    report.append("-" * 80)
    report.append("TEST SET RESULTS")
    report.append("-" * 80)
    report.append("")
    
    test_pck = aggregated['test']['pck_overall']
    test_pck_cats = aggregated['test']['pck_mean_categories']
    
    if test_pck['count'] > 0:
        report.append(f"PCK@0.2 (Overall):")
        report.append(f"  Mean:  {test_pck['mean']:.4f} ({test_pck['mean']*100:.2f}%)")
        report.append(f"  Std:   {test_pck['std']:.4f}")
        report.append(f"  Min:   {test_pck['min']:.4f} ({test_pck['min']*100:.2f}%)")
        report.append(f"  Max:   {test_pck['max']:.4f} ({test_pck['max']*100:.2f}%)")
        report.append(f"  Range: {(test_pck['max'] - test_pck['min'])*100:.2f}%")
        report.append("")
        
        report.append(f"PCK@0.2 (Mean across categories):")
        report.append(f"  Mean:  {test_pck_cats['mean']:.4f} ({test_pck_cats['mean']*100:.2f}%)")
        report.append(f"  Std:   {test_pck_cats['std']:.4f}")
        report.append(f"  Min:   {test_pck_cats['min']:.4f} ({test_pck_cats['min']*100:.2f}%)")
        report.append(f"  Max:   {test_pck_cats['max']:.4f} ({test_pck_cats['max']*100:.2f}%)")
        report.append("")
        
        report.append("Per-fold breakdown:")
        for fold in aggregated['per_fold_results']:
            test_pck_fold = fold.get('test_pck_overall')
            test_pck_cats_fold = fold.get('test_pck_mean_categories')
            if test_pck_fold is not None:
                report.append(f"  Split {fold['split']}: PCK={test_pck_fold:.4f} ({test_pck_fold*100:.2f}%), "
                            f"PCK_cats={test_pck_cats_fold:.4f} ({test_pck_cats_fold*100:.2f}%)")
    else:
        report.append("⚠️  No test results available")
    
    report.append("")
    
    # Validation results
    report.append("-" * 80)
    report.append("VALIDATION SET RESULTS")
    report.append("-" * 80)
    report.append("")
    
    val_pck = aggregated['val']['pck_overall']
    val_pck_cats = aggregated['val']['pck_mean_categories']
    
    if val_pck['count'] > 0:
        report.append(f"PCK@0.2 (Overall):")
        report.append(f"  Mean:  {val_pck['mean']:.4f} ({val_pck['mean']*100:.2f}%)")
        report.append(f"  Std:   {val_pck['std']:.4f}")
        report.append(f"  Min:   {val_pck['min']:.4f} ({val_pck['min']*100:.2f}%)")
        report.append(f"  Max:   {val_pck['max']:.4f} ({val_pck['max']*100:.2f}%)")
        report.append("")
        
        report.append(f"PCK@0.2 (Mean across categories):")
        report.append(f"  Mean:  {val_pck_cats['mean']:.4f} ({val_pck_cats['mean']*100:.2f}%)")
        report.append(f"  Std:   {val_pck_cats['std']:.4f}")
        report.append("")
        
        report.append("Per-fold breakdown:")
        for fold in aggregated['per_fold_results']:
            val_pck_fold = fold.get('val_pck_overall')
            val_pck_cats_fold = fold.get('val_pck_mean_categories')
            if val_pck_fold is not None:
                report.append(f"  Split {fold['split']}: PCK={val_pck_fold:.4f} ({val_pck_fold*100:.2f}%), "
                            f"PCK_cats={val_pck_cats_fold:.4f} ({val_pck_cats_fold*100:.2f}%)")
    else:
        report.append("⚠️  No validation results available")
    
    report.append("")
    report.append("=" * 80)
    report.append("")
    report.append("REPORTING GUIDELINES:")
    report.append("")
    report.append("For publication/benchmark comparison, report:")
    report.append(f"  Test PCK@0.2: {test_pck['mean']*100:.2f}% ± {test_pck['std']*100:.2f}%")
    report.append("")
    report.append("LaTeX format:")
    if test_pck['count'] > 0:
        report.append(f"  ${test_pck['mean']*100:.2f} \\pm {test_pck['std']*100:.2f}$")
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate k-fold cross-validation results for CAPE on MP-100'
    )
    parser.add_argument(
        '--input_base',
        type=str,
        required=True,
        help='Base directory containing split1/, split2/, ..., split5/ subdirectories'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output JSON file for aggregated results'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_base = Path(args.input_base)
    if not input_base.exists():
        print(f"❌ Error: Input directory not found: {input_base}")
        sys.exit(1)
    
    print("=" * 80)
    print("Aggregating K-Fold Cross-Validation Results")
    print("=" * 80)
    print(f"Input directory: {input_base}")
    print("")
    
    # Load results from all splits
    split_results = []
    for split_num in range(1, 6):
        split_dir = input_base / f"split{split_num}"
        
        if not split_dir.exists():
            print(f"⚠️  Split {split_num} directory not found: {split_dir}")
            continue
        
        print(f"Loading results for split {split_num}...")
        result = load_eval_results(split_dir, split_num)
        split_results.append(result)
    
    print("")
    
    if not split_results:
        print("❌ Error: No split results found!")
        sys.exit(1)
    
    print(f"✓ Loaded results from {len(split_results)} splits")
    print("")
    
    # Aggregate results
    print("Computing aggregated statistics...")
    aggregated = aggregate_results(split_results)
    
    # Save JSON results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"✓ Saved aggregated results to: {output_file}")
    print("")
    
    # Generate and save report
    report = format_report(aggregated)
    report_file = output_file.parent / 'kfold_report.txt'
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved report to: {report_file}")
    print("")
    
    # Print report to stdout
    print(report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

