"""
MP-100 Official Split Loader

This module extracts category splits directly from MP-100 annotation files,
eliminating the need for separate category_splits.json files.
"""

import json
from pathlib import Path


def load_mp100_split(dataset_root, split_id=1):
    """
    Load MP-100 category split directly from annotation files.

    Args:
        dataset_root: Root directory containing annotations/
        split_id: Which split to use (1-5)

    Returns:
        dict with:
            - 'train': List of train category IDs
            - 'test': List of test category IDs
            - 'split_id': Split number (1-5)
            - 'total_categories': Total number of categories
    """
    dataset_root = Path(dataset_root)

    # Load train and test annotations
    train_file = dataset_root / f'annotations/mp100_split{split_id}_train.json'
    test_file = dataset_root / f'annotations/mp100_split{split_id}_test.json'

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"MP-100 split {split_id} not found. "
            f"Expected: {train_file} and {test_file}"
        )

    # Extract category IDs
    with open(train_file) as f:
        train_data = json.load(f)
        train_categories = sorted([c['id'] for c in train_data['categories']])

    with open(test_file) as f:
        test_data = json.load(f)
        test_categories = sorted([c['id'] for c in test_data['categories']])

    # Verify no overlap
    overlap = set(train_categories) & set(test_categories)
    if overlap:
        raise ValueError(
            f"MP-100 split {split_id} has {len(overlap)} overlapping categories! "
            f"This should not happen with official splits."
        )

    return {
        'train': train_categories,
        'test': test_categories,
        'split_id': split_id,
        'train_count': len(train_categories),
        'test_count': len(test_categories),
        'total_categories': len(train_categories) + len(test_categories),
        'description': f'MP-100 Split {split_id} (official)',
    }


def get_all_mp100_splits(dataset_root):
    """
    Load all 5 MP-100 splits.

    Returns:
        List of 5 split dictionaries
    """
    return [load_mp100_split(dataset_root, i) for i in range(1, 6)]


def print_split_info(split_dict):
    """Print information about a split."""
    print(f"\n{split_dict['description']}")
    print(f"  Train categories: {split_dict['train_count']}")
    print(f"  Test categories:  {split_dict['test_count']}")
    print(f"  Total:            {split_dict['total_categories']}")
    print(f"  Train IDs: {split_dict['train'][:10]}... ({split_dict['train_count']} total)")
    print(f"  Test IDs:  {split_dict['test']}")


if __name__ == '__main__':
    import sys

    # Test
    dataset_root = '.' if len(sys.argv) < 2 else sys.argv[1]

    print("="*80)
    print("MP-100 Official Splits")
    print("="*80)

    for split_id in range(1, 6):
        split = load_mp100_split(dataset_root, split_id)
        print_split_info(split)

    print("\n" + "="*80)
    print("✓ All 5 splits loaded successfully!")
    print("  Use these splits for CAPE training/evaluation")
    print("="*80)
