"""
Tests to validate EOS token prediction after fixing the visibility mask bug.

These tests ensure that:
1. The model learns to predict EOS tokens during training
2. Generation stops at appropriate lengths (not max_len)
3. Keypoint counts match category expectations
4. No excessive trimming is needed

BUG CONTEXT (Nov 25, 2025):
- EOS token was excluded from classification loss (visibility_mask[EOS] = False)
- Model never learned to predict EOS
- Always generated 200 keypoints regardless of category (17-32 expected)
- Trimming operation silently discarded 168-183 extra predictions

FIX APPLIED:
- datasets/mp100_cape.py:766-769 - Include EOS in visibility_mask
- models/roomformer_v2.py:593-602 - Add warning for incomplete generations
- scripts/eval_cape_checkpoint.py:464-476 - Add assertion before trimming
- models/engine_cape.py:611-620 - Add assertion before trimming
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.mp100_cape import MP100CAPE
from datasets.episodic_sampler import build_episodic_dataloader
from datasets.token_types import TokenType
from models import build_model
from models.cape_model import build_cape_model
from util.sequence_utils import extract_keypoints_from_predictions
from models.engine_cape import extract_keypoints_from_sequence


@pytest.fixture
def setup_model_and_data():
    """Setup model and validation dataloader for testing."""
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Build dataset
    import albumentations as A
    transforms = A.Compose([
        A.Resize(height=512, width=512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    val_dataset = MP100CAPE(
        img_folder='data',
        ann_file='annotations/mp100_split1_val.json',
        transforms=transforms,
        split='val',
        vocab_size=2000,
        seq_len=200
    )
    
    tokenizer = val_dataset.tokenizer
    
    # Build dataloader
    val_dataloader = build_episodic_dataloader(
        val_dataset,
        category_split_file='category_splits.json',
        split='val',
        batch_size=2,
        num_queries_per_episode=2,
        episodes_per_epoch=5,
        num_workers=0
    )
    
    return device, tokenizer, val_dataloader


def test_eos_token_in_visibility_mask():
    """
    Test that EOS tokens are included in visibility mask.
    
    This test verifies the fix for the critical bug where EOS was excluded
    from loss computation, preventing the model from learning to stop generation.
    """
    import albumentations as A
    transforms = A.Compose([
        A.Resize(height=512, width=512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    dataset = MP100CAPE(
        img_folder='data',
        ann_file='annotations/mp100_split1_val.json',
        transforms=transforms,
        split='val',
        vocab_size=2000,
        seq_len=200
    )
    
    # Get a sample
    sample = dataset[0]
    
    # Check that visibility_mask includes EOS token
    seq_data = sample['seq_data']
    token_labels = seq_data['token_labels']
    visibility_mask = seq_data['visibility_mask']
    
    # Find EOS token position
    eos_positions = (token_labels == TokenType.eos.value).nonzero(as_tuple=True)[0]
    
    if len(eos_positions) > 0:
        eos_pos = eos_positions[0].item()
        # EOS must be included in visibility mask
        assert visibility_mask[eos_pos] == True, \
            f"EOS token at position {eos_pos} must be True in visibility_mask (got {visibility_mask[eos_pos]})"
        print(f"✓ EOS token at position {eos_pos} is correctly included in visibility_mask")
    else:
        pytest.fail("No EOS token found in token_labels")


@pytest.mark.skipif(not os.path.exists('outputs/cape_run'), reason="No trained checkpoints available")
def test_eos_prediction_rate_after_training(setup_model_and_data):
    """
    Test that a trained model predicts EOS tokens at a reasonable rate.
    
    After the fix, a properly trained model should:
    - Predict EOS for most sequences (>50%)
    - Not always reach max_len
    
    This test should FAIL on old checkpoints (before fix) and PASS on new checkpoints (after fix).
    """
    device, tokenizer, val_dataloader = setup_model_and_data
    
    # Find latest checkpoint
    checkpoint_dir = Path('outputs/cape_run')
    checkpoints = list(checkpoint_dir.glob('checkpoint_e*.pth'))
    if not checkpoints:
        pytest.skip("No checkpoints found")
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1][1:]))  # epoch number
    
    print(f"\nLoading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint['args']
    
    # Build model
    build_result = build_model(checkpoint_args, tokenizer=tokenizer, train=False)
    if isinstance(build_result, tuple):
        base_model, _ = build_result
    else:
        base_model = build_result
    model = build_cape_model(checkpoint_args, base_model)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Run inference on a few batches
    eos_count = 0
    total_sequences = 0
    sequences_at_max_len = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= 3:  # Test on 3 batches
                break
            
            support_coords = batch['support_coords'].to(device)
            support_masks = batch['support_masks'].to(device)
            query_images = batch['query_images'].to(device)
            support_skeletons = batch.get('support_skeletons', None)
            
            # Run inference
            predictions = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                skeleton_edges=support_skeletons
            )
            
            pred_logits = predictions['logits']
            pred_token_types = pred_logits.argmax(dim=-1)  # (B, seq_len)
            
            for i in range(pred_token_types.shape[0]):
                total_sequences += 1
                
                # Check if EOS is present
                has_eos = (pred_token_types[i] == TokenType.eos.value).any()
                if has_eos:
                    eos_count += 1
                    eos_pos = (pred_token_types[i] == TokenType.eos.value).nonzero(as_tuple=True)[0][0].item()
                    if eos_pos >= 199:  # Near max_len (200)
                        sequences_at_max_len += 1
                else:
                    sequences_at_max_len += 1
    
    eos_rate = eos_count / total_sequences if total_sequences > 0 else 0
    max_len_rate = sequences_at_max_len / total_sequences if total_sequences > 0 else 0
    
    print(f"\nEOS Prediction Statistics:")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Sequences with EOS: {eos_count} ({eos_rate:.1%})")
    print(f"  Sequences at max_len: {sequences_at_max_len} ({max_len_rate:.1%})")
    
    # After fix and retraining, EOS rate should be high
    # Note: This test may fail on checkpoints trained BEFORE the fix
    assert eos_rate > 0.5, \
        f"EOS prediction rate too low: {eos_rate:.1%}. " \
        f"Model may have been trained before the visibility_mask fix. " \
        f"Retrain the model to include EOS in classification loss."


@pytest.mark.skipif(not os.path.exists('outputs/cape_run'), reason="No trained checkpoints available")
def test_predicted_keypoint_count_reasonable(setup_model_and_data):
    """
    Test that predicted keypoint counts are reasonable (not always 200).
    
    After the fix:
    - Predicted keypoints should be close to expected count
    - Should NOT always be exactly max_len (200)
    - Should vary across categories
    """
    device, tokenizer, val_dataloader = setup_model_and_data
    
    # Find latest checkpoint
    checkpoint_dir = Path('outputs/cape_run')
    checkpoints = list(checkpoint_dir.glob('checkpoint_e*.pth'))
    if not checkpoints:
        pytest.skip("No checkpoints found")
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1][1:]))
    
    checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint['args']
    
    # Build model
    build_result = build_model(checkpoint_args, tokenizer=tokenizer, train=False)
    if isinstance(build_result, tuple):
        base_model, _ = build_result
    else:
        base_model = build_result
    model = build_cape_model(checkpoint_args, base_model)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Run inference
    pred_counts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= 3:
                break
            
            support_coords = batch['support_coords'].to(device)
            support_masks = batch['support_masks'].to(device)
            query_images = batch['query_images'].to(device)
            support_skeletons = batch.get('support_skeletons', None)
            query_metadata = batch.get('query_metadata', [])
            
            predictions = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                skeleton_edges=support_skeletons
            )
            
            pred_coords = predictions['coordinates']
            pred_logits = predictions['logits']
            pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)
            
            for idx in range(pred_kpts.shape[0]):
                pred_count = pred_kpts[idx].shape[0]
                pred_counts.append(pred_count)
                
                # Get expected count
                if idx < len(query_metadata):
                    expected_count = query_metadata[idx].get('num_keypoints', 20)
                    
                    # Allow some margin but should not be 200
                    assert pred_count < 150, \
                        f"Predicted {pred_count} keypoints (expected ~{expected_count}). " \
                        f"Model still generating max_len sequences. Retrain required."
    
    print(f"\nPredicted keypoint counts: min={min(pred_counts)}, max={max(pred_counts)}, avg={np.mean(pred_counts):.1f}")
    
    # Should have variation, not all 200
    assert len(set(pred_counts)) > 1, "All predictions have same count - model not learning proper lengths"
    assert max(pred_counts) < 150, f"Max predicted count ({max(pred_counts)}) suggests model reaching max_len"


@pytest.mark.skipif(not os.path.exists('outputs/cape_run'), reason="No trained checkpoints available")
def test_trimming_discards_minimal_predictions(setup_model_and_data):
    """
    Test that trimming operation discards minimal predictions after fix.
    
    Before fix: 168-183 keypoints discarded per sample
    After fix: Should discard <5 keypoints (only noise/error)
    """
    device, tokenizer, val_dataloader = setup_model_and_data
    
    checkpoint_dir = Path('outputs/cape_run')
    checkpoints = list(checkpoint_dir.glob('checkpoint_e*.pth'))
    if not checkpoints:
        pytest.skip("No checkpoints found")
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1][1:]))
    
    checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint['args']
    
    build_result = build_model(checkpoint_args, tokenizer=tokenizer, train=False)
    if isinstance(build_result, tuple):
        base_model, _ = build_result
    else:
        base_model = build_result
    model = build_cape_model(checkpoint_args, base_model)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    
    total_discarded = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= 3:
                break
            
            support_coords = batch['support_coords'].to(device)
            support_masks = batch['support_masks'].to(device)
            query_images = batch['query_images'].to(device)
            support_skeletons = batch.get('support_skeletons', None)
            query_metadata = batch.get('query_metadata', [])
            
            predictions = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                skeleton_edges=support_skeletons
            )
            
            pred_coords = predictions['coordinates']
            pred_logits = predictions['logits']
            pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)
            
            for idx in range(pred_kpts.shape[0]):
                if idx < len(query_metadata):
                    expected_count = query_metadata[idx].get('num_keypoints', 20)
                    pred_count = pred_kpts[idx].shape[0]
                    discarded = max(0, pred_count - expected_count)
                    total_discarded += discarded
                    total_samples += 1
    
    avg_discarded = total_discarded / total_samples if total_samples > 0 else 0
    
    print(f"\nTrimming statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total keypoints discarded: {total_discarded}")
    print(f"  Average discarded per sample: {avg_discarded:.1f}")
    
    # After fix, should discard very few keypoints
    assert avg_discarded < 5, \
        f"Average {avg_discarded:.1f} keypoints discarded per sample. " \
        f"Model still overgenerating. Expected <5 after fix."


def test_token_type_distribution_is_balanced():
    """
    Test that GT token sequences have balanced distribution including EOS.
    
    This validates that the dataset construction properly includes EOS tokens
    in the sequences.
    """
    import albumentations as A
    transforms = A.Compose([
        A.Resize(height=512, width=512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    dataset = MP100CAPE(
        img_folder='data',
        ann_file='annotations/mp100_split1_val.json',
        transforms=transforms,
        split='val',
        vocab_size=2000,
        seq_len=200
    )
    
    # Collect token statistics from first 20 samples
    token_counts = {
        'coord': 0,
        'sep': 0,
        'eos': 0,
        'cls': 0,
        'padding': 0
    }
    
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        seq_data = sample['seq_data']
        token_labels = seq_data['token_labels']
        visibility_mask = seq_data['visibility_mask']
        
        # Count token types
        for label, visible in zip(token_labels, visibility_mask):
            if label == TokenType.coord.value:
                token_counts['coord'] += 1
            elif label == TokenType.sep.value:
                token_counts['sep'] += 1
            elif label == TokenType.eos.value:
                token_counts['eos'] += 1
                # Check if EOS is marked as visible in mask
                if not visible:
                    pytest.fail(f"Sample {i}: EOS token not marked as visible in visibility_mask!")
            elif label == TokenType.cls.value:
                token_counts['cls'] += 1
            elif label == -1:  # Padding
                token_counts['padding'] += 1
    
    print(f"\nToken distribution across 20 samples:")
    for token_type, count in token_counts.items():
        print(f"  {token_type}: {count}")
    
    # EOS should appear once per sample (20 samples)
    assert token_counts['eos'] >= 15, \
        f"Expected ~20 EOS tokens (one per sample), got {token_counts['eos']}"
    
    # COORD tokens should be majority
    assert token_counts['coord'] > token_counts['eos'], \
        "Expected more COORD tokens than EOS tokens"


def test_visibility_mask_includes_all_visible_coords():
    """
    Test that visibility_mask correctly marks visible coordinates as True.
    """
    import albumentations as A
    transforms = A.Compose([
        A.Resize(height=512, width=512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    dataset = MP100CAPE(
        img_folder='data',
        ann_file='annotations/mp100_split1_val.json',
        transforms=transforms,
        split='val',
        vocab_size=2000,
        seq_len=200
    )
    
    # Check first sample
    sample = dataset[0]
    seq_data = sample['seq_data']
    token_labels = seq_data['token_labels']
    visibility_mask = seq_data['visibility_mask']
    
    # Count tokens
    coord_positions = (token_labels == TokenType.coord.value).nonzero(as_tuple=True)[0]
    coord_visible_count = visibility_mask[coord_positions].sum().item()
    
    print(f"\nVisibility mask statistics:")
    print(f"  Total COORD tokens: {len(coord_positions)}")
    print(f"  COORD tokens marked visible: {coord_visible_count}")
    print(f"  Total True in visibility_mask: {visibility_mask.sum().item()}")
    
    # At least some coordinates should be visible
    assert coord_visible_count > 0, "No COORD tokens marked as visible"
    
    # Check that EOS is included (this is the critical fix)
    eos_positions = (token_labels == TokenType.eos.value).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        eos_visible_count = visibility_mask[eos_positions].sum().item()
        assert eos_visible_count > 0, \
            "EOS token(s) not marked as visible - fix not applied correctly!"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

