#!/bin/bash
#
# K-Fold Cross-Validation for CAPE on MP-100
#
# This script runs training on all 5 MP-100 splits sequentially,
# then aggregates results to produce mean ± std PCK metrics.
#
# Usage:
#   ./scripts/run_kfold_cross_validation.sh [OPTIONS]
#
# Options:
#   --epochs N          Number of epochs per fold (default: 300)
#   --batch_size N      Batch size (default: 2)
#   --episodes N        Episodes per epoch (default: 500)
#   --output_dir DIR    Base output directory (default: outputs/kfold)
#   --resume_from N     Resume from split N (skip earlier splits)
#   --eval_only         Only run evaluation, skip training
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default values
EPOCHS=300
BATCH_SIZE=2
NUM_QUERIES=2
EPISODES_PER_EPOCH=500
OUTPUT_BASE="outputs/kfold_$(date +%Y%m%d_%H%M%S)"
RESUME_FROM=1
EVAL_ONLY=false
DEVICE="mps"  # Will auto-detect if not specified

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --episodes)
            EPISODES_PER_EPOCH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --eval_only)
            EVAL_ONLY=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

echo "=============================================================================="
echo "  MP-100 CAPE: 5-Fold Cross-Validation"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Epochs per fold:        $EPOCHS"
echo "  Batch size:             $BATCH_SIZE"
echo "  Episodes per epoch:     $EPISODES_PER_EPOCH"
echo "  Output directory:       $OUTPUT_BASE"
echo "  Resume from split:      $RESUME_FROM"
echo "  Eval only:              $EVAL_ONLY"
echo "  Device:                 $DEVICE"
echo ""
echo "Estimated time: $(($EPOCHS * 5 / 60)) hours (if 1 epoch = ~1 min)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Save configuration
cat > "$OUTPUT_BASE/kfold_config.txt" << EOF
K-Fold Cross-Validation Configuration
======================================
Date: $(date)
Epochs per fold: $EPOCHS
Batch size: $BATCH_SIZE
Episodes per epoch: $EPISODES_PER_EPOCH
Output directory: $OUTPUT_BASE
Resume from split: $RESUME_FROM
Eval only: $EVAL_ONLY
Device: $DEVICE
EOF

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ============================================================================
# Run K-Fold Cross-Validation
# ============================================================================

for SPLIT in {1..5}; do
    # Skip if resuming from a later split
    if [ $SPLIT -lt $RESUME_FROM ]; then
        echo "Skipping split $SPLIT (resume_from=$RESUME_FROM)"
        echo ""
        continue
    fi
    
    echo "=============================================================================="
    echo "  FOLD $SPLIT / 5"
    echo "=============================================================================="
    echo ""
    
    SPLIT_OUTPUT="$OUTPUT_BASE/split${SPLIT}"
    
    # ========================================================================
    # Training Phase
    # ========================================================================
    
    if [ "$EVAL_ONLY" = false ]; then
        echo "Training on split $SPLIT..."
        echo "Output: $SPLIT_OUTPUT"
        echo ""
        
        python models/train_cape_episodic.py \
            --mp100_split=$SPLIT \
            --output_dir="$SPLIT_OUTPUT" \
            --cape_mode \
            --poly2seq \
            --dataset_name=mp100 \
            --dataset_root="." \
            --category_split_file="category_splits.json" \
            --device="$DEVICE" \
            \
            --epochs=$EPOCHS \
            --batch_size=$BATCH_SIZE \
            --num_queries_per_episode=$NUM_QUERIES \
            --episodes_per_epoch=$EPISODES_PER_EPOCH \
            \
            --lr=1e-4 \
            --lr_backbone=1e-5 \
            --weight_decay=1e-4 \
            --lr_drop="200,250" \
            --clip_max_norm=0.1 \
            \
            --support_encoder_layers=3 \
            --support_fusion_method=cross_attention \
            \
            --backbone=resnet50 \
            --hidden_dim=256 \
            --nheads=8 \
            --enc_layers=6 \
            --dec_layers=6 \
            --dim_feedforward=1024 \
            --dropout=0.1 \
            \
            --image_size=256 \
            --vocab_size=2000 \
            --seq_len=200 \
            --num_queries=200 \
            --num_polys=1 \
            \
            --cls_loss_coef=2.0 \
            --coords_loss_coef=5.0 \
            --room_cls_loss_coef=0.0 \
            --eos_weight=20.0 \
            \
            --semantic_classes=70 \
            --num_feature_levels=4 \
            --dec_n_points=4 \
            --enc_n_points=4 \
            \
            --aux_loss \
            --with_poly_refine \
            --num_workers=2 \
            --seed=42 \
            --print_freq=10 \
            --job_name="kfold_split${SPLIT}"
        
        echo ""
        echo "✓ Training complete for split $SPLIT"
        echo ""
    else
        echo "Skipping training (eval_only=true)"
        echo ""
    fi
    
    # ========================================================================
    # Evaluation Phase
    # ========================================================================
    
    echo "Evaluating split $SPLIT on test set..."
    echo ""
    
    # Find the best checkpoint
    CHECKPOINT="$SPLIT_OUTPUT/checkpoint_best.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "⚠️  Best checkpoint not found, looking for latest checkpoint..."
        CHECKPOINT=$(ls -t "$SPLIT_OUTPUT"/checkpoint_*.pth 2>/dev/null | head -1)
        
        if [ -z "$CHECKPOINT" ]; then
            echo "❌ No checkpoint found for split $SPLIT"
            echo "   Skipping evaluation..."
            continue
        fi
    fi
    
    echo "Using checkpoint: $CHECKPOINT"
    echo ""
    
    # Evaluate on test set
    python scripts/eval_cape_checkpoint.py \
        --checkpoint="$CHECKPOINT" \
        --split=test \
        --num-episodes=100 \
        --num-queries-per-episode=2 \
        --output-dir="$SPLIT_OUTPUT/test_eval" \
        --num-visualizations=10 \
        --draw-skeleton \
        2>&1 | tee "$SPLIT_OUTPUT/test_eval.log"
    
    echo ""
    echo "✓ Evaluation complete for split $SPLIT"
    echo "  Results saved to: $SPLIT_OUTPUT/test_eval/"
    echo ""
    
    # Also evaluate on validation set for comparison
    echo "Evaluating split $SPLIT on validation set..."
    echo ""
    
    python scripts/eval_cape_checkpoint.py \
        --checkpoint="$CHECKPOINT" \
        --split=val \
        --num-episodes=50 \
        --num-queries-per-episode=2 \
        --output-dir="$SPLIT_OUTPUT/val_eval" \
        --num-visualizations=5 \
        2>&1 | tee "$SPLIT_OUTPUT/val_eval.log"
    
    echo ""
    echo "✓ Validation evaluation complete for split $SPLIT"
    echo ""
    
    echo "=============================================================================="
    echo ""
done

# ============================================================================
# Aggregate Results
# ============================================================================

echo "=============================================================================="
echo "  Aggregating Results Across All Folds"
echo "=============================================================================="
echo ""

python scripts/aggregate_kfold_results.py \
    --input_base="$OUTPUT_BASE" \
    --output_file="$OUTPUT_BASE/kfold_summary.json"

echo ""
echo "=============================================================================="
echo "  K-Fold Cross-Validation Complete!"
echo "=============================================================================="
echo ""
echo "Results summary: $OUTPUT_BASE/kfold_summary.json"
echo "Full report:     $OUTPUT_BASE/kfold_report.txt"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_BASE/kfold_summary.json"
echo "  cat $OUTPUT_BASE/kfold_report.txt"
echo ""

