#!/bin/bash
#
# Example usage of eval_cape_checkpoint.py
# 
# This script shows common evaluation scenarios for the CAPE model.
#

# ============================================================================
# Setup
# ============================================================================
PROJECT_ROOT="/Users/pavlosrousoglou/Desktop/Cornell/Deep Learning/category-agnostic-pose-estimation"
cd "$PROJECT_ROOT"

# ============================================================================
# Example 1: Quick Evaluation (5 episodes, 3 visualizations)
# ============================================================================
echo "================================================================================"
echo "Example 1: Quick Evaluation"
echo "================================================================================"
echo

python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
    --split val \
    --num-episodes 5 \
    --num-visualizations 3 \
    --draw-skeleton \
    --output-dir outputs/eval_quick \
    --device cpu

echo
echo "Results saved to: outputs/eval_quick/"
echo

# ============================================================================
# Example 2: Thorough Validation Evaluation (100 episodes, 50 visualizations)
# ============================================================================
echo "================================================================================"
echo "Example 2: Thorough Validation Evaluation"
echo "================================================================================"
echo

python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
    --split val \
    --num-episodes 100 \
    --num-visualizations 50 \
    --draw-skeleton \
    --output-dir outputs/eval_val_thorough \
    --device cpu

echo
echo "Results saved to: outputs/eval_val_thorough/"
echo

# ============================================================================
# Example 3: Evaluate All Queries in Episodes
# ============================================================================
echo "================================================================================"
echo "Example 3: Save All Queries per Episode"
echo "================================================================================"
echo

python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
    --split val \
    --num-episodes 10 \
    --num-visualizations 10 \
    --save-all-queries \
    --draw-skeleton \
    --output-dir outputs/eval_all_queries \
    --device cpu

echo
echo "Results saved to: outputs/eval_all_queries/"
echo "Note: This will generate 2x visualizations (2 queries per episode)"
echo

# ============================================================================
# Example 4: Different PCK Thresholds
# ============================================================================
echo "================================================================================"
echo "Example 4: Different PCK Thresholds"
echo "================================================================================"
echo

for threshold in 0.1 0.2 0.3; do
    echo "Evaluating with PCK threshold: $threshold"
    python scripts/eval_cape_checkpoint.py \
        --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
        --split val \
        --num-episodes 20 \
        --num-visualizations 5 \
        --pck-threshold $threshold \
        --output-dir outputs/eval_pck_${threshold} \
        --device cpu
    
    echo "PCK@$threshold:"
    cat outputs/eval_pck_${threshold}/metrics_val.json | grep -A 1 "pck_overall"
    echo
done

# ============================================================================
# Summary
# ============================================================================
echo "================================================================================"
echo "All examples complete!"
echo "================================================================================"
echo
echo "Check outputs:"
echo "  - outputs/eval_quick/"
echo "  - outputs/eval_val_thorough/"
echo "  - outputs/eval_all_queries/"
echo "  - outputs/eval_pck_0.1/"
echo "  - outputs/eval_pck_0.2/"
echo "  - outputs/eval_pck_0.3/"
echo

