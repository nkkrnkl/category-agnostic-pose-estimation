#!/bin/bash
#
# Compare Old vs New Support Encoders
#
# This script runs ablation experiments to compare:
# 1. Baseline (old SupportPoseGraphEncoder)
# 2. Geometric encoder without GCN
# 3. Geometric encoder with GCN
#
# Usage:
#   bash scripts/compare_encoders.sh

set -e

# Configuration
EPOCHS=10
BATCH_SIZE=2
NUM_QUERIES=2
DATASET_ROOT="."

echo "======================================================================"
echo "Support Encoder Comparison Study"
echo "======================================================================"
echo "This script will train 3 models:"
echo "  1. Baseline (old encoder)"
echo "  2. Geometric encoder (no GCN)"
echo "  3. Geometric encoder (with GCN)"
echo ""
echo "Configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Queries per episode: $NUM_QUERIES"
echo "======================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Experiment 1: Baseline (old encoder)
echo ""
echo "[1/3] Training baseline (old encoder)..."
echo "======================================================================"
python models/train_cape_episodic.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_queries_per_episode $NUM_QUERIES \
    --dataset_root $DATASET_ROOT \
    --output_dir outputs/compare_baseline \
    --job_name baseline_encoder \
    2>&1 | tee outputs/compare_baseline/train.log

# Experiment 2: Geometric encoder (no GCN)
echo ""
echo "[2/3] Training geometric encoder (no GCN)..."
echo "======================================================================"
python models/train_cape_episodic.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_queries_per_episode $NUM_QUERIES \
    --dataset_root $DATASET_ROOT \
    --use_geometric_encoder \
    --output_dir outputs/compare_geometric_no_gcn \
    --job_name geometric_no_gcn \
    2>&1 | tee outputs/compare_geometric_no_gcn/train.log

# Experiment 3: Geometric encoder (with GCN)
echo ""
echo "[3/3] Training geometric encoder (with GCN)..."
echo "======================================================================"
python models/train_cape_episodic.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_queries_per_episode $NUM_QUERIES \
    --dataset_root $DATASET_ROOT \
    --use_geometric_encoder \
    --use_gcn_preenc \
    --num_gcn_layers 2 \
    --output_dir outputs/compare_geometric_with_gcn \
    --job_name geometric_with_gcn \
    2>&1 | tee outputs/compare_geometric_with_gcn/train.log

# Summary
echo ""
echo "======================================================================"
echo "Comparison Complete!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - outputs/compare_baseline/"
echo "  - outputs/compare_geometric_no_gcn/"
echo "  - outputs/compare_geometric_with_gcn/"
echo ""
echo "To compare validation PCK curves:"
echo "  python scripts/plot_comparison.py \\"
echo "    outputs/compare_baseline/metrics.json \\"
echo "    outputs/compare_geometric_no_gcn/metrics.json \\"
echo "    outputs/compare_geometric_with_gcn/metrics.json"
echo ""
echo "Expected results:"
echo "  - Baseline: Should work (reference)"
echo "  - Geometric (no GCN): May be similar or slightly worse"
echo "  - Geometric (with GCN): Should be competitive or better"
echo ""
echo "If geometric encoder performs well, proceed to Stage 4 (decoder GCN)."
echo "If not, debug and tune hyperparameters before proceeding."
echo "======================================================================"

