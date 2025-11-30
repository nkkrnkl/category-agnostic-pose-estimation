#!/bin/bash

# START_CAPE_TRAINING.sh
# Quick start script for Category-Agnostic Pose Estimation (CAPE) training
#
# This script trains the CAPE model with support pose graph conditioning
# on MP-100 dataset with train/test category split for unseen category evaluation

set -e  # Exit on error

echo "=============================================================================="
echo "  Category-Agnostic Pose Estimation (CAPE) Training"
echo "=============================================================================="
echo ""
echo "Training setup:"
echo "  - 56 training categories (episodic sampling)"
echo "  - 14 test categories (unseen, for evaluation)"
echo "  - Support pose graph conditioning"
echo "  - Raster2Seq framework"
echo ""

# Set environment variable for MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "../venv" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check if running in quick mode
QUICK_MODE="${1:-normal}"

if [ "$QUICK_MODE" = "quick" ]; then
    echo "⚡ QUICK MODE: Running with reduced settings for fast testing"
    echo ""
    EPOCHS=5
    EPISODES_PER_EPOCH=100
    BATCH_SIZE=1
    NUM_QUERIES=1
else
    echo "🚀 NORMAL MODE: Full training configuration"
    echo ""
    EPOCHS=300
    EPISODES_PER_EPOCH=1000
    BATCH_SIZE=64
    NUM_QUERIES=2
fi

# Create output directory
OUTPUT_DIR="output/cape_episodic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Epochs:              $EPOCHS"
echo "  Episodes/epoch:      $EPISODES_PER_EPOCH"
echo "  Batch size:          $BATCH_SIZE"
echo "  Queries/episode:     $NUM_QUERIES"
echo "  Output directory:    $OUTPUT_DIR"
echo ""

# Check for required files
if [ ! -f "category_splits.json" ]; then
    echo "❌ Error: category_splits.json not found!"
    echo "   This file defines train/test category splits."
    exit 1
fi

if [ ! -d "annotations" ]; then
    echo "❌ Error: annotations/ directory not found!"
    echo "   Please ensure MP-100 annotations are in place."
    exit 1
fi

if [ ! -d "data" ]; then
    echo "❌ Error: data/ directory not found!"
    echo "   Please ensure MP-100 images are in place."
    exit 1
fi

echo "✓ All required files found"
echo ""

# Python environment check
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found!"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check for PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ Error: PyTorch not installed!"
    echo "   Please install: pip install torch torchvision"
    exit 1
fi

echo "✓ PyTorch found"
echo ""

# Detect device (prioritizes CUDA if available)
DEVICE=$(python3 -c "
import torch
import sys
if torch.cuda.is_available():
    print('cuda:0')
    sys.exit(0)
elif torch.backends.mps.is_available():
    # Skip MPS due to deformable attention backward compatibility
    print('cpu')
    sys.exit(0)
else:
    print('cpu')
    sys.exit(0)
")

echo "Detected device: $DEVICE"
echo ""
if [ "$DEVICE" = "cuda:0" ]; then
    echo "✓ CUDA detected - Using GPU for training"
    echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
    echo "  CUDA optimizations enabled:"
    echo "    - Mixed Precision Training (AMP): ~2x speedup"
    echo "    - cuDNN benchmark mode: Faster convolutions"
    echo "    - Non-blocking data transfers: Overlapped CPU/GPU work"
else
    echo "Note: Using CPU (CUDA not available)"
    echo "      MPS is skipped due to deformable attention compatibility issues"
fi
echo ""

# Launch training
echo "=============================================================================="
echo "  Starting CAPE Training..."
echo "=============================================================================="
echo ""

python3 -m models.train_cape_episodic \
    --cape_mode \
    --poly2seq \
    --dataset_name mp100 \
    --dataset_root "$SCRIPT_DIR" \
    --category_split_file "category_splits.json" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_queries_per_episode $NUM_QUERIES \
    --episodes_per_epoch $EPISODES_PER_EPOCH \
    \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4 \
    --lr_drop "200,250" \
    --clip_max_norm 0.1 \
    \
    --support_encoder_layers 3 \
    --support_fusion_method cross_attention \
    \
    --backbone resnet50 \
    --hidden_dim 256 \
    --nheads 8 \
    --enc_layers 6 \
    --dec_layers 6 \
    --dim_feedforward 1024 \
    --dropout 0.1 \
    \
    --image_size 256 \
    --vocab_size 2000 \
    --seq_len 200 \
    --num_queries 200 \
    --num_polys 1 \
    \
    --cls_loss_coef 2.0 \
    --coords_loss_coef 5.0 \
    --room_cls_loss_coef 0.0 \
    \
    --semantic_classes 70 \
    --num_feature_levels 4 \
    --dec_n_points 4 \
    --enc_n_points 4 \
    \
    --aux_loss \
    --with_poly_refine \
    --num_workers 2 \
    --seed 42 \
    --print_freq 10 \
    --job_name "cape_episodic_$(date +%Y%m%d)"

echo ""
echo "=============================================================================="
echo "  Training Complete!"
echo "=============================================================================="
echo ""
echo "Outputs saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check training logs in: $OUTPUT_DIR"
echo "  2. Evaluate on unseen categories:"
echo "     python evaluate_cape.py --checkpoint $OUTPUT_DIR/checkpoint_best.pth --split test"
echo "  3. Visualize predictions:"
echo "     python visualize_cape.py --checkpoint $OUTPUT_DIR/checkpoint_best.pth"
echo ""
