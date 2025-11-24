#!/bin/bash

# TRAINING.sh
# Quick start script for Category-Agnostic Pose Estimation (CAPE) training
#
# This script trains the Raster2Seq CAPE model with support pose conditioning
# on MP-100 dataset with train/test category split for unseen category evaluation

set -e  # Exit on error

echo "=============================================================================="
echo "  Category-Agnostic Pose Estimation (CAPE) Training"
echo "=============================================================================="
echo ""
echo "Training setup:"
echo "  - Episodic few-shot learning"
echo "  - Support pose graph conditioning"
echo "  - Raster2Seq framework"
echo ""

# Set environment variable for MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check if running in quick mode
QUICK_MODE="${1:-normal}"

if [ "$QUICK_MODE" = "quick" ]; then
    echo "⚡ QUICK MODE: Running with reduced settings for fast testing"
    echo ""
    PRELIMINARY_FLAG="--preliminary"
    CONFIG_FILE="${2:-configs/default.yaml}"
else
    echo "🚀 NORMAL MODE: Full training configuration"
    echo ""
    PRELIMINARY_FLAG=""
    CONFIG_FILE="${2:-configs/default.yaml}"
fi

# Create output directories
OUTPUT_DIR="output/cape_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Config file:         $CONFIG_FILE"
echo "  Mode:                $([ "$QUICK_MODE" = "quick" ] && echo "Preliminary (5 epochs)" || echo "Full training")"
echo "  Output directory:    $OUTPUT_DIR"
echo ""

# Check for required files
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "   Please ensure the config file exists."
    exit 1
fi

if [ ! -d "data/cleaned_annotations" ]; then
    echo "❌ Error: data/cleaned_annotations/ directory not found!"
    echo "   Please ensure MP-100 cleaned annotations are in place."
    exit 1
fi

if [ ! -d "data" ]; then
    echo "❌ Error: data/ directory not found!"
    echo "   Please ensure MP-100 images are in place."
    exit 1
fi

# Check for annotation files (at least one split)
ANNOTATION_FOUND=0
for split in 1 2 3 4 5; do
    if [ -f "data/cleaned_annotations/mp100_split${split}_train.json" ]; then
        ANNOTATION_FOUND=1
        break
    fi
done

if [ $ANNOTATION_FOUND -eq 0 ]; then
    echo "❌ Error: No MP-100 annotation files found in data/cleaned_annotations/ directory!"
    echo "   Expected files like: data/cleaned_annotations/mp100_split1_train.json"
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
    print('mps')
    sys.exit(0)
else:
    print('cpu')
    sys.exit(0)
")

echo "Detected device: $DEVICE"
echo ""

if [ "$DEVICE" = "cuda:0" ]; then
    echo "✓ CUDA detected - Using GPU for training"
    GPU_NAME=$(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo "N/A")
    GPU_MEMORY=$(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB") if torch.cuda.is_available() else print("N/A")' 2>/dev/null || echo "N/A")
    echo "  GPU: $GPU_NAME"
    echo "  Memory: $GPU_MEMORY"
elif [ "$DEVICE" = "mps" ]; then
    echo "✓ MPS detected - Using Apple Silicon GPU for training"
else
    echo "Note: Using CPU (CUDA/MPS not available)"
    echo "      Training will be slower on CPU"
fi
echo ""

# Launch training
echo "=============================================================================="
echo "  Starting CAPE Training..."
echo "=============================================================================="
echo ""

# Build training command
TRAIN_CMD="python3 train.py --config $CONFIG_FILE"

if [ -n "$PRELIMINARY_FLAG" ]; then
    TRAIN_CMD="$TRAIN_CMD $PRELIMINARY_FLAG"
fi

# Check if resume checkpoint is provided
if [ -n "$3" ] && [ -f "$3" ]; then
    echo "Resuming from checkpoint: $3"
    TRAIN_CMD="$TRAIN_CMD --resume $3"
fi

echo "Running command:"
echo "  $TRAIN_CMD"
echo ""

# Run training
$TRAIN_CMD

TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "  Training Complete!"
    echo "=============================================================================="
    echo ""
    
    # Find the most recent checkpoint directory from config
    CHECKPOINT_DIR=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
except:
    print('checkpoints')
    " 2>/dev/null || echo "checkpoints")
    
    LOG_DIR=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('paths', {}).get('log_dir', 'logs'))
except:
    print('logs')
    " 2>/dev/null || echo "logs")
    
    echo "Outputs saved to:"
    echo "  Checkpoints: $CHECKPOINT_DIR"
    echo "  Logs: $LOG_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Check training logs in: $LOG_DIR"
    echo "  2. Evaluate on test set:"
    echo "     python3 eval.py --checkpoint $CHECKPOINT_DIR/<checkpoint_name>.pth"
    echo "  3. View model checkpoints in: $CHECKPOINT_DIR"
else
    echo "  Training Failed with exit code: $TRAIN_EXIT_CODE"
    echo "=============================================================================="
    exit $TRAIN_EXIT_CODE
fi

echo ""

