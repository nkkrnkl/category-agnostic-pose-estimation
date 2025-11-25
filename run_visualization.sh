#!/bin/bash
# Convenience script to visualize CAPE predictions on test images
# Usage: ./run_visualization.sh [checkpoint_path] [num_samples]

# Activate virtual environment
source activate_venv.sh

# Default arguments
CHECKPOINT="${1:-outputs/cape_run/checkpoint_best_pck*.pth}"
NUM_SAMPLES="${2:-3}"
DEVICE="mps"  # Change to "cuda" or "cpu" if needed

# Expand wildcard in checkpoint path
CHECKPOINT_EXPANDED=$(ls -t $CHECKPOINT 2>/dev/null | head -1)

if [ -z "$CHECKPOINT_EXPANDED" ]; then
    echo "❌ Error: No checkpoint found matching: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -lh outputs/cape_run/*.pth 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "================================================================================"
echo "CAPE Visualization Runner"
echo "================================================================================"
echo "Checkpoint:   $CHECKPOINT_EXPANDED"
echo "Samples/cat:  $NUM_SAMPLES"
echo "Device:       $DEVICE"
echo "Output:       visualizations/"
echo "================================================================================"
echo ""

# Run visualization
python visualize_cape_predictions.py \
    --checkpoint "$CHECKPOINT_EXPANDED" \
    --dataset_root . \
    --device "$DEVICE" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir visualizations \
    2>&1 | tee visualization_output.log

echo ""
echo "================================================================================"
echo "✓ Visualization complete! Check visualizations/ for output images"
echo "================================================================================"
