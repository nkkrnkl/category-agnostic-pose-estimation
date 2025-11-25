#!/bin/bash
# Quick overfit test to verify model can learn
# Usage: ./run_overfit_test.sh [category_id]

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Default to category 40 (zebra) if not specified
CATEGORY=${1:-40}

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔍 DEBUG OVERFIT TEST - Category $CATEGORY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Purpose: Verify model can overfit on a single category"
echo "Expected: Training loss → 0 within ~20 epochs"
echo ""
echo "If loss stays high, there's a bug in model/data pipeline!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Activate venv if available
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run overfit test
python -m models.train_cape_episodic \
  --dataset_root . \
  --debug_overfit_category $CATEGORY \
  --debug_overfit_episodes 10 \
  --epochs 50 \
  --batch_size 2 \
  --accumulation_steps 2 \
  --lr 5e-4 \
  --early_stopping_patience 0 \
  --output_dir outputs/debug_overfit_cat${CATEGORY} \
  --print_freq 5 \
  2>&1 | tee overfit_cat${CATEGORY}.log

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ Overfit test complete!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Log saved to: overfit_cat${CATEGORY}.log"
echo "💾 Checkpoints saved to: outputs/debug_overfit_cat${CATEGORY}/"
echo ""
echo "Expected Results:"
echo "  - Epoch 10: Loss < 10.0"
echo "  - Epoch 20: Loss < 1.0"
echo "  - Epoch 50: Loss < 0.1"
echo ""
echo "If loss stays > 30, investigate:"
echo "  1. Check log for errors"
echo "  2. Enable debug mode: export DEBUG_CAPE=1"
echo "  3. Verify category $CATEGORY has images in your dataset"
echo ""
echo "To visualize results:"
echo "  ./run_simple_visualization.sh"
echo ""

