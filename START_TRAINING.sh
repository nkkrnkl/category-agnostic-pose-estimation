#!/bin/bash
# Quick start script for MP-100 CAPE training
# Usage: bash START_TRAINING.sh [mode]
# Modes: test, debug, quick, full, split1-5

set -e  # Exit on error

# Enable MPS fallback for unsupported operations on Mac
export PYTORCH_ENABLE_MPS_FALLBACK=1

MODE=${1:-test}
# Get the project root from the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MP-100 CAPE Training Script"
echo "=========================================="
echo "Mode: $MODE"
echo ""

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    # Try to find and activate venv (check current directory first, then parent)
    if [ -f ".venv/bin/activate" ]; then
        echo "Activating local .venv virtual environment..."
        source ".venv/bin/activate"
    elif [ -f "venv/bin/activate" ]; then
        echo "Activating local venv virtual environment..."
        source "venv/bin/activate"
    elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
        echo "Activating parent venv virtual environment..."
        source "$PROJECT_ROOT/venv/bin/activate"
    else
        echo "Warning: No virtual environment found. Using system Python."
        echo "  To create one: python3 -m venv .venv && source .venv/bin/activate"
    fi
fi

# Ensure data symlink exists for mounted GCS data
if [ ! -e "data" ] && [ -d "Raster2Seq_internal-main/data" ]; then
    echo "Creating symlink: data -> Raster2Seq_internal-main/data"
    ln -s Raster2Seq_internal-main/data data
fi

# Verify mount is active
if [ ! -d "Raster2Seq_internal-main/data" ] || [ -z "$(mount | grep 'Raster2Seq_internal-main/data')" ]; then
    echo "Warning: GCS mount may not be active. Make sure to run:"
    echo "  cd Raster2Seq_internal-main && ./mount_gcs.sh"
    echo ""
fi

# Determine Python command
PYTHON_CMD="python3"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python or activate a virtual environment."
    exit 1
fi

case $MODE in
    test)
        echo "Running dataset loading test..."
        $PYTHON_CMD test_mp100_loading.py
        ;;

    debug)
        echo "Running debug training (overfit 1 sample, 50 epochs)..."
        $PYTHON_CMD train_mp100_cape.py \
            --debug \
            --dataset_root . \
            --mp100_split 1 \
            --epochs 50 \
            --output_dir output/debug_test \
            --job_name debug
        ;;

    tiny)
        echo "Running tiny experiment (5 epochs, batch_size 8, for speed testing)..."
        $PYTHON_CMD train_mp100_cape.py \
            --dataset_root . \
            --mp100_split 2 \
            --batch_size 8 \
            --epochs 5 \
            --lr 1e-4 \
            --image_norm \
            --skip_missing_files \
            --output_dir output/tiny_test \
            --job_name tiny_test \
            --num_workers 4
        ;;

    quick)
        echo "Running quick experiment (20 epochs)..."
        $PYTHON_CMD train_mp100_cape.py \
            --dataset_root . \
            --mp100_split 1 \
            --batch_size 4 \
            --epochs 20 \
            --lr 1e-4 \
            --image_norm \
            --output_dir output/quick_experiment \
            --job_name quick_test \
            --num_workers 2
        ;;

    full)
        echo "Running full training on split 1 (300 epochs)..."
        $PYTHON_CMD train_mp100_cape.py \
            --dataset_root . \
            --mp100_split 1 \
            --batch_size 4 \
            --epochs 300 \
            --lr 1e-4 \
            --lr_drop 200,250 \
            --image_norm \
            --use_anchor \
            --dec_layer_type v5 \
            --output_dir output/split1_full \
            --job_name split1_v5 \
            --num_workers 4
        ;;

    split*)
        SPLIT_NUM=${MODE#split}
        echo "Running full training on split $SPLIT_NUM..."
        $PYTHON_CMD train_mp100_cape.py \
            --dataset_root . \
            --mp100_split $SPLIT_NUM \
            --batch_size 4 \
            --epochs 300 \
            --lr 1e-4 \
            --lr_drop 200,250 \
            --image_norm \
            --use_anchor \
            --dec_layer_type v5 \
            --output_dir output/split${SPLIT_NUM} \
            --job_name split${SPLIT_NUM}_v5 \
            --num_workers 4
        ;;

    all)
        echo "Running full training on ALL 5 splits..."
        for split in 1 2 3 4 5; do
            echo ""
            echo "=========================================="
            echo "Training Split $split / 5"
            echo "=========================================="
            $PYTHON_CMD train_mp100_cape.py \
                --dataset_root . \
                --mp100_split $split \
                --batch_size 4 \
                --epochs 300 \
                --lr 1e-4 \
                --lr_drop 200,250 \
                --image_norm \
                --use_anchor \
                --dec_layer_type v5 \
                --output_dir output/split${split} \
                --job_name split${split}_v5 \
                --num_workers 4
        done
        echo ""
        echo "All 5 splits completed!"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  test   - Test dataset loading"
        echo "  debug  - Debug training (overfit 1 sample)"
        echo "  tiny   - Tiny experiment (5 epochs, batch_size 8, ~30-60 min)"
        echo "  quick  - Quick experiment (20 epochs, batch_size 4, ~3-6 hours)"
        echo "  full   - Full training on split 1 (300 epochs, ~7-14 days)"
        echo "  split1 - Full training on split 1"
        echo "  split2 - Full training on split 2"
        echo "  split3 - Full training on split 3"
        echo "  split4 - Full training on split 4"
        echo "  split5 - Full training on split 5"
        echo "  all    - Train on all 5 splits"
        echo ""
        echo "Example: bash START_TRAINING.sh tiny"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
