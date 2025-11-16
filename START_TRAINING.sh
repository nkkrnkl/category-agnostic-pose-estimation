#!/bin/bash
# Quick start script for MP-100 CAPE training
# Usage: bash START_TRAINING.sh [mode]
# Modes: test, debug, quick, full, split1-5

set -e  # Exit on error

MODE=${1:-test}
PROJECT_ROOT="/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project"
cd "$PROJECT_ROOT/theodoros"

echo "=========================================="
echo "MP-100 CAPE Training Script"
echo "=========================================="
echo "Mode: $MODE"
echo ""

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

case $MODE in
    test)
        echo "Running dataset loading test..."
        python test_mp100_loading.py
        ;;

    debug)
        echo "Running debug training (overfit 1 sample, 50 epochs)..."
        python train_mp100_cape.py \
            --debug \
            --dataset_root . \
            --mp100_split 1 \
            --epochs 50 \
            --output_dir output/debug_test \
            --job_name debug
        ;;

    quick)
        echo "Running quick experiment (20 epochs)..."
        python train_mp100_cape.py \
            --dataset_root . \
            --mp100_split 1 \
            --batch_size 2 \
            --epochs 20 \
            --lr 1e-4 \
            --image_norm \
            --output_dir output/quick_experiment \
            --job_name quick_test \
            --num_workers 2
        ;;

    full)
        echo "Running full training on split 1 (300 epochs)..."
        python train_mp100_cape.py \
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
        python train_mp100_cape.py \
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
            python train_mp100_cape.py \
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
        echo "  quick  - Quick experiment (20 epochs)"
        echo "  full   - Full training on split 1"
        echo "  split1 - Full training on split 1"
        echo "  split2 - Full training on split 2"
        echo "  split3 - Full training on split 3"
        echo "  split4 - Full training on split 4"
        echo "  split5 - Full training on split 5"
        echo "  all    - Train on all 5 splits"
        echo ""
        echo "Example: bash START_TRAINING.sh debug"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
