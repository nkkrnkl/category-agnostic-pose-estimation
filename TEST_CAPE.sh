#!/bin/bash
# Quick test of CAPE training (minimal settings)

cd "$(dirname "$0")"
source ../venv/bin/activate

echo "Testing CAPE Implementation..."
echo "Running 1 epoch with 5 episodes for quick validation"
echo ""
echo "Note: Using CPU for compatibility with deformable attention"
echo "      (MPS has issues with grid_sampler backward pass)"
echo ""

python train_cape_episodic.py \
    --cape_mode \
    --dataset_root . \
    --category_split_file category_splits.json \
    --output_dir output/cape_test \
    --device cpu \
    --epochs 1 \
    --batch_size 1 \
    --num_queries_per_episode 1 \
    --episodes_per_epoch 5 \
    --lr 1e-4 \
    --support_encoder_layers 3 \
    --hidden_dim 256 \
    --vocab_size 2000 \
    --seq_len 200 \
    --semantic_classes 70 \
    --print_freq 1 \
    --num_workers 0
