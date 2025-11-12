#!/usr/bin/env bash

SPLIT=test
python plot_floor.py --dataset_name=r2g \
               --dataset_root=data/R2G_hr_dataset_processed_v1/ \
               --eval_set=${SPLIT} \
               --output_dir=output_gt_r2g/${SPLIT} \
               --semantic_classes=13 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_image_transform \
               --image_size 512 \
               --plot_gt_image \
               --crop_white_space \
               --image_scale 1 \
               --plot_gt \
               # --debug \
               # --plot_gt_image
               # --plot_density \