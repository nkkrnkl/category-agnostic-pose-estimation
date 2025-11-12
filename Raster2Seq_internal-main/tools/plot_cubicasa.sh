#!/usr/bin/env bash

SPLIT=test
python plot_floor.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined \
               --eval_set=${SPLIT} \
               --output_dir=output_gt_cc5k_refined_v4-1_wd/${SPLIT} \
               --semantic_classes=19 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_image_transform \
               --image_size 512 \
               --plot_gt_image \
               --crop_white_space \
               --image_scale 2 \
            #    --plot_gt \
               # --debug \
               # --plot_gt_image
               # --plot_density \