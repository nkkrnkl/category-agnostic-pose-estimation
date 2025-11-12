#!/usr/bin/env bash

# python plot_floor.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --eval_set=test \
#                --output_dir=output_gt_s3dpoint \
#                --semantic_classes=19 \
#                --input_channels 1 \
#                --disable_image_transform \
#                --image_size 256 \
#                --plot_gt_image \
#                --crop_white_space \
#                --image_scale 2 \

python plot_floor.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw/ \
               --eval_set=test \
               --output_dir=output_gt_s3dbw_onecolor/ \
               --semantic_classes=-1 \
               --input_channels 3 \
               --disable_image_transform \
               --poly2seq \
               --seq_len 512 \
               --num_bins 64 \
               --image_size 256 \
               --crop_white_space \
               --image_scale 2 \
               --plot_gt_image \
               --plot_gt \
               --one_color
            #    --debug
            #    --plot_polys