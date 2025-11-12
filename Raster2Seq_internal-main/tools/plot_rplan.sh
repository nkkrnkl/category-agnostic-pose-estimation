#!/usr/bin/env bash

python plot_floor.py --dataset_name=rplan \
               --dataset_root=/share/kuleshov/htp26/floorplan_datasets/coco_rplan_2/ \
               --eval_set=val \
               --output_dir=plot_gt_outputs \
               --semantic_classes=12 \
               --input_channels 3 \
               --disable_image_transform \
               --debug \
               --plot_gt \
            #    --plot_density \