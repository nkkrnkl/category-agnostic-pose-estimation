#!/usr/bin/env bash

SPLIT=val
EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_t1
python plot_scores_by_seqlen.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw/ \
               --eval_set=${SPLIT} \
               --output_dir=graph_plot/${SPLIT} \
               --semantic_classes=-1 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_image_transform \
            #    --num_subset_images 1000 \
               # --json_dir /home/htp26/RoomFormerTest/slurm_scripts/${EXP}/result_jsons
               # --plot_density \
            #    --plot_gt_image \