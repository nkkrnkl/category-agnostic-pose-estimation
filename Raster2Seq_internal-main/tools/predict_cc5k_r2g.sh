#!/usr/bin/env bash

# python eval.py --dataset_name=cubicasa \
#                --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_queries56x50_sem_v1/checkpoint0499.pth \
#                --output_dir=eval_cubi \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                # --debug \
#             #    --ema4eval \
#             #    --save_pred \

DATA=/share/elor/htp26/floorplan_datasets/R2G_dataset/
FOLDER=test 

# python predict.py \
#                --dataset_root=${DATA}/${FOLDER} \
#                --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_queries56x50_sem_v1/checkpoint0499.pth \
#                --output_dir=cc5k_${FOLDER}_preds \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                --drop_wd \

python predict.py \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv2_t1/checkpoint1899.pth \
               --output_dir=cc5kmodel_r2g_${FOLDER}_preds \
               --semantic_classes=12 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --use_anchor \
               --ema4eval \
               --per_token_sem_loss \
               --drop_wd \
               --one_color \
               --crop_white_space

               # --plot_text \
            #    --save_pred \
            #    --pre_decoder_pos_embed \
               # --debug \