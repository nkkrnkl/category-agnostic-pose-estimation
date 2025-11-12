#!/usr/bin/env bash

DATA=data/R2G_hr_dataset_processed_v1
FOLDER=test 

# python predict.py \
#                --dataset_root=${DATA}/${FOLDER} \
#                --checkpoint=/home/htp26/RoomFormerTest/output/r2g_queries56x50_sem13/checkpoint0799.pth \
#                --output_dir=r2g_${FOLDER}_preds \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=13 \
#                --input_channels 3 \
#                --drop_wd \
#                --image_scale 2 \
#                --crop_white_space \
#                --measure_time \
#                --batch_size 1 \

python predict.py \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=/home/htp26/RoomFormerTest/output/r2g_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls13_convertv3_from849_t1/checkpoint0549.pth \
               --output_dir=r2g_${FOLDER}_preds \
               --semantic_classes=13 \
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
               --crop_white_space \
               --image_scale 2 \
               --measure_time \
               --batch_size 1 \
               # --one_color \
               # --plot_text \
            #    --save_pred \
            #    --pre_decoder_pos_embed \
               # --debug \