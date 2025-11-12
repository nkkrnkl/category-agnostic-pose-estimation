#!/usr/bin/env bash

DATA=data/waffle/data/original_size_images/
FOLDER=00000

CKPT=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnoorder_t1/checkpoint0499.pth

python predict.py \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=${CKPT} \
               --output_dir=waffle_raster${FOLDER}_preds2 \
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
               --save_pred \
               --one_color \
               --image_scale 2 \
               --crop_white_space