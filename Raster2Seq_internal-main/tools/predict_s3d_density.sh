#!/usr/bin/env bash

DATA=data/stru3d/
FOLDER=test

python predict.py \
               --dataset_name=stru3d \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=/share/elor/htp26/roomformer/checkpoints/roomformer_stru3d_semantic_rich.pth \
               --output_dir=s3d_density_preds/roomformer_sem_${FOLDER}_preds \
               --semantic_classes=19 \
               --save_pred \
               --image_scale 1 \
               --input_channels 1 \
               --drop_wd \
               --num_queries=2800 \
               --num_polys=70 \
               # --crop_white_space \

# # EXP=s3d_projection_ddp_poly2seq_l512_bin32_nosem_bs32_coo20_cls2_anchor_deccatsrc_converterv3_t1
# EXP=s3d_projection_poly2seq_l512_sem1_bs32_coo20_cls2_anchor_deccatsrc_correct_numcls19_pts_finetune_convertv3_t2

# python predict.py \
#                --dataset_name=stru3d \
#                --dataset_root=${DATA}/${FOLDER} \
#                --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint0699.pth \
#                --output_dir=s3d_density_preds/ours_sem_${FOLDER}_preds \
#                --semantic_classes=19 \
#                --input_channels 1 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --use_anchor \
#                --ema4eval \
#                --save_pred \
#                --drop_wd \
#             #    --pre_decoder_pos_embed \
#             #    --image_scale 1 \
#             #    --crop_white_space \
#                # --per_token_sem_loss \
#                # --debug
#             #    --plot_text \
#                # --debug \
