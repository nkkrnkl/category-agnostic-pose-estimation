#!/usr/bin/env bash


DATA=data/coco_s3d_bw/
FOLDER=test

python predict.py \
               --dataset_name=stru3d \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=output/s3d_bw_ddp_queries40x30/checkpoint0499.pth \
               --output_dir=s3d_${FOLDER}_preds_wd/ \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --input_channels 3 \
               --image_scale 2 \
               --crop_white_space \
               --save_pred \

# python predict.py \
#                --dataset_name=stru3d \
#                --dataset_root=${DATA}/${FOLDER} \
#                --checkpoint=/home/htp26/RoomFormerTest/output/s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2/checkpoint0449.pth \
#                --output_dir=s3d_${FOLDER}_preds_wd \
#                --semantic_classes=19 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --use_anchor \
#                --ema4eval \
#                --per_token_sem_loss \
#                --save_pred \
#                --image_scale 2 \
#                --crop_white_space \
#                # --drop_wd \
#                # --debug
#             #    --plot_text \
#             #    --pre_decoder_pos_embed \
#                # --debug \

# EXP=s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_nopolyrefine_predecPE_ema4eval_v1/
# EXP=s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_nopolyrefine_predecPE_deccatsrc_v1
# EXP=s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_anchor_deccatsrc_correct_t1
# EXP=s3d_bw_ddp_poly2seq_l512_bin32_nosem_bs32_coo20_cls1_anchor_deccatsrc_converterv3_t1
# EXP=s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2

# python predict.py \
#                --dataset_name=stru3d \
#                --dataset_root=${DATA}/${FOLDER} \
#                --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint0449.pth \
#                --output_dir=s3d_${FOLDER}_speed \
#                --semantic_classes=19 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --use_anchor \
#                --ema4eval \
#                --save_pred \
#                --image_scale 1 \
#                --drop_wd \
#                --per_token_sem_loss \
#                --save_anchors \
#                # --one_color \
#                # --crop_white_space \
#                # --pre_decoder_pos_embed \
#                # --debug
#             #    --plot_text \
#                # --debug \
