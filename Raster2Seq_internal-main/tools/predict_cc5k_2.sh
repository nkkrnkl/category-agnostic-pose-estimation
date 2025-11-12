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

DATA=data/coco_cubicasa5k_nowalls_v4-1_refined/
FOLDER=test 

# python predict.py \
#                --dataset_name=cubicasa \
#                --dataset_root=${DATA}/${FOLDER} \
#                --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_queries56x50_sem_v1/checkpoint0499.pth \
#                --output_dir=cc5k_${FOLDER}_preds_wd \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                --save_pred \
#                --image_scale 2 \
#                --crop_white_space \
#                # --drop_wd \
#                # --plot_text \

# EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv2_t1
# EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnoorder_t1
EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnorder499_t1
# s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_nopolyrefine_predecPE_deccatsrc_v1
EPOCH=0499
python predict.py \
               --dataset_name=cubicasa \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint${EPOCH}.pth \
               --output_dir=cc5k_${FOLDER}_preds \
               --semantic_classes=12 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --use_anchor \
               --ema4eval \
               --per_token_sem_loss \
               --save_pred \
               --image_scale 2 \
               --crop_white_space \
               --drop_wd \
               --dec_attn_concat_src \
               # --plot_text \
            #    --pre_decoder_pos_embed \
               # --debug \