#!/usr/bin/env bash

EXP=s3d_bw_ddp_queries40x30
python eval.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --eval_set=test \
               --checkpoint=output/${EXP}/checkpoint0499.pth \
               --output_dir=slurm_scripts/${EXP}_2/ \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --input_channels 3 \
               --save_pred \
               # --measure_time \
               # --batch_size 1 \
            #    --debug \

# EXP=s3d_bw_poly2seq_onestage_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_convertv3_t1
# EPOCH=1349
# python eval.py --dataset_name=stru3d \
#                --dataset_root=data/coco_s3d_bw \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint${EPOCH}.pth \
#                --output_dir=slurm_scripts/${EXP}_${EPOCH} \
#                --num_queries=1200 \
#                --num_polys=30 \
#                --semantic_classes=19 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --ema4eval \
#                --use_anchor \
#                --per_token_sem_loss \
#                --converter_version v3 \
#                --measure_time \
#                --batch_size 1
#                # --pre_decoder_pos_embed \
#                # --debug \
#                # --add_cls_token \
#             #    --disable_sampling_cache
#                # --dec_qkv_proj \
