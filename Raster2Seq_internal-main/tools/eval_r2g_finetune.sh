#!/usr/bin/env bash

EXP=r2g_queries56x50_sem13
EPOCH=0799
python eval.py --dataset_name=r2g \
               --dataset_root=data/R2G_hr_dataset_processed_v1 \
               --eval_set=test \
               --checkpoint=output/r2g_queries56x50_sem13/checkpoint${EPOCH}.pth \
               --output_dir=slurm_scripts4/${EXP}/eval_epoch${EPOCH} \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=13 \
               --input_channels 3 \
               --ema4eval \
               --save_pred \
            #    --debug \

# EXP=r2g_poly2seq_l512_bin32_nosem_coo20_cls5_anchor_deccatsrc_ignorewd_smoothing_convertv3_fromckpt_t1
# EPOCH=0649
# python eval.py --dataset_name=r2g \
#                --dataset_root=data/R2G_hr_dataset_processed_v1 \
#                --eval_set=test \
#                --checkpoint=output/${EXP}/checkpoint${EPOCH}.pth \
#                --output_dir=slurm_scripts2/${EXP} \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=-1 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --ema4eval \
#                --use_anchor \
#                # --pre_decoder_pos_embed \
#             #    --debug \
#                # --save_pred \