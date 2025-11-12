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
#             #    --ema4eval \
#             #    --save_pred \
#             #    --debug \

EXP=cubi_v4-1refined_poly2seq_l512_bin32_nosem_coo20_cls5_anchor_deccatsrc_ignorewd_smoothing_convertv3_fromckpt_t1
EPOCH=0499
python eval.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
               --eval_set=test \
               --checkpoint=output/${EXP}/checkpoint${EPOCH}.pth \
               --output_dir=slurm_scripts2/${EXP} \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=-1 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
               # --pre_decoder_pos_embed \
            #    --debug \
               # --save_pred \