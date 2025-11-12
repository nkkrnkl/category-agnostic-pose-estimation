#!/usr/bin/env bash

# EXP=cubi_v4-1refined_queries56x50_sem_v1
# IOU=0.25
# python eval.py --dataset_name=waffle \
#                --dataset_root=data/waffle_benchmark_processed/ \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint0499.pth \
#                --output_dir=slurm_scripts3/${EXP}_iou${IOU} \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                --iou_thres ${IOU} \
#                --disable_sem_rich \
#                --save_pred \
#                # --debug \
#             #    --ema4eval \

# EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv2_t1
# EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnoorder_t1
# EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnorder499_t1
# EPOCH=0499
# IOU=0.25
# python eval.py --dataset_name=waffle \
#                --dataset_root=data/waffle_benchmark_processed/ \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint${EPOCH}.pth \
#                --output_dir=slurm_scripts3/${EXP}_iou${IOU} \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --ema4eval \
#                --per_token_sem_loss \
#                --use_anchor \
#                --iou_thres ${IOU} \
#                --disable_sem_rich \
#                --save_pred \
#             #    --converter_version v3 \
#                # --debug \
#                # --pre_decoder_pos_embed \


EXP=r2g_queries56x50_sem13
IOU=0.25
python eval.py --dataset_name=waffle \
               --dataset_root=data/waffle_benchmark_processed/ \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint0499.pth \
               --output_dir=slurm_scripts3/${EXP}_iou${IOU} \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=13 \
               --input_channels 3 \
               --iou_thres ${IOU} \
               --disable_sem_rich \
               --save_pred \
               --image_size=256
               # --debug \
            #    --ema4eval \


# EXP=r2g_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls13_convertv3_from849_t1
# EPOCH=0549
# IOU=0.25
# python eval.py --dataset_name=waffle \
#                --dataset_root=data/waffle_benchmark_processed/ \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint${EPOCH}.pth \
#                --output_dir=slurm_scripts3/${EXP}_iou${IOU} \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=13 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --ema4eval \
#                --per_token_sem_loss \
#                --use_anchor \
#                --iou_thres ${IOU} \
#                --disable_sem_rich \
#                --save_pred \
#                --image_size=256
#             #    --converter_version v3 \
#                # --debug \
#                # --pre_decoder_pos_embed \