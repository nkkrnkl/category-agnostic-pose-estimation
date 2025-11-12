#!/usr/bin/env bash

# python eval.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --eval_set=test \
#                --checkpoint=checkpoints/roomformer_stru3d_semantic_rich.pth \
#                --output_dir=eval_stru3d_sem_rich \
#                --num_queries=2800 \
#                --num_polys=70 \
#                --semantic_classes=19

EXP=s3d_projection_poly2seq_l512_sem1_bs32_coo20_cls2_anchor_deccatsrc_correct_numcls19_pts_finetune_convertv3_t2
# s3d_projection_ddp_poly2seq_l512_sem_bs32_coo20_cls5_anchor_deccatsrc_correct_smoothing1e-1_numcls19_pts_finetune_t1
# s3d_projection_ddp_poly2seq_l512_sem_bs32_coo20_cls5_anchor_deccatsrc_correct_smoothing1e-1_numcls19_pts_finetune_convertv2_t2
# s3d_projection_ddp_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_correct_smoothing1e-1_numcls19_pts_finetune_t1
EPOCH=0699
python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=output/${EXP}/checkpoint${EPOCH}.pth \
               --output_dir=slurm_scripts/${EXP}/epoch${EPOCH} \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --input_channels 1 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
               --per_token_sem_loss \
            #    --pre_decoder_pos_embed \