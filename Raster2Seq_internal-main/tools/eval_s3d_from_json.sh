#!/usr/bin/env bash

EXP=s3d_bw_ddp_queries40x30
python eval_from_json.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --eval_set=test \
               --output_dir=s3d_test_preds_wd/${EXP}/test/ \
               --semantic_classes=19 \
               --input_channels 3 \
               --input_json_dir slurm_scripts/s3d_bw_ddp_queries40x30/jsons/ \
               --num_workers 0 \
               --device cpu \
               --image_size 256 \
               --save_pred \

# EXP=s3dbw_frinet_ckpt
# python eval_from_json.py --dataset_name=stru3d \
#                --dataset_root=data/coco_s3d_bw \
#                --eval_set=val \
#                --output_dir=slurm_scripts4/${EXP}/eval \
#                --semantic_classes=-1 \
#                --input_channels 3 \
#                --input_json_dir /home/htp26/FRI-Net/eval_results/s3dbw_frinet_ckpt_last_val/json \
#                --num_workers 0 \
#                --device cpu \
#                --image_size 256 \
#                --save_pred \

# EXP=s3d_roomformerxpolydiffuse_nosem
# INPUT_DIR=/home/htp26/poly-diffuse/outputs/s3d_polydiffuse_step10_roomformer_nosem/npy
# EXP=s3d_oursxpolydiffuse_nosem
# INPUT_DIR=/home/htp26/poly-diffuse/outputs/s3d_polydiffuse_step10_ours_nosem/npy
# EXP=s3d_oursxpolydiffuse_sem
# INPUT_DIR=/home/htp26/poly-diffuse/outputs/s3d_polydiffuse_step10_ours_sem/npy
# EXP=s3d_roomformerxpolydiffuse_sem
# INPUT_DIR=/home/htp26/poly-diffuse/outputs/s3d_polydiffuse_step10_roomformer_sem/npy
# EXP=s3d_ours_sem
# INPUT_DIR=/home/htp26/RoomFormerTest/s3d_density_preds/ours_sem_test_preds/s3d_projection_poly2seq_l512_sem1_bs32_coo20_cls2_anchor_deccatsrc_correct_numcls19_pts_finetune_convertv3_t2/npy
# python eval_from_json.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --eval_set=test \
#                --output_dir=slurm_scripts5/${EXP}/eval \
#                --semantic_classes=-1 \
#                --input_channels 3 \
#                --input_json_dir ${INPUT_DIR} \
#                --num_workers 0 \
#                --device cpu \
#                --image_size 256 \
#                --input_file_type npy