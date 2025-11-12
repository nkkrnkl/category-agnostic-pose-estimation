#!/bin/bash
#SBATCH -J cc5k_sem_v3              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,elor         # Request partition
#SBATCH --constraint="[a6000|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=4             # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

export NCCL_P2P_LEVEL=NVL
 
# MASTER_PORT=13143
# WANDB_MODE=online python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=cubicasa \
#                --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --job_name=cubi_v4-1refined_queries56x50_sem_ignore8_t2 \
#                --batch_size 6 \
#                --input_channels=3 \
#                --output_dir /share/elor/htp26/roomformer/output \
#                --eval_every_epoch=20 \
#                --ckpt_every_epoch=20 \
#                --epochs 800 \
#                --ignore_index 8
#             #    --poly2seq \
#             #    --seq_len 1024 \
#             #    --num_bins 64 \
#             #    --debug \
#                # --start_from_checkpoint output/s3d_sem_rgb_ddp_queries40x30/checkpoint0499.pth
#                # --resume output/cubi_queries60x30_sem_debug_t3/checkpoint.pth

MASTER_PORT=14156
CLS_COEFF=5
COO_COEFF=20
SEM_COEFF=1
SEQ_LEN=512
NUM_BINS=32
CONVERTER=v3
DATA=data/coco_cubicasa5k_nowalls_v4-1_refined/
JOB=cubi_v4-1refined_poly2seq_l${SEQ_LEN}_bin${NUM_BINS}_sem${SEM_COEFF}_coo${COO_COEFF}_cls${CLS_COEFF}_raster1_anchor_deccatsrc_smoothing_cls12_convert${CONVERTER}_fromorder499_t1

# PRETRAIN=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_poly2seq_l512_bin32_nosem_coo20_cls1_anchor_deccatsrc_fromckpt2450_ignorewd_smoothing_clscoeffx5@6e-1_t1/checkpoint1999.pth
PRETRAIN=output/cubi_v4-1refined_poly2seq_l512_bin32_nosem_coo20_cls5_anchor_deccatsrc_ignorewd_smoothing_convertv3_fromckpt_t1/checkpoint0499.pth

WANDB_MODE=online torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=cubicasa \
               --dataset_root=${DATA} \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=12 \
               --job_name=${JOB} \
               --batch_size 56 \
               --input_channels=3 \
               --output_dir /share/elor/htp26/roomformer/output \
               --poly2seq \
               --seq_len $SEQ_LEN \
               --num_bins ${NUM_BINS} \
               --ckpt_every_epoch=50 \
               --eval_every_epoch=50 \
               --label_smoothing 0.1 \
               --epochs 600 \
               --lr_drop '1900' \
               --cls_loss_coef ${CLS_COEFF} \
               --coords_loss_coef ${COO_COEFF} \
               --room_cls_loss_coef ${SEM_COEFF} \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --per_token_sem_loss \
               --ema4eval \
               --jointly_train \
               --start_from_checkpoint ${PRETRAIN} \
               --resume output/${JOB}/checkpoint.pth \
               --use_anchor \
               --converter_version ${CONVERTER} \
               --raster_loss_coef 1.0 \
               # --disable_wd_as_line \
            #    --pre_decoder_pos_embed \
               # --dec_layer_type='v3' \