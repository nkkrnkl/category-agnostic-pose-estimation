#!/bin/bash
#SBATCH -J s3d_wd              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,elor         # Request partition
#SBATCH --constraint="[6000ada|a6000|a5000|a5500|3090|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=4              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# python main.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --num_queries=2800 \
#                --num_polys=70 \
#                --semantic_classes=19 \
#                --job_name=train_stru3d_sem_rich

MASTER_PORT=23560
SEQ_LEN=512
SEM_COEFF=1
CLS_COEFF=1
COO_COEFF=20
JOB=s3d_bw_ddp_poly2seq_l${SEQ_LEN}_sem_bs32_coo${COO_COEFF}_cls${CLS_COEFF}_anchor_deccatsrc_correct_smoothing1e-1_numcls19_pts_finetune_wdonly_debug
PRETRAIN=/home/htp26/RoomFormerTest/output/s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_anchor_deccatsrc_correct_t1/checkpoint2349.pth
NUM_GPUS=1

# s3d_bw_ddp_poly2seq_l512_sem${SEM_COEFF}_bs32_coo${COO_COEFF}_cls${CLS_COEFF}_nopolyrefine_predecPE_deccatsrc_pts_finetune_t1
WANDB_MODE=offline python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=$MASTER_PORT main_ddp.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --job_name=${JOB} \
               --batch_size 32 \
               --input_channels=3 \
               --output_dir /share/elor/htp26/roomformer/output/ \
               --poly2seq \
               --seq_len ${SEQ_LEN} \
               --num_bins 32 \
               --ckpt_every_epoch 50 \
               --eval_every_epoch 20 \
               --lr 2e-4 \
               --lr_backbone 2e-5 \
               --label_smoothing 0.1 \
               --epochs 1800 \
               --lr_drop '' \
               --cls_loss_coef ${CLS_COEFF} \
               --coords_loss_coef ${COO_COEFF} \
               --room_cls_loss_coef ${SEM_COEFF} \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --per_token_sem_loss \
               --ema4eval \
               --jointly_train \
               --start_from_checkpoint ${PRETRAIN} \
               --use_anchor \
               --resume output/${JOB}/checkpoint.pth \
               --wd_only \
               --debug
               # --pre_decoder_pos_embed \
               # --dec_layer_type='v4' \
               # --clip_max_norm 1.0 \
               # --dec_qkv_proj \