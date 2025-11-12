#!/bin/bash
#SBATCH -J s3d_org_sem_v3              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=elor,gpu         # Request partition
#SBATCH --constraint="[a6000|a100|6000ada]"
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

MASTER_PORT=13563
SEQ_LEN=512
SEM_COEFF=1
CLS_COEFF=2
COO_COEFF=20
CONVERTER=v3
JOB=s3d_projection_poly2seq_l${SEQ_LEN}_sem${SEM_COEFF}_bs32_coo${COO_COEFF}_cls${CLS_COEFF}_anchor_deccatsrc_correct_numcls19_pts_finetune_convert${CONVERTER}_t2
# PRETRAIN=output/s3d_projection_ddp_poly2seq_l512_bin32_nosem_bs32_coo20_cls5_anchor_deccatsrc_t1/checkpoint2349.pth
PRETRAIN=output/s3d_projection_ddp_poly2seq_l512_bin32_nosem_bs32_coo20_cls2_anchor_deccatsrc_converterv3_t1/checkpoint0499.pth
NUM_GPUS=1

WANDB_MODE=online python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=$MASTER_PORT main_ddp.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --job_name=${JOB} \
               --batch_size 32 \
               --input_channels=1 \
               --output_dir /share/elor/htp26/roomformer/output/ \
               --poly2seq \
               --seq_len ${SEQ_LEN} \
               --num_bins 32 \
               --ckpt_every_epoch 50 \
               --eval_every_epoch 20 \
               --lr 2e-4 \
               --lr_backbone 2e-5 \
               --label_smoothing 0.0 \
               --epochs 800 \
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
               --converter_version ${CONVERTER} \