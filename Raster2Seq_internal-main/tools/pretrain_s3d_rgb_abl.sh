#!/bin/bash
#SBATCH -J s3d_drop              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=elor,gpu         # Request partition
#SBATCH --constraint="[a6000|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=4              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# python main.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --num_queries=800 \
#                --num_polys=20 \
#                --semantic_classes=-1 \
#                --output_dir /share/kuleshov/htp26/roomformer/output/ \
#                --job_name=stru3d_bs10_org_ddp \
#                --resume=/share/kuleshov/htp26/roomformer/output/stru3d_bs10_org_ddp/checkpoint.pth

# MASTER_PORT=13548
# python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=stru3d \
#             --dataset_root=data/stru3d_sem_rich \
#             --num_queries=800 \
#             --num_polys=20 \
#             --semantic_classes=-1 \
#             --job_name=s3d_rgb_ddp_1debug_t1 \
#             --output_dir /share/kuleshov/htp26/roomformer/output/ \
#             --input_channels=3 \
#             --debug \
#             --eval_every_epoch 1
#             # --image_norm \
#             # --coords_loss_coef=20 \
#             # --set_cost_coords=20 \
#             # --resume=/share/kuleshov/htp26/roomformer/output/stru3d_bs10_org_ddp/checkpoint.pth

MASTER_PORT=13576
CLS_COEFF=1
COO_COEFF=20
SEQ_LEN=512
NUM_BINS=32
CONVERTER=v3
JOB=s3d_bw_ddp_poly2seq_l${SEQ_LEN}_bin${NUM_BINS}_nosem_bs32_coo${COO_COEFF}_cls${CLS_COEFF}_anchor_deccatsrc_converter${CONVERTER}_drop0.2_t1

WANDB_MODE=online python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=-1 \
               --job_name=${JOB} \
               --batch_size 32 \
               --input_channels=3 \
               --output_dir /share/elor/htp26/roomformer/output/ \
               --poly2seq \
               --seq_len ${SEQ_LEN} \
               --num_bins ${NUM_BINS} \
               --ckpt_every_epoch 50 \
               --eval_every_epoch 20 \
               --lr 2e-4 \
               --lr_backbone 2e-5 \
               --label_smoothing 0.0 \
               --epochs 1400 \
               --lr_drop '' \
               --cls_loss_coef ${CLS_COEFF} \
               --coords_loss_coef ${COO_COEFF} \
               --resume output/${JOB}/checkpoint.pth \
               --ema4eval \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --converter_version ${CONVERTER} \
               --use_anchor \
               --random_drop_rate 0.2 \
               # --freeze_anchor \
               # --pre_decoder_pos_embed \
               # --wd_only \
            #    --increase_cls_loss_coef_epoch_ratio 0.6 --increase_cls_loss_coef 5. \
               # --dec_layer_type='v6' \
               # --clip_max_norm 1.0 \
               # --dec_qkv_proj \