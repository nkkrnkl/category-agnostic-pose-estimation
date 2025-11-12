#!/bin/bash
#SBATCH -J rp              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,kuleshov         # Request partition
#SBATCH --constraint="[a6000|a5000|3090|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=2              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# python main.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --num_queries=2800 \
#                --num_polys=70 \
#                --semantic_classes=19 \
#                --job_name=train_stru3d_sem_rich

# MASTER_PORT=13597
# python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=rplan \
#                --dataset_root=data/coco_rplan_2 \
#                --num_queries=1200 \
#                --num_polys=40 \
#                --semantic_classes=12 \
#                --job_name=rp_ddp_queries40x30_1debug \
#                --batch_size 16 \
#                --input_channels=3 \
#                --output_dir /share/kuleshov/htp26/roomformer/output/ \
#                --debug \
#                #    --start_from_checkpoint checkpoints/roomformer_stru3d_semantic_rich.pth
#             #    --set_cost_coords=20 \
#                # --coords_loss_coef=20 \
#                # --lr 2e-5 \
#                # --lr_backbone 2e-5 \

#                # --resume /share/kuleshov/htp26/roomformer/output/s3d_sem_rgb_ddp_fromckpt/checkpoint.pth
#                # --eval_every_epoch 1

# WANDB_MODE=offline 
MASTER_PORT=13127
python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=rplan \
               --dataset_root=data/coco_rplan_2 \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=12 \
               --job_name=rp_ddp_queries40x30_sem_t1 \
               --batch_size 12 \
               --input_channels=3 \
               --output_dir /share/kuleshov/htp26/roomformer/output/ \
               --eval_every_epoch=20 \
               --epochs 500 \
               # --debug \
               # --resume output/rp_ddp_queries40x20_nosem_1debug_t3/checkpoint0059.pth
            #    --lr 5e-4 \