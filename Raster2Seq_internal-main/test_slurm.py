import os
import time
import subprocess

import numpy as np
import pandas as pd

slurm_template = """#!/bin/bash -e

export EPOCH_ID={epoch}
export EXP={exp}

echo "----------------------------"
echo $EPOCH_ID $EXP
echo "----------------------------"

CUDA_VISIBLE_DEVICES={device} python eval.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/{exp}/checkpoint{epoch}.pth \
               --output_dir={slurm_output}/{exp}_{epoch}/ \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
               --per_token_sem_loss \
               --save_pred \
               --converter_version v3 \
            #    --model_version 'v2' \
            #    --pre_decoder_pos_embed \

"""

###### ARGS
exp = "s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2"
device = "0"

config = pd.DataFrame({
    "epochs": ['0449'],
})
print(config)

###################################
slurm_file_path = f"slurm_scripts/{exp}/run.sh"
slurm_output = f"slurm_scripts/{exp}/"
os.makedirs(slurm_output, exist_ok=True)

for idx, row in config.iterrows():
    slurm_command = slurm_template.format(
        exp=exp,
        epoch=row.epochs,
        slurm_output=slurm_output,
        device=device,
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])