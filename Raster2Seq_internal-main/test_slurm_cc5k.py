import os
import time
import subprocess

import numpy as np
import pandas as pd


###### ARGS
exp = "cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_raster1_anchor_deccatsrc_smoothing_cls12_convertv3_fromorder499_t1"
device = "0"
num_classes = 12

config = pd.DataFrame({
    "epochs": ['499'],
})
print(config)


if num_classes > 0:
    slurm_template = """#!/bin/bash -e

    export EPOCH_ID={epoch}
    export EXP={exp}

    echo "----------------------------"
    echo $EPOCH_ID $EXP
    echo "----------------------------"

    CUDA_VISIBLE_DEVICES={device} python eval.py --dataset_name=cubicasa \
                --dataset_root=data/coco_cubicasa5k_nowalls_{data_versiion}_refined/ \
                --eval_set=test \
                --checkpoint=/home/htp26/RoomFormerTest/output/{exp}/checkpoint{epoch}.pth \
                --output_dir={slurm_output}/eval_cc5k_epoch{epoch} \
                --num_queries=2800 \
                --num_polys=50 \
                --semantic_classes={num_classes} \
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
                # --pre_decoder_pos_embed \

    """
else:
    slurm_template = """#!/bin/bash -e

    export EPOCH_ID={epoch}
    export EXP={exp}

    echo "----------------------------"
    echo $EPOCH_ID $EXP
    echo "----------------------------"

    CUDA_VISIBLE_DEVICES={device} python eval.py --dataset_name=cubicasa \
                --dataset_root=data/coco_cubicasa5k_nowalls_{data_versiion}_refined/ \
                --eval_set=test \
                --checkpoint=/home/htp26/RoomFormerTest/output/{exp}/checkpoint{epoch}.pth \
                --output_dir={slurm_output}/eval_cc5k_epoch{epoch} \
                --num_queries=2800 \
                --num_polys=50 \
                --semantic_classes={num_classes} \
                --input_channels 3 \
                --poly2seq \
                --seq_len 512 \
                --num_bins 32 \
                --disable_poly_refine \
                --dec_attn_concat_src \
                --ema4eval \
                --use_anchor \
                --save_pred \
                #    --per_token_sem_loss \
                # --pre_decoder_pos_embed \

    """

###################################
slurm_file_path = f"slurm_scripts4/{exp}/run.sh"
slurm_output = f"slurm_scripts4/{exp}/"
os.makedirs(slurm_output, exist_ok=True)

for idx, row in config.iterrows():
    slurm_command = slurm_template.format(
        exp=exp,
        epoch=str(row.epochs).zfill(4),
        slurm_output=slurm_output,
        device=device,
        num_classes=num_classes,
        data_versiion="v4-1",
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])