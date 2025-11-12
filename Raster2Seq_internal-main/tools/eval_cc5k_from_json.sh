#!/usr/bin/env bash

# EXP=r2g_orgin_ckpt
# python eval_from_json.py --dataset_name=r2g \
#                --dataset_root=data/R2G_hr_dataset_processed_v1 \
#                --eval_set=test \
#                --output_dir=slurm_scripts4/${EXP}/eval \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                --input_json_dir /home/htp26/Raster-to-Graph/output/r2g_org_test_2/jsons/ \
#                --num_workers 0 \
#                --device cpu \
#                --image_size 512 \
#                # --debug \
#             #    --save_pred \
#             #    --debug \


# EXP=cc5k_heat_256_ckpt
# python eval_from_json.py --dataset_name=cubicasa \
#                --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
#                --eval_set=test \
#                --output_dir=slurm_scripts4/${EXP}/eval \
#                --semantic_classes=-1 \
#                --input_channels 3 \
#                --input_json_dir /home/htp26/heat/results/npy_heat_cc5k_256/test \
#                --num_workers 0 \
#                --device cpu \
#                --image_size 256 \

EXP=cc5k_frinet_nowd_256_ckpt_test
python eval_from_json.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
               --eval_set=test \
               --output_dir=slurm_scripts4/${EXP}/test \
               --semantic_classes=-1 \
               --input_channels 3 \
               --input_json_dir /home/htp26/FRI-Net/eval_results/cc5k_frinet_nowd_ckpt_last/json \
               --num_workers 0 \
               --device cpu \
               --image_size 256 \
               --save_pred \

# EXP=r2g-cc5k_frinet_256_ckpt
# python eval_from_json.py --dataset_name=cubicasa \
#                --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
#                --eval_set=test \
#                --output_dir=cross_eval_out/${EXP}/eval \
#                --semantic_classes=-1 \
#                --input_channels 3 \
#                --input_json_dir /home/htp26/FRI-Net/eval_results/r2g-cc5k_frinet_ckpt_last/json \
#                --num_workers 0 \
#                --device cpu \
#                --image_size 256 \