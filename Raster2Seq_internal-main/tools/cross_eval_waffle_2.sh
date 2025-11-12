# python clipseg_eval_script.py clipseg_eval/config.yaml 0 \
#     slurm_scripts3/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnorder499_t1_iou0.25/jsons

# slurm_scripts3/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnoorder_t1_iou0.25/jsons/
# slurm_scripts3/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv2_t1_iou0.25/jsons/

# RoomFormer r2g-waffle
# python clipseg_eval_script.py clipseg_eval/config.yaml 0 \
#     slurm_scripts3/r2g_queries56x50_sem13_iou0.25/jsons

# Ours r2g-waffle
# python clipseg_eval_script.py clipseg_eval/config.yaml 0 \
#     slurm_scripts3/r2g_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls13_convertv3_from849_t1_iou0.25/jsons


# # FRI-Net cc5k-waffle
# python clipseg_eval_script.py clipseg_eval/config.yaml 0 \
#     /home/htp26/FRI-Net/eval_results/cc5k-waffle_frinet_nowd_ckpt_last/json

# FRI-Net cc5k-waffle
python clipseg_eval_script.py clipseg_eval/config.yaml 0 \
    /home/htp26/FRI-Net/eval_results/r2g-waffle_frinet_ckpt_last/json
