import os
from glob import glob
from html4vision import Col, imagetable

# CLASS2ID = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
#             'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
#             'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

max_samples = 100
split = 'test'

path_A = "waffle_rastertest_preds/cubi_v4-1refined_queries56x50_sem_v1/"
path_B = "waffle_rastertest_preds/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_t1/"

results_A = sorted(glob(f'{path_A}/*_pred_room_map.png'))[:max_samples]
results_B = sorted(glob(f'{path_B}/*_pred_room_map.png'))[:max_samples]
results_A2 = sorted(glob(f'{path_A}/*_pred_floorplan.png'))[:max_samples]
results_B2 = sorted(glob(f'{path_B}/*_pred_floorplan.png'))[:max_samples]

# Create IDs from filenames
ids = [os.path.basename(f).split('_')[0] for f in results_A]
gt_files = [f'{path_A}/{_id}.png' for _id in ids]

# table description
cols = [
    Col('id1', 'ID', ids),                                               # make a column of 1-based indices
    Col('img', 'Input Raster', gt_files),     # specify image content for column 3
    Col('img', 'RoomFormer', results_A),     # specify image content for column 3
    Col('img', 'RoomFormer Map', results_A2),     # specify image content for column 3
    Col('img', 'Poly2Seq', results_B), # specify image content for column 4
    Col('img', 'Poly2Seq Map', results_B2),     # specify image content for column 3
]

# html table generation
imagetable(cols, out_file='waffle_pred_vis.html', imsize=(1024, 1024))