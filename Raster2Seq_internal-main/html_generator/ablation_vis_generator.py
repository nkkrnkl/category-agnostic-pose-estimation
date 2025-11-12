import os
from glob import glob
from html4vision import Col, imagetable

CLASS2ID = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

max_samples = 100
gt_folder = "output_gt_s3dbw/"
base_path = "s3d_test_preds_abl/s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_nopolyrefine_predecPE_ema4eval_v1/"
pls_featfusion_path = "s3d_test_preds_abl/s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_nopolyrefine_predecPE_deccatsrc_v1/"
pls_anchor_path = "s3d_test_preds_abl/s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_anchor_deccatsrc_correct_t1/"
full_path = "s3d_test_preds_abl/s3d_bw_ddp_poly2seq_l512_bin32_nosem_bs32_coo20_cls1_anchor_deccatsrc_converterv3_t1/"

# Create elements from directory of images
gt_images = glob(f'{gt_folder}/*_gt_image.png')
image_ids = [os.path.basename(f).split('_')[0] for f in gt_images]
gt_floors = [f'{gt_folder}/0{_id}_floor.png' for _id in image_ids]

resA = [f'{base_path}/0{img_id}_pred_floorplan.png' for img_id in image_ids] 
resB = [f'{pls_featfusion_path}/0{img_id}_pred_floorplan.png' for img_id in image_ids] 
resC = [f'{pls_anchor_path}/0{img_id}_pred_floorplan.png' for img_id in image_ids] 
resD = [f'{full_path}/0{img_id}_pred_floorplan.png' for img_id in image_ids] 

# table description
cols = [
    Col('id1', 'ID', image_ids),                                               # make a column of 1-based indices
    Col('img', 'GT Raster', gt_images),             # specify image content for column 2
    Col('img', 'GT Vector', gt_floors),             # specify image content for column 2
    Col('img', 'Base', resA),             # specify image content for column 2
    Col('img', '+FeatFusion', resB),             # specify image content for column 2
    Col('img', '+FeatFusion+Anchor', resC),             # specify image content for column 2
    Col('img', '+FeatFusion+Anchor+Order', resD),             # specify image content for column 2
]

# html table generation
imagetable(cols, out_file='abl_vis.html', imsize=(512, 512))