import os
from glob import glob
from html4vision import Col, imagetable

# CLASS2ID = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
#             'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
#             'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

def get_image_paths_from_directory(directory_path):
    """
    Load all images from the specified directory.

    Args:
        directory_path (str): Path to the directory containing images.

    Returns:
        list: A list of PIL Image objects.
    """
    paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Add more extensions if needed

    # Iterate through all files in the directory
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(valid_extensions):  # Check for valid image extensions
                file_path = os.path.join(root, filename)
                paths.append(file_path)

    return paths

max_samples = 1000
split = 'test'

path_A = "r2g_test_preds/r2g_res256_vis" # "slurm_scripts4/r2g_res256_ckpt/eval/"
path_B = "r2g_test_preds/r2g_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls13_convertv3_from849_t1/"
path_C = "r2g_test_preds/r2g_queries56x50_sem13/"
path_D = "r2g_test_preds/heat_r2g_256_vis/"

data = "output_gt_r2g/test/"

image_files = sorted(glob(f'{data}/*_gt_image.png'))[:max_samples]


# Create IDs from filenames
ids = [os.path.basename(f).replace('_gt_image.png', '') for f in image_files]
gt_files = [f'{data}/{str(_id)}_floor.png' for _id in ids][:max_samples]
results_A2 = [f'{path_A}/{str(_id).zfill(6)}_pred_floorplan.png' for _id in ids][:max_samples]
results_B2 = [f'{path_B}/{str(_id).zfill(6)}_pred_floorplan.png' for _id in ids][:max_samples]
results_C2 = [f'{path_C}/{str(_id).zfill(6)}_pred_floorplan.png' for _id in ids][:max_samples]
results_D2 = [f'{path_D}/{str(_id).zfill(6)}_pred_floorplan.png' for _id in ids][:max_samples]


# table description
cols = [
    Col('id1', 'ID', list(range(len(ids)))),                                               # make a column of 1-based indices
    Col('img', 'Input Raster', image_files),     # specify image content for column 3
    Col('img', 'GT', gt_files),     # specify image content for column 3
    Col('img', 'HEAT', results_D2), # specify image content for column 4
    Col('img', 'RoomFormer', results_C2),     # specify image content for column 3
    Col('img', 'Raster2Graph Map', results_A2),     # specify image content for column 3
    Col('img', 'Raster2Seq Map', results_B2),     # specify image content for column 3
]

# html table generation
imagetable(cols, out_file='r2g_pred_res256_vis.html', imsize=(512, 512))