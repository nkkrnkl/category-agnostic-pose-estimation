import os
from glob import glob
from html4vision import Col, imagetable

root_path = "data/coco_cubicasa5k_nowalls_v3"
split = 'train'
max_samples = 100

# Create elements from directory of images
gt_images = sorted(glob(f'{root_path}/{split}/*.png'))[:max_samples]

# Create IDs from filenames
ids = [os.path.basename(f).split('.')[0] for f in gt_images]

col5 = [f'{root_path}/{split}_aux/{img_id}_org.png' for img_id in ids]
col6 = [f'{root_path}/{split}_aux/{img_id}_mask.png' for img_id in ids]
col1 = [f'{root_path}/{split}_aux/{img_id}_room.png' for img_id in ids] 
col2 = [f'{root_path}/{split}_aux/{img_id}_icon.png' for img_id in ids]
col3 = [f'{root_path}/{split}_aux/{img_id}_combined.png' for img_id in ids]
col4 = [f'{root_path}/{split}_aux/{img_id}_final.png' for img_id in ids]

# table description
cols = [
    Col('id1', 'ID', ids),                             # make a column of 1-based indices
    Col('img', 'Raster (org)', col5),             # specify image content for column 2
    Col('img', 'Mask', col6),             # specify image content for column 2
    Col('img', 'Clean Raster (after mask-out)', gt_images),             # specify image content for column 2
    Col('img', 'Room map (org)', col1),    # specify image content for column 2
    Col('img', 'Icon map (org)', col2),    # specify image content for column 3
    Col('img', 'Combined Room map with window, door', col3), # specify image content for column 4
    Col('img', 'Final Room map', col4), # specify image content for column 4
    # Col('text', 'Class Label', [str(CLASS2ID)] * max_samples)
]

# html table generation
# imsize=(256, 256)
imagetable(cols, out_file='gt_vis.html', imsize=[512,512])