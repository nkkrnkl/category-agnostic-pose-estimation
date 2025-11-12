import os
import json
from glob import glob
from html4vision import Col, imagetable

root_before = "data/waffle/benchmark"
root_after = "data/waffle_benchmark_processed"
split = 'test'
image_root = f"{root_after}/{split}/"

with open(f"{root_after}/annotations/{split}_image_id_mapping.json", 'r') as f:
    image_id_mapping = {x[-5:]: x[:-6] for x in f.read().splitlines()}

raster_paths_after = sorted(glob(f'{image_root}/*.png'))
image_ids = [os.path.basename(f).replace('.png', '') for f in raster_paths_after]

# Create IDs from filenames
mask_paths = [f'{root_after}/aux/{image_id_mapping[img_id]}_fg_mask.png' for img_id in image_ids]
seg_after_paths = [f'{root_after}/aux/{image_id_mapping[img_id]}_polylines.png' for img_id in image_ids]
seg_before_paths = [os.path.join(root_before, 'segmented_descrete_pngs', f'{image_id_mapping[img_id]}_seg_colors.png') for img_id in image_ids]
raster_paths_before = [os.path.join(root_before, 'pngs', f'{image_id_mapping[img_id]}.png') for img_id in image_ids]

# table description
cols = [
    Col('id1', 'ID', image_ids),                             # make a column of 1-based indices
    Col('img', 'Raster (before)', raster_paths_before),             # specify image content for column 2
    Col('img', 'Seg (before)', seg_before_paths),             # specify image content for column 2
    Col('img', 'Mask', mask_paths),             # specify image content for column 2
    Col('img', 'Raster (after)', raster_paths_after),             # specify image content for column 2
    Col('img', 'Seg (after)', seg_after_paths),    # specify image content for column 2
]

# html table generation
imagetable(cols, out_file='waffle_vis.html', imsize=[512,512])