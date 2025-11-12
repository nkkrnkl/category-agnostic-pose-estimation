import os
import json
from glob import glob
from html4vision import Col, imagetable

org_root_path = "data/coco_cubicasa5k_nowalls_v4"
root_path = "data/coco_cubicasa5k_nowalls_v4-1_refined"
split = 'test'
max_samples = 100

# Create elements from directory of images
gt_jsons = sorted(glob(f'{root_path}/annotations_json/{split}/*.json'))[:max_samples]
with open(f"{root_path}/annotations/{split}_image_id_mapping.json", 'r') as f:
    image_id_mapping = json.load(f)

# Create IDs from filenames
ids = [os.path.basename(f).split('.')[0] for f in gt_jsons]
org_ids = [os.path.basename(f).split('.')[0].split('_')[1] for f in gt_jsons]
new_ids = [os.path.basename(f).split('.')[0].split('_')[0] for f in gt_jsons]

col1 = [f'{org_root_path}/{split}/{img_id}.png' for img_id in org_ids]
col2 = [f'{root_path}/{split}_aux/{img_id}_polylines.png' for img_id in org_ids]
col3 = [f'{root_path}/{split}_aux/{img_id}_org_floor.png' for img_id in org_ids] 
col4 = [f'{root_path}/{split}_aux/{img_id}_floor.png' for img_id in ids] 
col5 = [f'{root_path}/{split}/{str(image_id_mapping[str(int(img_id))]).zfill(5)}.png' for img_id in new_ids]

# table description
cols = [
    Col('id1', 'ID', ids),                             # make a column of 1-based indices
    Col('img', 'Raster (org)', col1),             # specify image content for column 2
    Col('img', 'Floormap (org)', col3),             # specify image content for column 2
    Col('img', 'Polylines', col2),             # specify image content for column 2
    Col('img', 'Floormap (after)', col4),    # specify image content for column 2
    Col('img', 'Raster (after)', col5),    # specify image content for column 2
    # Col('text', 'Class Label', [str(CLASS2ID)] * max_samples)
]

# html table generation
imagetable(cols, out_file='gt_vis_2.html', imsize=[512,512])