import os
import json
from glob import glob
from html4vision import Col, imagetable

org_root_path = "data/coco_cubicasa5k_nowalls_v4-2_refined"
root_A = "output_gt_cc5k_refined_v4-1/"
root_B = "output_gt_cc5k_refined_v4-2/"
split = 'train'

with open(f"{org_root_path}/annotations/{split}_image_id_mapping.json", 'r') as f:
    image_id_mapping = json.load(f)

log_path = f"{org_root_path}/annotations_json/{split}_removed_ids.txt"
with open(log_path, 'r') as f:
    altered_file_ids = sorted([os.path.basename(x.split(' ')[0]).split('.')[0] for x in f.read().split('\n')])
    mapped_ids = [image_id_mapping[x.split('_')[0].lstrip('0')] for x in altered_file_ids]

# Create IDs from filenames
raster_paths = [f'{org_root_path}/{split}/{str(img_id).zfill(5)}.png' for img_id in mapped_ids] 
aux_paths = [f'{org_root_path}/{split}_aux2/{img_id}_overlap.png' for img_id in altered_file_ids]
paths_A = [f'{root_A}/{split}/{int(img_id)}_floor_nosem.png' for img_id in mapped_ids]
paths_B = [f'{root_B}/{split}/{int(img_id)}_floor_nosem.png' for img_id in mapped_ids]
# overlapping_paths = [f'{org_root_path}/{split}_aux2/{img_id}_{str(image_id_mapping[str(int(img_id))]).zfill(5)}.png' for img_id in new_ids]

# table description
cols = [
    Col('id1', 'ID', mapped_ids),                             # make a column of 1-based indices
    Col('img', 'Raster', raster_paths),             # specify image content for column 2
    Col('img', 'Floormap (before)', paths_A),             # specify image content for column 2
    Col('img', 'Floormap (after)', paths_B),    # specify image content for column 2
    Col('img', 'Overlapping', aux_paths),    # specify image content for column 2
]

# html table generation
imagetable(cols, out_file='gt_vis_3.html', imsize=[512,512])