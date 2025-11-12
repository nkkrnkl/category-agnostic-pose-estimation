import os
import json

fn = 'train.json'
path = f"/home/htp26/RoomFormerTest/data/coco_rplan/annotations/{fn}"
output_dir = f"/share/kuleshov/htp26/floorplan_datasets/coco_rplan_2/annotations"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{fn}"

with open(path, "r") as f:
    annot = json.load(f)

# for i, x in enumerate(annot['images']):
#     annot['images'][i]['file_name'] = f"{str(x['id']).zfill(5)}.png"

# with open(output_path, "wt") as f:
#     json.dump(annot, f)