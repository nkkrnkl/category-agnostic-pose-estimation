import argparse
import json
import os
import sys
from tqdm import tqdm
from rplan_utils import generate_density, normalize_annotations, parse_floor_plan_polys, generate_coco_dict

sys.path.append('../.')
from common_utils import read_scene_pc, export_density


### Note: Some scenes have missing/wrong annotations. These are the indices that you should additionally exclude 
### to be consistent with MonteFloor and HEAT:
invalid_scenes_ids = []

type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'entrance': 5,
            'dining room': 6, 'study room': 7, 'storage': 8, 'front door': 9, 'unknown': 10, 'interior_door': 11}
id2id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 
    7:6, 8:7, 10:8, 15:9, 16:10, 17:11}

original_id2type = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, 
              "dining room": 7, "study room": 8, "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}

def config():
    a = argparse.ArgumentParser(description='Generate coco format data for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output', default='coco_stru3d', type=str, help='path to output folder')
    
    args = a.parse_args()
    return args

def main(args):
    data_root = args.data_root
    data_parts = os.listdir(data_root)

    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    annotation_outFolder = os.path.join(outFolder, 'annotations')
    if not os.path.exists(annotation_outFolder):
        os.mkdir(annotation_outFolder)

    train_img_folder = os.path.join(outFolder, 'train')
    val_img_folder = os.path.join(outFolder, 'val')
    # test_img_folder = os.path.join(outFolder, 'test')

    # test_img_folder
    for img_folder in [train_img_folder, val_img_folder, ]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    coco_train_json_path = os.path.join(annotation_outFolder, 'train.json')
    coco_val_json_path = os.path.join(annotation_outFolder, 'val.json')
    # coco_test_json_path = os.path.join(annotation_outFolder, 'test.json')

    coco_train_dict = {"images":[],"annotations":[],"categories":[]}
    coco_val_dict = {"images":[],"annotations":[],"categories":[]}
    # coco_test_dict = {"images":[],"annotations":[],"categories":[]}

    for key, value in type2id.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        coco_train_dict["categories"].append(type_dict)
        coco_val_dict["categories"].append(type_dict)
        # coco_test_dict["categories"].append(type_dict)

    ### begin processing
    instance_id = 0
    scenes = sorted(os.listdir(data_root))[:15_000]
    count = 0
    for path_name in tqdm(scenes):
        scene_id = os.path.basename(path_name).split('.')[0]
        scene_path = os.path.join(data_root, path_name)
        with open(scene_path) as f:
            scene_annot = json.load(f)

        if int(scene_id) in invalid_scenes_ids:
            print('skip {}'.format(scene_path))
            continue
        
        # # load pre-generated point cloud 
        # ply_path = os.path.join(scene_path, 'point_cloud.ply')
        # points = read_scene_pc(ply_path)
        # xyz = points[:, :3]

        # ### project point cloud to density map
        # density, normalization_dict = generate_density(xyz, width=256, height=256)
        
        # ### rescale raw annotations
        # normalized_annos = normalize_annotations(scene_path, normalization_dict)

        ### prepare coco dict
        img_id = int(scene_id)
        img_dict = {}
        img_dict["file_name"] = str(scene_id).zfill(5) + '.jpg'
        img_dict["id"] = img_id
        img_dict["width"] = 256
        img_dict["height"] = 256

        # ### parse annotations
        # polys = parse_floor_plan_polys(normalized_annos)
        polygons_list, scene_image = generate_coco_dict(scene_annot, instance_id, img_id, ignore_types=[])

        instance_id += len(scene_annot['room_type'])

        ### train
        if count < 12_000:
            coco_train_dict["images"].append(img_dict)
            coco_train_dict["annotations"] += polygons_list
            # export_density(density, train_img_folder, scene_id)
            scene_image.save(f"{train_img_folder}/{str(scene_id).zfill(5)}.png")

        ### val
        else:
        # elif int(scene_id) >= 8000 and int(scene_id) < 3250:
            coco_val_dict["images"].append(img_dict)
            coco_val_dict["annotations"] += polygons_list
            # export_density(density, val_img_folder, scene_id)
            scene_image.save(f"{val_img_folder}/{str(scene_id).zfill(5)}.png")

        # ### test
        # else:
        #     coco_test_dict["images"].append(img_dict)
        #     coco_test_dict["annotations"] += polygons_list
        #     # export_density(density, test_img_folder, scene_id)
        
        print(scene_id)
        count += 1


    with open(coco_train_json_path, 'w') as f:
        json.dump(coco_train_dict, f)
    with open(coco_val_json_path, 'w') as f:
        json.dump(coco_val_dict, f)
    # with open(coco_test_json_path, 'w') as f:
    #     json.dump(coco_test_dict, f)


if __name__ == "__main__":

    main(config())