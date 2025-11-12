import sys
import os
from pathlib import Path
import glob
import shutil

import argparse
import cv2
import numpy as np
import json

from shapely.geometry import Polygon
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_utils import resort_corners
from create_coco_cc5k import create_coco_bounding_box


from util.plot_utils import plot_semantic_rich_floorplan_nicely, plot_semantic_rich_floorplan_opencv


def plot_floor(output_coco_polygons, categories_dict, img_w, img_h, save_path, door_window_index=[10, 9]):
    gt_sem_rich = []
    for j, (poly, poly_type) in enumerate(output_coco_polygons):
        corners = np.array(poly).reshape(-1, 2).astype(np.int32)
        # corners_flip_y = corners.copy()
        # corners_flip_y[:,1] = 255 - corners_flip_y[:,1]
        # corners = corners_flip_y
        gt_sem_rich.append([corners, poly_type])
    # plot_semantic_rich_floorplan_nicely(gt_sem_rich, save_path, prec=None, rec=None, 
    #                                     plot_text=True, is_bw=False,
    #                                     door_window_index=door_window_index, 
    #                                     img_w=img_w,
    #                                     img_h=img_h,
    #                                     semantics_label_mapping=get_dataset_class_labels(categories_dict),
    #                                     )
    plot_semantic_rich_floorplan_opencv(gt_sem_rich, save_path, img_w=img_w, img_h=img_h, 
                                        door_window_index=door_window_index, 
                                        semantics_label_mapping=get_dataset_class_labels(categories_dict), is_bw=False)



def prepare_dict(categories_dict):
    save_dict = {"images":[],"annotations":[],"categories":categories_dict}
    return save_dict


def extract_polygons_from_mask(binary_mask, output_mask_path):
    """
    Extract polygons from a binary mask where regions with value 1 are polygons
    and background regions have value 0.

    Args:
        binary_mask (numpy.ndarray): Binary mask with shape (H, W), where 1 represents
                                     the polygon regions and 0 represents the background.

    Returns:
        list: A list of polygons, where each polygon is represented as a list of (x, y) coordinates.
    """
    # Ensure the mask is binary (0 and 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract polygons from contours
    polygons = []
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.001 * cv2.arcLength(contour, True)  # Adjust epsilon for more/less detail
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(approx_polygon.squeeze().tolist())  # Convert to list of (x, y) points

    # Convert binary_mask to a 3-channel image to draw colored polylines
    binary_mask_colored = cv2.cvtColor(binary_mask*255, cv2.COLOR_GRAY2BGR)

    # Plot polygons on the binary mask with green color
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        cv2.polylines(binary_mask_colored, [points], isClosed=True, color=(0, 0, 255), thickness=10)
    
    cv2.imwrite(output_mask_path, binary_mask_colored)

    return polygons


def read_polygons_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        category_dict = data['categories']
        polygons = [data['annotations'][i]['segmentation'][0] for i in range(len(data['annotations']))]
        poly_types = [data['annotations'][i]['category_id'] for i in range(len(data['annotations']))]
        source_misc = [data['annotations'][i] for i in range(len(data['annotations']))]
        source_polygons = [(polygons[i], poly_types[i]) for i in range(len(polygons))]

    return source_polygons, source_misc, category_dict


def get_dataset_class_labels(category_dict):
    return {category_dict[i]['id']: category_dict[i]['name'] for i in range(len(category_dict))}


def check_all_window_door_inside(polygons, door_window_index):
    flag = all([poly_type in door_window_index for _, poly_type in polygons ])
    return flag


def extract_region_and_annotation(source_image, source_annot_path, region_polygons, 
                                  start_image_id,
                                  output_image_dir="output", output_annot_dir="annotations", output_aux_dir="output_aux",
                                  vis_aux=True):
    """
    Extract regions of the floorplan from the source image based on polygons
    and generate annotations.

    Args:
        source_image (numpy.ndarray): The source image (H, W, 3).
        polygons (list): List of polygons, where each polygon is a list of (x, y) coordinates.
        output_dir (str): Directory to save the extracted regions and annotations.

    Returns:
        list: A list of annotations for each extracted region.
    """
    door_window_index = [10, 9]
    source_polygons, source_misc, categories_dict = read_polygons_from_json(source_annot_path)
    source_img_id = os.path.basename(source_annot_path).split('.')[0].zfill(5)
    if vis_aux:
        gt_sem_rich_path = os.path.join(output_aux_dir, '{}_org_floor.png'.format(source_img_id))
        plot_floor(source_polygons, categories_dict, source_image.shape[1], source_image.shape[0], gt_sem_rich_path, door_window_index=door_window_index)
    margin = 10
    img_id = start_image_id

    # each region polygon corresponds to an image
    for i, polygon in enumerate(region_polygons):
        instance_id = 0
        output_coco_polygons = []
        # Create a mask for the current polygon
        mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # Crop the ROI to the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(points)
        # Expand the bounding box by the margin
        x_expanded = max(x - margin, 0)
        y_expanded = max(y - margin, 0)
        w_expanded = min(x + w + margin, source_image.shape[1]) - x_expanded
        h_expanded = min(y + h + margin, source_image.shape[0]) - y_expanded

        x, y, w, h = x_expanded, y_expanded, w_expanded, h_expanded
        cropped_roi = source_image[y:y+h, x:x+w]

        save_dict = prepare_dict(categories_dict)

        # Create an annotation for the extracted region
        img_dict = {}
        img_dict["file_name"] = f'{str(img_id).zfill(5)}_{source_img_id}.png'
        img_dict["id"] = img_id 
        img_dict["width"] = w
        img_dict["height"] = h

        # Save the cropped ROI
        roi_filename = f"{output_image_dir}/{str(img_id).zfill(5)}_{source_img_id}.png"
        cv2.imwrite(roi_filename, cropped_roi)

        bounding_box = np.array([x, y, x + w, y + h])

        # Convert source polygons to NumPy arrays for vectorized operations
        source_polygons_np = [np.array(src_poly[0]).reshape(-1, 2) for src_poly in source_polygons]
        assert (len(source_polygons_np) == len(source_polygons))

        coco_annotation_dict_list = []
        # Iterate through the polygons and filter those inside the bounding box
        for j, tmp in enumerate(source_polygons_np):
            # Compute the bounding box of the current polygon
            poly_bbox = np.hstack([np.min(tmp, axis=0), np.max(tmp, axis=0)])

            # Check if the polygon is outside the bounding box
            if np.any(poly_bbox[:2] < bounding_box[:2]) or np.any(poly_bbox[2:] > bounding_box[2:]):
                continue

            # Scale the polygon coordinates relative to the top-left corner of the bounding box
            scaled_polygon = tmp - bounding_box[:2]

            coco_seg_poly = []
            poly_sorted = resort_corners(scaled_polygon)
            # image = draw_polygon_on_image(image, poly_shapely, "test_poly.jpg")

            for p in poly_sorted:
                coco_seg_poly += list(p)

            if len(scaled_polygon) == 2:
                area = source_misc[j]['area']
                coco_bb = source_misc[j]['bbox']
                # shift the bounding box
                coco_bb[0] -= bounding_box[0]
                coco_bb[1] -= bounding_box[1]
            else:
                poly_shapely = Polygon(scaled_polygon)
                area = poly_shapely.area
                rectangle_shapely = poly_shapely.envelope

                # Slightly wider bounding box
                bb_x, bb_y = rectangle_shapely.exterior.xy
                coco_bb = create_coco_bounding_box(bb_x, bb_y, w, h, bound_pad=2)

            coco_annotation_dict = {
                    "segmentation": [coco_seg_poly],
                    "area": area,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": coco_bb,
                    "category_id": source_polygons[j][1],
                    "id": instance_id}
            coco_annotation_dict_list.append(coco_annotation_dict)
            output_coco_polygons.append([coco_seg_poly, source_polygons[j][1]])

            # Remove after obtaining the polygon
            # source_polygons.pop(j)
            # source_misc.pop(j)
            instance_id += 1
        
        # skip if just windows and doors are inside
        if check_all_window_door_inside(output_coco_polygons, door_window_index):
            instance_id -= len(coco_annotation_dict_list)
            continue

        save_dict['images'].append(img_dict)
        save_dict["annotations"] += coco_annotation_dict_list

        if vis_aux:
            gt_sem_rich_path = os.path.join(output_aux_dir, '{}_{}_floor.png'.format(str(img_id).zfill(5), source_img_id))
            plot_floor(output_coco_polygons, categories_dict, w, h, gt_sem_rich_path, door_window_index=door_window_index)

        # Save annotations to a JSON file
        json_path = f"{output_annot_dir}/{str(img_id).zfill(5)}_{source_img_id}.json"
        with open(json_path, "w") as f:
            json.dump(save_dict, f)
        
        img_id += 1
    
    start_image_id = img_id
        
    return start_image_id


def config():
    a = argparse.ArgumentParser(description='Generate coco format data for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output', default='coco_cubicasa5k', type=str, help='path to output folder')
    
    args = a.parse_args()
    return args




# Example usage
if __name__ == "__main__":
    args = config()

    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    annotation_outFolder = os.path.join(outFolder, 'annotations_json')
    if not os.path.exists(annotation_outFolder):
        os.mkdir(annotation_outFolder)
    
    annos_train_folder = os.path.join(annotation_outFolder, 'train')
    annos_val_folder = os.path.join(annotation_outFolder, 'val')
    annos_test_folder = os.path.join(annotation_outFolder, 'test')
    os.makedirs(annos_train_folder, exist_ok=True)
    os.makedirs(annos_val_folder, exist_ok=True)
    os.makedirs(annos_test_folder, exist_ok=True)

    train_img_folder = os.path.join(outFolder, 'train')
    val_img_folder = os.path.join(outFolder, 'val')
    test_img_folder = os.path.join(outFolder, 'test')

    for img_folder in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    ### begin processing
    start_image_id = 3500
    save_folders = [train_img_folder, val_img_folder, test_img_folder]
    annos_folders = [annos_train_folder, annos_val_folder, annos_test_folder]
    splits = ['train', 'val', 'test']

    def wrapper(index):
        image_path, annot_path, mask_path = packed_input_files[index]
        cur_image_id = int(os.path.basename(image_path).split('.')[0])
        binary_mask = cv2.imread(mask_path)[:,:,-1] 
        source_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Extract polygons
        region_polygons = extract_polygons_from_mask(binary_mask, output_mask_path=f'{save_aux_path}/{str(cur_image_id).zfill(5)}_polylines.png')

        return extract_region_and_annotation(source_image, annot_path, region_polygons, 
                                    start_image_id + index*10,
                                    save_path, save_anno_path, save_aux_path,
                                    vis_aux=True)

    def worker_init(input_files_object):
        # Store dataset as global to avoid pickling issues
        global packed_input_files
        packed_input_files = input_files_object

    for i, split in enumerate(splits):
        image_files = sorted(glob.glob(f"{args.data_root}/{split}/*.png"))
        image_id_list = [os.path.basename(image_path).split('.')[0] for image_path in image_files]
        anno_files = [f"{args.data_root}/annotations_json/{split}/{id_}.json" for id_ in image_id_list]
        mask_files = [f"{args.data_root}/{split}_aux/{id_}_mask.png" for id_ in image_id_list]
        save_path = save_folders[i]
        save_anno_path = annos_folders[i]
        save_aux_path = save_path.rstrip('/') + '_aux'
        os.makedirs(save_aux_path, exist_ok=True)

        # for j, (image_path, anno_path, mask_path) in enumerate(zip(image_files, anno_files, mask_files)):
        #     cur_image_id = int(os.path.basename(image_path).split('.')[0])
        #     binary_mask = cv2.imread(mask_path)[:,:,-1] 
        #     source_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #     # Extract polygons
        #     polygons = extract_polygons_from_mask(binary_mask, output_mask_path=f'{save_aux_path}/{str(cur_image_id).zfill(5)}_polylines.png')
        #     # # skip if only one polygon (floorplan)
        #     # if len(polygons) == 1:
        #     #     print(f"Skipping {image_path} with only one polygon")
        #     #     with open(anno_path, 'r') as f:
        #     #         data = json.load(f)
        #     #     # update image id
        #     #     data['images'][0]['id'] = start_image_id
        #     #     data['images'][0]["file_name"] = f'{str(start_image_id).zfill(5)}_{str(cur_image_id).zfill(5)}.png'
        #     #     for anno in data['annotations']:
        #     #         anno['image_id'] = start_image_id

        #     #     with open(f"{save_anno_path}/{str(start_image_id).zfill(5)}_{str(cur_image_id).zfill(5)}.json", 'w') as f:
        #     #         json.dump(data, f, indent=2)
        #     #     shutil.copy(image_path, f"{save_path}/{str(start_image_id).zfill(5)}_{str(cur_image_id).zfill(5)}.png")

        #     #     gt_sem_rich_path = os.path.join(save_aux_path, '{}_{}_floor.png'.format(str(start_image_id).zfill(5), str(cur_image_id).zfill(5)))
        #     #     output_coco_polygons = [(x['segmentation'][0], x['category_id']) for x in data['annotations']]
        #     #     plot_floor(output_coco_polygons, data['categories'], data['images'][0]['width'], data['images'][0]['height'], gt_sem_rich_path, door_window_index=[10, 9])

        #     #     start_image_id += 1
        #     #     continue

        #     # # Print the extracted polygons
        #     # print("Extracted polygons:")
        #     # for i, polygon in enumerate(polygons):
        #     #     print(f"Polygon {i + 1}: {polygon}")

        #     start_image_id = extract_region_and_annotation(source_image,
        #                                 anno_path,
        #                                 polygons,
        #                                 start_image_id,
        #                                 output_image_dir=save_path, 
        #                                 output_annot_dir=save_anno_path,
        #                                 output_aux_dir=save_aux_path,
        #                                 vis_aux=True)

        packed_input_files = list(zip(image_files, anno_files, mask_files))
        # for j in range(5):
        #     wrapper(j)
        # exit(0)

        num_processes = 16
        with Pool(num_processes, initializer=worker_init, initargs=(packed_input_files,)) as p:
            indices = [j for j in range(len(packed_input_files))]
            list(tqdm(p.imap(wrapper, indices), total=len(indices)))
        
        start_image_id += len(packed_input_files)*10