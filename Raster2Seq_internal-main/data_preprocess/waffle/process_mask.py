import os
import sys
from glob import glob
from pathlib import Path
import json

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_utils import resort_corners


def draw_polygon_on_image(image, polygons, class_to_color):
    """
    Draws polygons on the image based on the COLOR_TO_CLASS mapping.

    Args:
        image (numpy.ndarray): The image on which to draw.
        polygons (list of list of tuple): List of polygons, where each polygon is a list of (x, y) points.

    Returns:
        numpy.ndarray: The image with polygons drawn.
    """
    # Draw each polygon on the image
    for polygon, polygon_class in polygons:
        # Convert polygon points to numpy array
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        color = class_to_color[polygon_class]
        bgr = (color[2], color[1], color[0])  # Convert RGB to BGR for OpenCV
        # Draw filled polygon
        cv2.fillPoly(image, [pts], bgr)

    return image


def fill_mask(segmentation_mask):
    filled_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)

    # Iterate over each class index in the segmentation mask
    for class_index in np.unique(segmentation_mask):
        if class_index == 0:  # Skip the background
            continue

        # Create a binary mask for the current class
        binary_mask = (segmentation_mask == class_index).astype(np.uint8)

        # Find contours for the current class
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill each contour with white color in the single-channel mask
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled_mask


def to_bw_image(input_image):
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to convert the grayscale image to black and white
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return bw_image


def create_coco_bounding_box(bb_x, bb_y, image_width, image_height, bound_pad=2):
    bb_x = np.unique(bb_x)
    bb_y = np.unique(bb_y)
    bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
    bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

    bb_x_max = np.minimum(np.max(bb_x) + bound_pad, image_width - 1)
    bb_y_max = np.minimum(np.max(bb_y) + bound_pad, image_height - 1)

    bb_width = (bb_x_max - bb_x_min)
    bb_height = (bb_y_max - bb_y_min)

    coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]
    return coco_bb


def prepare_dict(categories_dict):
    save_dict = {"images":[],"annotations":[],"categories":[]}
    for key, value in categories_dict.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        save_dict["categories"].append(type_dict)
    return save_dict


def convert_numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":

    LABEL_NOTATIONS = {
        "Background": (0, 0, 0),       # Black
        "Interior": (255, 255, 255), # White
        "Walls": (255, 0, 0),          # Red
        "Doors": (0, 0, 255),          # Blue
        "Windows": (0, 255, 255)       # Cyan
    }

    CLASS2INDEX = {
        "Background": 0,       # Black
        "Interior": 1, # White
        # "Walls": 2,          # Red
        "Doors": 3,          # Blue
        "Windows": 4       # Cyan
    }

    # Create a mapping from RGB values to class indices
    COLOR_TO_CLASS = {
        (0, 0, 0): 0,       # Background
        (255, 255, 255): 1, # Interior
        (255, 0, 0): 2,     # Walls
        (0, 0, 255): 3,     # Doors
        (0, 255, 255): 4    # Windows
    }

    NEW_CLASS_MAPPING = {
        1: 0,
        3: 1,
        4: 2,
    }

    CLASS_TO_COLOR = {
        0: (255, 255, 255), # Interior
        1: (0, 0, 255),     # Doors
        2: (0, 255, 255),   # Windows
    }

    root = "/home/htp26/RoomFormerTest/data/waffle/benchmark/"
    image_dir = f"{root}/pngs"
    label_dir = f"{root}/segmented_descrete_pngs"
    input_paths = sorted(glob(f"{label_dir}/*.png"))
    output_dir = "/share/elor/htp26/floorplan_datasets/waffle_benchmark_processed/"
    output_aux_dir = f"{output_dir}/aux"
    output_image_dir = f"{output_dir}/test/"
    output_annot_dir = f"{output_dir}/annotations/"
    fn_mapping_log = f"{output_annot_dir}/test_image_id_mapping.json"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_aux_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annot_dir, exist_ok=True)

    instance_count = 0

    save_dict = prepare_dict(CLASS2INDEX)
    output_mappings = []

    for i, path in enumerate(input_paths):
        # if i > 5:
        #     exit(0)
        mask = Image.open(path).convert("RGB")
        fn = os.path.basename(path).replace("_seg_colors.png", "")
        new_fn = str(i).zfill(5)

        mask = np.array(mask)
        image = Image.open(os.path.join(image_dir, f"{fn}.png")).convert("RGB")
        image_width, image_height = image.size

        # Initialize an empty segmentation mask with the same height and width as the input mask
        segmentation_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        img_id = i
        img_dict = {}
        img_dict["file_name"] = str(img_id).zfill(5) + '.png'
        img_dict["id"] = img_id
        img_dict["width"] = image_width
        img_dict["height"] = image_height

        output_polygons = []
        coco_annotation_dict_list = []
        # Iterate over each pixel in the mask and assign the corresponding class index
        for color, class_index in COLOR_TO_CLASS.items():
            # Create a boolean mask for the current color
            color_mask = (mask == color).all(axis=-1)
            color_mask_uint8 = color_mask.astype(np.uint8)

            # Assign the class index to the segmentation mask
            segmentation_mask[color_mask] = class_index

            if class_index not in NEW_CLASS_MAPPING:
                continue
            class_index = NEW_CLASS_MAPPING[class_index]

            # Find contours for the current color mask
            contours, _ = cv2.findContours(color_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_contours = []
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.001 * peri, True)
                new_contours.append(approx)

            # Convert contours to polygon coordinates
            polygons = [contour.reshape(-1, 2) for contour in new_contours]

            for polygon in polygons:
                # Convert the polygon to a Shapely Polygon object
                if polygon.shape[0] < 3:
                    continue

                shapely_polygon = Polygon(polygon)
                area = shapely_polygon.area
                rectangle_shapely = shapely_polygon.envelope
                bb_x, bb_y = rectangle_shapely.exterior.xy
                coco_bb = create_coco_bounding_box(bb_x, bb_y, image_width, image_height, bound_pad=2)

                if class_index in [3, 4] and area < 1:
                    continue
                if class_index not in [3, 4] and area < 100:
                    continue

                coco_seg_poly = []
                poly_sorted = resort_corners(polygon)
                # image = draw_polygon_on_image(image, poly_shapely, "test_poly.jpg")

                for p in poly_sorted:
                    coco_seg_poly += list(p)

                # Create a dictionary for the COCO annotation
                coco_annotation_dict = {
                    "segmentation": [coco_seg_poly],
                    "area": area,
                    "iscrow": 0,
                    "image_id": i,
                    "bbox": coco_bb,
                    "category_id": class_index,
                    "id": instance_count,
                }
                coco_annotation_dict_list.append(coco_annotation_dict)
                instance_count += 1
                output_polygons.append([coco_seg_poly, class_index])

        save_dict['images'].append(img_dict)
        save_dict["annotations"] += coco_annotation_dict_list

        
        # Print the unique class indices in the segmentation mask to verify
        print(path)
        print(np.unique(segmentation_mask))

        filled_mask = fill_mask(segmentation_mask)

        clean_image = np.array(image)
        filled_mask_resized = cv2.resize(filled_mask, (clean_image.shape[1], clean_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{output_aux_dir}/{fn}_fg_mask.png", filled_mask_resized)

        clean_image = clean_image * np.array(filled_mask_resized[:, :, np.newaxis] / 255.).astype(bool)
        clean_image[filled_mask_resized == 0] = 255
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR)
        # clean_image = to_bw_image(clean_image)
        cv2.imwrite(f"{output_image_dir}/{new_fn}.png", clean_image)

        image_with_polygons = draw_polygon_on_image(np.zeros_like(clean_image), output_polygons, CLASS_TO_COLOR)
        cv2.imwrite(f"{output_aux_dir}/{fn}_polylines.png", image_with_polygons)

        output_mappings.append(f"{fn} {new_fn}")

    with open(fn_mapping_log, 'w') as f:
        for mapping in output_mappings:
            f.write(f"{mapping}\n")


    # Serialize save_dict to JSON
    json_path = f"{output_annot_dir}/test.json"
    with open(json_path, 'w') as f:
        json.dump(save_dict, f, default=convert_numpy_to_python)