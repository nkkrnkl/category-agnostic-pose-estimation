import os
from glob import glob

from matplotlib import cm
from matplotlib.colors import to_rgb
import argparse

import json
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from itertools import combinations

import cv2
from PIL import Image, ImageDraw
from PIL import ImageFont

# Initialize the colormap
colormap = cm.get_cmap('tab20', 20)  # Use 'tab20' colormap with 20 distinct colors

# Function to get a color from the colormap based on an ID
def get_color(id):
    normalized_id = id % 20  # Normalize ID to fit within the range of the colormap
    return tuple(int(c * 255) for c in to_rgb(colormap(normalized_id)))

def parse_polygon(segmentation):
    # COCO-style segmentation: list of flat coordinate list
    if not segmentation:
        return None
    coords = np.array(segmentation[0]).reshape(-1, 2) # list(zip(segmentation[0][::2], segmentation[0][1::2]))
    if coords.shape[0] < 3: # ignore window, door
        return None
    return Polygon(coords)

def compute_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0

def remove_high_overlap_annotations(annotations, iou_threshold=0.9, image_size=(2048, 2048), output_path="overlap.png"):
    polygons = [(ann['id'], parse_polygon(ann['segmentation'])) for ann in annotations]
    to_remove = set()

    # Create a single white image to plot all overlaps
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    # Create a dictionary to store colors for each ID
    id_to_color = {}
    drawn_ids = set()

    def draw_polygons(poly1, poly2, id1, id2, iou, plot_iou=True):
        # Assign colors for each ID if not already assigned
        if id1 not in id_to_color:
            id_to_color[id1] = get_color(id1)
        if id2 not in id_to_color:
            id_to_color[id2] = get_color(id2)
        
        # Check and draw polygons only if they haven't been drawn yet
        if id1 not in drawn_ids:
            poly1_coords = [(x, y) for x, y in zip(*poly1.exterior.xy)]
            draw.polygon(poly1_coords, outline=id_to_color[id1], fill=id_to_color[id1])
            drawn_ids.add(id1)  # Mark poly1 as drawn

        if id2 not in drawn_ids:
            poly2_coords = [(x, y) for x, y in zip(*poly2.exterior.xy)]
            draw.polygon(poly2_coords, outline=id_to_color[id2], fill=id_to_color[id2])
            drawn_ids.add(id2)  # Mark poly2 as drawn

        # Draw the overlapping region in gray
        if plot_iou:
            overlap = poly1.intersection(poly2)
            if not overlap.is_empty :
                if isinstance(overlap, (Polygon, MultiPolygon)):
                    polygons = [overlap] if isinstance(overlap, Polygon) else overlap.geoms

                    for polygon in polygons:
                        overlap_coords = [(x, y) for x, y in zip(*polygon.exterior.xy)]
                        # Draw the black outline
                        draw.line(overlap_coords + [overlap_coords[0]], fill="black", width=4)

                        temp_img = Image.new("RGBA", img.size, (0, 0, 0, 0))  # Fully transparent image
                        temp_draw = ImageDraw.Draw(temp_img)
                        transparent_gray = (0, 0, 0, 64)  # Semi-transparent gray
                        temp_draw.polygon(overlap_coords, fill=transparent_gray)
                        img.paste(temp_img, (0, 0), temp_img)

                        # Calculate the centroid and draw IoU
                        centroid = polygon.centroid
                        centroid_coords = (centroid.x, centroid.y)
                        try:
                            font = ImageFont.truetype(os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf'), size=20)
                        except OSError:
                            font = ImageFont.load_default()
                        draw.text(centroid_coords, f"{iou:.2f}", fill="white", font=font)

                # elif isinstance(overlap, GeometryCollection):
                #     for geom in overlap.geoms:
                #         if isinstance(geom, Polygon):
                #             overlap_coords = [(x, y) for x, y in zip(*geom.exterior.xy)]
                #             temp_img = Image.new("RGBA", img.size, (0, 0, 0, 0))  # Fully transparent image
                #             temp_draw = ImageDraw.Draw(temp_img)
                #             transparent_gray = (0, 0, 0, 64)  # Semi-transparent gray
                #             temp_draw.polygon(overlap_coords, fill=transparent_gray)
                #             img.paste(temp_img, (0, 0), temp_img)

                #             # Calculate the centroid and draw IoU
                #             centroid = geom.centroid
                #             centroid_coords = (centroid.x, centroid.y)
                #             try:
                #                 font = ImageFont.truetype(os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf'), size=20)
                #             except OSError:
                #                 font = ImageFont.load_default()
                #             draw.text(centroid_coords, f"{iou:.2f}", fill="white", font=font)


    for (id1, poly1), (id2, poly2) in combinations(polygons, 2):
        if id1 in to_remove or id2 in to_remove:
            continue
        if poly1 and poly2:
            iou = compute_iou(poly1, poly2) 
            draw_polygons(poly1, poly2, id1, id2, iou, plot_iou=iou>=iou_threshold) # add iou
            if iou >= iou_threshold:
                print(iou)
                # Remove the one with smaller area
                area1 = poly1.area
                area2 = poly2.area
                to_remove.add(id1 if area1 < area2 else id2)

    # Save the single image with all overlaps
    img.save(output_path)
    filtered_annotations = [ann for ann in annotations if ann['id'] not in to_remove]
    return filtered_annotations, to_remove


def config():
    a = argparse.ArgumentParser(description='Generate coco format data for Structured3D')
    a.add_argument('--input_dir', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output_dir', default='coco_cubicasa5k', type=str, help='path to output folder')
    
    args = a.parse_args()
    return args

if __name__ == "__main__":
    args = config()
    root_input_dir = args.input_dir # "/share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v4-1_refined"
    root_output_dir = args.output_dir # "/share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v4-2_refined"

    input_dir = f"{root_input_dir}/annotations_json/"
    output_dir = f"{root_output_dir}/annotations_json/"

    for split in ['test', 'val', 'train']:
        split_input_path = os.path.join(input_dir, split)
        split_files = glob(os.path.join(split_input_path, "*.json"))
        split_output_path = os.path.join(output_dir, split)
        aux_path = os.path.join(root_output_dir, f"{split}_aux2")

        mapping_id_path = os.path.join(root_input_dir, f"annotations/{split}_image_id_mapping.json")
        with open(mapping_id_path, 'r') as f:
            mapping_id_dict = json.load(f)

        removed_ids_list = []
        os.makedirs(split_output_path, exist_ok=True)
        os.makedirs(aux_path, exist_ok=True)
        for path in split_files:
            fn = os.path.basename(path)
            cur_id = fn.split('.')[0].split('_')[0].lstrip('0')

            if cur_id == '03590':
                breakpoint()
            cur_image = Image.open(os.path.join(root_input_dir, f"{split}/{str(mapping_id_dict[cur_id]).zfill(5)}.png"))
            image_size = cur_image.size
            with open(path, "r") as f:
                data = json.load(f)
                annotations = data['annotations']

            filtered, removed_ids = remove_high_overlap_annotations(annotations, iou_threshold=0.35,
                                                                    image_size=image_size,
                                                                    output_path=f"{aux_path}/{fn.split('.')[0]}_overlap.png")
            if len(removed_ids) != 0:
                print(path)
                print(f"{len(removed_ids)} are removed: {removed_ids}")
                assert len(filtered) != len(annotations)
                removed_ids_list.append(f"{split}/{fn} {str(removed_ids)}")

            data['annotations'] = filtered
            with open(os.path.join(split_output_path, fn), "w") as f:
                json.dump(data, f, indent=2)
        
        with open(os.path.join(output_dir, f"{split}_removed_ids.txt"), "w") as f:
            f.write("\n".join(removed_ids_list))
        
    # path = "/home/htp26/RoomFormerTest/data/coco_cubicasa5k_nowalls_v4-1_refined/annotations_json/test/49730_08123.json"
    # fn = os.path.basename(path)
    # with open(path, "r") as f:
    #     data = json.load(f)
    #     annotations = data['annotations']

    # filtered, removed_ids = remove_high_overlap_annotations(annotations, 
    #                                                         iou_threshold=0.35,
    #                                                         output_path=f"{fn.split('.')[0]}_overlap.png",)
    # if len(removed_ids) != 0:
    #     print(fn)
    #     print(f"{len(removed_ids)} are removed: {removed_ids}")

        
