import gc
import sys
import os
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon
import shutil
import cv2
from multiprocessing import Pool

from util.graph_utils import tensors_to_graphs_batch, get_cycle_basis_and_semantic, get_cycle_basis
from util.data_utils import data_to_cuda, delete_graphs, get_given_layers_random_region, get_random_region_targets, \
    draw_given_layers_on_tensors_random_region, initialize_tensors, nms, random_keep, is_stop, draw_preds_on_tensors, edge_inside, point_inside, \
    remove_points, remove_edge, get_edges_amount
from datasets.dataset import MyDataset


mean = [0.920, 0.913, 0.891]
std = [0.214, 0.216, 0.228]

ID2CLASS = {0: 'unknown', 1: 'living_room', 2: 'kitchen', 3: 'bedroom',
            4: 'bathroom', 5: 'restroom', 6: 'balcony', 7: 'closet',
            8: 'corridor', 9: 'washing_room', 10: 'PS', 11: 'outside',
            # 12: 'wall'
            }

def plot_room_map(preds, room_map, room_id=0, im_size=256, plot_text=True):
    """Draw room polygons overlaid on the density map
    """
    centroid_x = int(np.mean(preds[:, 0]))
    centroid_y = int(np.mean(preds[:, 1]))

    # Get text size to create a background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    text = str(room_id)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    border_color = (252, 252, 0)

    for i, corner in enumerate(preds):
        if i == len(preds)-1:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[0][0]), round(preds[0][1])), border_color, 2)
        else:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[i+1][0]), round(preds[i+1][1])), border_color, 2)
        cv2.circle(room_map, (round(corner[0]), round(corner[1])), 2, (0, 0, 255), 2)
        # cv2.putText(room_map, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw white background box with transparency
        # overlay = room_map.copy()
        # cv2.addWeighted(overlay, 0.7, room_map, 0.3, 0, room_map)  # 70% opacity

        # Draw text
        if plot_text:
            cv2.rectangle(room_map, 
                        (centroid_x - text_width//2 - 2, centroid_y - text_height//2 - 2),
                        (centroid_x + text_width//2 + 2, centroid_y + text_height//2 + 2),
                        (255, 255, 255), # (0, 0, 0), 
                        -1)  # Filled rectangle
            cv2.putText(room_map, 
                        text, 
                        (centroid_x - text_width//2, centroid_y + text_height//2), 
                        font, 
                        font_scale, 
                        (0, 100, 0),
                        thickness)
        
    return room_map


def plot_density_map(sample, image_size, room_polys, pred_room_label_per_scene, plot_text=True):
    if not isinstance(sample, np.ndarray):
        density_map = np.transpose(sample.cpu().numpy(), [1, 2, 0])
        # # Convert to grayscale if not already
        # if density_map.shape[2] > 1:
        #     density_map = cv2.cvtColor(density_map, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    else:
        density_map = sample
    if density_map.shape[2] == 3:
        density_map = density_map * (image_size - 1)
    else:
        density_map = np.repeat(density_map, 3, axis=2) * (image_size - 1)
    pred_room_map = np.zeros([image_size, image_size, 3])

    for room_poly, room_id in zip(room_polys, pred_room_label_per_scene):
        pred_room_map = plot_room_map(np.array(room_poly), pred_room_map, room_id, im_size=image_size, plot_text=plot_text)
    
    alpha = .4  # Adjust for desired transparency
    pred_room_map = cv2.addWeighted(density_map.astype(np.uint8), alpha, pred_room_map.astype(np.uint8), 1-alpha, 0)
    return pred_room_map


def is_clockwise(points):
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0

def resort_corners(corners):
    # re-find the starting point and sort corners clockwisely
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2 
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners

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

def prepare_dict():
    save_dict = {"images":[],"annotations":[],"categories":[]}
    for key, value in ID2CLASS.items():
        if key == 0:
            continue
        type_dict = {"supercategory": "room", "id": key, "name": value}
        save_dict["categories"].append(type_dict)
    return save_dict


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory',)
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the dataset directory',)
    # Add more arguments as needed
    return parser


def visualize_room_polygons(room_polygons, room_classes, image_size=512, save_path='cubicasa_debug.png'):
    """
    Visualize the extracted room polygons.
    
    Args:
        room_polygons: Dictionary of room polygons as returned by extract_room_polygons
        figsize: Figure size for the plot
    """
    # Set figure size to exactly 256x256 pixels
    dpi = 100  # Standard screen DPI
    figsize = (image_size/dpi, image_size/dpi)  # Convert pixels to inches
    class_names = [v for k, v in ID2CLASS.items()]

    # Get unique classes from the mask
    unique_classes = list(ID2CLASS.keys())

    # Create a discrete colormap
    cmap = plt.cm.get_cmap('gist_ncar', 256) # nipy_spectral
    norm = np.linspace(0, 1, 13) # int(max(unique_classes))+1

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot each room polygon and fill with color
    for polygon, room_cls in zip(room_polygons, room_classes):
        polygon_array = np.array(polygon).copy()
        polygon_array[:, 1] = image_size - 1 - polygon_array[:, 1]  # flip
        # Fill the polygon with its class color
        color = cmap(norm[int(room_cls)])
        ax.fill(polygon_array[:, 0], polygon_array[:, 1], color=color, alpha=0.4, zorder=1)
        # Draw the polygon border
        ax.plot(polygon_array[:, 0], polygon_array[:, 1], 'k-', linewidth=2, zorder=2)

        # Add room ID label at the centroid
        centroid_x = np.mean(polygon_array[:, 0])
        centroid_y = np.mean(polygon_array[:, 1])
        ax.text(centroid_x, centroid_y, str(room_cls), fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7), zorder=3)

    # Create custom legend elements
    legend_elements = []
    for i, cls in enumerate(sorted(unique_classes)):
        color = cmap(norm[int(cls)])
        cls_name = f"{int(cls)}_{class_names[int(cls)]}"
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f"{cls_name}", alpha=0.6))
    ax.legend(handles=legend_elements, loc='best', title="Classes", fontsize=10, markerscale=1, title_fontsize=12, framealpha=0.5)

    plt.tight_layout(pad=0)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_floorplan(image_set, split, source_data_path, save_dir, save_aux_dir, vis_fp=False):
    (img, target) = image_set
    img = img * torch.tensor(std)[:,None,None] + torch.tensor(mean)[:,None,None] # unnormalize
    graph = tensors_to_graphs_batch([target['graph']])
    del target['graph']

    tgt_this_preds = []
    tgt_this_edges = []
    for _ in range(len(target['points'])):
        tgt_p_d = {}
        tgt_p_d['scores'] = torch.tensor(1.0000, device='cpu')
        tgt_p_d['points'] = target['unnormalized_points'][_]
        tgt_p_d['edges'] = target['edges'][_]
        tgt_p_d['size'] = target['size']
        if 'semantic_left_up' in target:
            tgt_p_d['semantic_left_up'] = target['semantic_left_up'][_]
            tgt_p_d['semantic_right_up'] = target['semantic_right_up'][_]
            tgt_p_d['semantic_right_down'] = target['semantic_right_down'][_]
            tgt_p_d['semantic_left_down'] = target['semantic_left_down'][_]
        tgt_this_preds.append(tgt_p_d)
        for __ in range(4):
            adj = graph[0][tuple(tgt_p_d['points'].tolist())][__]
            if adj != (-1, -1):
                tgt_p_d1 = tgt_p_d
                tgt_p_d2 = {}
                indx = 99999
                for ___, up in enumerate(target['unnormalized_points'].tolist()):
                    if abs(up[0] - adj[0]) + abs(up[1] - adj[1]) <= 2:
                        indx = ___
                        break
                # assert indx != 99999
                if indx == 99999:  # No match found
                    # Log a warning or skip this iteration
                    print(f"Warning: No match found for adj {adj}")
                    continue  # Skip to the next iteration
                # tgt_p_d2['scores'] = torch.tensor(1.0000, device='cuda:0')
                tgt_p_d2['points'] = target['unnormalized_points'][indx]
                tgt_p_d2['edges'] = target['edges'][indx]
                tgt_p_d2['size'] = target['size']
                if 'semantic_left_up' in target:
                    tgt_p_d2['semantic_left_up'] = target['semantic_left_up'][indx]
                    tgt_p_d2['semantic_right_up'] = target['semantic_right_up'][indx]
                    tgt_p_d2['semantic_right_down'] = target['semantic_right_down'][indx]
                    tgt_p_d2['semantic_left_down'] = target['semantic_left_down'][indx]
                tgt_e_l = (tgt_p_d1, tgt_p_d2)
                if not edge_inside((tgt_p_d2, tgt_p_d1), tgt_this_edges):
                    tgt_this_edges.append(tgt_e_l)
    tgt = [(tgt_this_preds, [], tgt_this_edges)]
    target_d_rev, target_simple_cycles, target_results = \
        get_cycle_basis_and_semantic((2, 999999, tgt))
    
    # convert to coco format
    polys_list = []
    polys_semantic_list = []
    output_json = []

    image_width, image_height = target['size'][0].item(), target['size'][1].item()
    filename = target['file_name'].split('.')[0]
    img_id = int(target['image_id'])

    img_dict = {}
    img_dict["file_name"] = str(img_id).zfill(6) + '.png'
    img_dict["id"] = img_id
    img_dict["width"] = image_width
    img_dict["height"] = image_height
    save_dict = prepare_dict()

    os.makedirs(os.path.join(save_dir, split), exist_ok=True)
    os.makedirs(f"{save_dir}/{split}_jsons/", exist_ok=True)
    json_path = f"{save_dir}/{split}_jsons/{str(img_id).zfill(6)}.json"


    for instance_id, (poly, poly_cls) in enumerate(zip(target_simple_cycles, target_results)):
        t = [(int(pt[0]), int(pt[1])) for pt in poly]
        class_id = int(poly_cls)

        polys_list.append(t)
        polys_semantic_list.append(class_id)

        poly_shapely = Polygon(t)
        area = poly_shapely.area
        coco_seg_poly = []
        polygon = np.array(t)
        poly_sorted = resort_corners(polygon)

        for p in poly_sorted:
            coco_seg_poly += list(p)
        
        if area < 100:
            continue

        if class_id not in ID2CLASS:
            print(f"Warning: Class ID {class_id} not found in ID2CLASS mapping. Skipping instance.")
            continue

        # Slightly wider bounding box
        rectangle_shapely = poly_shapely.envelope
        bb_x, bb_y = rectangle_shapely.exterior.xy
        coco_bb = create_coco_bounding_box(bb_x, bb_y, image_width, image_height, bound_pad=2)

        output_json.append({'image_id': img_id, 
                        'segmentation': [coco_seg_poly],
                        'category_id': class_id,
                        'id': instance_id,
                        'area': area,
                        "bbox": coco_bb,
                        "iscrowd": 0,
                        })

    if vis_fp: 
        visualize_room_polygons(polys_list, polys_semantic_list, image_size=image_width, 
                                save_path=os.path.join(save_aux_dir, str(img_id).zfill(6) + '.png'))
        room_map = plot_density_map(img, image_width, 
                                             polys_list, polys_semantic_list, plot_text=False, )
        cv2.imwrite(os.path.join(save_aux_dir, str(img_id).zfill(6) + '_density_map.png'), room_map)

    print(f"Processed image {img_id} with {len(output_json)} instances.")
    # print(f"Class: {target_results}")
    # min_class_id = min(target_results)
    # max_class_id = max(target_results)
    # if max_class_id == 12:
    #     breakpoint()
    # print(f"Min class ID: {min_class_id}, Max class ID: {max_class_id}")
    save_dict['images'].append(img_dict)
    save_dict["annotations"] += output_json
    with open(json_path, 'w') as json_file:
        # Convert all numpy and torch types to native Python types for JSON serialization
        def convert(o):
            if isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            if isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            if isinstance(o, torch.Tensor):
                return o.item() if o.numel() == 1 else o.tolist()
            return str(o)
        json.dump(save_dict, json_file, default=convert)

    # rename image file
    shutil.copy(os.path.join(source_data_path, split, filename + '.png'), 
        os.path.join(save_dir, split,  str(img_id).zfill(6) + '.png'))

    # Write mapping from source file name to target file name (safe for parallel)
    mapping_line = f"{filename} {str(img_id).zfill(6)}\n"
    # Each process writes to its own temp file
    pid = os.getpid()
    os.makedirs(os.path.join(save_dir, f'{split}_logs'), exist_ok=True)
    mapping_file = os.path.join(save_dir, f'{split}_logs', f"{split}_file_mapping_{pid}.txt")
    with open(mapping_file, "a") as f:
        f.write(mapping_line)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    torch.set_printoptions(threshold=np.inf, linewidth=999999)
    np.set_printoptions(threshold=np.inf, linewidth=999999)
    gc.collect()
    torch.cuda.empty_cache()

    def wrapper(scene_id):
        try:
            image_set = dataset[scene_id]
        except:
            print(f"Error processing scene {scene_id}. Skipping...")
            return
        process_floorplan(image_set, split, args.dataset_path, args.output_dir, save_aux_dir, vis_fp=scene_id < 100)

    def worker_init(dataset_obj):
        # Store dataset as global to avoid pickling issues
        global dataset
        dataset = dataset_obj

    splits = ['train', 'val', 'test']
    for split in splits:
        dataset = MyDataset(args.dataset_path + f'/{split}', args.dataset_path + '/annot_json' + f'/instances_{split}.json', 
                            extract_roi=False)

        save_aux_dir = os.path.join(args.output_dir, f"{split}_aux")
        os.makedirs(save_aux_dir, exist_ok=True)

        # for i, image_set in enumerate(tqdm(dataset)):
        #     save_aux_dir = os.path.join(args.output_dir, f"{split}_aux")
        #     os.makedirs(save_aux_dir, exist_ok=True)
        #     process_floorplan(image_set, split, args.dataset_path, args.output_dir, save_aux_dir, vis_fp=i < 100)

        num_processes = 16
        with Pool(num_processes, initializer=worker_init, initargs=(dataset,)) as p:
            indices = range(len(dataset))
            list(tqdm(p.imap(wrapper, indices), total=len(dataset)))