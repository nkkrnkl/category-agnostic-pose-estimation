import os
import numpy as np
import cv2
import json
from PIL import Image
import argparse
from glob import glob

from util.plot_utils import plot_room_map, plot_semantic_rich_floorplan_tight
from util.plot_utils import plot_semantic_rich_floorplan_opencv, plot_floorplan_with_regions, plot_density_map
from util.plot_utils import S3D_LABEL, CC5K_LABEL, auto_crop_whitespace


def resize_and_pad(img, target_size, pad_value=(255,255,255), interp=Image.BICUBIC):
    """
    Resizes a NumPy image while preserving aspect ratio and then pads it to the target size.

    Args:
        img (numpy.ndarray): Input image as a NumPy array (H, W, C).
        target_size (tuple): Target size as (height, width).
        pad_value (int): Value to use for padding. Default is 0.
        interp (int): Interpolation method. Default is PIL.Image.BICUBIC.

    Returns:
        numpy.ndarray: Resized and padded image as a NumPy array.
    """
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize the image
    resized_img = np.array(Image.fromarray(img).resize((new_w, new_h), interp))

    # Calculate padding
    pad_h, pad_w = target_size[0] - new_h, target_size[1] - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad the image
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
    )

    return padded_img


def plot_polys(image, room_polys, room_ids, save_path):


    if image.shape[2] == 3:
        density_map = image
    else:
        density_map = np.repeat(image, 3, axis=2)
    pred_room_map = np.zeros(density_map.shape).astype(np.uint8)

    for poly, poly_id in zip(room_polys, room_ids):
        poly = poly.reshape(-1,2).astype(np.int32)
        pred_room_map = plot_room_map(poly, pred_room_map, poly_id)

    # Blend the overlay with the density map using alpha blending
    alpha = 0.6  # Adjust for desired transparency
    pred_room_map = cv2.addWeighted(density_map.astype(np.uint8), alpha, pred_room_map.astype(np.uint8), 1-alpha, 0)

    cv2.imwrite(save_path, pred_room_map)

    return pred_room_map

def plot_floor_map(image_size, room_polys, room_ids, save_path):
    # plot semantically-rich floorplan
    # image_size = image.shape[0]
    gt_sem_rich = []
    for j, poly in enumerate(room_polys):
        corners = poly.reshape(-1, 2).astype(np.int32)
        corners_flip_y = corners.copy()
        corners_flip_y[:,1] = image_size - 1 - corners_flip_y[:,1]
        corners = corners_flip_y
        gt_sem_rich.append([corners, room_ids[j]])

    plot_semantic_rich_floorplan_tight(gt_sem_rich, save_path, prec=-1, rec=-1, plot_text=True, is_bw=False, door_window_index=[10,9])

def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer', add_help=False)

    # dataset parameters
    parser.add_argument('--dataset_name', default='stru3d')
    parser.add_argument('--dataset_root', default='data/stru3d', type=str)
    parser.add_argument('--eval_set', default='test', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--crop_white_space', action='store_true')
    parser.add_argument('--one_color', action='store_true')

    parser.add_argument('--json_root', default='test', type=str)
    parser.add_argument('--save_dir', default='vis_from_json', type=str)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args_parser()
    semantics_label_mapping = None
    if args.dataset_name == 'stru3d':
        door_window_index = [16, 17]
        semantics_label_mapping = S3D_LABEL
    elif args.dataset_name == 'cubicasa':
        door_window_index = [10, 9]
        semantics_label_mapping = CC5K_LABEL
    elif args.dataset_name == 'waffle':
        door_window_index = [1, 2]
    else:
        door_window_index = [-1, -1]


    json_list = glob(os.path.join(args.json_root, "*.json"))
    save_dir = args.save_dir # 'vis_from_json'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # input_json= '/home/htp26/RoomFormerTest/data_preprocess/waffle/coco_annotations.json'
    image_root = os.path.join(args.dataset_root, args.eval_set) # '/home/htp26/RoomFormerTest/data/coco_cubicasa5k_nowalls_v4-1_refined/test'
    image_size = args.image_size
    for input_json in json_list:
        with open(input_json, 'r') as f:
            data = json.load(f) # ['annotations']
            if len(data) == 0:
                continue
            scene_id = data[0]['image_id']
            room_polys = [np.array(x['segmentation']) for x in data]
            room_ids = [x['category_id'] for x in data]
            sample_path = os.path.join(image_root, str(scene_id).zfill(5) + '.png')
            image = np.array(Image.open(sample_path))
            # Ensure the image has 3 channels (RGB)
            if len(image.shape) == 2:  # Grayscale image
                image = np.stack([image] * 3, axis=-1)  # Convert to RGB
            elif image.shape[-1] > 3:  # Drop alpha channel if present
                image = image[:, :, :3]
            image = resize_and_pad(image, (image_size, image_size), pad_value=(255,255,255))

        # poly_path = os.path.join(output_dir, '{}_pred_polys.png'.format(scene_id))
        # plot_polys(
        #     image, room_polys, room_ids, poly_path
        # )

        # pred_room_map = plot_density_map(image, image_size, 
        #                                     room_polys, room_ids, plot_text=False)
        # cv2.imwrite(os.path.join(save_dir, '{}_pred_room_map.png'.format(scene_id)), pred_room_map)

        floorplan_map = plot_semantic_rich_floorplan_opencv(zip(room_polys, room_ids), 
            None, door_window_index=door_window_index,
            semantics_label_mapping=semantics_label_mapping, 
            plot_text=False, one_color=args.one_color,
            img_h=image_size,
            img_w=image_size, scale=2)

        if args.crop_white_space:
            image, cropped_box = auto_crop_whitespace(image)
            _x,_y,_w,_h = cropped_box # [ele * args.image_scale for ele in cropped_box]
            floorplan_map = floorplan_map[_y:_y+_h, _x:_x+_w].copy()

        cv2.imwrite(os.path.join(save_dir, '{}_pred_floorplan.png'.format(scene_id)), floorplan_map)
        cv2.imwrite(os.path.join(save_dir, '{}_gt.png'.format(scene_id)), image)

        # plot_floor_map(
        #     512, room_polys, room_ids, map_path
        # )
