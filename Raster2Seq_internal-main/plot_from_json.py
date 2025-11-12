import os
import numpy as np
import cv2
import json
from PIL import Image

from util.plot_utils import plot_room_map, plot_semantic_rich_floorplan_tight


def resize_and_pad(img, target_size, pad_value=0, interp=Image.BICUBIC):
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

def plot_floor_map(image, room_polys, room_ids, save_path):
    # plot semantically-rich floorplan
    image_size = image.shape[0]
    gt_sem_rich = []
    for j, poly in enumerate(room_polys):
        corners = poly.reshape(-1, 2).astype(np.int32)
        corners_flip_y = corners.copy()
        corners_flip_y[:,1] = image_size - 1 - corners_flip_y[:,1]
        corners = corners_flip_y
        gt_sem_rich.append([corners, room_ids[j]])

    plot_semantic_rich_floorplan_tight(gt_sem_rich, save_path, prec=-1, rec=-1, plot_text=True, is_bw=False, door_window_index=[10,9])


if __name__ == "__main__":
    input_json='akori_church_plan_annotations.json'
    output_dir = 'vis_from_json'
    image_root='/home/htp26/RoomFormerTest/data/coco_cubicasa5k_nowalls_v4-1_refined/test'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_json, 'r') as f:
        data = json.load(f)
        scene_id = data[0]['image_id']
        room_polys = [np.array(x['segmentation']) for x in data]
        room_ids = [x['category_id'] for x in data]
        sample_path = os.path.join(image_root, str(scene_id).zfill(5) + '.png')
        image = np.array(Image.open(sample_path))
        if image.shape[-1] > 3: # drop alpha channel
            image = image[:, :, :3]
        image = resize_and_pad(image, (256, 256), pad_value=(255,255,255))

    poly_path = os.path.join(output_dir, '{}_pred_polys.png'.format(scene_id))
    plot_polys(
        image, room_polys, room_ids, poly_path

    )

    map_path = os.path.join(output_dir, '{}_floor_map.png'.format(scene_id))
    plot_floor_map(
        image, room_polys, room_ids, map_path
    )