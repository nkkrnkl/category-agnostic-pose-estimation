import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib import patheffects


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

def get_room_color_segments(num_rooms, colormap='jet'):
    """
    Divide the colormap into segments for each room.
    
    Args:
        num_rooms (int): Number of rooms
        colormap (str): Matplotlib colormap name
        
    Returns:
        list: List of color segments, each containing start and end values
    """
    if num_rooms == 1:
        return [(0.0, 1.0)]
    
    segments = []
    step = 1.0 / num_rooms
    for i in range(num_rooms):
        start = i * step
        end = (i + 1) * step
        segments.append((start, end))
    
    return segments

def get_corner_colors(num_corners, color_segment, colormap='jet'):
    """
    Get colors for corners within a room's color segment.
    
    Args:
        num_corners (int): Number of corners in the room
        color_segment (tuple): (start, end) values for the color segment
        colormap (str): Matplotlib colormap name
        
    Returns:
        list: List of RGBA colors for each corner
    """
    cmap = cm.get_cmap(colormap)
    start, end = color_segment
    
    if num_corners == 1:
        color_values = [start + (end - start) * 0.5]
    else:
        color_values = [start + (end - start) * i / (num_corners - 1) for i in range(num_corners)]
    
    colors = [cmap(val) for val in color_values]
    return colors

# def plot_polygon_sequence_with_corners(image, room_polys, room_ids, save_dir, scene_id, 
#                                      corner_size=8, colormap='jet'):
#     """
#     Plots each polygon sequentially with colored corners to illustrate the generation process.

#     Args:
#         image (numpy.ndarray): The base image.
#         room_polys (list): List of room polygons.
#         room_ids (list): List of room IDs corresponding to the polygons.
#         save_dir (str): Directory to save the sequential plots.
#         scene_id (str): Scene ID for naming the output files.
#         corner_size (int): Size of corner markers.
#         colormap (str): Matplotlib colormap name.
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Get color segments for rooms
#     num_rooms = len(room_polys)
#     color_segments = get_room_color_segments(num_rooms, colormap)

#     # Initialize the figure
#     pred_img_paths = []
#     for step, (poly, room_id) in enumerate(zip(room_polys, room_ids), start=1):
#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.imshow(image)
#         ax.axis('off')
#         ax.set_xlim(0, image.shape[1])
#         ax.set_ylim(image.shape[0], 0)
        
#         # Draw the polygon outline
#         polygon = Polygon(poly, closed=True, edgecolor='black', facecolor='none', linewidth=3)
#         ax.add_patch(polygon)
        
#         # Get colors for corners in this room
#         room_color_segment = color_segments[step - 1]
#         corner_colors = get_corner_colors(len(poly), room_color_segment, colormap)
        
#         # Draw corners with colors indicating order
#         for corner_idx, (corner, color) in enumerate(zip(poly, corner_colors)):
#             x, y = corner
#             circle = patches.Circle((x, y), corner_size, color=color, zorder=10)
#             ax.add_patch(circle)
            
#             # Add corner number as text
#             ax.text(x, y, str(corner_idx + 1), ha='center', va='center', 
#                    fontsize=16, fontweight='bold', color='white', zorder=11)
        
#         output_path = os.path.join(save_dir, f"{scene_id}_step_{step}_with_corners.png")
#         plt.savefig(output_path, bbox_inches='tight', dpi=150)
#         plt.close(fig)
#         pred_img_paths.append(output_path)

#     # Plot the original image and predictions in a 2x3 grid
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes[0, 0].imshow(image)
#     axes[0, 0].set_title("Input", fontsize=12, fontweight='bold')
#     axes[0, 0].axis('off')
    
#     for idx, img_path in enumerate(pred_img_paths):
#         row = (idx + 1) // 3
#         col = (idx + 1) % 3
#         pred_img = mpimg.imread(img_path)
#         axes[row, col].imshow(pred_img)
#         axes[row, col].set_title(f"Room {idx+1} (Index: {room_ids[idx]})", 
#                                 fontsize=12, fontweight='bold')
#         axes[row, col].axis('off')
    
#     # Hide any unused subplot
#     if len(pred_img_paths) < 5:
#         axes[1, 2].axis('off')

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"{scene_id}_grid_2x3_with_corners.png"), 
#                 bbox_inches='tight', dpi=150)
#     plt.close(fig)

def plot_polygon_sequence_with_corners(image, room_polys, room_ids, save_dir, scene_id, 
                                     corner_size=10, colormap='jet'):
    """
    Plots each polygon sequentially with colored corners to illustrate the generation process.

    Args:
        image (numpy.ndarray): The base image.
        room_polys (list): List of room polygons.
        room_ids (list): List of room IDs corresponding to the polygons.
        save_dir (str): Directory to save the sequential plots.
        scene_id (str): Scene ID for naming the output files.
        corner_size (int): Size of corner markers.
        colormap (str): Matplotlib colormap name.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get color segments for rooms
    num_rooms = len(room_polys)
    color_segments = get_room_color_segments(num_rooms, colormap)

    # Initialize the figure
    pred_img_paths = []
    for step, (poly, room_id) in enumerate(zip(room_polys, room_ids), start=1):
        # Create figure with space for colorbar
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.imshow(image, alpha=0.3) # make image transparent
        ax.axis('off')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        
        # Draw the polygon outline
        polygon = Polygon(poly, closed=True, edgecolor='black', facecolor='none', linewidth=10)
        ax.add_patch(polygon)
        
        # Get colors for corners in this room
        room_color_segment = color_segments[step - 1]
        corner_colors = get_corner_colors(len(poly), room_color_segment, colormap)
        
        # Draw corners with colors indicating order
        for corner_idx, (corner, color) in enumerate(zip(poly, corner_colors)):
            x, y = corner
            circle = patches.Circle((x, y), corner_size, color=color, zorder=10)
            ax.add_patch(circle)
            
            # Add corner number as text
            ax.text(x, y, str(corner_idx + 1), ha='center', va='center', 
                   fontsize=4*corner_size, fontweight='bold', color='white', zorder=11,
                   path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
        
        output_path = os.path.join(save_dir, f"{scene_id}_step_{step}_with_corners.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        pred_img_paths.append(output_path)
    
    # Calculate dynamic figure size based on grid dimensions
    num_rows, num_cols = 1, num_rooms + 1
    
    # Dynamic figsize calculation
    base_width_per_col = 4   # Base width per column (adjust as needed)
    base_height_per_row = 8  # Base height per row (adjust as needed)
    min_width = 12           # Minimum figure width
    min_height = 6           # Minimum figure height
    
    # Calculate figure dimensions
    fig_width = max(min_width, num_cols * base_width_per_col)
    fig_height = max(min_height, num_rows * base_height_per_row)
    
    # Plot the original image and predictions with dynamic sizing
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(num_rows, num_cols, hspace=0.1, wspace=0.1)
    
    # First subplot - Input image
    font_size = 24
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.set_title("Rasterized Image", fontsize=font_size, fontweight='bold')
    ax0.axis('off')
    
    # Prediction subplots
    for idx, img_path in enumerate(pred_img_paths):
        row = (idx + 1) // num_rows if num_rows != 1 else num_rows - 1
        col = (idx + 1) % num_cols
        ax = fig.add_subplot(gs[row, col])
        pred_img = mpimg.imread(img_path)
        ax.imshow(pred_img)
        ax.set_title(f"Room {idx+1}",
                    fontsize=font_size, fontweight='bold')
        ax.axis('off')
    
    # # Hide any unused subplot
    # if len(pred_img_paths) < 5:
    #     fig.add_subplot(gs[1, 2]).axis('off')
    
    # # Add vertical colorbar on the right side of the entire grid
    # cmap = cm.get_cmap(colormap)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_rooms))
    # sm.set_array([])
    
    # # Create colorbar on the right side
    # cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    # cbar.set_label('Room Order', rotation=270, labelpad=20, fontsize=18)

    # # Set integer-only ticks
    # cbar.set_ticks(range(1, num_rooms + 1))
    # cbar.ax.tick_params(labelsize=font_size*0.8)
    # cbar.ax.invert_yaxis() # reverse colorbar: Room 1 at top, last room at bottom

    plt.savefig(os.path.join(save_dir, f"{scene_id}_grid_2x3_with_corners.png"), 
                bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_all_rooms_with_corners(image, room_polys, room_ids, save_dir, scene_id, 
                               corner_size=8, colormap='jet'):
    """
    Plot all rooms together with colored corners showing the order.
    
    Args:
        image (numpy.ndarray): The base image.
        room_polys (list): List of room polygons.
        room_ids (list): List of room IDs corresponding to the polygons.
        save_dir (str): Directory to save the plots.
        scene_id (str): Scene ID for naming the output files.
        corner_size (int): Size of corner markers.
        colormap (str): Matplotlib colormap name.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get color segments for rooms
    num_rooms = len(room_polys)
    color_segments = get_room_color_segments(num_rooms, colormap)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    ax.axis('off')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title(f"All Rooms with Corner Order Visualization", fontsize=14, fontweight='bold')
    
    # Draw all rooms
    for room_idx, (poly, room_id) in enumerate(zip(room_polys, room_ids)):
        # Draw the polygon outline
        polygon = Polygon(poly, closed=True, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(polygon)
        
        # Get colors for corners in this room
        room_color_segment = color_segments[room_idx]
        corner_colors = get_corner_colors(len(poly), room_color_segment, colormap)
        
        # Draw corners with colors indicating order
        for corner_idx, (corner, color) in enumerate(zip(poly, corner_colors)):
            x, y = corner
            circle = patches.Circle((x, y), corner_size, color=color, zorder=10)
            ax.add_patch(circle)
            
            # Add corner number as text
            ax.text(x, y, f"R{room_idx+1}C{corner_idx+1}", ha='center', va='center', 
                   fontsize=6, fontweight='bold', color='white', zorder=11)
    
    # Add colorbar to show the mapping
    cmap = cm.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=num_rooms))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Room Order', rotation=270, labelpad=20)
    
    output_path = os.path.join(save_dir, f"{scene_id}_all_rooms_with_corners.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# Original function kept for compatibility
def plot_polygon_sequence(image, room_polys, room_ids, save_dir, scene_id):
    """
    Original function - plots each polygon sequentially without corner visualization.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the figure
    pred_img_paths = []
    for step, (poly, room_id) in enumerate(zip(room_polys, room_ids), start=1):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        polygon = Polygon(poly, closed=True, edgecolor='red', facecolor='none', linewidth=4)
        ax.add_patch(polygon)
        output_path = os.path.join(save_dir, f"{scene_id}_step_{step}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        pred_img_paths.append(output_path)

    # Plot the original image and predictions in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input")
    axes[0, 0].axis('off')
    for idx, img_path in enumerate(pred_img_paths):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        pred_img = mpimg.imread(img_path)
        axes[row, col].imshow(pred_img)
        axes[row, col].set_title(f"RoomID {idx+1}")
        axes[row, col].axis('off')
    # Hide any unused subplot (last cell if only 6 images)
    if len(pred_img_paths) < 5:
        axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{scene_id}_grid_2x3.png"))
    plt.close(fig)

# Main processing logic
data_root = "/home/htp26/RoomFormerTest/s3d_test_preds/s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2/jsons/"
image_root = "data/coco_s3d_bw/test/"
image_size = 256
save_dir = "sequential_plots_2"
from glob import glob
json_list = glob(os.path.join(data_root, "*.json"))
json_list = [f"{data_root}/03383_pred.json"]

for input_json in json_list:
    # input_json = "03383_pred.json"
    print(input_json)

    with open(input_json, 'r') as f:
        data = json.load(f)
        if len(data) == 0:
            raise ValueError("No data found in the JSON file.")
        scene_id = data[0]['image_id']
        room_polys = [np.array(x['segmentation']).reshape(-1, 2) for x in data]
        room_ids = [x['category_id'] for x in data]
        sample_path = os.path.join(image_root, str(scene_id).zfill(5) + '.png')
        image = np.array(Image.open(sample_path))

        # Ensure the image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
        elif image.shape[-1] > 3:  # Drop alpha channel if present
            image = image[:, :, :3]
        image = resize_and_pad(image, (image_size, image_size), pad_value=(255, 255, 255))

        # Plot polygons sequentially with corner visualization
        print("Generating sequential plots with corner visualization...")
        plot_polygon_sequence_with_corners(image, room_polys, room_ids, save_dir, scene_id)
        
        # # Plot all rooms together with corner visualization
        # print("Generating combined plot with all rooms and corners...")
        # plot_all_rooms_with_corners(image, room_polys, room_ids, save_dir, scene_id)
        
        # # Original visualization (kept for compatibility)
        # print("Generating original plots...")
        # plot_polygon_sequence(image, room_polys, room_ids, save_dir, scene_id)
        
        print(f"All visualizations saved to: {save_dir}")