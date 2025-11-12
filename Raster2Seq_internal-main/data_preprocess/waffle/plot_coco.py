import json
import numpy as np
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import random
import cv2


def visualize_coco_polygons(coco_json_path, image_path=None, save_path='visualization.png', 
                                   show_bbox=True, show_labels=True):
    """
    Visualize COCO polygon annotations in headless environment.
    """
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get image dimensions
    image_info = coco_data['images'][0]
    img_width = image_info['width']
    img_height = image_info['height']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Load background image if provided, otherwise create blank canvas
    if image_path:
        try:
            img = Image.open(image_path)
            img = np.array(img)
            ax.imshow(img)
        except:
            print(f"Could not load image {image_path}, using blank canvas")
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)
            ax.set_aspect('equal')
    else:
        blank_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        ax.imshow(blank_img)
    
    # Generate random colors
    colors = generate_colors(len(coco_data['annotations']))
    
    # Plot each annotation
    for idx, annotation in enumerate(coco_data['annotations']):
        color = colors[idx]
        
        # Plot segmentation polygon(s)
        if 'segmentation' in annotation:
            for seg in annotation['segmentation']:
                points = np.array(seg).reshape(-1, 2)
                
                # Create polygon patch
                poly = Polygon(points, 
                             facecolor=color + (0.3,),
                             edgecolor=color,
                             linewidth=2)
                ax.add_patch(poly)
                
                # Plot vertices
                ax.plot(points[:, 0], points[:, 1], 'o', 
                       color=color, markersize=4)
        
        # Plot bounding box if requested
        if show_bbox and 'bbox' in annotation:
            bbox = annotation['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                   linewidth=1, edgecolor=color,
                                   facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        # Add label if requested
        if show_labels:
            if 'bbox' in annotation:
                label_x = annotation['bbox'][0]
                label_y = annotation['bbox'][1]
            else:
                seg = annotation['segmentation'][0]
                points = np.array(seg).reshape(-1, 2)
                label_x = np.mean(points[:, 0])
                label_y = np.min(points[:, 1])
            
            ax.text(label_x, label_y - 5, f"ID: {annotation['id']}", 
                   fontsize=8, color='black', 
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', alpha=0.7))
    
    ax.set_title(f"COCO Annotations - {len(coco_data['annotations'])} objects")
    ax.axis('off')
    
    # Save without displaying
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Important: close the figure to free memory
    print(f"Visualization saved to {save_path}")


def visualize_single_annotation(coco_json_path, annotation_id, save_path=None):
    """
    Visualize a single annotation in detail.
    """
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Find the specific annotation
    annotation = None
    for ann in coco_data['annotations']:
        if ann['id'] == annotation_id:
            annotation = ann
            break
    
    if not annotation:
        print(f"Annotation with ID {annotation_id} not found")
        return
    
    # Get image dimensions
    image_info = coco_data['images'][0]
    img_width = image_info['width']
    img_height = image_info['height']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Set canvas
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('lightgray')
    
    # Plot segmentation
    if 'segmentation' in annotation:
        for seg in annotation['segmentation']:
            points = np.array(seg).reshape(-1, 2)
            
            # Plot filled polygon
            poly = Polygon(points, 
                         facecolor='blue',
                         alpha=0.3,
                         edgecolor='blue',
                         linewidth=3)
            ax.add_patch(poly)
            
            # Plot vertices with indices
            for i, (x, y) in enumerate(points):
                ax.plot(x, y, 'ro', markersize=8)
                ax.text(x + 5, y + 5, str(i), fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='white', alpha=0.8))
    
    # Plot bounding box
    if 'bbox' in annotation:
        bbox = annotation['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                               linewidth=2, edgecolor='red',
                               facecolor='none', linestyle='--')
        ax.add_patch(rect)
    
    # Add annotation info
    info_text = f"Annotation ID: {annotation['id']}\n"
    info_text += f"Area: {annotation.get('area', 'N/A'):.2f}\n"
    info_text += f"Category: {annotation.get('category_id', 'N/A')}\n"
    info_text += f"Vertices: {len(points)}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", 
                   facecolor='white', alpha=0.9))
    
    ax.set_title(f"COCO Annotation ID: {annotation_id}")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_colors(n):
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        colors.append(plt.cm.hsv(hue)[:3])  # RGB only
    return colors


def create_overlay_visualization(coco_json_path, output_path='coco_overlay.png'):
    """
    Create an overlay visualization using OpenCV for better performance with many polygons.
    """
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get image dimensions
    image_info = coco_data['images'][0]
    img_width = image_info['width']
    img_height = image_info['height']
    
    # Create blank canvas
    canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    overlay = canvas.copy()
    
    # Generate random colors
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
              for _ in range(len(coco_data['annotations']))]
    
    # Draw each annotation
    for idx, annotation in enumerate(coco_data['annotations']):
        color = colors[idx]
        
        # Draw segmentation
        if 'segmentation' in annotation:
            for seg in annotation['segmentation']:
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                
                # Draw filled polygon on overlay
                cv2.fillPoly(overlay, [points], color)
                
                # Draw polygon outline
                cv2.polylines(canvas, [points], True, color, 2)
                
                # Draw vertices
                for point in points:
                    cv2.circle(canvas, tuple(point), 4, color, -1)
        
        # Draw bounding box
        if 'bbox' in annotation:
            bbox = annotation['bbox']
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
            
            # Add label
            label = f"ID: {annotation['id']}"
            cv2.putText(canvas, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Blend overlay with canvas
    result = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"Overlay visualization saved to {output_path}")
    
    # Display if desired
    cv2.imshow('COCO Annotations', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_statistics(coco_json_path):
    """
    Plot statistics about the COCO annotations.
    """
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Gather statistics
    areas = []
    vertex_counts = []
    
    for annotation in coco_data['annotations']:
        areas.append(annotation.get('area', 0))
        if 'segmentation' in annotation:
            for seg in annotation['segmentation']:
                vertex_counts.append(len(seg) // 2)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot area distribution
    ax1.hist(areas, bins=30, edgecolor='black')
    ax1.set_title('Distribution of Polygon Areas')
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Count')
    
    # Plot vertex count distribution
    ax2.hist(vertex_counts, bins=20, edgecolor='black')
    ax2.set_title('Distribution of Vertex Counts')
    ax2.set_xlabel('Number of Vertices')
    ax2.set_ylabel('Count')
    
    # Plot area vs vertex count
    for annotation in coco_data['annotations']:
        if 'segmentation' in annotation:
            area = annotation.get('area', 0)
            for seg in annotation['segmentation']:
                vertices = len(seg) // 2
                ax3.scatter(vertices, area, alpha=0.6)
    
    ax3.set_title('Area vs Vertex Count')
    ax3.set_xlabel('Number of Vertices')
    ax3.set_ylabel('Area')
    
    # Summary statistics
    summary_text = f"Total Annotations: {len(coco_data['annotations'])}\n"
    summary_text += f"Average Area: {np.mean(areas):.2f}\n"
    summary_text += f"Average Vertices: {np.mean(vertex_counts):.2f}\n"
    summary_text += f"Min/Max Area: {min(areas):.2f} / {max(areas):.2f}\n"
    summary_text += f"Min/Max Vertices: {min(vertex_counts)} / {max(vertex_counts)}"
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
            fontsize=14, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightgray', alpha=0.8))
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Basic visualization
    coco_json = "coco_annotations.json"
    
    # Visualize all polygons
    visualize_coco_polygons(coco_json, save_path="all_polygons.png")
    
    # Visualize specific annotation
    visualize_single_annotation(coco_json, annotation_id=1, save_path="annotation_1.png")
    
    # Create overlay visualization (better for many polygons)
    create_overlay_visualization(coco_json)
    
    # Plot statistics
    plot_statistics(coco_json)