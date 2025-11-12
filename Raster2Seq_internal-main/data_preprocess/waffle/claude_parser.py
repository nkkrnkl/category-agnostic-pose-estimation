import xml.etree.ElementTree as ET
import json
import re
import numpy as np
from pathlib import Path


def parse_svg_path_to_points(path_data):
    """Parse SVG path data into a list of coordinate points."""
    points = []
    
    # Remove extra whitespace and split the path data
    path_data = re.sub(r'\s+', ' ', path_data.strip())
    
    # Initialize current position
    current_x, current_y = 0, 0
    
    # Split path into commands and parameters
    commands = re.findall(r'[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*', path_data)
    
    for cmd in commands:
        command = cmd[0]
        params = re.findall(r'-?\d*\.?\d+', cmd[1:])
        params = [float(p) for p in params]
        
        if command == 'M':  # Move to (absolute)
            if len(params) >= 2:
                current_x, current_y = params[0], params[1]
                points.append([current_x, current_y])
                # Handle multiple coordinate pairs after M
                for i in range(2, len(params), 2):
                    if i + 1 < len(params):
                        current_x, current_y = params[i], params[i + 1]
                        points.append([current_x, current_y])
        
        elif command == 'm':  # Move to (relative)
            if len(params) >= 2:
                current_x += params[0]
                current_y += params[1]
                points.append([current_x, current_y])
                # Handle multiple coordinate pairs after m
                for i in range(2, len(params), 2):
                    if i + 1 < len(params):
                        current_x += params[i]
                        current_y += params[i + 1]
                        points.append([current_x, current_y])
        
        elif command == 'L':  # Line to (absolute)
            for i in range(0, len(params), 2):
                if i + 1 < len(params):
                    current_x, current_y = params[i], params[i + 1]
                    points.append([current_x, current_y])
        
        elif command == 'l':  # Line to (relative)
            for i in range(0, len(params), 2):
                if i + 1 < len(params):
                    current_x += params[i]
                    current_y += params[i + 1]
                    points.append([current_x, current_y])
        
        elif command == 'H':  # Horizontal line to (absolute)
            for param in params:
                current_x = param
                points.append([current_x, current_y])
        
        elif command == 'h':  # Horizontal line to (relative)
            for param in params:
                current_x += param
                points.append([current_x, current_y])
        
        elif command == 'V':  # Vertical line to (absolute)
            for param in params:
                current_y = param
                points.append([current_x, current_y])
        
        elif command == 'v':  # Vertical line to (relative)
            for param in params:
                current_y += param
                points.append([current_x, current_y])
        
        elif command in ['C', 'c', 'S', 's', 'Q', 'q', 'T', 't']:
            # For curves, we'll sample points along the curve
            # This is a simplified approach - just using end points
            if command.isupper():  # Absolute
                if len(params) >= 2:
                    current_x, current_y = params[-2], params[-1]
                    points.append([current_x, current_y])
            else:  # Relative
                if len(params) >= 2:
                    current_x += params[-2]
                    current_y += params[-1]
                    points.append([current_x, current_y])
        
        elif command in ['A', 'a']:
            # Arc commands - simplified to endpoint
            if len(params) >= 7:
                if command == 'A':  # Absolute
                    current_x, current_y = params[5], params[6]
                else:  # Relative
                    current_x += params[5]
                    current_y += params[6]
                points.append([current_x, current_y])
        
        elif command in ['Z', 'z']:
            # Close path - connect to first point
            if points:
                points.append(points[0])
    
    return points


def rect_to_polygon(x, y, width, height):
    """Convert rectangle parameters to polygon points."""
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
        [x, y]
    ]


def ellipse_to_polygon(cx, cy, rx, ry, num_points=32):
    """Convert ellipse parameters to polygon points."""
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = cx + rx * np.cos(angle)
        y = cy + ry * np.sin(angle)
        points.append([x, y])
    points.append(points[0])  # Close the polygon
    return points


def parse_svg_to_coco(svg_file, output_json):
    """Parse SVG file and convert to COCO segmentation format."""
    
    # Parse the SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Extract SVG namespace if present
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # Get SVG dimensions
    width = float(root.get('width', '0').replace('mm', '').replace('px', '') or 0)
    height = float(root.get('height', '0').replace('mm', '').replace('px', '') or 0)
    
    # Handle viewBox if dimensions are not directly specified
    if width == 0 or height == 0:
        viewbox = root.get('viewBox', '').split()
        if len(viewbox) == 4:
            width = float(viewbox[2])
            height = float(viewbox[3])
    
    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": "SVG to COCO conversion",
            "version": "1.0",
            "year": 2024
        },
        "images": [{
            "id": 1,
            "width": int(width),
            "height": int(height),
            "file_name": Path(svg_file).name
        }],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "shape",
            "supercategory": "object"
        }]
    }
    
    annotation_id = 1
    
    # Find all shape elements
    for elem in root.iter():
        segmentation = []
        bbox = None
        
        # Handle paths
        if elem.tag.endswith('path'):
            d = elem.get('d')
            if d:
                points = parse_svg_path_to_points(d)
                if len(points) > 2:
                    # Flatten points for COCO format
                    segmentation = [coord for point in points for coord in point]
        
        # Handle rectangles
        elif elem.tag.endswith('rect'):
            x = float(elem.get('x', 0))
            y = float(elem.get('y', 0))
            width = float(elem.get('width', 0))
            height = float(elem.get('height', 0))
            if width > 0 and height > 0:
                points = rect_to_polygon(x, y, width, height)
                segmentation = [coord for point in points for coord in point]
        
        # Handle ellipses and circles
        elif elem.tag.endswith('ellipse') or elem.tag.endswith('circle'):
            if elem.tag.endswith('ellipse'):
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                rx = float(elem.get('rx', 0))
                ry = float(elem.get('ry', 0))
            else:  # circle
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                r = float(elem.get('r', 0))
                rx = ry = r
            
            if rx > 0 and ry > 0:
                points = ellipse_to_polygon(cx, cy, rx, ry)
                segmentation = [coord for point in points for coord in point]
        
        # Handle polygons
        elif elem.tag.endswith('polygon'):
            points_str = elem.get('points', '')
            if points_str:
                points = []
                coords = points_str.split()
                for coord in coords:
                    if ',' in coord:
                        x, y = coord.split(',')
                        points.append([float(x), float(y)])
                if points:
                    points.append(points[0])  # Close the polygon
                    segmentation = [coord for point in points for coord in point]
        
        # If we have a valid segmentation, create annotation
        if segmentation and len(segmentation) >= 6:  # At least 3 points
            # Calculate bounding box
            xs = segmentation[0::2]
            ys = segmentation[1::2]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            
            # Calculate area
            area = (max_x - min_x) * (max_y - min_y)
            
            # Skip very small shapes (likely noise)
            if area > 1:
                annotation = {
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Converted {len(coco_data['annotations'])} shapes to COCO format")
    print(f"Output saved to: {output_json}")


def main():
    # Example usage
    svg_file = "/home/htp26/RoomFormerTest/data/waffle/data/svg_files/Abbazia_di_Sant%27%27Andrea_in_Flumine%2C_chiesa_-_Pianta.svg"  # Your SVG file
    output_json = "coco_annotations.json"
    
    parse_svg_to_coco(svg_file, output_json)
    
    # Optionally, display statistics
    with open(output_json, 'r') as f:
        data = json.load(f)
        print(f"\nImage dimensions: {data['images'][0]['width']} x {data['images'][0]['height']}")
        print(f"Total annotations: {len(data['annotations'])}")
        
        # Show first few annotations
        print("\nFirst 3 annotations:")
        for i, ann in enumerate(data['annotations'][:3]):
            print(f"  Annotation {ann['id']}:")
            print(f"    Bbox: {ann['bbox']}")
            print(f"    Area: {ann['area']}")
            print(f"    Points: {len(ann['segmentation'][0]) // 2}")


if __name__ == "__main__":
    main()