import os
# from pathlib import Path
import cairosvg
import io
from PIL import Image
from glob import glob
import numpy as np

import svgpathtools
from svgpathtools import svg2paths, Document, Path, Line
from svgpathtools import parse_path
from shapely.geometry import Polygon
from xml.dom import minidom
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from svgelements import SVG, Path, Shape, Circle, Rect, Ellipse, Polygon as SVGPolygon, Polyline

def find_axis_scales(doc, target_size):
    # Get SVG dimensions
    svg_elem = doc.getElementsByTagName('svg')[0]
    
    # Get original width and height
    try:
        orig_width = float(svg_elem.getAttribute('width').replace('px', '') or 0)
        orig_height = float(svg_elem.getAttribute('height').replace('px', '') or 0)
        
        # If width/height not specified, try viewBox
        if orig_width == 0 or orig_height == 0:
            viewbox = svg_elem.getAttribute('viewBox')
            if viewbox:
                viewbox_parts = viewbox.split()
                if len(viewbox_parts) >= 4:
                    orig_width = float(viewbox_parts[2])
                    orig_height = float(viewbox_parts[3])
        
        # Calculate scaling factor to preserve aspect ratio
        scale = min(target_size / orig_width, target_size / orig_height) if orig_width > 0 and orig_height > 0 else 1
    except Exception as e:
        print(f"Error determining SVG dimensions: {str(e)}")
        scale = 1
    
    return scale, (int(scale*orig_width), int(scale*orig_height))


def extract_svg_polygons_with_ids(svg_file, target_size=512, 
                                  sample_points=100, simplify_tolerance=0.5):
    """
    Extract polygons from SVG file with their corresponding group IDs using established libraries.
    
    Args:
        svg_file: Path to the SVG file
        sample_points: Number of points to sample along each path
        simplify_tolerance: Tolerance for Shapely simplification 
        
    Returns:
        Dictionary mapping element IDs to their polygons as numpy arrays
    """
    # Parse the SVG document to get groups and IDs
    doc = minidom.parse(svg_file)

    # maintain aspect ratio with a target size
    scale, new_size = find_axis_scales(doc, target_size=target_size)
    scale_x, scale_y = scale, scale
    
    # Dictionary to store results
    polygons_by_id = {}
    
    # Process all group elements with IDs
    groups = doc.getElementsByTagName('g')
    for group in groups:
        group_id = group.getAttribute('id')
        if not group_id:
            continue
        
        # Store polygons for this group
        polygons_by_id[group_id] = []
        
        # Process paths in this group
        paths = group.getElementsByTagName('path')
        for path_elem in paths:
            # Get the path data
            d = path_elem.getAttribute('d')
            if not d:
                continue
                
            try:
                # Parse the path data using svgpathtools
                path = svgpathtools.parse_path(d)

                # Check if path contains segments
                if len(path) == 0:
                    print(f"Warning: Path in {group_id} contains no segments, skipping")
                    continue
                
                # Sample points along the path
                points = []
                for t in np.linspace(0, 1, sample_points):
                    point = path.point(t)

                    # Scale the coordinates
                    scaled_x = point.real * scale_x
                    scaled_y = point.imag * scale_y
                    
                    points.append((scaled_x, scaled_y))
                
                # Simplify using Shapely if we have enough points
                if len(points) >= 3:
                    # Try to create a polygon - may fail if points don't form valid polygon
                    try:
                        shapely_poly = Polygon(points)
                        if not shapely_poly.is_valid:
                            print(f"Warning: Invalid polygon from path in {group_id}, attempting to fix")
                            shapely_poly = shapely_poly.buffer(0)  # Buffer trick to fix self-intersections
                        
                        if shapely_poly.is_valid:
                            simplified_poly = shapely_poly.simplify(simplify_tolerance, preserve_topology=True)

                            # Handle both Polygon and MultiPolygon cases
                            if simplified_poly.geom_type == 'Polygon':
                                simplified_points = np.array(simplified_poly.exterior.coords)
                                polygons_by_id[group_id].append(simplified_points)
                            elif simplified_poly.geom_type == 'MultiPolygon':
                                # Add each polygon in the multipolygon separately
                                print(f"Note: Path in {group_id} created a MultiPolygon with {len(simplified_poly.geoms)} parts")
                                for geom in simplified_poly.geoms:
                                    if geom.geom_type == 'Polygon':
                                        poly_points = np.array(geom.exterior.coords)
                                        polygons_by_id[group_id].append(poly_points)
                            else:
                                print(f"Warning: Unexpected geometry type {simplified_poly.geom_type} in {group_id}")
                        else:
                            print(f"Warning: Could not create valid polygon from path in {group_id}")
                    except Exception as polygon_error:
                        print(f"Error creating polygon from path in {group_id}: {str(polygon_error)}")
                        # Fall back to adding points as a polyline instead of polygon
                        if len(points) >= 2:
                            polygons_by_id[group_id].append(np.array(points))
            except Exception as e:
                print(f"Error processing path in {group_id}: {str(e)}")
        
        # Process rectangles in this group
        rects = group.getElementsByTagName('rect')
        for rect in rects:
            try:
                x = float(rect.getAttribute('x') or 0)
                y = float(rect.getAttribute('y') or 0)
                width = float(rect.getAttribute('width') or 0)
                height = float(rect.getAttribute('height') or 0)

                # Scale the rectangle coordinates
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                scaled_width = width * scale_x
                scaled_height = height * scale_y

                # Create scaled rectangle as polygon points
                rect_points = np.array([
                    [scaled_x, scaled_y],
                    [scaled_x + scaled_width, scaled_y],
                    [scaled_x + scaled_width, scaled_y + scaled_height],
                    [scaled_x, scaled_y + scaled_height],
                    [scaled_x, scaled_y]  # Close the polygon
                ])
                
                polygons_by_id[group_id].append(rect_points)
            except Exception as e:
                print(f"Error processing rectangle in {group_id}: {str(e)}")
        
        # Process polygons in this group
        polygons = group.getElementsByTagName('polygon')
        for polygon in polygons:
            try:
                points_str = polygon.getAttribute('points')
                if not points_str:
                    continue
                
                # Parse points string into coordinates
                # SVG format is "x1,y1 x2,y2 x3,y3 ..."
                coords = []
                point_pairs = points_str.strip().split()
                for pair in point_pairs:
                    if ',' in pair:
                        x, y = pair.split(',')
                        coords.append((float(x) * scale_x, float(y) * scale_y))
                
                if len(coords) >= 3:
                    # Convert to numpy array
                    poly_points = np.array(coords)
                    # Add the first point at the end if it's not already there to close the polygon
                    if not np.array_equal(poly_points[0], poly_points[-1]):
                        poly_points = np.vstack([poly_points, poly_points[0]])
                    
                    # Simplify if needed
                    shapely_poly = Polygon(poly_points)
                    simplified_poly = shapely_poly.simplify(simplify_tolerance, preserve_topology=True)
                    simplified_points = np.array(simplified_poly.exterior.coords)
                    polygons_by_id[group_id].append(simplified_points)
            except Exception as e:
                print(f"Error processing polygon in {group_id}: {str(e)}")
            

        # Process polylines in this group
        polylines = group.getElementsByTagName('polyline')
        for polyline in polylines:
            try:
                points_str = polyline.getAttribute('points')
                if not points_str:
                    continue
                
                # Parse points string into coordinates (same format as polygon)
                coords = []
                point_pairs = points_str.strip().split()
                for pair in point_pairs:
                    if ',' in pair:
                        x, y = pair.split(',')
                        coords.append((float(x) * scale_x, float(y) * scale_y))
                
                if len(coords) >= 2:
                    # Convert to numpy array
                    line_points = np.array(coords)
                    polygons_by_id[group_id].append(line_points)
            except Exception as e:
                print(f"Error processing polyline in {group_id}: {str(e)}")


        # Process lines in this group
        lines = group.getElementsByTagName('line')
        for line in lines:
            try:
                x1 = float(line.getAttribute('x1') or 0)
                y1 = float(line.getAttribute('y1') or 0)
                x2 = float(line.getAttribute('x2') or 0)
                y2 = float(line.getAttribute('y2') or 0)
                
                # Scale the line coordinates
                scaled_x1 = x1 * scale_x
                scaled_y1 = y1 * scale_y
                scaled_x2 = x2 * scale_x
                scaled_y2 = y2 * scale_y
                
                # Create a line as a simple two-point array
                line_points = np.array([
                    [scaled_x1, scaled_y1],
                    [scaled_x2, scaled_y2]
                ])
                
                polygons_by_id[group_id].append(line_points)
            except Exception as e:
                print(f"Error processing line in {group_id}: {str(e)}")
    
    # Clean up
    doc.unlink()
    
    return polygons_by_id, new_size


def svg_to_jpg(svg_file, img_file, width=None, height=None):
    # Convert SVG to PNG in memory
    with open(svg_file, "rb") as f:
        png_data = cairosvg.svg2png(
            file_obj=f,
            write_to=img_file,
            output_width=width,
            output_height=height
        )
    
    # # Open PNG with PIL and convert to JPG
    # image = Image.open(io.BytesIO(png_data))
    
    # # Convert to RGB mode if needed (in case of transparency)
    # if image.mode != 'RGB':
    #     image = image.convert('RGB')
    
    # # Save as JPG
    # image.save(jpg_file, 'JPEG', quality=90)
    
    # return jpg_file


def visualize_polygons(polygons_by_id, figsize=(12, 10), save_path=None, 
                      random_colors=True, show_ids=True, alpha=0.6, 
                      edge_color='black', edge_width=0.5, svg_height=None):
    """
    Visualize the extracted polygons with different colors for each group ID.
    
    Args:
        polygons_by_id: Dictionary mapping group IDs to lists of polygon arrays
        figsize: Size of the figure in inches (width, height)
        save_path: Path to save the figure (if None, figure is displayed)
        random_colors: If True, use random colors; otherwise use color cycle
        show_ids: If True, show group IDs as labels
        alpha: Transparency level for polygons
        edge_color: Color for polygon edges
        edge_width: Width of polygon edges
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate a set of colors
    if random_colors:
        # Use a consistent color for each group ID
        color_map = {group_id: np.random.rand(3,) for group_id in polygons_by_id.keys()}
    else:
        # Use a color cycle
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20', len(polygons_by_id))
        color_map = {group_id: cmap(i)[:3] for i, group_id in enumerate(polygons_by_id.keys())}
    
    # Keep track of group centers for label placement
    group_centers = {}
    
    for group_id, polygons in polygons_by_id.items():
        all_x = []
        all_y = []
        
        for i, poly_points in enumerate(polygons):
            # Skip empty polygons
            if len(poly_points) == 0:
                continue
                
            # Create a copy of the points to modify
            flipped_points = poly_points.copy()
            
            # Flip Y coordinates (SVG has 0 at top, matplotlib has 0 at bottom)
            flipped_points[:, 1] = svg_height - flipped_points[:, 1]

            poly_points = flipped_points

            # Skip polygons with less than 3 points (can't form a polygon)
            if len(poly_points) < 3:
                # Draw as a line instead
                ax.plot(poly_points[:, 0], poly_points[:, 1], 
                        color=color_map[group_id], 
                        linewidth=edge_width*2, 
                        alpha=alpha)
                continue
            
            # Create a Matplotlib polygon patch
            patch = MplPolygon(
                poly_points, 
                closed=True,
                facecolor=color_map[group_id],
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=alpha
            )
            
            # Add the patch to the axes
            ax.add_patch(patch)
            
            # Collect coordinates for calculating group center
            all_x.extend(poly_points[:, 0])
            all_y.extend(poly_points[:, 1])
        
        # Calculate center of the group for label placement
        if all_x and all_y:
            group_centers[group_id] = (np.mean(all_x), np.mean(all_y))
    
    # Add labels if requested
    if show_ids:
        for group_id, center in group_centers.items():
            ax.text(center[0], center[1], group_id, 
                   ha='center', va='center', 
                   fontsize=8, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Set up the axes
    ax.set_aspect('equal')
    ax.autoscale_view()
    
    # Add a title
    plt.title(f'SVG Polygon Visualization ({len(polygons_by_id)} groups)')
    
    # Add a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=np.array(color_map[group_id]), 
               markersize=10, label=group_id)
        for group_id in sorted(polygons_by_id.keys())
    ]
    
    if len(legend_elements) > 20:
        # If there are too many groups, don't show the legend
        pass
    else:
        ax.legend(handles=legend_elements, loc='best', 
                 fontsize=8, framealpha=0.8)
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Either save or show the figure
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()


# def extract_svg_polygons_with_ids(svg_file, target_size=512):
#     """
#     Extract polygons from SVG file with their corresponding group IDs using svgelements.
    
#     Args:
#         svg_file: Path to the SVG file
#         target_size: Target size for scaling the output polygons
        
#     Returns:
#         Dictionary mapping element IDs to their polygons as numpy arrays
#     """
#     # Parse the SVG document
#     svg = SVG.parse(svg_file)
    
#     # Calculate scaling factor
#     svg_width = svg.width
#     svg_height = svg.height
    
#     if svg_width > svg_height:
#         scale_factor = target_size / svg_width
#     else:
#         scale_factor = target_size / svg_height
    
#     # Dictionary to store results
#     polygons_by_id = {}

#     # Helper function to find group ID
#     def find_group_id(element):
#         # Check if element itself has an ID
#         if hasattr(element, 'id') and element.id:
#             return element.id
            
#         # In svgelements, we can use the .values attribute to access parent info
#         # This is implementation-specific and may not always work as expected
#         if hasattr(element, 'values'):
#             if 'id' in element.values:
#                 return element.values['id']
                
#         # For some elements, we can check the owner document
#         # Note: This approach depends on the specific structure of your SVG
#         return None
    
#     # Process all elements
#     for element in svg.elements():
#         # Skip elements without direct IDs
#         if not hasattr(element, 'id') or not element.id:
#             continue
            
#         group_id = element.id
        
#         # Initialize the group if not already present
#         if group_id not in polygons_by_id:
#             polygons_by_id[group_id] = []
        
#         # Extract points based on element type
#         points = None
        
#         if isinstance(element, Path):
#             # Get high-precision points from the path
#             # smaller distance = higher precision
#             path_points = element.points(distance=0.5)
#             if len(path_points) >= 3:
#                 points = np.array(path_points)
                
#         elif isinstance(element, SVGPolygon):
#             # Get points directly from polygon
#             poly_points = list(element.points)
#             if len(poly_points) >= 3:
#                 points = np.array(poly_points)
                
#         elif isinstance(element, Rect):
#             # Convert rectangle to polygon points
#             x, y = element.x, element.y
#             w, h = element.width, element.height
#             points = np.array([
#                 [x, y],
#                 [x + w, y],
#                 [x + w, y + h],
#                 [x, y + h],
#                 [x, y]  # Close the polygon
#             ])
            
#         elif isinstance(element, Circle):
#             # Approximate circle with polygon points
#             cx, cy = element.cx, element.cy
#             r = element.r
#             theta = np.linspace(0, 2*np.pi, 72, endpoint=True)
#             x = cx + r * np.cos(theta)
#             y = cy + r * np.sin(theta)
#             points = np.column_stack((x, y))
            
#         elif isinstance(element, Ellipse):
#             # Approximate ellipse with polygon points
#             cx, cy = element.cx, element.cy
#             rx, ry = element.rx, element.ry
#             theta = np.linspace(0, 2*np.pi, 72, endpoint=True)
#             x = cx + rx * np.cos(theta)
#             y = cy + ry * np.sin(theta)
#             points = np.column_stack((x, y))
            
#         elif isinstance(element, Polyline):
#             # Get points directly from polyline
#             line_points = list(element.points)
#             if len(line_points) >= 2:
#                 points = np.array(line_points)
        
#         # Scale the points if we have them
#         if points is not None and len(points) > 0:
#             # Scale coordinates
#             scaled_points = points * scale_factor
#             polygons_by_id[group_id].append(scaled_points)
    
#     return polygons_by_id, (int(scale_factor*svg_width), int(scale_factor*svg_height))


if __name__ == "__main__":
    svg_file = "/home/htp26/RoomFormerTest/data/waffle/data/svg_files/Topkapi_Palace_overview_EN.svg"
    svg_to_jpg(svg_file, 'png_debug.png', width=512)
    # polygons, attributes = svg_to_polygons(str(svg_file))
    polygons, svg_size = extract_svg_polygons_with_ids(svg_file)

    visualize_polygons(
        polygons,
        figsize=(15, 12),
        save_path="plot_debug_2.jpg",  # Set to None to display instead of save
        random_colors=True,
        show_ids=True,
        svg_height=svg_size[1],
    )
