import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
import sys

# Import the SVG parser
from svg_parser import SVGPolygonExtractor

def visualize_path(path_data, title="SVG Path Visualization", output_file=None, show_plot=True):
    """
    Visualize an SVG path by parsing it and plotting the resulting points.
    
    Args:
        path_data (str): SVG path data string
        title (str): Title for the plot
        output_file (str, optional): File to save the visualization to
        show_plot (bool): Whether to display the plot
    """
    # Create the extractor and parse the path
    extractor = SVGPolygonExtractor()
    points = extractor.parse_svg_path(path_data, curve_sample_points=30)
    
    if not points:
        print("Error: No points extracted from path data")
        return

    # Extract x and y coordinates
    x_coords = [point['x'] for point in points]
    y_coords = [point['y'] for point in points]
    
    # Create figure and plot
    plt.figure(figsize=(10, 8))
    
    # Plot the path
    plt.plot(x_coords, y_coords, 'b-', linewidth=2)
    
    # Plot the points
    plt.scatter(x_coords, y_coords, c='red', s=20, zorder=2)
    
    # Mark the first point
    plt.scatter([x_coords[0]], [y_coords[0]], c='green', s=100, marker='o', zorder=3)
    
    # Add labels
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Add path data as text
    max_path_length = 50
    path_display = path_data if len(path_data) <= max_path_length else path_data[:max_path_length] + "..."
    plt.figtext(0.5, 0.01, f"Path data: {path_display}", ha='center', fontsize=10)
    
    # Set axis equal to preserve shape
    plt.axis('equal')
    
    # Invert y-axis to match SVG coordinate system (origin at top-left)
    plt.gca().invert_yaxis()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    plt.close()

def create_visualization(svg_elements, output_dir="visualizations"):
    """
    Create visualizations for SVG elements extracted by the parser.
    
    Args:
        svg_elements (dict): Dictionary with extracted SVG elements
        output_dir (str): Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize paths
    for i, path in enumerate(svg_elements.get('paths', [])):
        path_id = path.get('id', f'path_{i}')
        path_data = path.get('path_data', '')
        points = path.get('points', [])
        
        if not points:
            continue
        
        # Extract x and y coordinates
        x_coords = [point['x'] for point in points]
        y_coords = [point['y'] for point in points]
        
        # Create figure and plot
        plt.figure(figsize=(10, 8))
        
        # Plot the path
        plt.plot(x_coords, y_coords, 'b-', linewidth=2)
        
        # Plot the points
        plt.scatter(x_coords, y_coords, c='red', s=20, zorder=2)
        
        # Mark the first point
        plt.scatter([x_coords[0]], [y_coords[0]], c='green', s=100, marker='o', zorder=3)
        
        # Add labels
        plt.title(f"Path: {path_id}")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Add path data as text
        max_path_length = 50
        path_display = path_data if len(path_data) <= max_path_length else path_data[:max_path_length] + "..."
        plt.figtext(0.5, 0.01, f"Path data: {path_display}", ha='center', fontsize=10)
        
        # Set axis equal to preserve shape
        plt.axis('equal')
        
        # Invert y-axis to match SVG coordinate system (origin at top-left)
        plt.gca().invert_yaxis()
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(output_dir, f"{path_id}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Path visualization saved to {output_file}")
        
        plt.close()
    
    # Visualize rectangles
    for i, rect in enumerate(svg_elements.get('rectangles', [])):
        rect_id = rect.get('id', f'rect_{i}')
        points = rect.get('points', [])
        
        if not points:
            continue
        
        # Extract x and y coordinates
        x_coords = [point['x'] for point in points]
        y_coords = [point['y'] for point in points]
        
        # Create figure and plot
        plt.figure(figsize=(10, 8))
        
        # Plot the rectangle
        plt.plot(x_coords, y_coords, 'b-', linewidth=2)
        
        # Plot the points
        plt.scatter(x_coords, y_coords, c='red', s=20, zorder=2)
        
        # Add labels
        plt.title(f"Rectangle: {rect_id}")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Set axis equal to preserve shape
        plt.axis('equal')
        
        # Invert y-axis to match SVG coordinate system (origin at top-left)
        plt.gca().invert_yaxis()
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(output_dir, f"{rect_id}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Rectangle visualization saved to {output_file}")
        
        plt.close()
    
    # Create a summary visualization with all elements
    plt.figure(figsize=(12, 10))
    
    # Plot paths
    for i, path in enumerate(svg_elements.get('paths', [])):
        points = path.get('points', [])
        if not points:
            continue
        
        x_coords = [point['x'] for point in points]
        y_coords = [point['y'] for point in points]
        plt.plot(x_coords, y_coords, '-', linewidth=1.5, alpha=0.7, label=f"Path {i+1}" if i < 5 else None)
    
    # Plot rectangles
    for i, rect in enumerate(svg_elements.get('rectangles', [])):
        points = rect.get('points', [])
        if not points:
            continue
        
        x_coords = [point['x'] for point in points]
        y_coords = [point['y'] for point in points]
        plt.plot(x_coords, y_coords, '--', linewidth=1.5, alpha=0.7, label=f"Rectangle {i+1}" if i < 5 else None)
    
    # Add labels
    plt.title("SVG Elements Overview")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Set axis equal to preserve shape
    plt.axis('equal')
    
    # Invert y-axis to match SVG coordinate system (origin at top-left)
    plt.gca().invert_yaxis()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if there are elements
    if svg_elements.get('paths') or svg_elements.get('rectangles'):
        plt.legend(loc='upper right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save visualization
    output_file = os.path.join(output_dir, "overview.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Overview visualization saved to {output_file}")
    
    plt.close()

def visualize_examples():
    """
    Visualize the predefined SVG path examples.
    """
    # Create the extractor and get examples
    extractor = SVGPolygonExtractor()
    examples = extractor.extract_examples()
    
    # Create a directory for examples
    os.makedirs("examples", exist_ok=True)
    
    # Visualize each example
    for name, example in examples.items():
        print(f"Visualizing {name}...")
        visualize_path(
            example['path_data'],
            title=f"{name}: {example['description']}",
            output_file=f"examples/{name}.png",
            show_plot=False
        )
    
    print(f"Example visualizations saved to 'examples' directory")

def visualize_commands():
    """
    Visualize individual SVG path commands with simple examples.
    """
    commands = {
        "M_absolute": {
            "description": "Move to absolute position",
            "path_data": "M 100 100 L 200 100 L 200 200 L 100 200 Z"
        },
        "m_relative": {
            "description": "Move to relative position",
            "path_data": "M 100 100 m 100 0 l 0 100 l -100 0 Z"
        },
        "L_absolute": {
            "description": "Line to absolute position",
            "path_data": "M 100 100 L 200 100 L 200 200 L 100 200 Z"
        },
        "l_relative": {
            "description": "Line to relative position",
            "path_data": "M 100 100 l 100 0 l 0 100 l -100 0 Z"
        },
        "H_absolute": {
            "description": "Horizontal line to absolute X",
            "path_data": "M 100 100 H 200 V 200 H 100 Z"
        },
        "h_relative": {
            "description": "Horizontal line relative X",
            "path_data": "M 100 100 h 100 v 100 h -100 Z"
        },
        "V_absolute": {
            "description": "Vertical line to absolute Y",
            "path_data": "M 100 100 H 200 V 200 H 100 Z"
        },
        "v_relative": {
            "description": "Vertical line relative Y",
            "path_data": "M 100 100 h 100 v 100 h -100 Z"
        },
        "C_cubic": {
            "description": "Cubic Bézier curve absolute",
            "path_data": "M 100 200 C 100 100, 250 100, 250 200 S 400 300, 400 200"
        },
        "c_cubic_relative": {
            "description": "Cubic Bézier curve relative",
            "path_data": "M 100 200 c 0 -100, 150 -100, 150 0 s 150 100, 150 0"
        },
        "S_smooth_cubic": {
            "description": "Smooth cubic Bézier curve absolute",
            "path_data": "M 100 200 C 100 100, 250 100, 250 200 S 400 300, 400 200"
        },
        "s_smooth_cubic_relative": {
            "description": "Smooth cubic Bézier curve relative",
            "path_data": "M 100 200 c 0 -100, 150 -100, 150 0 s 150 100, 150 0"
        },
        "Q_quadratic": {
            "description": "Quadratic Bézier curve absolute",
            "path_data": "M 100 200 Q 200 100, 300 200 T 500 200"
        },
        "q_quadratic_relative": {
            "description": "Quadratic Bézier curve relative",
            "path_data": "M 100 200 q 100 -100, 200 0 t 200 0"
        },
        "T_smooth_quadratic": {
            "description": "Smooth quadratic Bézier curve absolute",
            "path_data": "M 100 200 Q 200 100, 300 200 T 500 200"
        },
        "t_smooth_quadratic_relative": {
            "description": "Smooth quadratic Bézier curve relative",
            "path_data": "M 100 200 q 100 -100, 200 0 t 200 0"
        },
        "A_arc": {
            "description": "Elliptical arc absolute",
            "path_data": "M 100 200 A 100 50 0 1 0 300 200"
        },
        "a_arc_relative": {
            "description": "Elliptical arc relative",
            "path_data": "M 100 200 a 100 50 0 1 0 200 0"
        },
        "Z_closepath": {
            "description": "Close path command",
            "path_data": "M 100 100 L 200 100 L 150 200 Z"
        }
    }
    
    # Create a directory for command examples
    os.makedirs("commands", exist_ok=True)
    
    # Visualize each command example
    for name, command in commands.items():
        print(f"Visualizing {name}...")
        visualize_path(
            command['path_data'],
            title=f"{name}: {command['description']}",
            output_file=f"commands/{name}.png",
            show_plot=False
        )
    
    print(f"Command visualizations saved to 'commands' directory")

def main():
    parser = argparse.ArgumentParser(description='Visualize SVG paths')
    parser.add_argument('--path', help='SVG path data to visualize')
    parser.add_argument('--file', help='SVG or JSON file with path data to visualize')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--examples', action='store_true', help='Visualize predefined examples')
    parser.add_argument('--commands', action='store_true', help='Visualize SVG path commands')
    parser.add_argument('--visualize-svg', help='Process an SVG file and create visualizations for all elements')
    parser.add_argument('--output-dir', default='visualizations', help='Directory to save visualizations (default: visualizations)')
    
    args = parser.parse_args()
    
    if args.examples:
        visualize_examples()
        return
    
    if args.commands:
        visualize_commands()
        return
    
    if args.path:
        visualize_path(args.path, output_file=args.output)
        return
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found.")
            return
        
        # Check if it's a JSON file
        if args.file.lower().endswith('.json'):
            try:
                with open(args.file, 'r') as f:
                    data = json.load(f)
                
                if 'path_data' in data:
                    # Single path
                    visualize_path(data['path_data'], output_file=args.output)
                elif isinstance(data, dict) and all(isinstance(item, dict) for item in data.values()):
                    # Multiple paths
                    for name, item in data.items():
                        if 'path_data' in item:
                            output_file = f"{os.path.splitext(args.output)[0]}_{name}.png" if args.output else None
                            visualize_path(
                                item['path_data'], 
                                title=f"{name}: {item.get('description', '')}",
                                output_file=output_file
                            )
                else:
                    print("Error: Invalid JSON format. Expected 'path_data' field or dictionary of paths.")
            except json.JSONDecodeError:
                print("Error: Invalid JSON file.")
            
            return
        
        # If it's an SVG file, process it
        if args.file.lower().endswith('.svg'):
            extractor = SVGPolygonExtractor(svg_file_path=args.file)
            svg_elements = extractor.process_full_svg()
            create_visualization(svg_elements, output_dir=args.output_dir)
            return
    
    if args.visualize_svg:
        if not os.path.exists(args.visualize_svg):
            print(f"Error: SVG file {args.visualize_svg} not found.")
            return
        
        extractor = SVGPolygonExtractor(svg_file_path=args.visualize_svg)
        svg_elements = extractor.process_full_svg()
        create_visualization(svg_elements, output_dir=args.output_dir)
        return
    
    # If no arguments provided, show help
    parser.print_help()

if __name__ == "__main__":
    main()