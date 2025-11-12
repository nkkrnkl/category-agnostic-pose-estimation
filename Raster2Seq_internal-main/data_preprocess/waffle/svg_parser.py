import re
import xml.etree.ElementTree as ET
import json
import math
import numpy as np

class SVGPolygonExtractor:
    """
    A class to extract polygon coordinates from SVG files.
    Handles all path commands including curves to convert to (x,y) coordinate points.
    """
    
    def __init__(self, svg_file_path=None, svg_string=None):
        """
        Initialize with either a file path or an SVG string.
        
        Args:
            svg_file_path (str, optional): Path to the SVG file
            svg_string (str, optional): SVG content as a string
        """
        self.svg_content = None
        
        if svg_file_path:
            with open(svg_file_path, 'r', encoding='utf-8') as file:
                self.svg_content = file.read()
        elif svg_string:
            self.svg_content = svg_string
        
        if self.svg_content:
            # Parse the XML
            self.root = ET.fromstring(self.svg_content)
            # Define namespace mapping if needed
            self.namespaces = {
                'svg': 'http://www.w3.org/2000/svg',
                'xlink': 'http://www.w3.org/1999/xlink'
            }
    
    def parse_svg_path(self, path_data, curve_sample_points=10):
        """
        Parse SVG path data to extract polygon coordinates, including curve commands.
        
        Args:
            path_data (str): The 'd' attribute of a path element
            curve_sample_points (int): Number of points to sample for curve commands
            
        Returns:
            list: List of dictionaries with x,y coordinates
        """
        if not path_data or not path_data.strip():
            return []
            
        # Normalize the path data to make parsing easier
        # Add spaces before and after commands and remove commas
        path_data = self._normalize_path_data(path_data)
        
        # Regular expression to match commands and their parameters
        command_pattern = re.compile(r'([a-zA-Z])([^a-zA-Z]*)')
        
        commands = command_pattern.findall(path_data)
        points = []
        subpath_start_x, subpath_start_y = 0, 0  # For closepath command
        current_x, current_y = 0, 0
        
        # Keep track of control points for smooth curve commands
        last_control_point_x, last_control_point_y = None, None
        last_command = None
    
    def _normalize_path_data(self, path_data):
        """
        Normalize SVG path data to make parsing easier.
        - Add spaces around commands
        - Replace commas with spaces
        - Remove consecutive spaces
        
        Args:
            path_data (str): Original path data
            
        Returns:
            str: Normalized path data
        """
        # Add space before each command
        for cmd in 'MmLlHhVvCcSsQqTtAaZz':
            path_data = path_data.replace(cmd, f" {cmd} ")
        
        # Replace commas with spaces
        path_data = path_data.replace(',', ' ')
        
        # Replace multiple spaces with a single space
        path_data = re.sub(r'\s+', ' ', path_data).strip()
        
        return path_data
        
        for cmd_type, args_str in commands:
            # Parse arguments
            args = [float(arg) for arg in re.findall(r'[-+]?[0-9]*\.?[0-9]+', args_str)]
            
            if cmd_type == 'M':  # Move to (absolute)
                current_x, current_y = args[0], args[1]
                points.append({'x': current_x, 'y': current_y})
                
                # Handle multiple coordinate pairs after M as implicit L commands
                i = 2
                while i < len(args):
                    current_x, current_y = args[i], args[i+1]
                    points.append({'x': current_x, 'y': current_y})
                    i += 2
            
            elif cmd_type == 'm':  # Move to (relative)
                if not points:  # First command - treat as absolute
                    current_x, current_y = args[0], args[1]
                else:
                    current_x += args[0]
                    current_y += args[1]
                points.append({'x': current_x, 'y': current_y})
                
                # Handle multiple coordinate pairs after m as implicit l commands
                i = 2
                while i < len(args):
                    current_x += args[i]
                    current_y += args[i+1]
                    points.append({'x': current_x, 'y': current_y})
                    i += 2
            
            elif cmd_type == 'L':  # Line to (absolute)
                i = 0
                while i < len(args):
                    current_x, current_y = args[i], args[i+1]
                    points.append({'x': current_x, 'y': current_y})
                    i += 2
            
            elif cmd_type == 'l':  # Line to (relative)
                i = 0
                while i < len(args):
                    current_x += args[i]
                    current_y += args[i+1]
                    points.append({'x': current_x, 'y': current_y})
                    i += 2
            
            elif cmd_type == 'H':  # Horizontal line to (absolute)
                for x in args:
                    current_x = x
                    points.append({'x': current_x, 'y': current_y})
            
            elif cmd_type == 'h':  # Horizontal line to (relative)
                for x_offset in args:
                    current_x += x_offset
                    points.append({'x': current_x, 'y': current_y})
            
            elif cmd_type == 'V':  # Vertical line to (absolute)
                for y in args:
                    current_y = y
                    points.append({'x': current_x, 'y': current_y})
            
            elif cmd_type == 'v':  # Vertical line to (relative)
                for y_offset in args:
                    current_y += y_offset
                    points.append({'x': current_x, 'y': current_y})
            
            elif cmd_type in ['C', 'c']:  # Cubic Bezier curve
                i = 0
                while i < len(args):
                    if cmd_type == 'C':  # Absolute
                        x1, y1 = args[i], args[i+1]
                        x2, y2 = args[i+2], args[i+3]
                        x, y = args[i+4], args[i+5]
                    else:  # Relative (c)
                        x1, y1 = current_x + args[i], current_y + args[i+1]
                        x2, y2 = current_x + args[i+2], current_y + args[i+3]
                        x, y = current_x + args[i+4], current_y + args[i+5]
                    
                    # Save last control point for potential smooth curve command
                    last_control_point_x, last_control_point_y = x2, y2
                    
                    # Sample points along the cubic Bezier curve
                    for t in np.linspace(0, 1, curve_sample_points):
                        # Cubic Bezier formula
                        px = (1-t)**3 * current_x + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x
                        py = (1-t)**3 * current_y + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y
                        
                        if t > 0:  # Skip t=0 as it's the current point
                            points.append({'x': px, 'y': py})
                    
                    current_x, current_y = x, y
                    i += 6
                    
                last_command = cmd_type
            
            elif cmd_type in ['S', 's']:  # Smooth cubic Bezier curve
                i = 0
                while i < len(args):
                    # Calculate first control point as reflection of previous control point
                    if last_command in ['C', 'c', 'S', 's'] and last_control_point_x is not None:
                        x1 = 2 * current_x - last_control_point_x
                        y1 = 2 * current_y - last_control_point_y
                    else:
                        x1, y1 = current_x, current_y
                    
                    if cmd_type == 'S':  # Absolute
                        x2, y2 = args[i], args[i+1]
                        x, y = args[i+2], args[i+3]
                    else:  # Relative (s)
                        x2, y2 = current_x + args[i], current_y + args[i+1]
                        x, y = current_x + args[i+2], current_y + args[i+3]
                    
                    # Save last control point for potential smooth curve command
                    last_control_point_x, last_control_point_y = x2, y2
                    
                    # Sample points along the cubic Bezier curve
                    for t in np.linspace(0, 1, curve_sample_points):
                        # Cubic Bezier formula
                        px = (1-t)**3 * current_x + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x
                        py = (1-t)**3 * current_y + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y
                        
                        if t > 0:  # Skip t=0 as it's the current point
                            points.append({'x': px, 'y': py})
                    
                    current_x, current_y = x, y
                    i += 4
                    
                last_command = cmd_type
            
            elif cmd_type in ['Q', 'q']:  # Quadratic Bezier curve
                i = 0
                while i < len(args):
                    if cmd_type == 'Q':  # Absolute
                        x1, y1 = args[i], args[i+1]
                        x, y = args[i+2], args[i+3]
                    else:  # Relative (q)
                        x1, y1 = current_x + args[i], current_y + args[i+1]
                        x, y = current_x + args[i+2], current_y + args[i+3]
                    
                    # Save control point for potential smooth curve command
                    last_control_point_x, last_control_point_y = x1, y1
                    
                    # Sample points along the quadratic Bezier curve
                    for t in np.linspace(0, 1, curve_sample_points):
                        # Quadratic Bezier formula
                        px = (1-t)**2 * current_x + 2*(1-t)*t * x1 + t**2 * x
                        py = (1-t)**2 * current_y + 2*(1-t)*t * y1 + t**2 * y
                        
                        if t > 0:  # Skip t=0 as it's the current point
                            points.append({'x': px, 'y': py})
                    
                    current_x, current_y = x, y
                    i += 4
                    
                last_command = cmd_type
            
            elif cmd_type in ['T', 't']:  # Smooth quadratic Bezier curve
                i = 0
                while i < len(args):
                    # Calculate control point as reflection of previous control point
                    if last_command in ['Q', 'q', 'T', 't'] and last_control_point_x is not None:
                        x1 = 2 * current_x - last_control_point_x
                        y1 = 2 * current_y - last_control_point_y
                    else:
                        x1, y1 = current_x, current_y
                    
                    if cmd_type == 'T':  # Absolute
                        x, y = args[i], args[i+1]
                    else:  # Relative (t)
                        x, y = current_x + args[i], current_y + args[i+1]
                    
                    # Save control point for potential smooth curve command
                    last_control_point_x, last_control_point_y = x1, y1
                    
                    # Sample points along the quadratic Bezier curve
                    for t in np.linspace(0, 1, curve_sample_points):
                        # Quadratic Bezier formula
                        px = (1-t)**2 * current_x + 2*(1-t)*t * x1 + t**2 * x
                        py = (1-t)**2 * current_y + 2*(1-t)*t * y1 + t**2 * y
                        
                        if t > 0:  # Skip t=0 as it's the current point
                            points.append({'x': px, 'y': py})
                    
                    current_x, current_y = x, y
                    i += 2
                    
                last_command = cmd_type
            
            elif cmd_type in ['A', 'a']:  # Elliptical arc
                i = 0
                while i < len(args):
                    rx, ry = args[i], args[i+1]
                    x_axis_rotation = args[i+2]
                    large_arc_flag = args[i+3]
                    sweep_flag = args[i+4]
                    
                    if cmd_type == 'A':  # Absolute
                        x, y = args[i+5], args[i+6]
                    else:  # Relative (a)
                        x, y = current_x + args[i+5], current_y + args[i+6]
                    
                    # Convert arc to points
                    arc_points = self._elliptical_arc_to_points(
                        current_x, current_y, rx, ry, x_axis_rotation, 
                        large_arc_flag, sweep_flag, x, y, curve_sample_points
                    )
                    
                    # Add arc points
                    for point in arc_points:
                        points.append(point)
                    
                    current_x, current_y = x, y
                    i += 7
                    
                last_command = cmd_type
            
            elif cmd_type in ['Z', 'z']:  # Close path
                # Add the first point again to close the path
                if points:
                    points.append({'x': points[0]['x'], 'y': points[0]['y']})
                    current_x, current_y = points[0]['x'], points[0]['y']
            
            else:
                print(f"Warning: Unsupported SVG command: {cmd_type}")
            
        return points
    
    def _elliptical_arc_to_points(self, x1, y1, rx, ry, angle, large_arc_flag, sweep_flag, x2, y2, num_points=10):
        """
        Convert an elliptical arc to a series of points.
        
        Args:
            x1, y1: Starting point coordinates
            rx, ry: Radii of the ellipse
            angle: X-axis rotation in degrees
            large_arc_flag: 0 for small arc, 1 for large arc
            sweep_flag: 0 for counterclockwise, 1 for clockwise
            x2, y2: End point coordinates
            num_points: Number of points to generate
            
        Returns:
            list: List of dictionaries with x,y coordinates
        """
        # Ensure rx and ry are positive
        rx, ry = abs(rx), abs(ry)
        
        # If rx or ry is 0, treat as a straight line
        if rx == 0 or ry == 0:
            return [{'x': x2, 'y': y2}]
        
        # Convert angle from degrees to radians
        angle_rad = math.radians(angle)
        
        # Step 1: Compute (x1′, y1′) - the transformed start point
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        dx = (x1 - x2) / 2
        dy = (y1 - y2) / 2
        x1_prime = cos_angle * dx + sin_angle * dy
        y1_prime = -sin_angle * dx + cos_angle * dy
        
        # Ensure radii are large enough
        lambda_value = (x1_prime ** 2) / (rx ** 2) + (y1_prime ** 2) / (ry ** 2)
        if lambda_value > 1:
            rx *= math.sqrt(lambda_value)
            ry *= math.sqrt(lambda_value)
        
        # Step 2: Compute (cx′, cy′) - the transformed center point
        sign = -1 if large_arc_flag == sweep_flag else 1
        sq = ((rx**2 * ry**2) - (rx**2 * y1_prime**2) - (ry**2 * x1_prime**2)) / ((rx**2 * y1_prime**2) + (ry**2 * x1_prime**2))
        sq = 0 if sq < 0 else sq
        coef = sign * math.sqrt(sq)
        cx_prime = coef * ((rx * y1_prime) / ry)
        cy_prime = coef * (-(ry * x1_prime) / rx)
        
        # Step 3: Compute (cx, cy) from (cx′, cy′) - the center point
        cx = cos_angle * cx_prime - sin_angle * cy_prime + (x1 + x2) / 2
        cy = sin_angle * cx_prime + cos_angle * cy_prime + (y1 + y2) / 2
        
        # Step 4: Calculate the start and end angles
        start_vector_x = (x1_prime - cx_prime) / rx
        start_vector_y = (y1_prime - cy_prime) / ry
        end_vector_x = (-x1_prime - cx_prime) / rx
        end_vector_y = (-y1_prime - cy_prime) / ry
        
        start_angle = self._vector_angle([1, 0], [start_vector_x, start_vector_y])
        delta_angle = self._vector_angle([start_vector_x, start_vector_y], [end_vector_x, end_vector_y])
        
        if sweep_flag == 0 and delta_angle > 0:
            delta_angle -= 2 * math.pi
        elif sweep_flag == 1 and delta_angle < 0:
            delta_angle += 2 * math.pi
        
        # Step 5: Generate points along the arc
        points = []
        for i in range(1, num_points):  # Skip the first point as it's the current point
            t = start_angle + i * delta_angle / num_points
            
            # Coordinates on the ellipse
            ellipse_x = rx * math.cos(t)
            ellipse_y = ry * math.sin(t)
            
            # Rotate and translate back
            x = cos_angle * ellipse_x - sin_angle * ellipse_y + cx
            y = sin_angle * ellipse_x + cos_angle * ellipse_y + cy
            
            points.append({'x': x, 'y': y})
        
        return points
    
    def _vector_angle(self, u, v):
        """
        Calculate the angle between two vectors.
        
        Args:
            u, v: Two vectors
            
        Returns:
            float: Angle in radians
        """
        dot = u[0] * v[0] + u[1] * v[1]
        len_u = math.sqrt(u[0] ** 2 + u[1] ** 2)
        len_v = math.sqrt(v[0] ** 2 + v[1] ** 2)
        
        if len_u * len_v == 0:
            return 0
        
        cosine = max(-1, min(1, dot / (len_u * len_v)))
        angle = math.acos(cosine)
        
        if u[0] * v[1] - u[1] * v[0] < 0:
            angle = -angle
        
        return angle
    
    def extract_all_paths(self, curve_sample_points=10):
        """
        Extract all paths from the SVG.
        
        Args:
            curve_sample_points (int): Number of points to sample for curve commands
            
        Returns:
            list: List of dictionaries with path data and points
        """
        paths = []
        
        # Use XPath to find all path elements
        path_elements = self.root.findall('.//svg:path', self.namespaces)
        if not path_elements:
            # Try without namespace if no elements found
            path_elements = self.root.findall('.//path')
        
        for i, path_elem in enumerate(path_elements):
            path_data = path_elem.get('d')
            if path_data:
                # Get style/fill information if available
                style = path_elem.get('style', '')
                fill = path_elem.get('fill', '')
                
                # Extract ID if available
                path_id = path_elem.get('id', f'path_{i}')
                
                points = self.parse_svg_path(path_data, curve_sample_points)
                paths.append({
                    'id': path_id,
                    'path_data': path_data,
                    'style': style,
                    'fill': fill,
                    'points': points
                })
        
        return paths
    
    # Rest of the class methods remain the same
    def extract_rectangles(self):
        """
        Extract all rectangles from the SVG.
        
        Returns:
            list: List of dictionaries with rectangle data and corner points
        """
        rectangles = []
        
        # Use XPath to find all rectangle elements
        rect_elements = self.root.findall('.//svg:rect', self.namespaces)
        if not rect_elements:
            # Try without namespace if no elements found
            rect_elements = self.root.findall('.//rect')
        
        for i, rect_elem in enumerate(rect_elements):
            try:
                x = float(rect_elem.get('x', 0))
                y = float(rect_elem.get('y', 0))
                width = float(rect_elem.get('width', 0))
                height = float(rect_elem.get('height', 0))
                
                # Extract ID if available
                rect_id = rect_elem.get('id', f'rect_{i}')
                
                # Get style/fill information if available
                style = rect_elem.get('style', '')
                fill = rect_elem.get('fill', '')
                
                # Create polygon points from rectangle
                points = [
                    {'x': x, 'y': y},  # Top-left
                    {'x': x + width, 'y': y},  # Top-right
                    {'x': x + width, 'y': y + height},  # Bottom-right
                    {'x': x, 'y': y + height},  # Bottom-left
                    {'x': x, 'y': y}  # Close the path (back to top-left)
                ]
                
                rectangles.append({
                    'id': rect_id,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'style': style,
                    'fill': fill,
                    'points': points
                })
            except (TypeError, ValueError) as e:
                print(f"Error processing rectangle {i}: {e}")
        
        return rectangles
    
    def extract_polygons(self):
        """
        Extract all polygon elements from the SVG.
        
        Returns:
            list: List of dictionaries with polygon data and points
        """
        polygons = []
        
        # Use XPath to find all polygon elements
        polygon_elements = self.root.findall('.//svg:polygon', self.namespaces)
        if not polygon_elements:
            # Try without namespace if no elements found
            polygon_elements = self.root.findall('.//polygon')
        
        for i, polygon_elem in enumerate(polygon_elements):
            points_str = polygon_elem.get('points', '')
            if points_str:
                # Extract ID if available
                polygon_id = polygon_elem.get('id', f'polygon_{i}')
                
                # Get style/fill information if available
                style = polygon_elem.get('style', '')
                fill = polygon_elem.get('fill', '')
                
                # Parse points string into coordinates
                coord_pairs = re.findall(r'([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)', points_str)
                points = [{'x': float(x), 'y': float(y)} for x, y in coord_pairs]
                
                polygons.append({
                    'id': polygon_id,
                    'points_str': points_str,
                    'style': style,
                    'fill': fill,
                    'points': points
                })
        
        return polygons
    
    def extract_groups(self, curve_sample_points=10):
        """
        Extract groups and their children elements.
        
        Args:
            curve_sample_points (int): Number of points to sample for curve commands
            
        Returns:
            list: List of dictionaries with group data and children elements
        """
        groups = []
        
        # Use XPath to find all group elements
        group_elements = self.root.findall('.//svg:g', self.namespaces)
        if not group_elements:
            # Try without namespace if no elements found
            group_elements = self.root.findall('.//g')
        
        for i, group_elem in enumerate(group_elements):
            # Extract ID and label if available
            group_id = group_elem.get('id', f'group_{i}')
            group_label = group_elem.get('inkscape:label', '')
            
            # Find all path, rect, and polygon elements in this group
            paths = []
            # Try with namespace
            for path_elem in group_elem.findall('./svg:path', self.namespaces):
                path_data = path_elem.get('d')
                if path_data:
                    points = self.parse_svg_path(path_data, curve_sample_points)
                    paths.append({
                        'path_data': path_data,
                        'points': points
                    })
            # Try without namespace if no elements found
            if not paths:
                for path_elem in group_elem.findall('./path'):
                    path_data = path_elem.get('d')
                    if path_data:
                        points = self.parse_svg_path(path_data, curve_sample_points)
                        paths.append({
                            'path_data': path_data,
                            'points': points
                        })
            
            rects = []
            # Try with namespace
            for rect_elem in group_elem.findall('./svg:rect', self.namespaces):
                try:
                    x = float(rect_elem.get('x', 0))
                    y = float(rect_elem.get('y', 0))
                    width = float(rect_elem.get('width', 0))
                    height = float(rect_elem.get('height', 0))
                    
                    points = [
                        {'x': x, 'y': y},
                        {'x': x + width, 'y': y},
                        {'x': x + width, 'y': y + height},
                        {'x': x, 'y': y + height},
                        {'x': x, 'y': y}
                    ]
                    
                    rects.append({
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'points': points
                    })
                except (TypeError, ValueError):
                    pass
            # Try without namespace if no elements found
            if not rects:
                for rect_elem in group_elem.findall('./rect'):
                    try:
                        x = float(rect_elem.get('x', 0))
                        y = float(rect_elem.get('y', 0))
                        width = float(rect_elem.get('width', 0))
                        height = float(rect_elem.get('height', 0))
                        
                        points = [
                            {'x': x, 'y': y},
                            {'x': x + width, 'y': y},
                            {'x': x + width, 'y': y + height},
                            {'x': x, 'y': y + height},
                            {'x': x, 'y': y}
                        ]
                        
                        rects.append({
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'points': points
                        })
                    except (TypeError, ValueError):
                        pass
            
            polygons = []
            # Try with namespace
            for polygon_elem in group_elem.findall('./svg:polygon', self.namespaces):
                points_str = polygon_elem.get('points', '')
                if points_str:
                    coord_pairs = re.findall(r'([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)', points_str)
                    points = [{'x': float(x), 'y': float(y)} for x, y in coord_pairs]
                    polygons.append({
                        'points_str': points_str,
                        'points': points
                    })
            # Try without namespace if no elements found
            if not polygons:
                for polygon_elem in group_elem.findall('./polygon'):
                    points_str = polygon_elem.get('points', '')
                    if points_str:
                        coord_pairs = re.findall(r'([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)', points_str)
                        points = [{'x': float(x), 'y': float(y)} for x, y in coord_pairs]
                        polygons.append({
                            'points_str': points_str,
                            'points': points
                        })
            
            groups.append({
                'id': group_id,
                'label': group_label,
                'paths': paths,
                'rectangles': rects,
                'polygons': polygons
            })
        
        return groups
    
    def process_full_svg(self, curve_sample_points=10):
        """
        Process the full SVG to extract all relevant elements.
        
        Args:
            curve_sample_points (int): Number of points to sample for curve commands
            
        Returns:
            dict: Dictionary with all extracted elements
        """
        if not self.svg_content:
            return {
                'error': 'No SVG content provided'
            }
        
        paths = self.extract_all_paths(curve_sample_points)
        rectangles = self.extract_rectangles()
        polygons = self.extract_polygons()
        groups = self.extract_groups(curve_sample_points)
        
        return {
            'paths': paths,
            'rectangles': rectangles,
            'polygons': polygons,
            'groups': groups,
            'summary': {
                'paths_count': len(paths),
                'rectangles_count': len(rectangles),
                'polygons_count': len(polygons),
                'groups_count': len(groups)
            }
        }
    
    def save_result_to_json(self, output_file_path, result=None):
        """
        Save the extraction result to a JSON file.
        
        Args:
            output_file_path (str): Path to save the JSON output
            result (dict, optional): Result to save, if None process_full_svg() will be called
        """
        if result is None:
            result = self.process_full_svg()
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to {output_file_path}")
    
    def extract_examples(self):
        """
        Extract and process the example SVG paths from the SVG specs.
        
        Returns:
            dict: Dictionary with processed examples
        """
        examples = {
            "triangle": {
                "description": "A simple triangle path",
                "path_data": "M150 5 L75 200 L225 200 Z",
                "points": self.parse_svg_path("M150 5 L75 200 L225 200 Z")
            },
            "quadratic_bezier": {
                "description": "Quadratic Bézier curve example",
                "path_data": "M 100 350 q 150 -300 300 0",
                "points": self.parse_svg_path("M 100 350 q 150 -300 300 0")
            },
            "cubic_bezier": {
                "description": "Cubic Bézier curve example",
                "path_data": "M 100 350 C 100 250, 400 250, 400 350",
                "points": self.parse_svg_path("M 100 350 C 100 250, 400 250, 400 350")
            },
            "smooth_cubic_bezier": {
                "description": "Smooth cubic Bézier curve example",
                "path_data": "M 100 350 C 100 250, 250 250, 250 350 S 400 450, 400 350",
                "points": self.parse_svg_path("M 100 350 C 100 250, 250 250, 250 350 S 400 450, 400 350")
            },
            "elliptical_arc": {
                "description": "Elliptical arc example",
                "path_data": "M 100 350 A 150 150 0 1 0 400 350",
                "points": self.parse_svg_path("M 100 350 A 150 150 0 1 0 400 350")
            },
            "combined_path": {
                "description": "Path with multiple command types",
                "path_data": "M 100 350 L 150 200 H 250 V 100 Q 300 50, 350 100 T 400 200 C 450 250, 450 350, 400 350 Z",
                "points": self.parse_svg_path("M 100 350 L 150 200 H 250 V 100 Q 300 50, 350 100 T 400 200 C 450 250, 450 350, 400 350 Z")
            }
        }
        
        return examples


# Command-line interface
if __name__ == "__main__":
    import argparse
    import os
    import sys
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract polygon coordinates from SVG files')
    parser.add_argument('input_file', nargs='?', help='Path to the SVG file to process')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed information')
    parser.add_argument('-c', '--curve-samples', type=int, default=10, 
                        help='Number of sample points for curve commands (default: 10)')
    parser.add_argument('--examples', action='store_true', help='Process SVG path examples')
    
    args = parser.parse_args()
    
    # Process examples if requested
    if args.examples:
        print("Processing SVG path examples...")
        extractor = SVGPolygonExtractor()
        examples = extractor.extract_examples()
        
        for name, example in examples.items():
            print(f"\nExample: {name}")
            print(f"Description: {example['description']}")
            print(f"Path data: {example['path_data']}")
            print(f"Points ({len(example['points'])}): ")
            for i, point in enumerate(example['points']):
                if i < 5 or i >= len(example['points']) - 5:
                    print(f"  {i}: ({point['x']}, {point['y']})")
                elif i == 5:
                    print("  ...")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2)
            print(f"\nExamples saved to {args.output}")
        
        sys.exit(0)
    
    # If input file is provided via command line
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File {args.input_file} not found.")
            sys.exit(1)
        
        # Set default output filename if not provided
        output_file = args.output if args.output else os.path.splitext(args.input_file)[0] + '_coordinates.json'
        
        print(f"Processing SVG file: {args.input_file}")
        print(f"Curve sample points: {args.curve_samples}")
        extractor = SVGPolygonExtractor(svg_file_path=args.input_file)
        breakpoint()
        result = extractor.process_full_svg(curve_sample_points=args.curve_samples)
        
        # Save to JSON
        extractor.save_result_to_json(output_file, result)
        
        # If verbose mode is enabled, print summary
        if args.verbose:
            summary = result['summary']
            print("\nExtraction summary:")
            print(f"  Found {summary['paths_count']} paths")
            print(f"  Found {summary['rectangles_count']} rectangles")
            print(f"  Found {summary['polygons_count']} polygons")
            print(f"  Found {summary['groups_count']} groups")
            
            # Print a sample of extracted paths
            if result['paths']:
                print("\nSample path coordinates:")
                sample_path = result['paths'][0]
                print(f"  Path ID: {sample_path['id']}")
                print("  First 5 points:")
                for i, point in enumerate(sample_path['points'][:5]):
                    print(f"    ({point['x']}, {point['y']})")
                if len(sample_path['points']) > 5:
                    print(f"    ... and {len(sample_path['points'])-5} more points")
    
    # Interactive mode when no input file is provided
    else:
        print("No input file provided. Enter SVG file path or 'q' to quit:")
        while True:
            file_path = input("> ")
            if file_path.lower() in ('q', 'quit', 'exit'):
                break
                
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found. Try again or enter 'q' to quit.")
                continue
                
            # Process the file
            print(f"Processing SVG file: {file_path}")
            extractor = SVGPolygonExtractor(svg_file_path=file_path)
            result = extractor.process_full_svg(curve_sample_points=args.curve_samples)
            
            # Default output filename
            output_file = os.path.splitext(file_path)[0] + '_coordinates.json'
            output_choice = input(f"Save to {output_file}? (Y/n): ")
            
            if output_choice.lower() not in ('n', 'no'):
                extractor.save_result_to_json(output_file, result)
            else:
                custom_output = input("Enter custom output path: ")
                if custom_output:
                    extractor.save_result_to_json(custom_output, result)
                else:
                    print("Output not saved.")
            
            print("\nProcess another file? Enter path or 'q' to quit:")