import re
import xml.etree.ElementTree as ET
import json
import math
import numpy as np
from collections import defaultdict

class EnhancedSVGParser:
    """
    Enhanced parser for SVG files, with special handling for architectural floor plans.
    Properly handles SVG path commands, transformations, styles, and nested groups.
    """
    
    def __init__(self, svg_file_path=None, svg_string=None):
        """
        Initialize with either a file path or an SVG string.
        
        Args:
            svg_file_path (str, optional): Path to the SVG file
            svg_string (str, optional): SVG content as a string
        """
        self.svg_content = None
        self.root = None
        self.namespaces = {
            'svg': 'http://www.w3.org/2000/svg',
            'xlink': 'http://www.w3.org/1999/xlink',
            'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd',
            'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
        }
        
        # Track transformations
        self.transform_stack = []
        
        # Dictionary to store all elements by type
        self.elements = defaultdict(list)
        
        # Parse the SVG content
        if svg_file_path:
            with open(svg_file_path, 'r', encoding='utf-8') as file:
                self.svg_content = file.read()
        elif svg_string:
            self.svg_content = svg_string
        
        if self.svg_content:
            # Register namespaces to preserve them in output
            for prefix, uri in self.namespaces.items():
                ET.register_namespace(prefix, uri)
            
            # Parse the XML
            self.root = ET.fromstring(self.svg_content)
            
            # Extract document properties
            self.viewbox = self._extract_viewbox()
            self.width, self.height = self._extract_dimensions()
            
            # Extract all element definitions
            self.defs = self._extract_defs()
    
    def _extract_viewbox(self):
        """Extract viewBox information from SVG root element"""
        if self.root is None:
            return None
            
        viewbox = self.root.get('viewBox')
        if viewbox:
            return [float(val) for val in viewbox.split()]
        return None
    
    def _extract_dimensions(self):
        """Extract width and height from SVG root element"""
        if self.root is None:
            return None, None
            
        width = self.root.get('width')
        height = self.root.get('height')
        
        # Convert to numeric values, stripping units if present
        if width:
            width = float(re.match(r'[\d.]+', width).group(0))
        if height:
            height = float(re.match(r'[\d.]+', height).group(0))
            
        return width, height
    
    def _extract_defs(self):
        """Extract definitions (patterns, gradients, etc.) from the SVG"""
        defs = {}
        
        # Find all definition elements
        def_elements = self.root.findall('.//svg:defs', self.namespaces)
        if not def_elements:
            def_elements = self.root.findall('.//defs')
        
        for defs_elem in def_elements:
            # Process patterns
            for pattern in defs_elem.findall('.//svg:pattern', self.namespaces) or defs_elem.findall('.//pattern'):
                pattern_id = pattern.get('id')
                if pattern_id:
                    defs[f'#{pattern_id}'] = {'type': 'pattern', 'element': pattern}
            
            # Process gradients
            for gradient in defs_elem.findall('.//svg:linearGradient', self.namespaces) or defs_elem.findall('.//linearGradient'):
                gradient_id = gradient.get('id')
                if gradient_id:
                    defs[f'#{gradient_id}'] = {'type': 'linearGradient', 'element': gradient}
            
            for gradient in defs_elem.findall('.//svg:radialGradient', self.namespaces) or defs_elem.findall('.//radialGradient'):
                gradient_id = gradient.get('id')
                if gradient_id:
                    defs[f'#{gradient_id}'] = {'type': 'radialGradient', 'element': gradient}
        
        return defs
    
    def _parse_transform(self, transform_str):
        """
        Parse SVG transform attribute into a transformation matrix.
        
        Args:
            transform_str (str): SVG transform attribute value
            
        Returns:
            list: 3x3 transformation matrix as a list of lists
        """
        if not transform_str:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Identity matrix
        
        # Initialize with identity matrix
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        # Regular expressions to match transform functions
        matrix_re = r'matrix\(\s*([^,\s]+)\s*,\s*([^,\s]+)\s*,\s*([^,\s]+)\s*,\s*([^,\s]+)\s*,\s*([^,\s]+)\s*,\s*([^,\s]+)\s*\)'
        translate_re = r'translate\(\s*([^,\s]+)(?:\s*,\s*([^,\s]+))?\s*\)'
        scale_re = r'scale\(\s*([^,\s]+)(?:\s*,\s*([^,\s]+))?\s*\)'
        rotate_re = r'rotate\(\s*([^,\s]+)(?:\s*,\s*([^,\s]+)\s*,\s*([^,\s]+))?\s*\)'
        skewX_re = r'skewX\(\s*([^,\s]+)\s*\)'
        skewY_re = r'skewY\(\s*([^,\s]+)\s*\)'
        
        # Find all transform functions
        matches = []
        for pattern in [matrix_re, translate_re, scale_re, rotate_re, skewX_re, skewY_re]:
            matches.extend(re.finditer(pattern, transform_str))
        
        # Sort matches by position in string to preserve order
        matches.sort(key=lambda m: m.start())
        
        # Process each transform function
        for match in matches:
            transform_func = match.group(0)
            
            if transform_func.startswith('matrix'):
                # Matrix transform: matrix(a, b, c, d, e, f)
                a, b, c, d, e, f = map(float, match.groups())
                transform_matrix = [[a, c, e], [b, d, f], [0, 0, 1]]
                matrix = self._multiply_matrices(matrix, transform_matrix)
                
            elif transform_func.startswith('translate'):
                # Translate transform: translate(tx [, ty])
                tx = float(match.group(1))
                ty = float(match.group(2)) if match.group(2) else 0
                transform_matrix = [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
                matrix = self._multiply_matrices(matrix, transform_matrix)
                
            elif transform_func.startswith('scale'):
                # Scale transform: scale(sx [, sy])
                sx = float(match.group(1))
                sy = float(match.group(2)) if match.group(2) else sx
                transform_matrix = [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]
                matrix = self._multiply_matrices(matrix, transform_matrix)
                
            elif transform_func.startswith('rotate'):
                # Rotate transform: rotate(angle [, cx, cy])
                angle = float(match.group(1)) * math.pi / 180  # Convert to radians
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                
                if match.group(2) and match.group(3):
                    # Rotation around a specific point
                    cx, cy = float(match.group(2)), float(match.group(3))
                    transform_matrix = [
                        [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
                        [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
                        [0, 0, 1]
                    ]
                else:
                    # Rotation around the origin
                    transform_matrix = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
                    
                matrix = self._multiply_matrices(matrix, transform_matrix)
                
            elif transform_func.startswith('skewX'):
                # SkewX transform: skewX(angle)
                angle = float(match.group(1)) * math.pi / 180  # Convert to radians
                transform_matrix = [[1, math.tan(angle), 0], [0, 1, 0], [0, 0, 1]]
                matrix = self._multiply_matrices(matrix, transform_matrix)
                
            elif transform_func.startswith('skewY'):
                # SkewY transform: skewY(angle)
                angle = float(match.group(1)) * math.pi / 180  # Convert to radians
                transform_matrix = [[1, 0, 0], [math.tan(angle), 1, 0], [0, 0, 1]]
                matrix = self._multiply_matrices(matrix, transform_matrix)
        
        return matrix
    
    def _multiply_matrices(self, a, b):
        """
        Multiply two 3x3 transformation matrices.
        
        Args:
            a (list): First 3x3 matrix as a list of lists
            b (list): Second 3x3 matrix as a list of lists
            
        Returns:
            list: Resulting 3x3 matrix as a list of lists
        """
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result
    
    def _apply_transform(self, points, transform_matrix):
        """
        Apply transformation matrix to a list of points.
        
        Args:
            points (list): List of dictionaries with x,y coordinates
            transform_matrix (list): 3x3 transformation matrix as a list of lists
            
        Returns:
            list: Transformed points
        """
        transformed_points = []
        
        for point in points:
            x, y = point['x'], point['y']
            
            # Apply transformation
            new_x = transform_matrix[0][0] * x + transform_matrix[0][1] * y + transform_matrix[0][2]
            new_y = transform_matrix[1][0] * x + transform_matrix[1][1] * y + transform_matrix[1][2]
            
            transformed_points.append({'x': new_x, 'y': new_y})
        
        return transformed_points
    
    def _extract_style(self, element):
        """
        Extract style information from an SVG element.
        
        Args:
            element: SVG element
            
        Returns:
            dict: Style information
        """
        style = {}
        
        # Extract style attributes
        for attr in ['fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'opacity',
                     'fill-opacity', 'stroke-opacity', 'display', 'visibility']:
            if element.get(attr):
                style[attr] = element.get(attr)
        
        # Process style attribute
        style_attr = element.get('style')
        if style_attr:
            for item in style_attr.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    style[key.strip()] = value.strip()
        
        return style
    
    def _normalize_path_data(self, path_data):
        """
        Normalize SVG path data for easier parsing.
        
        Args:
            path_data (str): SVG path data
            
        Returns:
            str: Normalized path data
        """
        # Remove newlines and tabs
        path_data = path_data.replace('\n', ' ').replace('\t', ' ')
        
        # Add spaces around path commands
        for cmd in 'MmLlHhVvCcSsQqTtAaZz':
            path_data = path_data.replace(cmd, f" {cmd} ")
        
        # Replace commas with spaces
        path_data = path_data.replace(',', ' ')
        
        # Replace multiple spaces with a single space
        path_data = re.sub(r'\s+', ' ', path_data).strip()
        
        return path_data
    
    def parse_path(self, path_data, transform_matrix=None, curve_sample_points=30):
        """
        Parse SVG path data into a list of points.
        
        Args:
            path_data (str): SVG path data
            transform_matrix (list, optional): Transformation matrix to apply
            curve_sample_points (int): Number of points to sample for curve commands
            
        Returns:
            list: List of dictionaries with x,y coordinates
        """
        if not path_data or not path_data.strip():
            return []
        
        # Normalize path data
        path_data = self._normalize_path_data(path_data)
        
        # Split into command and parameter tokens
        tokens = path_data.split()
        if not tokens:
            return []
        
        # Initialize variables
        points = []
        current_x, current_y = 0, 0
        subpath_start_x, subpath_start_y = 0, 0
        last_control_x, last_control_y = None, None
        last_command = None
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            i += 1
            
            # Skip empty tokens
            if not token:
                continue
            
            # Check if token is a command
            if token in 'MmLlHhVvCcSsQqTtAaZz':
                command = token
                
                # Process command
                if command == 'M':  # Absolute moveto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        x = float(tokens[i])
                        i += 1
                        if i < len(tokens) and self._is_numeric(tokens[i]):
                            y = float(tokens[i])
                            i += 1
                            
                            current_x, current_y = x, y
                            subpath_start_x, subpath_start_y = x, y
                            points.append({'x': x, 'y': y})
                            
                            # If there are more coordinates, treat as L command
                            command = 'L'
                        else:
                            break
                
                elif command == 'm':  # Relative moveto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        dx = float(tokens[i])
                        i += 1
                        if i < len(tokens) and self._is_numeric(tokens[i]):
                            dy = float(tokens[i])
                            i += 1
                            
                            # First 'm' command is treated as absolute if it's the first command
                            if not points:
                                current_x, current_y = dx, dy
                            else:
                                current_x += dx
                                current_y += dy
                                
                            subpath_start_x, subpath_start_y = current_x, current_y
                            points.append({'x': current_x, 'y': current_y})
                            
                            # If there are more coordinates, treat as l command
                            command = 'l'
                        else:
                            break
                
                elif command == 'L':  # Absolute lineto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        x = float(tokens[i])
                        i += 1
                        if i < len(tokens) and self._is_numeric(tokens[i]):
                            y = float(tokens[i])
                            i += 1
                            
                            current_x, current_y = x, y
                            points.append({'x': x, 'y': y})
                        else:
                            break
                
                elif command == 'l':  # Relative lineto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        dx = float(tokens[i])
                        i += 1
                        if i < len(tokens) and self._is_numeric(tokens[i]):
                            dy = float(tokens[i])
                            i += 1
                            
                            current_x += dx
                            current_y += dy
                            points.append({'x': current_x, 'y': current_y})
                        else:
                            break
                
                elif command == 'H':  # Absolute horizontal lineto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        x = float(tokens[i])
                        i += 1
                        
                        current_x = x
                        points.append({'x': current_x, 'y': current_y})
                
                elif command == 'h':  # Relative horizontal lineto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        dx = float(tokens[i])
                        i += 1
                        
                        current_x += dx
                        points.append({'x': current_x, 'y': current_y})
                
                elif command == 'V':  # Absolute vertical lineto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        y = float(tokens[i])
                        i += 1
                        
                        current_y = y
                        points.append({'x': current_x, 'y': current_y})
                
                elif command == 'v':  # Relative vertical lineto
                    while i < len(tokens) and self._is_numeric(tokens[i]):
                        dy = float(tokens[i])
                        i += 1
                        
                        current_y += dy
                        points.append({'x': current_x, 'y': current_y})
                
                elif command == 'C':  # Absolute cubic Bézier curve
                    while i + 5 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(6)):
                        x1 = float(tokens[i])
                        y1 = float(tokens[i+1])
                        x2 = float(tokens[i+2])
                        y2 = float(tokens[i+3])
                        x = float(tokens[i+4])
                        y = float(tokens[i+5])
                        i += 6
                        
                        # Save control point for potential S command
                        last_control_x, last_control_y = x2, y2
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Cubic Bézier formula
                            px = (1-t)**3 * current_x + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x
                            py = (1-t)**3 * current_y + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 'c':  # Relative cubic Bézier curve
                    while i + 5 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(6)):
                        dx1 = float(tokens[i])
                        dy1 = float(tokens[i+1])
                        dx2 = float(tokens[i+2])
                        dy2 = float(tokens[i+3])
                        dx = float(tokens[i+4])
                        dy = float(tokens[i+5])
                        i += 6
                        
                        x1 = current_x + dx1
                        y1 = current_y + dy1
                        x2 = current_x + dx2
                        y2 = current_y + dy2
                        x = current_x + dx
                        y = current_y + dy
                        
                        # Save control point for potential s command
                        last_control_x, last_control_y = x2, y2
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Cubic Bézier formula
                            px = (1-t)**3 * current_x + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x
                            py = (1-t)**3 * current_y + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 'S':  # Absolute smooth cubic Bézier curve
                    while i + 3 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(4)):
                        x2 = float(tokens[i])
                        y2 = float(tokens[i+1])
                        x = float(tokens[i+2])
                        y = float(tokens[i+3])
                        i += 4
                        
                        # Calculate first control point as reflection of previous curve's second control point
                        if last_command in 'CcSs' and last_control_x is not None:
                            x1 = 2 * current_x - last_control_x
                            y1 = 2 * current_y - last_control_y
                        else:
                            x1, y1 = current_x, current_y
                        
                        # Save control point for potential S command
                        last_control_x, last_control_y = x2, y2
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Cubic Bézier formula
                            px = (1-t)**3 * current_x + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x
                            py = (1-t)**3 * current_y + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 's':  # Relative smooth cubic Bézier curve
                    while i + 3 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(4)):
                        dx2 = float(tokens[i])
                        dy2 = float(tokens[i+1])
                        dx = float(tokens[i+2])
                        dy = float(tokens[i+3])
                        i += 4
                        
                        x2 = current_x + dx2
                        y2 = current_y + dy2
                        x = current_x + dx
                        y = current_y + dy
                        
                        # Calculate first control point as reflection of previous curve's second control point
                        if last_command in 'CcSs' and last_control_x is not None:
                            x1 = 2 * current_x - last_control_x
                            y1 = 2 * current_y - last_control_y
                        else:
                            x1, y1 = current_x, current_y
                        
                        # Save control point for potential s command
                        last_control_x, last_control_y = x2, y2
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Cubic Bézier formula
                            px = (1-t)**3 * current_x + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x
                            py = (1-t)**3 * current_y + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 'Q':  # Absolute quadratic Bézier curve
                    while i + 3 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(4)):
                        x1 = float(tokens[i])
                        y1 = float(tokens[i+1])
                        x = float(tokens[i+2])
                        y = float(tokens[i+3])
                        i += 4
                        
                        # Save control point for potential T command
                        last_control_x, last_control_y = x1, y1
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Quadratic Bézier formula
                            px = (1-t)**2 * current_x + 2*(1-t)*t * x1 + t**2 * x
                            py = (1-t)**2 * current_y + 2*(1-t)*t * y1 + t**2 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 'q':  # Relative quadratic Bézier curve
                    while i + 3 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(4)):
                        dx1 = float(tokens[i])
                        dy1 = float(tokens[i+1])
                        dx = float(tokens[i+2])
                        dy = float(tokens[i+3])
                        i += 4
                        
                        x1 = current_x + dx1
                        y1 = current_y + dy1
                        x = current_x + dx
                        y = current_y + dy
                        
                        # Save control point for potential t command
                        last_control_x, last_control_y = x1, y1
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Quadratic Bézier formula
                            px = (1-t)**2 * current_x + 2*(1-t)*t * x1 + t**2 * x
                            py = (1-t)**2 * current_y + 2*(1-t)*t * y1 + t**2 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 'T':  # Absolute smooth quadratic Bézier curve
                    while i + 1 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(2)):
                        x = float(tokens[i])
                        y = float(tokens[i+1])
                        i += 2
                        
                        # Calculate control point as reflection of previous curve's control point
                        if last_command in 'QqTt' and last_control_x is not None:
                            x1 = 2 * current_x - last_control_x
                            y1 = 2 * current_y - last_control_y
                        else:
                            x1, y1 = current_x, current_y
                        
                        # Save control point for potential t command
                        last_control_x, last_control_y = x1, y1
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Quadratic Bézier formula
                            px = (1-t)**2 * current_x + 2*(1-t)*t * x1 + t**2 * x
                            py = (1-t)**2 * current_y + 2*(1-t)*t * y1 + t**2 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 'A':  # Absolute elliptical arc
                    while i + 6 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(7)):
                        rx = float(tokens[i])
                        ry = float(tokens[i+1])
                        x_axis_rotation = float(tokens[i+2])
                        large_arc_flag = int(float(tokens[i+3]))
                        sweep_flag = int(float(tokens[i+4]))
                        x = float(tokens[i+5])
                        y = float(tokens[i+6])
                        i += 7
                        
                        # Generate points along the arc
                        arc_points = self._elliptical_arc_to_points(
                            current_x, current_y, rx, ry, x_axis_rotation,
                            large_arc_flag, sweep_flag, x, y, curve_sample_points
                        )
                        
                        points.extend(arc_points)
                        current_x, current_y = x, y
                
                elif command == 'a':  # Relative elliptical arc
                    while i + 6 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(7)):
                        rx = float(tokens[i])
                        ry = float(tokens[i+1])
                        x_axis_rotation = float(tokens[i+2])
                        large_arc_flag = int(float(tokens[i+3]))
                        sweep_flag = int(float(tokens[i+4]))
                        dx = float(tokens[i+5])
                        dy = float(tokens[i+6])
                        i += 7
                        
                        x = current_x + dx
                        y = current_y + dy
                        
                        # Generate points along the arc
                        arc_points = self._elliptical_arc_to_points(
                            current_x, current_y, rx, ry, x_axis_rotation,
                            large_arc_flag, sweep_flag, x, y, curve_sample_points
                        )
                        
                        points.extend(arc_points)
                        current_x, current_y = x, y
                
                elif command in ['Z', 'z']:  # Close path
                    if points and (current_x != subpath_start_x or current_y != subpath_start_y):
                        points.append({'x': subpath_start_x, 'y': subpath_start_y})
                        current_x, current_y = subpath_start_x, subpath_start_y
                
                last_command = command
            
            else:
                # Not a command, so it must be a number following a previous command
                i -= 1  # Back up so we can read it again as a parameter
        
        # Apply transformation if provided
        if transform_matrix:
            points = self._apply_transform(points, transform_matrix)
        
        return points
    
    def _is_numeric(self, value):
        """Check if a string can be converted to a float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _elliptical_arc_to_points(self, x1, y1, rx, ry, phi_degrees, large_arc, sweep, x2, y2, num_points=20):
        """
        Convert an SVG elliptical arc to a series of points.
        
        Args:
            x1, y1: Starting point
            rx, ry: Radii of the ellipse
            phi_degrees: Rotation of the ellipse in degrees
            large_arc: Large arc flag (0 or 1)
            sweep: Sweep flag (0 or 1)
            x2, y2: End point
            num_points: Number of points to generate
            
        Returns:
            list: List of dictionaries with x,y coordinates
        """
        # Ensure rx and ry are positive
        rx, ry = abs(rx), abs(ry)
        
        # Handle edge cases
        if rx == 0 or ry == 0:
            return [{'x': x2, 'y': y2}]
        
        # Convert angle to radians
        phi = phi_degrees * math.pi / 180.0
        
        # Step 1: Transform to the origin
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        # Inverse transform 1
        x1p = cos_phi * (x1 - x2) / 2 + sin_phi * (y1 - y2) / 2
        y1p = -sin_phi * (x1 - x2) / 2 + cos_phi * (y1 - y2) / 2
        
        # Ensure radii are large enough
        lambda_value = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
        if lambda_value > 1:
            rx *= math.sqrt(lambda_value)
            ry *= math.sqrt(lambda_value)
        
        # Step 2: Compute center parameters
        sign = -1 if large_arc == sweep else 1
        temp = (rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p) / (rx * rx * y1p * y1p + ry * ry * x1p * x1p)
        temp = max(0, temp)  # Ensure non-negative
        radicand = temp
        factor = sign * math.sqrt(radicand)
        
        cxp = factor * rx * y1p / ry
        cyp = -factor * ry * x1p / rx
        
        # Step 3: Transform back
        cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
        cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2
        
        # Step 4: Compute the angle range
        ux = (x1p - cxp) / rx
        uy = (y1p - cyp) / ry
        vx = (-x1p - cxp) / rx
        vy = (-y1p - cyp) / ry
        
        # Compute the angle start
        n = math.sqrt(ux * ux + uy * uy)
        p = ux  # (1 * ux) + (0 * uy)
        theta = math.acos(p / n)
        if uy < 0:
            theta = -theta
        
        start_angle = theta
        
        # Compute the angle extent
        n1 = math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
        p1 = ux * vx + uy * vy
        if n1 != 0:
            theta1 = math.acos(p1 / n1)
        else:
            theta1 = 0
        
        if (ux * vy - uy * vx) < 0:
            theta1 = -theta1
        
        angle_extent = theta1
        if not sweep and angle_extent > 0:
            angle_extent -= 2 * math.pi
        elif sweep and angle_extent < 0:
            angle_extent += 2 * math.pi
        
        # Generate points along the arc
        points = []
        for i in range(1, num_points):  # Skip the first point (it's the current point)
            t = start_angle + (i / num_points) * angle_extent
            
            # Compute point on the ellipse
            x = rx * math.cos(t)
            y = ry * math.sin(t)
            
            # Rotate and translate back
            px = cos_phi * x - sin_phi * y + cx
            py = sin_phi * x + cos_phi * y + cy
            
            points.append({'x': px, 'y': py})
        
        return points
    
    def parse_svg(self, curve_sample_points=30):
        """
        Parse the SVG file and extract all elements.
        
        Args:
            curve_sample_points (int): Number of points to sample for curve commands
            
        Returns:
            dict: Dictionary with extracted SVG elements
        """
        if not self.root:
            return {'error': 'No SVG data available'}
        
        # Extract all elements
        all_elements = {
            'paths': [],
            'polygons': [],
            'polylines': [],
            'rects': [],
            'circles': [],
            'ellipses': [],
            'lines': [],
            'groups': [],
            'viewBox': self.viewbox,
            'width': self.width,
            'height': self.height
        }
        
        # Process paths
        for path_elem in self.root.findall('.//svg:path', self.namespaces) or self.root.findall('.//path'):
            path_data = path_elem.get('d', '')
            if not path_data:
                continue
            
            # Get transform if available
            transform_str = path_elem.get('transform', '')
            transform_matrix = self._parse_transform(transform_str) if transform_str else None
            
            # Extract style information
            style = self._extract_style(path_elem)
            
            # Get ID if available
            path_id = path_elem.get('id', f'path_{len(all_elements["paths"])}')
            
            # Parse path data
            points = self.parse_path(path_data, transform_matrix, curve_sample_points)
            
            # Add to elements
            all_elements['paths'].append({
                'id': path_id,
                'path_data': path_data,
                'points': points,
                'style': style,
                'transform': transform_str
            })
        
        # Process rectangles
        for rect_elem in self.root.findall('.//svg:rect', self.namespaces) or self.root.findall('.//rect'):
            try:
                x = float(rect_elem.get('x', '0'))
                y = float(rect_elem.get('y', '0'))
                width = float(rect_elem.get('width', '0'))
                height = float(rect_elem.get('height', '0'))
                rx = float(rect_elem.get('rx', '0'))
                ry = float(rect_elem.get('ry', '0'))
                
                # Get transform if available
                transform_str = rect_elem.get('transform', '')
                transform_matrix = self._parse_transform(transform_str) if transform_str else None
                
                # Extract style information
                style = self._extract_style(rect_elem)
                
                # Get ID if available
                rect_id = rect_elem.get('id', f'rect_{len(all_elements["rects"])}')
                
                # Create points for the rectangle
                if rx <= 0 and ry <= 0:
                    # Regular rectangle
                    points = [
                        {'x': x, 'y': y},
                        {'x': x + width, 'y': y},
                        {'x': x + width, 'y': y + height},
                        {'x': x, 'y': y + height},
                        {'x': x, 'y': y}  # Close the path
                    ]
                else:
                    # Rounded rectangle (approximated with lines and arcs)
                    if rx <= 0:
                        rx = ry
                    if ry <= 0:
                        ry = rx
                    
                    # Ensure rx and ry are not larger than half width/height
                    rx = min(rx, width / 2)
                    ry = min(ry, height / 2)
                    
                    # Create path data for rounded rectangle
                    path_data = f"M {x+rx} {y} "
                    path_data += f"L {x+width-rx} {y} "
                    path_data += f"A {rx} {ry} 0 0 1 {x+width} {y+ry} "
                    path_data += f"L {x+width} {y+height-ry} "
                    path_data += f"A {rx} {ry} 0 0 1 {x+width-rx} {y+height} "
                    path_data += f"L {x+rx} {y+height} "
                    path_data += f"A {rx} {ry} 0 0 1 {x} {y+height-ry} "
                    path_data += f"L {x} {y+ry} "
                    path_data += f"A {rx} {ry} 0 0 1 {x+rx} {y} "
                    path_data += "Z"
                    
                    # Parse the path data
                    points = self.parse_path(path_data, transform_matrix, curve_sample_points)
                else:
                    # Apply transformation if available
                    if transform_matrix:
                        points = self._apply_transform(points, transform_matrix)
                
                # Add to elements
                all_elements['rects'].append({
                    'id': rect_id,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'rx': rx,
                    'ry': ry,
                    'points': points,
                    'style': style,
                    'transform': transform_str
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing rectangle: {e}")
        
        # Process circles
        for circle_elem in self.root.findall('.//svg:circle', self.namespaces) or self.root.findall('.//circle'):
            try:
                cx = float(circle_elem.get('cx', '0'))
                cy = float(circle_elem.get('cy', '0'))
                r = float(circle_elem.get('r', '0'))
                
                # Get transform if available
                transform_str = circle_elem.get('transform', '')
                transform_matrix = self._parse_transform(transform_str) if transform_str else None
                
                # Extract style information
                style = self._extract_style(circle_elem)
                
                # Get ID if available
                circle_id = circle_elem.get('id', f'circle_{len(all_elements["circles"])}')
                
                # Create points approximating the circle
                points = []
                for i in range(curve_sample_points):
                    angle = 2 * math.pi * i / curve_sample_points
                    x = cx + r * math.cos(angle)
                    y = cy + r * math.sin(angle)
                    points.append({'x': x, 'y': y})
                
                # Close the path
                points.append(points[0].copy())
                
                # Apply transformation if available
                if transform_matrix:
                    points = self._apply_transform(points, transform_matrix)
                
                # Add to elements
                all_elements['circles'].append({
                    'id': circle_id,
                    'cx': cx,
                    'cy': cy,
                    'r': r,
                    'points': points,
                    'style': style,
                    'transform': transform_str
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing circle: {e}")
        
        # Process ellipses
        for ellipse_elem in self.root.findall('.//svg:ellipse', self.namespaces) or self.root.findall('.//ellipse'):
            try:
                cx = float(ellipse_elem.get('cx', '0'))
                cy = float(ellipse_elem.get('cy', '0'))
                rx = float(ellipse_elem.get('rx', '0'))
                ry = float(ellipse_elem.get('ry', '0'))
                
                # Get transform if available
                transform_str = ellipse_elem.get('transform', '')
                transform_matrix = self._parse_transform(transform_str) if transform_str else None
                
                # Extract style information
                style = self._extract_style(ellipse_elem)
                
                # Get ID if available
                ellipse_id = ellipse_elem.get('id', f'ellipse_{len(all_elements["ellipses"])}')
                
                # Create points approximating the ellipse
                points = []
                for i in range(curve_sample_points):
                    angle = 2 * math.pi * i / curve_sample_points
                    x = cx + rx * math.cos(angle)
                    y = cy + ry * math.sin(angle)
                    points.append({'x': x, 'y': y})
                
                # Close the path
                points.append(points[0].copy())
                
                # Apply transformation if available
                if transform_matrix:
                    points = self._apply_transform(points, transform_matrix)
                
                # Add to elements
                all_elements['ellipses'].append({
                    'id': ellipse_id,
                    'cx': cx,
                    'cy': cy,
                    'rx': rx,
                    'ry': ry,
                    'points': points,
                    'style': style,
                    'transform': transform_str
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing ellipse: {e}")
        
        # Process lines
        for line_elem in self.root.findall('.//svg:line', self.namespaces) or self.root.findall('.//line'):
            try:
                x1 = float(line_elem.get('x1', '0'))
                y1 = float(line_elem.get('y1', '0'))
                x2 = float(line_elem.get('x2', '0'))
                y2 = float(line_elem.get('y2', '0'))
                
                # Get transform if available
                transform_str = line_elem.get('transform', '')
                transform_matrix = self._parse_transform(transform_str) if transform_str else None
                
                # Extract style information
                style = self._extract_style(line_elem)
                
                # Get ID if available
                line_id = line_elem.get('id', f'line_{len(all_elements["lines"])}')
                
                # Create points for the line
                points = [
                    {'x': x1, 'y': y1},
                    {'x': x2, 'y': y2}
                ]
                
                # Apply transformation if available
                if transform_matrix:
                    points = self._apply_transform(points, transform_matrix)
                
                # Add to elements
                all_elements['lines'].append({
                    'id': line_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'points': points,
                    'style': style,
                    'transform': transform_str
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing line: {e}")
        
        # Process polylines
        for polyline_elem in self.root.findall('.//svg:polyline', self.namespaces) or self.root.findall('.//polyline'):
            try:
                points_str = polyline_elem.get('points', '')
                if not points_str:
                    continue
                
                # Parse points
                point_pairs = re.findall(r'([-+]?[\d\.]+)[,\s]+([-+]?[\d\.]+)', points_str)
                if not point_pairs:
                    continue
                
                # Get transform if available
                transform_str = polyline_elem.get('transform', '')
                transform_matrix = self._parse_transform(transform_str) if transform_str else None
                
                # Extract style information
                style = self._extract_style(polyline_elem)
                
                # Get ID if available
                polyline_id = polyline_elem.get('id', f'polyline_{len(all_elements["polylines"])}')
                
                # Create points
                points = [{'x': float(x), 'y': float(y)} for x, y in point_pairs]
                
                # Apply transformation if available
                if transform_matrix:
                    points = self._apply_transform(points, transform_matrix)
                
                # Add to elements
                all_elements['polylines'].append({
                    'id': polyline_id,
                    'points_str': points_str,
                    'points': points,
                    'style': style,
                    'transform': transform_str
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing polyline: {e}")
        
        # Process polygons
        for polygon_elem in self.root.findall('.//svg:polygon', self.namespaces) or self.root.findall('.//polygon'):
            try:
                points_str = polygon_elem.get('points', '')
                if not points_str:
                    continue
                
                # Parse points
                point_pairs = re.findall(r'([-+]?[\d\.]+)[,\s]+([-+]?[\d\.]+)', points_str)
                if not point_pairs:
                    continue
                
                # Get transform if available
                transform_str = polygon_elem.get('transform', '')
                transform_matrix = self._parse_transform(transform_str) if transform_str else None
                
                # Extract style information
                style = self._extract_style(polygon_elem)
                
                # Get ID if available
                polygon_id = polygon_elem.get('id', f'polygon_{len(all_elements["polygons"])}')
                
                # Create points
                points = [{'x': float(x), 'y': float(y)} for x, y in point_pairs]
                
                # Close the polygon
                if points and (points[0]['x'] != points[-1]['x'] or points[0]['y'] != points[-1]['y']):
                    points.append(points[0].copy())
                
                # Apply transformation if available
                if transform_matrix:
                    points = self._apply_transform(points, transform_matrix)
                
                # Add to elements
                all_elements['polygons'].append({
                    'id': polygon_id,
                    'points_str': points_str,
                    'points': points,
                    'style': style,
                    'transform': transform_str
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing polygon: {e}")
        
        # Process groups
        self._process_groups(self.root, all_elements, curve_sample_points)
        
        return all_elements
    
    def _process_groups(self, parent_elem, all_elements, curve_sample_points, parent_transform=None):
        """
        Process group elements recursively.
        
        Args:
            parent_elem: Parent XML element
            all_elements: Dictionary to store results
            curve_sample_points: Number of points to sample for curve commands
            parent_transform: Parent transformation matrix
            
        Returns:
            list: List of group elements
        """
        # Find all group elements
        group_elems = parent_elem.findall('.//svg:g', self.namespaces) or parent_elem.findall('.//g')
        
        # Process each group
        for i, group_elem in enumerate(group_elems):
            # Skip if this group is a child of another group that will be processed later
            if group_elem.getparent() != parent_elem and group_elem.getparent() in group_elems:
                continue
            
            # Get transform
            transform_str = group_elem.get('transform', '')
            transform_matrix = self._parse_transform(transform_str) if transform_str else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
            # Combine with parent transform if any
            if parent_transform:
                transform_matrix = self._multiply_matrices(parent_transform, transform_matrix)
            
            # Get style information
            style = self._extract_style(group_elem)
            
            # Get ID or label
            group_id = group_elem.get('id', '')
            group_label = group_elem.get(f'{{{self.namespaces["inkscape"]}}}label', '') if 'inkscape' in self.namespaces else ''
            
            if not group_id:
                group_id = f'group_{len(all_elements["groups"])}'
            
            # Group elements
            group_elements = {
                'id': group_id,
                'label': group_label,
                'style': style,
                'transform': transform_str,
                'paths': [],
                'rects': [],
                'circles': [],
                'ellipses': [],
                'lines': [],
                'polylines': [],
                'polygons': [],
                'groups': []
            }
            
            # Process child elements
            for child in group_elem:
                tag = child.tag
                if '}' in tag:
                    tag = tag.split('}', 1)[1]  # Remove namespace prefix
                
                if tag == 'path':
                    path_data = child.get('d', '')
                    if not path_data:
                        continue
                    
                    # Get child transform
                    child_transform_str = child.get('transform', '')
                    child_transform = self._parse_transform(child_transform_str) if child_transform_str else None
                    
                    # Combine transforms
                    combined_transform = transform_matrix
                    if child_transform:
                        combined_transform = self._multiply_matrices(transform_matrix, child_transform)
                    
                    # Extract style information
                    style = self._extract_style(child)
                    
                    # Get ID if available
                    path_id = child.get('id', f'{group_id}_path_{len(group_elements["paths"])}')
                    
                    # Parse path data
                    points = self.parse_path(path_data, combined_transform, curve_sample_points)
                    
                    # Add to group elements
                    group_elements['paths'].append({
                        'id': path_id,
                        'path_data': path_data,
                        'points': points,
                        'style': style,
                        'transform': child_transform_str
                    })
                
                elif tag == 'rect':
                    try:
                        x = float(child.get('x', '0'))
                        y = float(child.get('y', '0'))
                        width = float(child.get('width', '0'))
                        height = float(child.get('height', '0'))
                        rx = float(child.get('rx', '0'))
                        ry = float(child.get('ry', '0'))
                        
                        # Get child transform
                        child_transform_str = child.get('transform', '')
                        child_transform = self._parse_transform(child_transform_str) if child_transform_str else None
                        
                        # Combine transforms
                        combined_transform = transform_matrix
                        if child_transform:
                            combined_transform = self._multiply_matrices(transform_matrix, child_transform)
                        
                        # Extract style information
                        style = self._extract_style(child)
                        
                        # Get ID if available
                        rect_id = child.get('id', f'{group_id}_rect_{len(group_elements["rects"])}')
                        
                        # Create points for the rectangle
                        if rx <= 0 and ry <= 0:
                            # Regular rectangle
                            points = [
                                {'x': x, 'y': y},
                                {'x': x + width, 'y': y},
                                {'x': x + width, 'y': y + height},
                                {'x': x, 'y': y + height},
                                {'x': x, 'y': y}  # Close the path
                            ]
                            
                            # Apply transformation
                            points = self._apply_transform(points, combined_transform)
                        else:
                            # Rounded rectangle (approximated with lines and arcs)
                            if rx <= 0:
                                rx = ry
                            if ry <= 0:
                                ry = rx
                            
                            # Ensure rx and ry are not larger than half width/height
                            rx = min(rx, width / 2)
                            ry = min(ry, height / 2)
                            
                            # Create path data for rounded rectangle
                            path_data = f"M {x+rx} {y} "
                            path_data += f"L {x+width-rx} {y} "
                            path_data += f"A {rx} {ry} 0 0 1 {x+width} {y+ry} "
                            path_data += f"L {x+width} {y+height-ry} "
                            path_data += f"A {rx} {ry} 0 0 1 {x+width-rx} {y+height} "
                            path_data += f"L {x+rx} {y+height} "
                            path_data += f"A {rx} {ry} 0 0 1 {x} {y+height-ry} "
                            path_data += f"L {x} {y+ry} "
                            path_data += f"A {rx} {ry} 0 0 1 {x+rx} {y} "
                            path_data += "Z"
                            
                            # Parse the path data
                            points = self.parse_path(path_data, combined_transform, curve_sample_points)
                        
                        # Add to group elements
                        group_elements['rects'].append({
                            'id': rect_id,
                            'x': x,
                            'y': y,
                            , current_y
                        
                        # Save control point for potential T command
                        last_control_x, last_control_y = x1, y1
                        
                        # Generate points along the curve
                        for t in np.linspace(0, 1, curve_sample_points):
                            # Quadratic Bézier formula
                            px = (1-t)**2 * current_x + 2*(1-t)*t * x1 + t**2 * x
                            py = (1-t)**2 * current_y + 2*(1-t)*t * y1 + t**2 * y
                            
                            if t > 0:  # Skip t=0 as it's the current point
                                points.append({'x': px, 'y': py})
                        
                        current_x, current_y = x, y
                
                elif command == 't':  # Relative smooth quadratic Bézier curve
                    while i + 1 < len(tokens) and all(self._is_numeric(tokens[i+j]) for j in range(2)):
                        dx = float(tokens[i])
                        dy = float(tokens[i+1])
                        i += 2
                        
                        x = current_x + dx
                        y = current_y + dy
                        
                        # Calculate control point as reflection of previous curve's control point
                        if last_command in 'QqTt' and last_control_x is not None:
                            x1 = 2 * current_x - last_control_x
                            y1 = 2 * current_y - last_control_y
                        else:
                            x1, y1 = current_x