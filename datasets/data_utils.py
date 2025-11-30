import numpy as np
import matplotlib.pyplot as plt
def compute_centroid(polygon):
    """
    Compute centroid of a polygon.
    
    Args:
        polygon: List of (x, y) coordinates
    
    Returns:
        Tuple of (x, y) centroid coordinates
    """
    polygon = np.array(polygon)
    x = np.mean(polygon[:, 0])
    y = np.mean(polygon[:, 1])
    return (x, y)
def get_top_left(polygon):
    """
    Get top-left point of a polygon.
    
    Args:
        polygon: List of (x, y) coordinates
    
    Returns:
        Top-left point (x, y)
    """
    return min(polygon, key=lambda p: (p[1], p[0]))
def sort_polygons(polygons, tolerance=20, reverse=False):
    """
    Sort polygons by rows and columns.
    
    Args:
        polygons: List of polygons
        tolerance: Tolerance for grouping into rows
        reverse: Whether to reverse the order
    
    Returns:
        Tuple of (sorted_polygons, sorted_indices)
    """
    indexed = [(i, get_top_left(p), p) for i, p in enumerate(polygons)]
    indexed.sort(key=lambda x: x[1][1])
    rows = []
    for idx, corner, poly in indexed:
        y = corner[1]
        added = False
        for row in rows:
            if abs(row[0][1][1] - y) <= tolerance:
                row.append((idx, corner, poly))
                added = True
                break
        if not added:
            rows.append([(idx, corner, poly)])
    for row in rows:
        row.sort(key=lambda x: x[1][0])
    sorted_indices = [idx for row in rows for idx, _, _ in row]
    if reverse:
        sorted_indices = sorted_indices[::-1]
    sorted_polygons = [polygons[idx] for idx in sorted_indices]
    return sorted_polygons, sorted_indices
def plot_polygons(polygons, save_path):
    """
    Plot polygons and save to file.
    
    Args:
        polygons: List of polygons
        save_path: Path to save the plot
    """
    plt.figure(figsize=(6, 6))
    for i, poly in enumerate(polygons):
        poly = np.array(poly)
        plt.fill(poly[:, 0], poly[:, 1], alpha=0.5, label=f'Polygon {i+1}')
        centroid = compute_centroid(poly)
        plt.text(centroid[0], centroid[1], f'C{i+1}', fontsize=10, ha='center')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path)