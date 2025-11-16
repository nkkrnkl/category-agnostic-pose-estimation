import numpy as np
import matplotlib.pyplot as plt

def compute_centroid(polygon):
    """Compute centroid of a polygon given as list of (x, y)."""
    polygon = np.array(polygon)
    x = np.mean(polygon[:, 0])
    y = np.mean(polygon[:, 1])
    return (x, y)

def get_top_left(polygon):
    return min(polygon, key=lambda p: (p[1], p[0]))  # y ascending, x ascending

# def sort_polygons(polygons):
#     """Sort polygons from top-to-bottom, then left-to-right."""
#     # Step 1: compute centroids
#     # indexed_polygons = [(i, compute_centroid(poly), poly) for i, poly in enumerate(polygons)]
#     indexed_polygons = [(i, get_top_left_corner(poly), poly) for i, poly in enumerate(polygons)]

#     # Step 2: sort by y (top to bottom), then x (left to right)
#     indexed_polygons.sort(key=lambda x: (x[1][1], x[1][0])) # y first, then x

#     # Step 3: return sorted polygons
#     return [poly for _, _, poly in indexed_polygons]


def sort_polygons(polygons, tolerance=20, reverse=False):
    # Step 1: Get top-left corner and original index
    indexed = [(i, get_top_left(p), p) for i, p in enumerate(polygons)]

    # Step 2: Sort by Y (top to bottom)
    indexed.sort(key=lambda x: x[1][1])

    # Step 3: Group into rows
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

    # Step 4: Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda x: x[1][0])  # sort by x

    # Step 5: Flatten and return indices
    sorted_indices = [idx for row in rows for idx, _, _ in row]
    if reverse:
        sorted_indices = sorted_indices[::-1]
    sorted_polygons = [polygons[idx] for idx in sorted_indices]

    return sorted_polygons, sorted_indices


def plot_polygons(polygons, save_path):
    plt.figure(figsize=(6, 6))
    for i, poly in enumerate(polygons):
        poly = np.array(poly)
        plt.fill(poly[:, 0], poly[:, 1], alpha=0.5, label=f'Polygon {i+1}')
        centroid = compute_centroid(poly)
        plt.text(centroid[0], centroid[1], f'C{i+1}', fontsize=10, ha='center')
    # plt.title(title)
    # plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path)
    # plt.show()