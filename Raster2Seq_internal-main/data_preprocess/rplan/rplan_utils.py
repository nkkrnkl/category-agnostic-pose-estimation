"""
This code is an adaptation that uses Structured 3D for the code base.

Reference: https://github.com/bertjiazheng/Structured3D
"""
import io
import PIL.Image as Image
from PIL import ImageColor
import cv2
import numpy as np
from shapely.geometry import Polygon
import os
import json
import sys

# import drawSvg as drawsvg
import drawsvg
import cairosvg

sys.path.append('../data_preprocess')
from common_utils import resort_corners

type2id = {'living_room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'entrance': 5,
            'dining_room': 6, 'study_room': 7, 'storage': 8, 'front_door': 9, 'unknown': 10, 'interior_door': 11}
id2id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 
    7:6, 8:7, 10:8, 15:9, 16:10, 17:11}

original_id2type = {1:"living_room", 2:"kitchen", 3:"bedroom", 4:"bathroom", 5:"balcony", 6:"entrance",
              7:"dining_room", 8:"study_room", 10:"storage" , 15:"front_door", 16:"unknown", 17:"interior_door"}

ID_COLOR = {0: '#EE4D4D', 1: '#C67C7B', 2: '#FFD274', 3: '#BEBEBE', 4: '#BFE3E8',
            5: '#7BA779', 6: '#E87A90', 7: '#FF8C69', 8: '#1F849B', 9: '#727171',
            10: '#D3A2C7', 11: '#785A67'}


def generate_density(point_cloud, width=256, height=256):

    ps = point_cloud * -1
    ps[:,0] *= -1
    ps[:,1] *= -1

    image_res = np.array((width, height))

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords

    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res


    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    coordinates = \
        np.round(
            (ps[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * image_res[None])
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                image_res - 1)

    density = np.zeros((height, width), dtype=np.float32)

    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    # print(np.unique(counts))
    # counts = np.minimum(counts, 1e2)

    unique_coordinates = unique_coordinates.astype(np.int32)

    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)


    return density, normalization_dict




def normalize_point(point, normalization_dict):

    min_coords = normalization_dict["min_coords"]
    max_coords = normalization_dict["max_coords"]
    image_res = normalization_dict["image_res"]

    point_2d = \
        np.round(
            (point[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * image_res)
    point_2d = np.minimum(np.maximum(point_2d, np.zeros_like(image_res)),
                            image_res - 1)

    point[:2] = point_2d.tolist()

    return point

def normalize_annotations(scene_path, normalization_dict):
    annotation_path = os.path.join(scene_path, "annotation_3d.json")
    with open(annotation_path, "r") as f:
        annotation_json = json.load(f)

    for line in annotation_json["lines"]:
        point = line["point"]
        point = normalize_point(point, normalization_dict)
        line["point"] = point

    for junction in annotation_json["junctions"]:
        point = junction["coordinate"]
        point = normalize_point(point, normalization_dict)
        junction["coordinate"] = point

    return annotation_json

def parse_floor_plan_polys(annos):
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

    # outerwall_floor = []
    # for planeID in outerwall_planes:
    #     lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
    #     lineIDs = np.setdiff1d(lineIDs, lines_holes)
    #     junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
    #     for start, end in junction_pairs:
    #         if start in junction_floor and end in junction_floor:
    #             outerwall_floor.append([start, end])

    # outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    # polygons.append([outerwall_polygon[0], 'outwall'])

    return polygons

def convert_lines_to_vertices(lines):
    """
    convert line representation to polygon vertices

    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


def bbox_to_polygon(x1, y1, x2, y2):
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])

def generate_coco_dict(annos, curr_instance_id, curr_img_id, ignore_types):

    # junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    draw_color = drawsvg.Drawing(256, 256, displayInline=False)
    draw_color.append(drawsvg.Rectangle(0,0,256,256, fill='white'))

    coco_annotation_dict_list = []
    image = np.zeros((256, 256), dtype=np.uint8)

    for poly_ind, (bbox, org_poly_id) in enumerate(zip(annos['boxes'], annos['room_type'])):
        poly_type = original_id2type[org_poly_id]
        if poly_type in ignore_types:
            continue

        # polygon = junctions[np.array(polygon)]

        polygon = bbox_to_polygon(*bbox)
        poly_shapely = Polygon(polygon)
        #TODO: alter y coordinates by 255 - y
        polygon[:, 1] = 255 - polygon[:, 1]
        area = poly_shapely.area
        # print(area, poly_type)

        # assert area > 10
        # if area < 100:
        # if poly_type not in ['door', 'window'] and area < 100:
        #     continue
        # if poly_type in ['door', 'window'] and area < 1:
        #     continue

        draw_color.append(drawsvg.Lines(*polygon.flatten().tolist(), close=True, fill=ID_COLOR[type2id[poly_type]], fill_opacity=1.0, stroke='black', stroke_width=1))
        rectangle_shapely = poly_shapely.envelope

        # ### here we convert door/window annotation into a single line
        # if poly_type in ['door', 'window']:
        #     assert polygon.shape[0] == 4
        #     midp_1 = (polygon[0] + polygon[1])/2
        #     midp_2 = (polygon[1] + polygon[2])/2
        #     midp_3 = (polygon[2] + polygon[3])/2
        #     midp_4 = (polygon[3] + polygon[0])/2

        #     dist_1_3 = np.square(midp_1 -midp_3).sum()
        #     dist_2_4 = np.square(midp_2 -midp_4).sum()
        #     if dist_1_3 > dist_2_4:
        #         polygon = np.row_stack([midp_1, midp_3])
        #     else:
        #         polygon = np.row_stack([midp_2, midp_4])

        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)
        # image = draw_polygon_on_image(image, poly_shapely, "test_poly.jpg")

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bound_pad = 2
        bb_x, bb_y = rectangle_shapely.exterior.xy
        bb_x = np.unique(bb_x)
        bb_y = np.unique(bb_y)
        bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
        bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

        bb_x_max = np.minimum(np.max(bb_x) + bound_pad, 256 - 1)
        bb_y_max = np.minimum(np.max(bb_y) + bound_pad, 256 - 1)

        bb_width = (bb_x_max - bb_x_min)
        bb_height = (bb_y_max - bb_y_min)

        coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

        coco_annotation_dict = {
                "segmentation": [coco_seg_poly],
                "area": area,
                "iscrowd": 0,
                "image_id": curr_img_id,
                "bbox": coco_bb,
                "category_id": type2id[poly_type],
                "id": curr_instance_id}
        
        coco_annotation_dict_list.append(coco_annotation_dict)
        curr_instance_id += 1

    scene_image = Image.open(io.BytesIO(cairosvg.svg2png(draw_color.as_svg())))

    return coco_annotation_dict_list, scene_image


def draw_polygon_on_image(image, polygon, output_path):
    if image is None:
        # Load image
        image = np.zeros((256, 256), dtype=np.uint8)
    
    # Convert Shapely polygon to NumPy array
    pts = np.array(polygon.exterior.coords, dtype=np.int32)

    # Draw polygon on the image
    cv2.polylines(image, [pts], isClosed=True, color=255, thickness=2)

    # Save and show the image
    cv2.imwrite(output_path, image)
    return image