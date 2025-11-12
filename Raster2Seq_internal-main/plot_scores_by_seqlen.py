import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path
import copy

from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from models import build_model
import cv2

from datasets import get_dataset_class_labels

from util.poly_ops import pad_gt_polys
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan, plot_semantic_rich_floorplan_tight, plot_semantic_rich_floorplan_nicely


def plot_gt_floor(data_loader, device, output_dir, plot_gt=True, semantic_rich=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for batched_inputs, _ in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]

        # draw GT map
        if plot_gt:
            for i, gt_inst in enumerate(gt_instances):
                if not semantic_rich:
                    # plot regular room floorplan
                    gt_polys = []
                    density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
                    density_map = np.repeat(density_map, 3, axis=2)

                    gt_corner_map = np.zeros([256, 256, 3])
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        corners = poly[0].reshape(-1, 2)
                        if len(corners) < 3:
                            continue
                        gt_polys.append(corners)
                        
                    gt_room_polys = [np.array(r) for r in gt_polys]
                    gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, scale=1000)
                    cv2.imwrite(os.path.join(output_dir, '{}_floor_nosem.png'.format(scene_ids[i])), gt_floorplan_map)
                else:
                    # plot semantically-rich floorplan
                    gt_sem_rich = []
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        # if gt_inst.gt_classes.cpu().numpy()[j] not in [1, 9, 11]:
                        #     continue
                        corners = poly[0].reshape(-1, 2).astype(np.int32)
                        corners_flip_y = corners.copy()
                        corners_flip_y[:,1] = 255 - corners_flip_y[:,1]
                        corners = corners_flip_y
                        gt_sem_rich.append([corners, gt_inst.gt_classes.cpu().numpy()[j]])

                    gt_sem_rich_path = os.path.join(output_dir, '{}_floor.png'.format(str(scene_ids[i]).zfill(5)))
                    plot_semantic_rich_floorplan_nicely(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1, plot_text=False, is_bw=False,
                                                       door_window_index=[10, 9], 
                                                       img_w=samples[i].shape[2], 
                                                       img_h=samples[i].shape[1],
                                                       semantics_label_mapping=get_dataset_class_labels(args.dataset_name))


def plot_polys(data_loader, device, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for batched_inputs, _ in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        

        for i in range(len(samples)):
            density_map = np.transpose((samples[i]).cpu().numpy(), [1, 2, 0])
            if density_map.shape[2] == 3:
                density_map = density_map * 255
            else:
                density_map = np.repeat(density_map, 3, axis=2) * 255
            pred_room_map = np.zeros(density_map.shape).astype(np.uint8)

            room_polys = gt_instances[i].gt_masks.polygons
            room_ids = gt_instances[i].gt_classes.detach().cpu().numpy()
            for poly, poly_id in zip(room_polys, room_ids):
                poly = poly[0].reshape(-1,2).astype(np.int32)
                pred_room_map = plot_room_map(poly, pred_room_map, poly_id)

            # Blend the overlay with the density map using alpha blending
            alpha = 0.6  # Adjust for desired transparency
            pred_room_map = cv2.addWeighted(density_map.astype(np.uint8), alpha, pred_room_map.astype(np.uint8), 1-alpha, 0)

            # # plot predicted polygon overlaid on the density map
            # pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
            cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), pred_room_map)


def plot_gt_image(data_loader, device, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for batched_inputs, _ in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        

        for i in range(len(samples)):
            density_map = np.transpose((samples[i]).cpu().numpy(), [1, 2, 0])
            if density_map.shape[2] == 3:
                density_map = density_map * 255
            else:
                density_map = np.repeat(density_map, 3, axis=2) * 255

            # # plot predicted polygon overlaid on the density map
            # pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
            cv2.imwrite(os.path.join(output_dir, '{}_gt_image.png'.format(scene_ids[i])), density_map)


def plot_line_graph_combined(cc5k_list, s3d_list, cc5k_bin_sizes, s3d_bin_sizes, method_names=['Ours', 'RoomFormer'], output_path=""):
    # Create the figure
    # fig = go.Figure()
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False,
                        vertical_spacing=0.3,
                        row_heights=[0.5, 0.5],
                        # subplot_titles=("By Polygon Counts", "By Corner Counts", "By Polygon Counts", "By Corner Counts")
                        )

    # x, y, y2
    def get_line_traces(x, ys_list, bin_size, showlegend=True):
        num_bins = max(x) + 1
        bin_edges = [int(i * bin_size) for i in range(num_bins + 1)]
        bin_labels = [f"{bin_edges[i]}–{bin_edges[i+1]}" for i in range(num_bins)]

        colors = ['blue', 'red', 'green']
        trace_list = []
        for i, name in enumerate(method_names):
            trace = go.Scatter(
                x=bin_labels,
                y=ys_list[i],
                mode='lines+markers',
                name=name,
                line=dict(shape='linear', color=colors[i]),
                showlegend=showlegend,
            )

            trace_list.append(trace)

        # trace1 = go.Scatter(
        #     x=bin_labels,
        #     y=y,
        #     mode='lines+markers',
        #     name='Ours',
        #     line=dict(shape='linear', color='blue'),
        #     showlegend=showlegend,
        # )

        # trace2 = go.Scatter(
        #     x=bin_labels,
        #     y=y2,
        #     mode='lines+markers',
        #     name='RoomFormer',
        #     line=dict(shape='linear', color='red'),
        #     showlegend=showlegend,
        # )

        # return [trace1, trace2]
        return trace_list
    
    def add_traces_to_row(data_list, bin_sizes, row, showlegend=True):
        poly_data = extract_xy_list(data_list[0])
        poly_traces = get_line_traces(*poly_data, bin_sizes[0], showlegend=False)
        for i in range(len(poly_traces)):
            fig.add_trace(poly_traces[i], row=row, col=1)
        # fig.add_trace(poly_traces[0], row=row, col=1)
        # fig.add_trace(poly_traces[1], row=row, col=1)

        corner_data = extract_xy_list(data_list[1])
        corner_traces = get_line_traces(*corner_data, bin_sizes[1], showlegend=showlegend)
        for i in range(len(poly_traces)):
            fig.add_trace(corner_traces[i], row=row, col=2)
        # fig.add_trace(corner_traces[0], row=row, col=2)
        # fig.add_trace(corner_traces[1], row=row, col=2)
    
    add_traces_to_row(s3d_list, s3d_bin_sizes, 1, showlegend=False)
    add_traces_to_row(cc5k_list, cc5k_bin_sizes, 2, showlegend=True)

    fig.update_xaxes(tickangle=30, row=1, col=1, )  # match rotation with subplot 1
    fig.update_xaxes(tickangle=30, row=1, col=2, )  # match rotation with subplot 2
    fig.update_xaxes(tickangle=30, row=2, col=1, )  # match rotation with subplot 3
    fig.update_xaxes(tickangle=30, row=2, col=2, )  # match rotation with subplot 4
    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))

    # Update layout
    fig.update_layout(
        template='plotly_white',
        height=600,
        yaxis1=dict(
            title="Room F1",
            title_font=dict(size=20)
        ),
        yaxis3=dict(
            title="Room F1",
            title_font=dict(size=20)
        ),
        margin=dict(b=100),

        # legend=dict(
        #     orientation='h',
        #     yanchor='top',
        #     y=-0.25,              # a bit above the top edge of the plot area
        #     xanchor='center',
        #     x=0.5,
        #     font=dict(size=16),
        # ),
        legend=dict(
            orientation="h",           # vertical layout
            x=0.5,                          # center horizontally
            y=-0.32,                         # position below the plot
            xanchor="center",
            yanchor="top",
            # bgcolor="rgba(255,255,255,0.6)",  # optional: semi-transparent background
            # bordercolor="black",
            # borderwidth=1,
            font=dict(size=20)         # optional: adjust font size
        ),
        # margin=dict(l=100,r=100),
        annotations=[
            dict(
                text="<b>Structured3D-B</b>",  # Title for row 1
                xref="paper", yref="paper",
                x=0.5, y=1.03,  # left-aligned above row 1
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=24)
                # x=-0.1, y=0.78,           # x<0 places it to the left of the plot
                # xanchor='right', yanchor='middle',
                # showarrow=False,
                # font=dict(size=18)
            ),
            dict(
                text="<b>CubiCasa5K</b>",  # Title for row 2
                xref="paper", yref="paper",
                x=0.5, y=0.43,  # left-aligned above row 2
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=24)

                # x=-0.1, y=0.23,           # lower y for row 2
                # xanchor='right', yanchor='middle',
                # showarrow=False,
                # font=dict(size=18)
            ),
            # bottom xaxis titles
            dict(
                text="No. Polygons",
                xref="paper", yref="paper",
                x=0.23, y=-0.18,  # centered below left column
                showarrow=False,
                xanchor='center',
                yanchor='top',
                font=dict(size=20),
            ),
            dict(
                text="No. Corners",
                xref="paper", yref="paper",
                x=0.77, y=-0.18,  # centered below right column
                showarrow=False,
                xanchor='center',
                yanchor='top',
                font=dict(size=20),
            )

        ],

    )
    # Fix Y-axis range for consistency
    for i in range(1, 5):
        fig['layout'][f'yaxis{i}'].update(range=[0.4, 1.1])

    # fig.show()
    # Save the figure as an image
    fig.write_image(output_path, scale=3)
    print(f"Figure saved to {output_path}")


def plot_line_graph(x, y, y2, bin_size, plot_title, xaxis_title='No. Polygons', output_path=""):
    # Create the figure
    fig = go.Figure()

    num_bins = max(x) + 1
    bin_edges = [int(i * bin_size) for i in range(num_bins + 1)]  # e.g., [0.0, 0.1, ..., 1.0]
    bin_labels = [f"{bin_edges[i]}–{bin_edges[i+1]}" for i in range(num_bins)]

    fig.add_trace(go.Scatter(
        x=bin_labels,
        y=y,
        mode='lines+markers',
        name='Ours',
        line=dict(shape='linear')
    ))

    fig.add_trace(go.Scatter(
        x=bin_labels,
        y=y2,
        mode='lines+markers',
        name='RoomFormer',
        line=dict(shape='linear')
    ))

    # Update layout
    fig.update_layout(
        title=plot_title,
        xaxis_title=xaxis_title,
        yaxis_title='Room F1',
        yaxis=dict(range=[0.0, 1.0]),
        template='plotly_white',
        bargap=0.5,  # Add gap between bars (0.5 = 50% of bar width)
        legend=dict(
            x=0.10,
            y=0.10,
            xanchor='left',
            yanchor='bottom',
            # bgcolor='rgba(255,255,255,0.6)',  # semi-transparent background
            # bordercolor='black',
            borderwidth=1
        )
    )

    # fig.show()
    # Save the figure as an image
    fig.write_image(output_path, scale=3)
    print(f"Figure saved to {output_path}")


def plot_histogram(count_dict, title, output_path):
    # Plot the histogram using Plotly
    keys = list(count_dict.keys())
    values = list(count_dict.values())

    # Determine the maximum value for the y-axis
    max_y = max(values)
    # Adjust y-axis ticks dynamically for large ranges
    tick_interval = max(1, max_y // 10)  # Divide the range into 10 intervals
    tickvals_y = list(range(0, max_y + tick_interval, tick_interval))

    # Determine tick values for x-axis dynamically
    tickvals_x = keys  # Use the keys (number of points in polygons) as tick values

    fig = go.Figure(data=[
        go.Bar(x=keys, y=values, 
               text=values, textposition='outside', marker=dict(color='blue'), 
               width=0.5)
    ])

    fig.update_layout(
        title={
            'text': f'Histogram of {title}',
            'font': {'size': 24},  # Increase title font size
            'x': 0.5,  # Center the title
        },
        xaxis_title={
            'text': f'Number of {title}',
            'font': {'size': 18}  # Increase x-axis label font size
        },
        yaxis_title={
            'text': 'Frequency',
            'font': {'size': 18}  # Increase y-axis label font size
        },
        xaxis=dict(
            tickmode='array',  # Use custom tick values
            tickvals=tickvals_x,  # Set custom tick values
            ticktext=[str(val) for val in tickvals_x],  # Set custom tick labels
            tickfont=dict(size=10),  # Increase x-axis tick font size
            tickangle=45,
        ),
        yaxis=dict(
            tickvals=tickvals_y,  # Set custom tick values
            ticktext=[str(val) for val in tickvals_y],  # Set custom tick labels
            tickfont=dict(size=14),  # Increase y-axis tick font size
        ),
        template='plotly_white',
        bargap=0.5,  # Add gap between bars (0.5 = 50% of bar width)
        # Increase figure width for a long x-axis
        width=max(600, 30 * len(keys)),  # Dynamic width based on number of bars
    )
    # Save the figure as an image
    fig.write_image(output_path, scale=3)
    print(f"Figure saved to {output_path}")

    # fig.show()


# def extract_xy_list(input_a, input_b):
def extract_xy_list(input_list):
    xs = sorted(input_list[0].keys())
    ys = []
    ys2 = []
    ys_list = [[] for _ in range(len(input_list))]

    for _x in xs:
        for i, _input in enumerate(input_list):
            # s = np.mean(input_a[_x])
            # s2 = np.mean(input_b[_x])
            # ys.append(s)
            # ys2.append(s2)

            s = np.mean(_input[_x])
            ys_list[i].append(s)
    
    # return xs, ys, ys2
    return xs, ys_list


def loop_data(data_loader, eval_set, device, json_dirs, output_dir, is_s3d=True):
    max_num_points = -1
    max_num_polys = -1
    count_pts_dict = defaultdict(lambda: 0)
    count_length_dict = defaultdict(lambda: 0)

    score_by_no_polys_list = [defaultdict(lambda: []) for _ in range(len(json_dirs))]
    score_by_no_corners_list = [defaultdict(lambda: []) for _ in range(len(json_dirs))]

    bin_size_list = [5, 15] if is_s3d else [5, 30]

    def read_scores(json_path_list): # , json_path, json_path2
        score_list = []
        for json_path in json_path_list:
            with open(json_path, 'r') as f:
                score = json.load(f)['room_f1']
            score_list.append(score)
        # with open(json_path2, 'r') as f:
        #     score2 = json.load(f)['room_f1']
        # return score, score2

        return score_list

    def assign_scores(num_polys, num_corners, score_list, by_poly_dict_list, by_corner_dict_list, bin_size, corner_bin_size):
        # by polys
        # by_poly_dict_list[0][num_polys // bin_size].append(score_list[0])
        # by_poly_dict_list[1][num_polys // bin_size].append(score_list[1])
        for i in range(len(score_list)):
            by_poly_dict_list[i][num_polys // bin_size].append(score_list[i])

        # by corner
        # by_corner_dict_list[0][num_corners // corner_bin_size].append(score_list[0])
        # by_corner_dict_list[1][num_corners // corner_bin_size].append(score_list[1])
        for i in range(len(score_list)):
            by_corner_dict_list[i][num_corners // corner_bin_size].append(score_list[i])

        return by_poly_dict_list, by_corner_dict_list


    for batched_inputs, batched_extras in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        image_filenames = [x["file_name"] for x in batched_inputs]

        for i in range(len(samples)):
            # json_path = os.path.join(json_dirs[0], f"{str(scene_ids[i]).zfill(5)}_pred.json")
            # json_path2 = os.path.join(json_dirs[1], f"{str(scene_ids[i]).zfill(5)}_pred.json")
            # scores = read_scores(json_path, json_path2)

            json_path_list = [os.path.join(jdir, f"{str(scene_ids[i]).zfill(5)}_pred.json")for jdir in json_dirs]
            scores = read_scores(json_path_list)

            if batched_extras is not None:
                t = batched_extras['mask'][i].sum().item()
                count_length_dict[t] += 1
                num_corners = (batched_extras['token_labels'][i] == 0).sum().item()

            room_polys = gt_instances[i].gt_masks.polygons
            room_ids = gt_instances[i].gt_classes.detach().cpu().numpy()

            score_by_no_polys_list, score_by_no_corners_list = assign_scores(len(room_ids), num_corners, scores, 
                             score_by_no_polys_list, score_by_no_corners_list, bin_size_list[0], bin_size_list[1]
                             )
            # cc5k_score_by_no_polys_list, cc5k_score_by_no_corners_list = assign_scores(len(room_ids), num_corners, cc5k_scores, 
            #                  cc5k_score_by_no_polys_list, cc5k_score_by_no_corners_list, cc5k_bin_size, cc5k_corner_bin_size)

            # count_room_dict[len(room_ids) // bin_size] += 1
    #         for poly, poly_id in zip(room_polys, room_ids):
    #             poly = poly[0].reshape(-1,2).astype(np.int32)
    #             count_pts_dict[len(poly)] += 1
    #             if len(poly) > max_num_points:
    #                 max_num_points = len(poly)
    #         if len(room_ids) > max_num_polys:
    #             max_num_polys = len(room_ids)
        
    # print(f"Max pts: {max_num_points}, Max polys: {max_num_polys}")
    
    # xs, ys, ys2 = extract_xy_list(score_dict_by_no_polys, score_dict_by_no_polys2)
    # plot_line_graph(xs, ys, ys2, bin_size, 
    #                 "Model Performance by Polygon Counts",
    #                 'No. Polygons',
    #                 os.path.join(output_dir, f"{eval_set}_graph.png"))

    # xs, ys, ys2 = extract_xy_list(score_dict_by_no_corners, score_dict_by_no_corners2)
    # plot_line_graph(xs, ys, ys2, corner_bin_size, 
    #                 "Model Performance by Corner Counts",
    #                 'No. Corners',
    #                 os.path.join(output_dir, f"{eval_set}_corner_graph.png"))

    return score_by_no_polys_list, score_by_no_corners_list, bin_size_list
    # plot_line_graph_combined([cc5k_score_by_no_polys_list, cc5k_score_by_no_corners_list], 
    #                          [s3d_score_by_no_polys_list, s3d_score_by_no_corners_list],
    #                          [cc5k_bin_size, cc5k_corner_bin_size], 
    #                          [s3d_bin_size, s3d_corner_bin_size],
    #                          os.path.join(output_dir, f"combined_{eval_set}_graph.png"))


def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)

    # new
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--add_cls_token', action='store_true')
    parser.add_argument('--per_token_sem_loss', action='store_true')
    parser.add_argument('--json_dir', type=str, default='')
    parser.add_argument('--wd_only', action='store_true')
    parser.add_argument('--num_subset_images', type=int, default=-1)

    # poly2seq
    parser.add_argument('--poly2seq', action='store_true')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--num_bins', type=int, default=64)

    # backbone
    parser.add_argument('--input_channels', default=1, type=int)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    parser.add_argument('--image_norm', action='store_true')
    parser.add_argument('--disable_image_transform', action='store_true')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=800, type=int,
                        help="Number of query slots (num_polys * max. number of corner per poly)")
    parser.add_argument('--num_polys', default=20, type=int,
                        help="Number of maximum number of room polygons")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="Type of query pos in decoder - \
                        1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                        2. sine: since embedding from reference points (so if references points update, query_pos also \
                        3. none: remove query_pos")
    parser.add_argument('--with_poly_refine', default=True, action='store_true',
                        help="iteratively refine reference points (i.e. positional part of polygon queries)")
    parser.add_argument('--masked_attn', default=False, action='store_true',
                        help="if true, the query in one room will not be allowed to attend other room")
    parser.add_argument('--semantic_classes', default=-1, type=int,
                        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty")
    parser.add_argument('--use_room_attn_at_last_dec_layer', default=False, action='store_true', help="use room-wise attention in last decoder layer")

    # aux
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    

    # dataset parameters
    parser.add_argument('--dataset_name', default='stru3d')
    parser.add_argument('--dataset_root', default='data/stru3d', type=str)
    parser.add_argument('--eval_set', default='test', type=str)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/roomformer_scenecad.pth', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='eval_stru3d',
                        help='path where to save result')

    # visualization options
    parser.add_argument('--plot_density', default=False, action='store_true', help="plot predicited room polygons overlaid on the density map")
    parser.add_argument('--plot_gt', default=False, action='store_true', help="plot ground truth floorplan")
    parser.add_argument('--plot_gt_image', default=False, action='store_true', help="plot ground truth image")


    return parser



def main(args):

    device = 'cpu' # torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # # build model
    # model = build_model(args, train=False)
    # model.to(device)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    # build dataset and dataloader
    args.dataset_name = 'cubicasa'
    args.dataset_root = "data/coco_cubicasa5k_nowalls_v4-1_refined"
    cc5k_dataset_eval = build_dataset(image_set=args.eval_set, args=args)

    args2 = copy.deepcopy(args)
    args2.dataset_name = 'stru3d'
    args2.dataset_root = "data/coco_s3d_bw"
    s3d_dataset_eval = build_dataset(image_set=args2.eval_set, args=args2)

    # for test
    if args.debug:
        dataset_eval = torch.utils.data.Subset(dataset_eval, list(range(0, args.batch_size, 1)))
    if args.num_subset_images > 0 and args.num_subset_images < len(dataset_eval):
        s3d_dataset_eval = torch.utils.data.Subset(s3d_dataset_eval, range(args.num_subset_images))
    cc5k_sampler_eval = torch.utils.data.SequentialSampler(cc5k_dataset_eval)
    s3d_sampler_eval = torch.utils.data.SequentialSampler(s3d_dataset_eval)


    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        if 'target_seq' in batch[0]:
            # Concatenate tensors for each key in the batch
            delta_x1 = torch.stack([item['delta_x1'] for item in batch], dim=0)
            delta_x2 = torch.stack([item['delta_x2'] for item in batch], dim=0)
            delta_y1 = torch.stack([item['delta_y1'] for item in batch], dim=0)
            delta_y2 = torch.stack([item['delta_y2'] for item in batch], dim=0)
            seq11 = torch.stack([item['seq11'] for item in batch], dim=0)
            seq21 = torch.stack([item['seq21'] for item in batch], dim=0)
            seq12 = torch.stack([item['seq12'] for item in batch], dim=0)
            seq22 = torch.stack([item['seq22'] for item in batch], dim=0)
            target_seq = torch.stack([item['target_seq'] for item in batch], dim=0)
            token_labels = torch.stack([item['token_labels'] for item in batch], dim=0)
            mask = torch.stack([item['mask'] for item in batch], dim=0)

            # Delete the keys from the batch
            for item in batch:
                del item['delta_x1']
                del item['delta_x2']
                del item['delta_y1']
                del item['delta_y2']
                del item['seq11']
                del item['seq21']
                del item['seq12']
                del item['seq22']
                del item['target_seq']
                del item['token_labels']
                del item['mask']

            # Return the concatenated batch
            return batch, {
                'delta_x1': delta_x1,
                'delta_x2': delta_x2,
                'delta_y1': delta_y1,
                'delta_y2': delta_y2,
                'seq11': seq11,
                'seq21': seq21,
                'seq12': seq12,
                'seq22': seq22,
                'target_seq': target_seq,
                'token_labels': token_labels,
                'mask': mask,
            }
            
        return batch, None

    cc5k_data_loader_eval = DataLoader(cc5k_dataset_eval, args.batch_size, sampler=cc5k_sampler_eval,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                 pin_memory=True)
    s3d_data_loader_eval = DataLoader(s3d_dataset_eval, args.batch_size, sampler=s3d_sampler_eval,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                 pin_memory=True)

    # for n, p in model.named_parameters():
    #     print(n)

    output_dir = Path(args.output_dir)

    # checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    # unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    # if len(missing_keys) > 0:
    #     print('Missing Keys: {}'.format(missing_keys))
    # if len(unexpected_keys) > 0:
    #     print('Unexpected Keys: {}'.format(unexpected_keys))

    save_dir = output_dir # os.path.join(os.path.dirname(args.checkpoint), output_dir)
    os.makedirs(save_dir, exist_ok=True)

    # if args.plot_gt:
    #     plot_gt_floor(
    #                 data_loader_eval, 
    #                 device, save_dir, 
    #                 plot_gt=args.plot_gt,
    #                 semantic_rich=args.semantic_classes>0
    #                 )
    
    # if args.plot_density:
    #     plot_polys(data_loader_eval, device, save_dir)

    # if args.plot_gt_image:
    #     plot_gt_image(data_loader_eval, device, save_dir)

    if args.eval_set == 'train':
        json_dir = "slurm_scripts4/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_t1/train/result_jsons"
        json_dir2 = "slurm_scripts4/cubi_v4-1refined_queries56x50_sem_v1/train/result_jsons"
    elif args.eval_set == 'val':
        cc5k_json_dir = "slurm_scripts4/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_t1/val/result_jsons"
        cc5k_json_dir2 = "slurm_scripts4/cubi_v4-1refined_queries56x50_sem_v1/val/result_jsons"
        cc5k_json_dir3 = "slurm_scripts4/cc5k_frinet_nowd_256_ckpt/eval/result_jsons"

        s3d_json_dir = "slurm_scripts/s3d_bw_ddp_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_correct_smoothing1e-1_numcls19_pts_finetune_t3/val/result_jsons/"
        s3d_json_dir2 = "slurm_scripts/s3d_bw_ddp_queries40x30/result_jsons/"
        s3d_json_dir3 = "slurm_scripts4/s3dbw_frinet_ckpt/eval/result_jsons"

    # json_dir = "/home/htp26/RoomFormerTest/slurm_scripts/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_t1/result_jsons"
    # json_dir2 = "/home/htp26/RoomFormerTest/slurm_scripts/cubi_v4-1refined_queries56x50_sem_v1/result_jsons"

    s3d_score_by_no_polys_list, s3d_score_by_no_corners_list, s3d_bin_size_list = loop_data(s3d_data_loader_eval, args.eval_set, device, [s3d_json_dir, s3d_json_dir2, s3d_json_dir3], save_dir, is_s3d=True)
    cc5k_score_by_no_polys_list, cc5k_score_by_no_corners_list, cc5k_bin_size_list = loop_data(cc5k_data_loader_eval, args.eval_set, device, [cc5k_json_dir, cc5k_json_dir2, cc5k_json_dir3], save_dir, is_s3d=False)

    plot_line_graph_combined([cc5k_score_by_no_polys_list, cc5k_score_by_no_corners_list], 
                             [s3d_score_by_no_polys_list, s3d_score_by_no_corners_list],
                             cc5k_bin_size_list, 
                             s3d_bin_size_list,
                             ["Ours", "RoomFormer", "FRI-Net"],
                             os.path.join(output_dir, f"combined_{args.eval_set}_graph.png"))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
