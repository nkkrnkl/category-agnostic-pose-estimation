import cv2
import copy
import json
import math
import os
import sys
import time
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon
import torch

import util.misc as utils

# Floorplan-specific evaluators (optional for CAPE/MP-100)
try:
    from s3d_floorplan_eval.Evaluator.Evaluator import Evaluator
    from s3d_floorplan_eval.options import MCSSOptions
    from s3d_floorplan_eval.DataRW.S3DRW import S3DRW
    from s3d_floorplan_eval.DataRW.wrong_annotatios import wrong_s3d_annotations_list
    from scenecad_eval.Evaluator import Evaluator_SceneCAD
    from rplan_eval.Evaluator import Evaluator_RPlan
    FLOORPLAN_EVAL_AVAILABLE = True
except ImportError:
    FLOORPLAN_EVAL_AVAILABLE = False
    print("Warning: Floorplan evaluators not available (OK for MP-100 CAPE)")

from util.poly_ops import pad_gt_polys
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan, plot_semantic_rich_floorplan_tight, sort_polygons_by_matching
from util.plot_utils import plot_density_map, plot_semantic_rich_floorplan_opencv
from util.plot_utils import S3D_LABEL, CC5K_LABEL
from util.eval_utils import compute_f1

from datasets import get_dataset_class_labels

# Initialize options only if evaluators are available
if FLOORPLAN_EVAL_AVAILABLE:
    options = MCSSOptions()
    opts = options.parse()
else:
    options = None
    opts = None

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, poly2seq: bool = False, ema_model=None, **kwargs):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100  # Reduced verbosity: print every 100 iterations instead of 10
    model_obj = model if not hasattr(model, 'module') else model.module

    for batched_inputs, batched_extras in metric_logger.log_every(data_loader, print_freq, header):
        samples = [x["image"].to(device) for x in batched_inputs]
        # Handle instances (may be None for datasets like MP-100 CAPE)
        gt_instances = [x["instances"].to(device) if x["instances"] is not None else None for x in batched_inputs]
        # print("length: ", batched_extras['mask'].sum(1).tolist())
        # print("Max #polys: ", max([len(gt_instances[i].gt_masks.polygons) for i in range(len(gt_instances))]))
        # print("Max #corners: ", max([gt_instances[i].gt_masks.polygons for i in range(gt_instances)]))
        if not poly2seq:
            room_targets = pad_gt_polys(gt_instances, model_obj.num_queries_per_poly, samples[0].shape[1], drop_rate=kwargs.get("drop_rate", 0.), device=device)
            outputs = model(samples)
        else:
            for key in batched_extras.keys():
                batched_extras[key] = batched_extras[key].to(device)
            room_targets = batched_extras
            outputs = model(samples, batched_extras)

        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        if ema_model is not None:
            utils.update_ema(ema_model, model.module, 0.999)

        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, dataset_name, data_loader, device, plot_density=False, output_dir=None, epoch=None, poly2seq: bool = False):
    model.eval()
    criterion.eval()

    if dataset_name == 'stru3d':
        door_window_index = [16, 17]
    elif dataset_name == 'cubicasa':
        door_window_index = [10, 9]
    else:
        door_window_index = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model_obj = model if not hasattr(model, 'module') else model.module

    for batched_inputs, batched_extras in metric_logger.log_every(data_loader, 10, header):
        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"]for x in batched_inputs]
        gt_instances = [x["instances"].to(device) if x["instances"] is not None else None for x in batched_inputs]
        if not poly2seq:
            room_targets = pad_gt_polys(gt_instances, model_obj.num_queries_per_poly, samples[0].shape[1], drop_rate=0., device=device)
            outputs = model(samples)
        else:
            room_targets = batched_extras
            outputs = model(samples, batched_extras)

        image_size = samples[0].size(2)
        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict

        # Compute scaled and unscaled loss dicts for logging
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in loss_dict.items()}

        # For MP-100 CAPE, skip detailed polygon evaluation (only compute loss)
        if dataset_name == 'mp100':
            # Just accumulate loss metrics and skip polygon evaluation
            metric_logger.update(loss=sum(loss_dict_scaled.values()),
                                 **loss_dict_scaled,
                                 **loss_dict_unscaled)
            continue

        bs = outputs['pred_logits'].shape[0]
        pred_logits = outputs['pred_logits']
        pred_corners = outputs['pred_coords']
        if poly2seq:
            pass
        else:
            fg_mask = torch.sigmoid(pred_logits) > 0.5 # select valid corners

        if 'pred_room_logits' in outputs:
            prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
            _, pred_room_label = prob[..., :-1].max(-1)

        # process per scene
        for i in range(bs):

            if dataset_name == 'stru3d':
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode = "online_eval")
                evaluator = Evaluator(curr_data_rw, curr_opts)
            elif dataset_name == 'scenecad':
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1,2).astype(np.int32)]
                evaluator = Evaluator_SceneCAD()
            elif dataset_name == 'rplan':
                gt_polys = [x[0].reshape(-1,2).astype(np.int32) for x in gt_instances[i].gt_masks.polygons]
                gt_polys_types = gt_instances[i].gt_classes.detach().cpu().tolist()
                gt_window_doors = []
                gt_window_doors_types = []
                evaluator = Evaluator_RPlan()
            elif dataset_name in ['cubicasa', 'r2g']:
                gt_polys, gt_polys_types = [], []
                gt_window_doors = []
                gt_window_doors_types = []
                for gt_poly, gt_id in zip(gt_instances[i].gt_masks.polygons, gt_instances[i].gt_classes.detach().cpu().tolist()):
                    gt_poly = gt_poly[0].reshape(-1,2).astype(np.int32)
                    if gt_id in door_window_index:
                        gt_window_doors.append(gt_poly)
                        gt_window_doors_types.append(gt_id)
                    else:
                        gt_polys.append(gt_poly)
                        gt_polys_types.append(gt_id)
                evaluator = Evaluator_RPlan()
            
            print("Running Evaluation for scene %s" % scene_ids[i])

            fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]

            room_polys = []
            
            semantic_rich = 'pred_room_logits' in outputs
            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy()
            
            # process per room
            for j in range(fg_mask_per_scene.shape[0]):
                fg_mask_per_room = fg_mask_per_scene[j]
                pred_corners_per_room = pred_corners_per_scene[j]
                valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
                if len(valid_corners_per_room)>0:
                    corners = (valid_corners_per_room * (image_size - 1)).cpu().numpy()
                    corners = np.around(corners).astype(np.int32)

                    if not semantic_rich:
                        # only regular rooms
                        if len(corners)>=4 and Polygon(corners).area >= 100:
                                room_polys.append(corners)
                    else:
                        # regular rooms
                        if pred_room_label_per_scene[j] not in door_window_index:
                            if len(corners)>=4 and Polygon(corners).area >= 100:
                                room_polys.append(corners)
                                room_types.append(pred_room_label_per_scene[j])
                        # window / door
                        elif len(corners)==2:
                            window_doors.append(corners)
                            window_doors_types.append(pred_room_label_per_scene[j])

            if not semantic_rich:
                pred_room_label_per_scene = len(room_polys) * [-1]
                    
            if dataset_name == 'stru3d':
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                                                            room_polys=room_polys, 
                                                            room_types=room_types, 
                                                            window_door_lines=window_doors, 
                                                            window_door_lines_types=window_doors_types)
            elif dataset_name == 'scenecad':
                quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)

            elif dataset_name in ['rplan', 'cubicasa', 'r2g']:
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=None, gt_polys_types=gt_polys_types,
                                                                img_size=(image_size, image_size),
                                                                    )
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=room_types, gt_polys_types=gt_polys_types,
                                                                window_door_lines=window_doors, gt_window_doors_list=gt_window_doors,
                                                                window_door_lines_types=window_doors_types, gt_window_doors_type_list=gt_window_doors_types,
                                                                img_size=(image_size, image_size),
                                                                )

            if 'room_iou' in quant_result_dict_scene:
                metric_logger.update(room_iou=quant_result_dict_scene['room_iou'])
            
            metric_logger.update(room_prec=quant_result_dict_scene['room_prec'])
            metric_logger.update(room_rec=quant_result_dict_scene['room_rec'])
            metric_logger.update(corner_prec=quant_result_dict_scene['corner_prec'])
            metric_logger.update(corner_rec=quant_result_dict_scene['corner_rec'])
            metric_logger.update(angles_prec=quant_result_dict_scene['angles_prec'])
            metric_logger.update(angles_rec=quant_result_dict_scene['angles_rec'])

            if semantic_rich:
                metric_logger.update(room_sem_prec=quant_result_dict_scene['room_sem_prec'])
                metric_logger.update(room_sem_rec=quant_result_dict_scene['room_sem_rec'])
                metric_logger.update(window_door_prec=quant_result_dict_scene['window_door_prec'])
                metric_logger.update(window_door_rec=quant_result_dict_scene['window_door_rec'])

        # plot last sample
        if plot_density and len(room_polys) > 0:
            pred_room_map = plot_density_map(samples[i], image_size, room_polys, pred_room_label_per_scene)
            cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map_{}.png'.format(scene_ids[i], epoch)), pred_room_map)

            plot_density = False # only plot once

        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        metric_logger.update(loss=sum(loss_dict_scaled.values()),
                             **loss_dict_scaled,
                             **loss_dict_unscaled)

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

@torch.no_grad()
def evaluate_v2(model, criterion, dataset_name, data_loader, device, plot_density=False, output_dir=None, epoch=None, poly2seq: bool = False, add_cls_token=False, 
                per_token_sem_loss=False, wd_as_line=True):
    model.eval()
    criterion.eval()
    if dataset_name == 'stru3d':
        door_window_index = [16, 17]
    elif dataset_name == 'cubicasa':
        door_window_index = [10, 9]
    else:
        door_window_index = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model_obj = model if not hasattr(model, 'module') else model.module

    for batched_inputs, batched_extras in metric_logger.log_every(data_loader, 10, header):
        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"]for x in batched_inputs]
        gt_instances = [x["instances"].to(device) if x["instances"] is not None else None for x in batched_inputs]
        if not poly2seq:
            room_targets = pad_gt_polys(gt_instances, model_obj.num_queries_per_poly, samples[0].shape[1], drop_rate=0., device=device)
            outputs = model(samples)
        else:
            for key in batched_extras.keys():
                batched_extras[key] = batched_extras[key].to(device)
            room_targets = batched_extras
            outputs = model(samples, batched_extras)

        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict

        image_size = samples[0].size(2)
        if poly2seq:
            outputs = model_obj.forward_inference(samples)
            pred_corners = outputs['gen_out']
            bs = outputs['pred_logits'].shape[0]
        else:
            bs = outputs['pred_logits'].shape[0]
            pred_logits = outputs['pred_logits']
            pred_corners = outputs['pred_coords']
            fg_mask = torch.sigmoid(pred_logits) > 0.5 # select valid corners

        if 'pred_room_logits' in outputs:
            prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
            num_classes = prob.shape[-1]
            _, pred_room_label = prob[..., :-1].max(-1)

        # process per scene
        for i in range(bs):

            gt_polys, gt_polys_types = [], []
            gt_window_doors = []
            gt_window_doors_types = []
            for gt_poly, gt_id in zip(gt_instances[i].gt_masks.polygons, gt_instances[i].gt_classes.detach().cpu().tolist()):
                gt_poly = gt_poly[0].reshape(-1,2).astype(np.int32)
                if gt_id in door_window_index:
                    gt_window_doors.append(gt_poly)
                    gt_window_doors_types.append(gt_id)
                else:
                    gt_polys.append(gt_poly)
                    gt_polys_types.append(gt_id)

            if dataset_name == 'stru3d':
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode = "online_eval")
                evaluator = Evaluator(curr_data_rw, curr_opts, disable_overlapping_filter=True)
            elif dataset_name == 'scenecad':
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1,2).astype(np.int32)]
                evaluator = Evaluator_SceneCAD(disable_overlapping_filter=True)
            elif dataset_name == 'rplan':
                gt_polys = [x[0].reshape(-1,2).astype(np.int32) for x in gt_instances[i].gt_masks.polygons]
                gt_polys_types = gt_instances[i].gt_classes.detach().cpu().tolist()
                gt_window_doors = []
                gt_window_doors_types = []
                evaluator = Evaluator_RPlan(disable_overlapping_filter=True)
            elif dataset_name in ['cubicasa', 'r2g']:
                evaluator = Evaluator_RPlan(disable_overlapping_filter=True,
                                            wd_as_line=wd_as_line)
            
            print("Running Evaluation for scene %s" % scene_ids[i])

            # fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]

            room_polys = []
            
            semantic_rich = 'pred_room_logits' in outputs
            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy().tolist()

            all_room_polys = []
            tmp = []
            all_length_list = [0]
            for j in range(len(pred_corners_per_scene)):
                if isinstance(pred_corners_per_scene[j], int):
                    if pred_corners_per_scene[j] == 2 and tmp: # sep
                        all_room_polys.append(tmp)
                        all_length_list.append(len(tmp)+1+add_cls_token)
                        tmp = []
                    continue
                tmp.append(pred_corners_per_scene[j])
            
            if len(tmp):
                all_room_polys.append(tmp)
                all_length_list.append(len(tmp)+1+add_cls_token)
            start_poly_indices = np.cumsum(all_length_list)

            final_pred_classes = []
            for j, poly in enumerate(all_room_polys):
                if len(poly) < 2:
                    continue
                corners = np.array(poly, dtype=np.float32) * (image_size - 1)
                corners = np.around(corners).astype(np.int32)

                if not semantic_rich:
                    # only regular rooms
                    if len(corners)>=4: # and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                else:
                    if per_token_sem_loss:
                        pred_classes, counts = np.unique(pred_room_label_per_scene[start_poly_indices[j]:start_poly_indices[j+1]][:-1], return_counts=True)
                        pred_class = pred_classes[np.argmax(counts)]
                    else:
                        pred_class = pred_room_label_per_scene[start_poly_indices[j+1]-1] # get last cls token in the seq
                    final_pred_classes.append(pred_class)

                    # if len(corners)>=4:
                    #     # if len(corners)>=4 and Polygon(corners).area >= 100:
                    #     room_polys.append(corners)
                    #     room_types.append(pred_class)
                    # # window / door
                    # elif len(corners)==2:
                    #     window_doors.append(corners)
                    #     window_doors_types.append(pred_class)

                    if wd_as_line:
                        # regular rooms
                        if len(corners)>=4:
                        # if len(corners)>=4 and Polygon(corners).area >= 100:
                            room_polys.append(corners)
                            room_types.append(pred_class)
                        # window / door
                        elif len(corners)==2:
                            window_doors.append(corners)
                            window_doors_types.append(pred_class)
                    else:
                        # regular rooms
                        if pred_class not in door_window_index:
                            room_polys.append(corners)
                            room_types.append(pred_class)
                        else:
                            window_doors.append(corners)
                            window_doors_types.append(pred_class)
            
            if not semantic_rich:
                pred_room_label_per_scene = len(all_room_polys) * [-1]
            else:
                pred_room_label_per_scene = final_pred_classes

            if dataset_name == 'stru3d':
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                                                            room_polys=room_polys, 
                                                            room_types=room_types, 
                                                            window_door_lines=window_doors, 
                                                            window_door_lines_types=window_doors_types)
            elif dataset_name == 'scenecad':
                quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)

            elif dataset_name in ['rplan', 'cubicasa', 'r2g']:
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=None, gt_polys_types=gt_polys_types,
                                                                img_size=(image_size, image_size),
                                                                    )
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=room_types, gt_polys_types=gt_polys_types,
                                                                window_door_lines=window_doors, gt_window_doors_list=gt_window_doors,
                                                                window_door_lines_types=window_doors_types, gt_window_doors_type_list=gt_window_doors_types,
                                                                img_size=(image_size, image_size),
                                                                )

            if 'room_iou' in quant_result_dict_scene:
                metric_logger.update(room_iou=quant_result_dict_scene['room_iou'])
            
            metric_logger.update(room_prec=quant_result_dict_scene['room_prec'])
            metric_logger.update(room_rec=quant_result_dict_scene['room_rec'])
            metric_logger.update(corner_prec=quant_result_dict_scene['corner_prec'])
            metric_logger.update(corner_rec=quant_result_dict_scene['corner_rec'])
            metric_logger.update(angles_prec=quant_result_dict_scene['angles_prec'])
            metric_logger.update(angles_rec=quant_result_dict_scene['angles_rec'])

            if semantic_rich:
                metric_logger.update(room_sem_prec=quant_result_dict_scene['room_sem_prec'])
                metric_logger.update(room_sem_rec=quant_result_dict_scene['room_sem_rec'])
                metric_logger.update(window_door_prec=quant_result_dict_scene['window_door_prec'])
                metric_logger.update(window_door_rec=quant_result_dict_scene['window_door_rec'])

        # plot last sample
        if plot_density and len(room_polys) > 0:
            pred_room_map = plot_density_map(samples[i], image_size, room_polys, pred_room_label_per_scene)
            cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map_{}.png'.format(scene_ids[i], epoch)), pred_room_map)

            plot_density = False # only plot once

        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        metric_logger.update(loss=sum(loss_dict_scaled.values()),
                             **loss_dict_scaled,
                             **loss_dict_unscaled)

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

@torch.no_grad()
def evaluate_floor(model, dataset_name, data_loader, device, output_dir, plot_pred=True, plot_density=True, plot_gt=True, semantic_rich=False,
                   save_pred=False, iou_thres=0.5):
    model.eval()
    if dataset_name == 'stru3d':
        door_window_index = [16, 17]
    elif dataset_name == 'cubicasa':
        door_window_index = [10, 9]
    elif dataset_name == 'waffle':
        door_window_index = [1, 2]
    else:
        door_window_index = []

    metric_category = ['room','corner','angles']
    if semantic_rich:
        metric_category += ['room_sem','window_door']

    quant_result_dict = None
    scene_counter = 0
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for batched_inputs, batched_extras in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) if x["instances"] is not None else None for x in batched_inputs]

        image_size = samples[0].size(2)

        # # draw GT map
        # if plot_gt:
        #     for i, gt_inst in enumerate(gt_instances):
        #         if not semantic_rich:
        #             # plot regular room floorplan
        #             gt_polys = []
        #             density_map = np.transpose((samples[i] * (image_size - 1)).cpu().numpy(), [1, 2, 0])
        #             density_map = np.repeat(density_map, 3, axis=2)

        #             gt_corner_map = np.zeros([image_size, image_size, 3])
        #             for j, poly in enumerate(gt_inst.gt_masks.polygons):
        #                 corners = poly[0].reshape(-1, 2)
        #                 gt_polys.append(corners)
                        
        #             gt_room_polys = [np.array(r) for r in gt_polys]
        #             gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, scale=1000)
        #             cv2.imwrite(os.path.join(output_dir, '{}_gt.png'.format(scene_ids[i])), gt_floorplan_map)
        #         else:
        #             # plot semantically-rich floorplan
        #             gt_sem_rich = []
        #             for j, poly in enumerate(gt_inst.gt_masks.polygons):
        #                 corners = poly[0].reshape(-1, 2).astype(np.int32)
        #                 corners_flip_y = corners.copy()
        #                 corners_flip_y[:,1] = image_size - 1 - corners_flip_y[:,1]
        #                 corners = corners_flip_y
        #                 gt_sem_rich.append([corners, gt_inst.gt_classes.cpu().numpy()[j]])

        #             gt_sem_rich_path = os.path.join(output_dir, '{}_sem_rich_gt.png'.format(scene_ids[i]))
        #             # plot_semantic_rich_flooplan(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1) 
        #             plot_semantic_rich_floorplan_tight(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1, plot_text=True, is_bw=False, door_window_index=door_window_index)

        outputs = model(samples)
        pred_logits = outputs['pred_logits']
        pred_corners = outputs['pred_coords']
        fg_mask = torch.sigmoid(pred_logits) > 0.5 # select valid corners

        if 'pred_room_logits' in outputs:
            prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
            _, pred_room_label = prob[..., :-1].max(-1)

        # process per scene
        for i in range(pred_logits.shape[0]):
            gt_polys, gt_polys_types = [], []
            gt_window_doors = []
            gt_window_doors_types = []
            for gt_poly, gt_id in zip(gt_instances[i].gt_masks.polygons, gt_instances[i].gt_classes.detach().cpu().tolist()):
                gt_poly = gt_poly[0].reshape(-1,2).astype(np.int32)
                if gt_id in door_window_index:
                    gt_window_doors.append(gt_poly)
                    gt_window_doors_types.append(gt_id)
                else:
                    gt_polys.append(gt_poly)
                    gt_polys_types.append(gt_id)
            
            if dataset_name == 'stru3d':
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode = "test") # TODO: "test"
                evaluator = Evaluator(curr_data_rw, curr_opts)
            elif dataset_name == 'scenecad':
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1,2).astype(np.int32)]
                evaluator = Evaluator_SceneCAD()
            elif dataset_name == 'rplan':
                gt_polys = [x[0].reshape(-1,2).astype(np.int32) for x in gt_instances[i].gt_masks.polygons]
                gt_polys_types = gt_instances[i].gt_classes.detach().cpu().tolist()
                evaluator = Evaluator_RPlan()
            elif dataset_name in ['cubicasa', 'waffle', 'r2g']:
                evaluator = Evaluator_RPlan(iou_thres=iou_thres)

            print("Running Evaluation for scene %s" % scene_ids[i])

            fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]
            room_polys = []

            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

            # process per room
            for j in range(fg_mask_per_scene.shape[0]):
                fg_mask_per_room = fg_mask_per_scene[j]
                pred_corners_per_room = pred_corners_per_scene[j]
                valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
                if len(valid_corners_per_room)>0:
                    corners = (valid_corners_per_room * (image_size - 1)).cpu().numpy()
                    corners = np.around(corners).astype(np.int32)

                    if not semantic_rich:
                        # only regular rooms
                        if len(corners)>=4 and Polygon(corners).area >= 100:
                            room_polys.append(corners)
                    else:
                        # regular rooms
                        # if pred_room_label_per_scene[j] not in door_window_index:
                        if len(corners)>=3 and Polygon(corners).area >= 100:
                            room_polys.append(corners)
                            room_types.append(pred_room_label_per_scene[j])
                        # window / door
                        elif len(corners)==2 and dataset_name != 'r2g':
                            window_doors.append(corners)
                            window_doors_types.append(pred_room_label_per_scene[j])
            
            if not semantic_rich:
                pred_room_label_per_scene = len(room_polys) * [-1]

            if dataset_name == 'stru3d':
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                                                            room_polys=room_polys, 
                                                            room_types=room_types, 
                                                            window_door_lines=window_doors, 
                                                            window_door_lines_types=window_doors_types)
    
            elif dataset_name == 'scenecad':
                quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)
            elif dataset_name in ['rplan', 'cubicasa', 'waffle', 'r2g']:
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=None, gt_polys_types=gt_polys_types,
                                                                img_size=(image_size, image_size),
                                                                    )
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=room_types, gt_polys_types=gt_polys_types,
                                                                window_door_lines=window_doors, gt_window_doors_list=gt_window_doors,
                                                                window_door_lines_types=window_doors_types, gt_window_doors_type_list=gt_window_doors_types,
                                                                img_size=(image_size, image_size),
                                                                )


            if quant_result_dict is None:
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

            # plot regular room floorplan
            gt_room_polys = [np.array(poly) for poly in gt_polys]
            room_polys = [np.array(poly) for poly in room_polys]

            if 'gt_polys_sorted_indcs' in quant_result_dict_scene:
                gt_polys_sorted_indcs = quant_result_dict_scene['gt_polys_sorted_indcs']
                del quant_result_dict_scene['gt_polys_sorted_indcs']
                gt_room_polys = [gt_room_polys[ind] for ind in gt_polys_sorted_indcs]
                    

            if 'pred2gt_indices' in quant_result_dict_scene:
                pred2gt_indices = quant_result_dict_scene['pred2gt_indices']
                del quant_result_dict_scene['pred2gt_indices']
                room_polys, gt_room_polys, pred_mask, gt_mask = sort_polygons_by_matching(pred2gt_indices, room_polys, gt_room_polys)
            else:
                pred_mask, gt_mask = None, None

            prec, rec = quant_result_dict_scene['room_prec'], quant_result_dict_scene['room_rec']
            f1 = 2*prec*rec/(prec+rec+1e-5)
            missing_rate = quant_result_dict_scene['room_missing_ratio']
            plot_statistics = {
                'f1': f1, 'prec': prec, 'rec': rec, 'missing_rate': missing_rate, 
                'num_preds': len(room_polys), 'num_gt': len(gt_polys), 'num_matched_preds': sum([x != -1 for x in pred2gt_indices])
            }

            if plot_pred:
                gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, matching_labels=gt_mask, base_scale=image_size, scale=1024)
                floorplan_map = plot_floorplan_with_regions(room_polys, matching_labels=pred_mask, base_scale=image_size, scale=1024)

                concatenated_map = concat_floorplan_maps(gt_floorplan_map, floorplan_map, plot_statistics)
                cv2.imwrite(os.path.join(output_dir, '{}_pred_floorplan.png'.format(scene_ids[i])), concatenated_map)


            if save_pred:
                # Save room_polys as JSON
                json_path = os.path.join(output_dir, 'jsons', '{}.json'.format(str(scene_ids[i]).zfill(5)))
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                polys_list = [poly.astype(float).tolist() for poly in room_polys]
                if semantic_rich:
                    polys_list += [window_door.astype(float).tolist() for window_door in window_doors]
                    types_list = room_types + window_doors_types
                else:
                    types_list = [-1] * len(polys_list)
                
                output_json = [{'image_id': str(scene_ids[i]).zfill(5), 
                                'segmentation': polys_list[instance_id],
                                'category_id': int(types_list[instance_id]),
                                'id': instance_id,
                                } for instance_id in range(len(polys_list))]
                with open(json_path, 'w') as json_file:
                    json.dump(output_json, json_file)


                json_result_path = os.path.join(output_dir, 'result_jsons', '{}.json'.format(str(scene_ids[i]).zfill(5)))
                new_quant_result_dict_scene = compute_f1(copy.deepcopy(quant_result_dict_scene), metric_category)
                os.makedirs(os.path.dirname(json_result_path), exist_ok=True)
                with open(json_result_path, 'w') as json_file:
                    json.dump(new_quant_result_dict_scene, json_file)

            if plot_density:
                pred_room_map = plot_density_map(samples[i], image_size, room_polys, pred_room_label_per_scene, plot_text=False)
                gt_room_map = plot_density_map(samples[i], image_size, gt_polys, gt_polys_types, plot_text=False)

                concatenated_map = concat_floorplan_maps(gt_room_map, pred_room_map, plot_statistics)
                cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), concatenated_map)


    for k in quant_result_dict.keys():
        quant_result_dict[k] /= float(scene_counter)
    quant_result_dict = compute_f1(quant_result_dict, metric_category)

    print("*************************************************")
    print(quant_result_dict)
    print("*************************************************")

    with open(os.path.join(output_dir, 'results.txt'), 'w') as file:
        file.write(json.dumps(quant_result_dict))


@torch.no_grad()
def evaluate_floor_v2(model, dataset_name, data_loader, device, output_dir, plot_pred=True, plot_density=True, plot_gt=True, semantic_rich=False,
                      save_pred=False, add_cls_token=False, per_token_sem_loss=False, iou_thres=0.5,
                      wd_as_line=True):
    model.eval()

    if dataset_name == 'stru3d':
        door_window_index = [16, 17]
    elif dataset_name == 'cubicasa':
        door_window_index = [10, 9]
    elif dataset_name == 'waffle':
        door_window_index = [1, 2]
    else:
        door_window_index = []

    metric_category = ['room','corner','angles']
    if semantic_rich:
        metric_category += ['room_sem','window_door']
    
    quant_result_dict = None
    scene_counter = 0
    merge = False
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for batched_inputs, batched_extras in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) if x["instances"] is not None else None for x in batched_inputs]

        image_size = samples[0].size(2)
        # # draw GT map
        # if plot_gt:
        #     for i, gt_inst in enumerate(gt_instances):
        #         if not semantic_rich:
        #             # plot regular room floorplan
        #             gt_polys = []
        #             density_map = np.transpose((samples[i] * (image_size - 1)).cpu().numpy(), [1, 2, 0])
        #             density_map = np.repeat(density_map, 3, axis=2)

        #             gt_corner_map = np.zeros([image_size, image_size, 3])
        #             for j, poly in enumerate(gt_inst.gt_masks.polygons):
        #                 corners = poly[0].reshape(-1, 2).astype(np.int32)
        #                 gt_polys.append(corners)
        #             gt_room_polys = gt_polys # [np.array(r) for r in gt_polys]
        #             gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, scale=1000)
        #             cv2.imwrite(os.path.join(output_dir, '{}_gt.png'.format(scene_ids[i])), gt_floorplan_map)
        #         else:
        #             # plot semantically-rich floorplan
        #             gt_sem_rich = []
        #             for j, poly in enumerate(gt_inst.gt_masks.polygons):
        #                 corners = poly[0].reshape(-1, 2).astype(np.int32)
        #                 corners_flip_y = corners.copy()
        #                 corners_flip_y[:,1] = image_size - 1 - corners_flip_y[:,1]
        #                 corners = corners_flip_y
        #                 gt_sem_rich.append([corners, gt_inst.gt_classes.cpu().numpy()[j]])

        #             gt_sem_rich_path = os.path.join(output_dir, '{}_sem_rich_gt.png'.format(scene_ids[i]))
        #             # plot_semantic_rich_flooplan(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1) 
        #             plot_semantic_rich_floorplan_tight(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1, plot_text=True, is_bw=False, door_window_index=door_window_index)

        outputs = model.forward_inference(samples)
        pred_corners = outputs['gen_out']
        bs = outputs['pred_logits'].shape[0]
        np_softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

        if 'pred_room_logits' in outputs:
            prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
            num_classes = prob.shape[-1]
            _, pred_room_label = prob[..., :-1].max(-1)
            pred_room_logits = outputs['pred_room_logits']

        # process per scene
        for i in range(bs):

            gt_polys, gt_polys_types = [], []
            gt_window_doors = []
            gt_window_doors_types = []
            for gt_poly, gt_id in zip(gt_instances[i].gt_masks.polygons, gt_instances[i].gt_classes.detach().cpu().tolist()):
                gt_poly = gt_poly[0].reshape(-1,2).astype(np.int32)
                if gt_id in door_window_index:
                    gt_window_doors.append(gt_poly)
                    gt_window_doors_types.append(gt_id)
                else:
                    gt_polys.append(gt_poly)
                    gt_polys_types.append(gt_id)
            
            if dataset_name == 'stru3d':
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode = "test") # TODO: return to "test"
                evaluator = Evaluator(curr_data_rw, curr_opts, disable_overlapping_filter=True)
            elif dataset_name == 'scenecad':
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1,2).astype(np.int32)]
                evaluator = Evaluator_SceneCAD(disable_overlapping_filter=True)
            elif dataset_name == 'rplan':
                gt_polys = [x[0].reshape(-1,2).astype(np.int32) for x in gt_instances[i].gt_masks.polygons]
                gt_polys_types = gt_instances[i].gt_classes.detach().cpu().tolist()
                evaluator = Evaluator_RPlan(disable_overlapping_filter=True)
            elif dataset_name in ['cubicasa', 'waffle', 'r2g']:
                evaluator = Evaluator_RPlan(disable_overlapping_filter=True, iou_thres=iou_thres,
                                            wd_as_line=wd_as_line)

            print("Running Evaluation for scene %s" % scene_ids[i])

            # fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]
            room_polys = []

            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy()
                pred_room_logit_per_scene = pred_room_logits[i].cpu().numpy()

            all_room_polys = []
            tmp = []
            all_length_list = [0]
            for j in range(len(pred_corners_per_scene)):
                if isinstance(pred_corners_per_scene[j], int):
                    if pred_corners_per_scene[j] == 2 and tmp: # sep
                        all_room_polys.append(tmp)
                        all_length_list.append(len(tmp)+1+add_cls_token)
                        tmp = []
                    continue
                tmp.append(pred_corners_per_scene[j])
            
            if len(tmp):
                all_room_polys.append(tmp)
                all_length_list.append(len(tmp)+1+add_cls_token)
            start_poly_indices = np.cumsum(all_length_list)

            final_pred_classes = []
            for j, poly in enumerate(all_room_polys):
                if len(poly) < 2:
                    continue
                corners = np.array(poly, dtype=np.float32) * (image_size - 1)
                corners = np.around(corners).astype(np.int32)
                # print(len(corners))

                if not semantic_rich:
                    # only regular rooms
                    if len(corners)>=4 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                    # room_polys.append(corners)
                else:
                    if per_token_sem_loss:
                        pred_classes, counts = np.unique(pred_room_label_per_scene[start_poly_indices[j]:start_poly_indices[j+1]][:-1], return_counts=True)
                        pred_class = pred_classes[np.argmax(counts)]
                        pred_logit = pred_room_logit_per_scene[start_poly_indices[j]:start_poly_indices[j+1]][:-1]
                    else:
                        pred_class = pred_room_label_per_scene[start_poly_indices[j+1]-1] # get last cls token in the seq
                    final_pred_classes.append(pred_class)

                    # # regular rooms
                    # if len(corners)>=4 and Polygon(corners).area >= 100:
                    #     room_polys.append(corners)
                    #     room_types.append(pred_class)
                    # # window / door
                    # elif len(corners)==2:
                    #     window_doors.append(corners)
                    #     window_doors_types.append(pred_class)

                    if wd_as_line:
                        # regular rooms
                        if (len(corners)>=3 and Polygon(corners).area >= 100):
                            room_polys.append(corners)
                            room_types.append(pred_class)
                        # window / door
                        elif len(corners)==2 and dataset_name != 'r2g':
                            window_doors.append(corners)
                            if pred_class not in door_window_index:
                                wd_prob = np_softmax(pred_logit[:, door_window_index].sum(0))
                                pred_class = door_window_index[wd_prob.argmax()]
                                # pred_class = np.random.choice(door_window_index, size=1, p=wd_prob)
                                # pred_class = door_window_index[np.random.rand() > 0.5]
                            
                            window_doors_types.append(pred_class)
                    else:
                        # regular rooms
                        if pred_class not in door_window_index:
                            room_polys.append(corners)
                            room_types.append(pred_class)
                        else:
                            window_doors.append(corners)
                            window_doors_types.append(pred_class)
            
            if not semantic_rich:
                pred_room_label_per_scene = len(all_room_polys) * [-1]
            else:
                pred_room_label_per_scene = final_pred_classes

            if dataset_name == 'stru3d':
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                                                            room_polys=room_polys, 
                                                            room_types=room_types, 
                                                            window_door_lines=window_doors, 
                                                            window_door_lines_types=window_doors_types)
    
            elif dataset_name == 'scenecad':
                quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)

            elif dataset_name in ['rplan', 'cubicasa', 'waffle', 'r2g']:
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=None, gt_polys_types=gt_polys_types,
                                                                img_size=(image_size, image_size),
                                                                    )
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys,
                                                                room_types=room_types, gt_polys_types=gt_polys_types,
                                                                window_door_lines=window_doors, gt_window_doors_list=gt_window_doors,
                                                                window_door_lines_types=window_doors_types, gt_window_doors_type_list=gt_window_doors_types,
                                                                img_size=(image_size, image_size),
                                                                )


            if quant_result_dict is None:
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

            # plot regular room floorplan
            gt_room_polys = [np.array(poly) for poly in gt_polys]
            room_polys = [np.array(poly) for poly in room_polys]

            if 'gt_polys_sorted_indcs' in quant_result_dict_scene:
                gt_polys_sorted_indcs = quant_result_dict_scene['gt_polys_sorted_indcs']
                del quant_result_dict_scene['gt_polys_sorted_indcs']
                gt_room_polys = [gt_room_polys[ind] for ind in gt_polys_sorted_indcs]
                    

            if 'pred2gt_indices' in quant_result_dict_scene:
                pred2gt_indices = quant_result_dict_scene['pred2gt_indices']
                del quant_result_dict_scene['pred2gt_indices']
                room_polys, gt_room_polys, pred_mask, gt_mask = sort_polygons_by_matching(pred2gt_indices, room_polys, gt_room_polys)
            else:
                pred_mask, gt_mask = None, None

            prec, rec = quant_result_dict_scene['room_prec'], quant_result_dict_scene['room_rec']
            f1 = 2*prec*rec/(prec+rec+1e-5)
            missing_rate = quant_result_dict_scene['room_missing_ratio']
            plot_statistics = {
                'f1': f1, 'prec': prec, 'rec': rec, 'missing_rate': missing_rate, 
                'num_preds': len(room_polys), 'num_gt': len(gt_polys), 'num_matched_preds': sum([x != -1 for x in pred2gt_indices])
            }

            if plot_pred:
                gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, matching_labels=gt_mask, base_scale=image_size, scale=1024)
                floorplan_map = plot_floorplan_with_regions(room_polys, matching_labels=pred_mask, base_scale=image_size, scale=1024)
                if merge:
                    concatenated_map = concat_floorplan_maps(gt_floorplan_map, floorplan_map, plot_statistics)
                    cv2.imwrite(os.path.join(output_dir, '{}_pred_floorplan.png'.format(scene_ids[i])), concatenated_map)
                else:
                    cv2.imwrite(os.path.join(output_dir, '{}_pred_floorplan.png'.format(scene_ids[i])), floorplan_map)
                    cv2.imwrite(os.path.join(output_dir, '{}_gt_floorplan.png'.format(scene_ids[i])), gt_floorplan_map)


                if semantic_rich:
                    _, ID2CLASS_LABEL = get_dataset_class_labels(dataset_name)
                    floorplan_map = plot_semantic_rich_floorplan_opencv(zip(room_polys+window_doors, room_types+window_doors_types), 
                        os.path.join(output_dir, '{}_pred_floorplan_sem.png'.format(scene_ids[i])), door_window_index=door_window_index,
                        semantics_label_mapping=ID2CLASS_LABEL, img_w=image_size, img_h=image_size, scale=1,
                        plot_text=False)

            if save_pred:
                # Save room_polys as JSON
                json_path = os.path.join(output_dir, 'jsons', '{}.json'.format(str(scene_ids[i]).zfill(5)))
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                polys_list = [poly.astype(float).tolist() for poly in room_polys]
                if semantic_rich:
                    polys_list += [window_door.astype(float).tolist() for window_door in window_doors]
                    types_list = room_types + window_doors_types
                else:
                    types_list = [-1] * len(polys_list)
                
                output_json = [{'image_id': str(scene_ids[i]).zfill(5), 
                                'segmentation': polys_list[instance_id],
                                'category_id': int(types_list[instance_id]),
                                'id': instance_id,
                                } for instance_id in range(len(polys_list))]
                with open(json_path, 'w') as json_file:
                    json.dump(output_json, json_file)

                json_result_path = os.path.join(output_dir, 'result_jsons', '{}.json'.format(str(scene_ids[i]).zfill(5)))
                new_quant_result_dict_scene = compute_f1(copy.deepcopy(quant_result_dict_scene), metric_category)
                os.makedirs(os.path.dirname(json_result_path), exist_ok=True)
                with open(json_result_path, 'w') as json_file:
                    json.dump(new_quant_result_dict_scene, json_file)
            
            if plot_gt:
                gt_image = np.transpose(samples[i].cpu().numpy(), (1, 2, 0))
                gt_image = (gt_image * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, '{}_gt.png'.format(scene_ids[i])), gt_image)

            if plot_density:
                pred_room_map = plot_density_map(samples[i], image_size, room_polys, pred_room_label_per_scene, plot_text=False)
                gt_room_map = plot_density_map(samples[i], image_size, gt_polys, gt_polys_types, plot_text=False)

                if merge:
                    concatenated_map = concat_floorplan_maps(gt_room_map, pred_room_map, plot_statistics)
                    cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), concatenated_map)
                else:
                    cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), pred_room_map)
                    cv2.imwrite(os.path.join(output_dir, '{}_gt_room_map.png'.format(scene_ids[i])), gt_room_map)


    for k in quant_result_dict.keys():
        quant_result_dict[k] /= float(scene_counter)
    quant_result_dict = compute_f1(quant_result_dict, metric_category)

    print("*************************************************")
    print(quant_result_dict)
    print("*************************************************")

    with open(os.path.join(output_dir, 'results.txt'), 'w') as file:
        file.write(json.dumps(quant_result_dict))


def generate(model, samples, semantic_rich=False, drop_wd=False):
    model.eval()
    outputs = model(samples)
    pred_corners = outputs['pred_coords']
    pred_logits = outputs['pred_logits']
    fg_mask = torch.sigmoid(pred_logits) > 0.5 # select valid corners

    bs = outputs['pred_logits'].shape[0]
    image_size = samples[0].size(2)

    if 'pred_room_logits' in outputs:
        prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
        _, pred_room_label = prob[..., :-1].max(-1)
    
    outputs = []
    output_classes = []

    # process per scene
    for i in range(bs):
        fg_mask_per_scene = fg_mask[i]
        pred_corners_per_scene = pred_corners[i]
        room_polys = []

        if semantic_rich:
            room_types = []
            window_doors = []
            window_doors_types = []
            pred_room_label_per_scene = pred_room_label[i].cpu().numpy()
        else:
            window_doors = None
            room_types = None


        # process per room
        for j in range(fg_mask_per_scene.shape[0]):
            fg_mask_per_room = fg_mask_per_scene[j]
            pred_corners_per_room = pred_corners_per_scene[j]
            valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
            if len(valid_corners_per_room)>0:
                corners = (valid_corners_per_room * (image_size - 1)).cpu().numpy()
                corners = np.around(corners).astype(np.int32)

                if not semantic_rich:
                    # only regular rooms
                    if len(corners)>=4 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                else:
                    # regular rooms
                    if len(corners)>=3 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                        room_types.append(pred_room_label_per_scene[j])
                    # window / door
                    elif len(corners)==2:
                        window_doors.append(corners)
                        window_doors_types.append(pred_room_label_per_scene[j])
        
        if not semantic_rich:
            pred_room_label_per_scene = len(room_polys) * [-1]
        else:
            pred_room_label_per_scene = pred_room_label_per_scene.tolist()

        if window_doors and not drop_wd:
            outputs.append(room_polys + window_doors)
            output_classes.append(room_types + window_doors_types)
        else:
            outputs.append(room_polys)
            output_classes.append(room_types)
    
    return {
        'room': outputs, 
        'labels': output_classes
    }


def generate_v2(model, samples, semantic_rich=False, use_cache=True, per_token_sem_loss=False, drop_wd=False, return_anchors=False):
    model.eval()
    outputs = model.forward_inference(samples, use_cache)
    pred_corners = outputs['gen_out']

    bs = outputs['pred_logits'].shape[0]
    image_size = samples[0].size(2)
    anchors = outputs.get('anchors', None)
    if anchors is not None:
        anchors = (torch.sigmoid(anchors) * image_size).cpu().numpy().astype(np.int32)
        anchors = anchors.tolist()

    if 'pred_room_logits' in outputs:
        prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
        _, pred_room_label = prob[..., :-1].max(-1)
    
    outputs = []
    output_classes = []
    output_anchors = []

    # process per scene
    for i in range(bs):
        pred_corners_per_scene = pred_corners[i]
        room_polys = []

        if semantic_rich:
            room_types = []
            window_doors = []
            window_doors_types = []
            pred_room_label_per_scene = pred_room_label[i].cpu().numpy()
        else:
            window_doors = None
            room_types = None
            pred_room_label_per_scene = [-1] * len(pred_corners_per_scene)

        all_room_polys = []
        tmp = []
        anchor_tmp = []
        all_length_list = [0]
        all_anchor_list = []
        for j in range(len(pred_corners_per_scene)):
            if isinstance(pred_corners_per_scene[j], int):
                if pred_corners_per_scene[j] == 2 and tmp: # sep
                    all_room_polys.append(tmp)
                    all_length_list.append(len(tmp)+1)
                    all_anchor_list.append(anchor_tmp)
                    tmp = []
                    anchor_tmp = []
                continue
            tmp.append(pred_corners_per_scene[j])
            anchor_tmp.append(anchors[j])
        
        if len(tmp):
            all_room_polys.append(tmp)
            all_length_list.append(len(tmp)+1)
            all_anchor_list.append(anchor_tmp)
        start_poly_indices = np.cumsum(all_length_list)

        final_pred_classes = []
        pred_room_anchors = []
        pred_window_anchors = []
        for j, poly in enumerate(all_room_polys):
            if len(poly) < 2:
                continue
            corners = np.array(poly, dtype=np.float32) * (image_size - 1)
            corners = np.around(corners).astype(np.int32)
            # print(len(corners))

            if not semantic_rich:
                # only regular rooms
                if len(corners)>=4 and Polygon(corners).area >= 100:
                    room_polys.append(corners)
                    if all_anchor_list:
                        pred_room_anchors.append(all_anchor_list[j])
            else:
                if per_token_sem_loss:
                    pred_classes, counts = np.unique(pred_room_label_per_scene[start_poly_indices[j]:start_poly_indices[j+1]][:-1], return_counts=True)
                    pred_class = pred_classes[np.argmax(counts)]
                else:
                    pred_class = pred_room_label_per_scene[start_poly_indices[j+1]-1] # get last cls token in the seq
                final_pred_classes.append(pred_class)

                # regular rooms
                if len(corners)>=3 and Polygon(corners).area >= 100:
                    room_polys.append(corners)
                    room_types.append(pred_class)
                    if all_anchor_list:
                        pred_room_anchors.append(all_anchor_list[j])
                # window / door
                elif len(corners)==2:
                    window_doors.append(corners)
                    window_doors_types.append(pred_class)
                    if all_anchor_list:
                        pred_window_anchors.append(all_anchor_list[j])
            
            
        if not semantic_rich:
            pred_room_label_per_scene = len(all_room_polys) * [-1]
        else:
            pred_room_label_per_scene = final_pred_classes

        if not drop_wd and window_doors:
            outputs.append(room_polys + window_doors)
            output_classes.append(room_types + window_doors_types)
            output_anchors.append(pred_room_anchors+pred_window_anchors)
        else:
            outputs.append(room_polys)
            output_classes.append(room_types)
            output_anchors.append(pred_room_anchors)
    
    out_dict = {
        'room': outputs, 
        'labels': output_classes
    }
    if return_anchors:
        out_dict.update({'anchors': output_anchors})

    return out_dict

def concat_floorplan_maps(gt_floorplan_map, floorplan_map, plot_statistics={}):
    pad_color = (0,0,0) if gt_floorplan_map.shape[2] == 3 else (0,0,0,0)
    padding = np.full((gt_floorplan_map.shape[0], 10, gt_floorplan_map.shape[2]), pad_color, dtype=np.uint8)
    # Concatenate pred_room_map, padding, and gt_room_map
    concatenated_map = cv2.hconcat([gt_floorplan_map, padding, floorplan_map])
    top_padding = np.full((100, concatenated_map.shape[1], concatenated_map.shape[2]), pad_color, dtype=np.uint8)

    # Add text for f1 and missing_rate
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255) if gt_floorplan_map.shape[2] == 3 else (0, 0, 255, 255) # White text
    thickness = 2
    line_type = cv2.LINE_AA

    # Position for the text
    text_f1 = f"F1: {plot_statistics['f1']:.2f}, Prec: {plot_statistics['prec']:.2f}, Rec: {plot_statistics['rec']:.2f}"
    text_missing_rate = f"Missing Rate: {plot_statistics['missing_rate']:.2f}, {plot_statistics['num_preds']}/{plot_statistics['num_matched_preds']}/{plot_statistics['num_gt']}"
    text_position_f1 = (10, 30)  # Position within the top padding
    text_position_missing_rate = (10, 70)  # Adjusted position for the second line

    # Overlay text on the top padding
    cv2.putText(top_padding, text_f1, text_position_f1, font, font_scale, font_color, thickness, line_type)
    cv2.putText(top_padding, text_missing_rate, text_position_missing_rate, font, font_scale, font_color, thickness, line_type)

    # Concatenate the top padding with the concatenated_map
    final_map = cv2.vconcat([top_padding, concatenated_map])
    return final_map
