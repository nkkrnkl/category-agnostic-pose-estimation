import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path
import copy
from tqdm import trange

import numpy as np
from shapely.geometry import Polygon
import torch
from torch.utils.data import DataLoader
import cv2


from datasets import build_dataset

from s3d_floorplan_eval.Evaluator.Evaluator import Evaluator
from s3d_floorplan_eval.options import MCSSOptions
from s3d_floorplan_eval.DataRW.S3DRW import S3DRW
from s3d_floorplan_eval.DataRW.wrong_annotatios import wrong_s3d_annotations_list

from scenecad_eval.Evaluator import Evaluator_SceneCAD
from rplan_eval.Evaluator import Evaluator_RPlan
from engine import concat_floorplan_maps

import util.misc as utils
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan, plot_semantic_rich_floorplan_tight, sort_polygons_by_matching
from util.plot_utils import plot_density_map, plot_semantic_rich_floorplan_opencv
from util.plot_utils import S3D_LABEL, CC5K_LABEL
from util.eval_utils import compute_f1

from datasets import get_dataset_class_labels

options = MCSSOptions()
opts = options.parse()

def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)

    # new
    parser.add_argument('--input_json_dir', type=str)
    parser.add_argument('--input_file_type', default='json')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--input_channels', default=1, type=int)
    parser.add_argument('--image_norm', action='store_true')
    parser.add_argument('--eval_every_epoch', type=int, default=20)
    parser.add_argument('--ckpt_every_epoch', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.)
    parser.add_argument('--ignore_index', type=int, default=-1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ema4eval', action='store_true')
    parser.add_argument('--measure_time', action='store_true')
    parser.add_argument('--disable_sampling_cache', action='store_true')
    parser.add_argument('--use_anchor', action='store_true')
    parser.add_argument('--iou_thres', type=float, default=0.5)
    parser.add_argument('--disable_sem_rich', action='store_true')
    parser.add_argument('--wd_only', action='store_true')
    parser.add_argument('--disable_image_transform', action='store_true')
    parser.add_argument('--num_subset_images', type=int, default=-1)
    parser.add_argument('--model_version', type=str, default='v1')
    parser.add_argument('--converter_version', type=str, default='v1')
    parser.add_argument('--inject_cls_embed', action='store_true')

    # poly2seq
    parser.add_argument('--poly2seq', action='store_true')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--num_bins', type=int, default=64)
    parser.add_argument('--pre_decoder_pos_embed', action='store_true')
    parser.add_argument('--learnable_dec_pe', action='store_true')
    parser.add_argument('--dec_qkv_proj', action='store_true')
    parser.add_argument('--dec_attn_concat_src', action='store_true')
    parser.add_argument('--dec_layer_type', type=str, default='v1')
    parser.add_argument('--per_token_sem_loss', action='store_true')
    parser.add_argument('--add_cls_token', action='store_true')

    # backbone
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
    parser.add_argument('--disable_poly_refine', action='store_true',
                        help="iteratively refine reference points (i.e. positional part of polygon queries)")

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
    parser.add_argument('--plot_pred', default=True, type=bool, help="plot predicted floorplan")
    parser.add_argument('--plot_density', default=True, type=bool, help="plot predicited room polygons overlaid on the density map")
    parser.add_argument('--plot_gt', default=True, type=bool, help="plot ground truth floorplan")
    parser.add_argument('--save_pred', action='store_true', help="save_pred")

    return parser


def evaluate_floor(json_root, dataset_name, data_loader, device, output_dir, plot_pred=True, plot_density=True, plot_gt=True, semantic_rich=False,
                   save_pred=False, iou_thres=0.5, args=None):
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
        gt_instances = [x["instances"].to(device) for x in batched_inputs]

        image_size = samples[0].size(2)

        # process per scene
        for i in range(len(samples)):
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
                evaluator = Evaluator(curr_data_rw, curr_opts, disable_overlapping_filter=True)
            elif dataset_name == 'scenecad':
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1,2).astype(np.int32)]
                evaluator = Evaluator_SceneCAD(disable_overlapping_filter=True)
            elif dataset_name == 'rplan':
                gt_polys = [x[0].reshape(-1,2).astype(np.int32) for x in gt_instances[i].gt_masks.polygons]
                gt_polys_types = gt_instances[i].gt_classes.detach().cpu().tolist()
                evaluator = Evaluator_RPlan(disable_overlapping_filter=True)
            elif dataset_name in ['cubicasa', 'waffle', 'r2g']:
                evaluator = Evaluator_RPlan(disable_overlapping_filter=True, iou_thres=iou_thres)

            print("Running Evaluation for scene %s" % scene_ids[i])

            room_polys = []

            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []

            if dataset_name == 'r2g':
                pred_path = os.path.join(json_root, str(scene_ids[i]).zfill(6)) + f'.{args.input_file_type}'
            else:
                pred_path = os.path.join(json_root, str(scene_ids[i]).zfill(5)) + f'.{args.input_file_type}'

            if not os.path.exists(pred_path):
                continue
            if args.input_file_type == 'npy':
                data = np.load(pred_path, allow_pickle=True)
                pred_corners = [np.around(np.array(x, dtype=np.int32)).reshape(-1,2) for x in data]
                pred_room_label_per_scene = [-1] * len(pred_corners)
            else:
                with open(pred_path, 'r') as f:
                    pred_data = json.load(f)
                    pred_corners = [np.around(np.array(x['segmentation'])).astype(np.int32).reshape(-1, 2) for x in pred_data]
                    pred_room_label_per_scene = np.array([x['category_id'] for x in pred_data])

            # process per room
            for j in range(len(pred_corners)):
                corners = pred_corners[j]
                if np.all(corners[0] == corners[-1]):
                    corners = corners[:-1]
                pred_class_id = pred_room_label_per_scene[j]
                if not semantic_rich:
                    # only regular rooms
                    if len(corners)>=4 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                else:
                    # regular rooms
                    # if pred_room_label_per_scene[j] not in door_window_index:
                    if len(corners)>=3 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                        room_types.append(pred_class_id)
                    # window / door
                    elif len(corners)==2 and dataset_name != 'r2g':
                        window_doors.append(corners)
                        window_doors_types.append(pred_class_id)
            
            if not semantic_rich:
                room_types = len(room_polys) * [-1]
                
            
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
                json_path = os.path.join(output_dir, 'jsons', '{}_pred.json'.format(str(scene_ids[i]).zfill(5)))
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
                pred_room_map = plot_density_map(samples[i], image_size, room_polys, room_types, plot_text=False)
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


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build dataset and dataloader
    dataset_eval = build_dataset(image_set=args.eval_set, args=args)

    tokenizer = None
    if args.poly2seq:
        args.vocab_size = dataset_eval.get_vocab_size()
        tokenizer = dataset_eval.get_tokenizer()

    # overfit one sample
    if args.debug:
        dataset_eval = torch.utils.data.Subset(dataset_eval, [480])
        dataset_eval[0]

    if args.num_subset_images > 0 and args.num_subset_images < len(dataset_eval):
        dataset_eval = torch.utils.data.Subset(dataset_eval, range(args.num_subset_images))
        
    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch, None

    data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                 pin_memory=True)

    output_dir = Path(args.output_dir)
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    evaluate_floor(
                args.input_json_dir, args.dataset_name, data_loader_eval, 
                device, save_dir, 
                plot_pred=args.plot_pred, 
                plot_density=args.plot_density, 
                plot_gt=args.plot_gt,
                semantic_rich=(args.semantic_classes>0 and not args.disable_sem_rich),
                save_pred=args.save_pred,
                iou_thres=args.iou_thres,
                args=args
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug:
        args.batch_size = 1
    if args.disable_poly_refine:
        args.with_poly_refine = False

    main(args)
