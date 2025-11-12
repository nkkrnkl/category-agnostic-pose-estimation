import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path
import copy
from tqdm import trange, tqdm

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from detectron2.data import transforms as T
import torchvision

from PIL import Image

import util.misc as utils
from datasets.transforms import ResizeAndPad
from datasets import build_dataset
from datasets.data_utils import sort_polygons
from datasets.discrete_tokenizer import DiscreteTokenizer
from engine import evaluate_floor, evaluate_floor_v2, plot_density_map, plot_floorplan_with_regions
from util.plot_utils import plot_semantic_rich_floorplan_opencv
from util.plot_utils import S3D_LABEL, CC5K_LABEL, auto_crop_whitespace
from engine import generate, generate_v2
from models import build_model


class ImageDataset(Dataset):
    def __init__(self, image_paths, num_image_channels=3, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.num_image_channels = num_image_channels

    def __len__(self):
        return len(self.image_paths)

    def _expand_image_dims(self, x):
        if len(x.shape) == 2:
            exp_img = np.expand_dims(x, 0)
        else:
            exp_img = x.transpose((2, 0, 1)) # (h,w,c) -> (c,h,w)
        return exp_img

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to fetch.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        img_path = self.image_paths[idx]
        if self.num_image_channels == 3:
            image = np.array(Image.open(img_path).convert("RGB"))  # Ensure 3-channel RGB
        else:
            image = np.array(Image.open(img_path))  # Ensure 1-channel RGB
        if self.transform:
            aug_input = T.AugInput(image)
            _ = self.transform(aug_input)
            image = aug_input.image

        image = (1/255) * torch.as_tensor(np.array(self._expand_image_dims(image)))
        return {
            'file_name': img_path,
            'image': image,
            }


def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)

    # new
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
    parser.add_argument('--drop_wd', action='store_true')
    parser.add_argument('--plot_text', action='store_true')
    parser.add_argument('--image_scale', type=int, default=2)
    parser.add_argument('--one_color', action='store_true')
    parser.add_argument('--crop_white_space', action='store_true')
    parser.add_argument('--save_anchors', action='store_true')

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
    parser.add_argument('--plot_gt', default=False, type=bool, help="plot ground truth floorplan")
    parser.add_argument('--save_pred', action='store_true', help="save_pred")

    return parser


def get_image_paths_from_directory(directory_path):
    """
    Load all images from the specified directory.

    Args:
        directory_path (str): Path to the directory containing images.

    Returns:
        list: A list of PIL Image objects.
    """
    paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Add more extensions if needed

    # Iterate through all files in the directory
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(valid_extensions):  # Check for valid image extensions
                file_path = os.path.join(root, filename)
                paths.append(file_path)

    return paths


def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    image_paths = get_image_paths_from_directory(args.dataset_root)
    data_transform = T.AugmentationList([
        ResizeAndPad((args.image_size, args.image_size), pad_value=255),
    ])
    dataset_eval = ImageDataset(image_paths, num_image_channels=args.input_channels, transform=data_transform)

    tokenizer = None
    if args.poly2seq:
        tokenizer = DiscreteTokenizer(args.num_bins, args.seq_len, add_cls=args.add_cls_token)
        args.vocab_size = len(tokenizer)

    # overfit one sample
    if args.debug:
        idx = 0
        for i, x in enumerate(dataset_eval):
            if '3252' in x['file_name']:
                idx = i
        dataset_eval = torch.utils.data.Subset(dataset_eval, [idx])

    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)
    data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval,
                                 drop_last=False, num_workers=args.num_workers,
                                 pin_memory=True)

    # build model
    model = build_model(args, train=False, tokenizer=tokenizer)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if args.ema4eval:
        ckpt_state_dict = copy.deepcopy(checkpoint['ema'])
    else:
        ckpt_state_dict = copy.deepcopy(checkpoint['model'])
    for key, value in checkpoint['model'].items():
        if key.startswith('module.'):
            ckpt_state_dict[key[7:]] = checkpoint['model'][key]
            del ckpt_state_dict[key]
    missing_keys, unexpected_keys = model.load_state_dict(ckpt_state_dict, strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    # disable grad
    for param in model.parameters():
        param.requires_grad = False

    save_dir = os.path.join(args.output_dir, os.path.dirname(args.checkpoint).split('/')[-1])
    os.makedirs(save_dir, exist_ok=True)

    semantics_label_mapping = None
    if args.dataset_name == 'stru3d':
        door_window_index = [16, 17]
        semantics_label_mapping = S3D_LABEL
    elif args.dataset_name == 'cubicasa':
        door_window_index = [10, 9]
        semantics_label_mapping = CC5K_LABEL
    elif args.dataset_name == 'waffle':
        door_window_index = [1, 2]
    else:
        door_window_index = []

    if args.measure_time:
        images = torch.rand(args.batch_size, 3, args.image_size, args.image_size).to(device)
        if args.poly2seq:
            model = torch.compile(model) # compile model is not compatible with RoomFormer
        # GPU-WARM-UP
        for _ in trange(10, desc="GPU-WARM-UP"):
            if not args.poly2seq:
                _ = model(images)
            else:
                _ = model.forward_inference(images)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)

    total_time = 0.
    for batch_images in tqdm(data_loader_eval):
        starter.record()
        x = batch_images['image'].to(device)
        filenames = batch_images['file_name']
        if not args.poly2seq:
            outputs = generate(model,
                    x,
                    semantic_rich=args.semantic_classes>0, 
                    drop_wd=args.drop_wd,
                    )
        else:
            outputs = generate_v2(model, 
                    x,
                    semantic_rich=args.semantic_classes>0, 
                    use_cache=True,
                    per_token_sem_loss=args.per_token_sem_loss,
                    drop_wd=args.drop_wd,
                    return_anchors=args.save_anchors,
                    )
        ender.record()
        torch.cuda.synchronize()
        total_time += starter.elapsed_time(ender) / len(data_loader_eval)

        pred_rooms = outputs['room']
        pred_labels = outputs['labels']

        # _, sorted_indices = sort_polygons(pred_rooms)
        # pred_rooms = [ for p in pred_rooms]

        image_size = x.shape[-2]
        for j, (pred_rm, pred_cls) in enumerate(zip(pred_rooms, pred_labels)):
            if pred_cls is None: pred_cls = [-1] * len(pred_rm)
            fn = os.path.basename(filenames[j]).split('.')[0]
            pred_room_map = plot_density_map(x[j], image_size, 
                                             pred_rm, pred_cls, plot_text=args.plot_text,)

            floorplan_map = plot_semantic_rich_floorplan_opencv(zip(pred_rm, pred_cls), 
                None, door_window_index=door_window_index,
                semantics_label_mapping=semantics_label_mapping, 
                plot_text=args.plot_text, one_color=args.one_color,
                is_sem=args.semantic_classes > 0,
                img_w=image_size*args.image_scale, img_h=image_size*args.image_scale,
                # img_w=image_size, img_h=image_size,
                scale=args.image_scale)

            # floorplan_map2 = plot_floorplan_with_regions(pred_rm, scale=image_size*args.image_scale, matching_labels=pred_cls,
            #                             regions_type=pred_cls, plot_text=args.plot_text, semantics_label_mapping=semantics_label_mapping)

            image = x[j].permute(1, 2, 0).cpu().numpy() * 255
            if args.crop_white_space:
                image = cv2.resize(image, (args.image_scale*args.image_size, args.image_scale*args.image_size), interpolation=cv2.INTER_NEAREST)  
                image, cropped_box = auto_crop_whitespace(image)
                # _x,_y,_w,_h = [ele * args.image_scale for ele in cropped_box]
                _x,_y,_w,_h = [ele for ele in cropped_box]

                floorplan_map = floorplan_map[_y:_y+_h, _x:_x+_w].copy()
                floorplan_map2 = floorplan_map[_y:_y+_h, _x:_x+_w].copy()


            # Ensure images are not empty before saving
            if pred_room_map is not None and pred_room_map.size > 0:
                cv2.imwrite(os.path.join(save_dir, '{}_pred_room_map.png'.format(fn)), pred_room_map)
            else:
                print("Warning: pred_room_map is empty, skipping save.")

            if floorplan_map is not None and floorplan_map.size > 0:
                cv2.imwrite(os.path.join(save_dir, '{}_pred_floorplan.png'.format(fn)), floorplan_map)
            else:
                print("Warning: floorplan_map is empty, skipping save.")

            if image is not None and image.size > 0:
                cv2.imwrite(os.path.join(save_dir, '{}.png'.format(fn)), image)
            else:
                print("Warning: image is empty, skipping save.")

            # cv2.imwrite(os.path.join(save_dir, '{}_pred_room_map.png'.format(fn)), pred_room_map)
            # cv2.imwrite(os.path.join(save_dir, '{}_pred_floorplan.png'.format(fn)), floorplan_map)
            # cv2.imwrite(os.path.join(save_dir, '{}.png'.format(fn)), image)
            # cv2.imwrite(os.path.join(save_dir, '{}_pred_floorplan_nice.png'.format(fn)), floorplan_map2)

            if args.save_pred:
                # Save room_polys as JSON
                json_path = os.path.join(save_dir, 'jsons', '{}.json'.format(fn))
                npy_path = os.path.join(save_dir, 'npy', '{}.npy'.format(fn))
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                polys_list = [poly.astype(float).tolist() for poly in pred_rm]
                types_list = pred_cls
                # else:
                #     types_list = [-1] * len(polys_list)
                
                if not args.save_anchors:
                    output_json = [{'image_id': fn, 
                                    'segmentation': polys_list[instance_id],
                                    'category_id': int(types_list[instance_id]),
                                    'id': instance_id,
                                    } for instance_id in range(len(polys_list))]
                else:
                    image_anchors = outputs['anchors'][j]
                    output_json = [{'image_id': fn, 
                                    'segmentation': polys_list[instance_id],
                                    'category_id': int(types_list[instance_id]),
                                    'id': instance_id,
                                    'anchors': image_anchors[instance_id],
                                    } for instance_id in range(len(polys_list))]
                
                with open(json_path, 'w') as json_file:
                    json.dump(output_json, json_file)

                polys_list = [np.array(poly).reshape(-1,2) for poly in polys_list]
                np.save(npy_path, np.array(polys_list, dtype=object))

                

    print(f"Total inference time: {total_time:.2f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug:
        args.batch_size = 1
    if args.disable_poly_refine:
        args.with_poly_refine = False

    main(args)
