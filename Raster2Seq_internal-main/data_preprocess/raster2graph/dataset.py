import copy
import json
import os
from collections import defaultdict
import numpy as np
import torch
import torch.utils.data
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from util.data_utils import l1_dist
from util.graph_utils import graph_to_tensor
from util.image_id_dict import d
from util.mean_std import mean, std
from util.semantics_dict import semantics_dict

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class MyDataset(Dataset):
    def __init__(self, img_path, annot_path, extract_roi, image_size=512):
        self.img_path = img_path
        self.quadtree_path = '/'.join(img_path.split('/')[:-1]) + '/annot_npy'
        self.mode = img_path.split('/')[-1]
        self.image_size = image_size

        # load annotation
        with open(annot_path, 'r') as f:
            dataset = json.load(f)
        # images
        self.imgs = {}
        for img in dataset['images']:
            self.imgs[img['id']] = img
        self.imgToAnns = defaultdict(list)
        for ann in dataset['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)
        self.ids = list(sorted(self.imgs.keys()))
        if '0c-10-c468a57377ff8ef63d3b26a6d1fa-0002' in self.ids:
            self.ids.remove('0c-10-c468a57377ff8ef63d3b26a6d1fa-0002')
        if '0c-10-8486f08035ba152d5244ac54099c-0001' in self.ids:
            self.ids.remove('0c-10-8486f08035ba152d5244ac54099c-0001')


    def __getitem__(self, index):
        img_id = self.ids[index]
        img_file_name = self.imgs[img_id]['file_name'].replace('.jpg', '.png')
        img = Image.open(os.path.join(self.img_path, img_file_name)).convert('RGB')
        image_scale = self.image_size / img.size[0]
        if img.size[0] != self.image_size or img.size[1] != self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if 1:
            # get structure annotations
            anns = self.imgToAnns[img_id]
            new_anns = []
            for ann in anns:
                new_ann = copy.deepcopy(ann)
                new_ann['point'] = [int(ann['point'][0] * image_scale), int(ann['point'][1] * image_scale)]
                new_anns.append(new_ann)
            target = {'image_id': img_id, 'annotations': new_anns}
            orig_quadtree = np.load(os.path.join(self.quadtree_path,
                                                 img_file_name[:-4] + '.npy'), allow_pickle=True).item()['quatree'][0]
            quadtree = {}
            for k, v in orig_quadtree.items():
                new_k = k
                new_v = []
                for pos in v:
                    new_pos = (int(pos[0] * image_scale), int(pos[1] * image_scale))
                    new_v.append(new_pos)
                quadtree[new_k] = new_v

            orig_graph = np.load(os.path.join(self.quadtree_path,
                                              img_file_name[:-4] + '.npy'), allow_pickle=True).item()
            del orig_graph['quatree']
            new_graph = {}
            for k, v in orig_graph.items():
                new_k = (int(k[0] * image_scale), int(k[1] * image_scale))
                new_v = []
                for adj in v:
                    if adj == (-1, -1):
                        new_v.append((-1, -1))
                    else:
                        new_v.append((int(adj[0] * image_scale), int(adj[1] * image_scale)))
                new_graph[new_k] = new_v

            target_layers = []
            for layer, layer_points in quadtree.items():
                target_layer = []
                for layer_point in layer_points:
                    for target_i in target['annotations']:
                        if l1_dist(target_i['point'], list(layer_point)) <= 2:
                            target_layer.append(target_i)
                            break
                target_layers.extend(target_layer)
            layer_indices = []
            count = 0
            for k, v in quadtree.items():
                if k == 0:
                    layer_indices.append(0)
                else:
                    layer_indices.append(count)
                count += len(v)

            image_id = torch.tensor([d[img_id]])

            points = [obj['point'] for obj in target_layers]
            points = torch.as_tensor(points, dtype=torch.int64).reshape(-1, 2)
            edges = [obj['edge_code'] for obj in target_layers]
            edges = torch.tensor(edges, dtype=torch.int64)

            # get semantic annotations
            semantic_left_up = [semantics_dict[obj['semantic'][0]] for obj in target_layers]
            semantic_right_up = [semantics_dict[obj['semantic'][1]] for obj in target_layers]
            semantic_right_down = [semantics_dict[obj['semantic'][2]] for obj in target_layers]
            semantic_left_down = [semantics_dict[obj['semantic'][3]] for obj in target_layers]
            semantic_left_up = torch.tensor(semantic_left_up, dtype=torch.int64)
            semantic_right_up = torch.tensor(semantic_right_up, dtype=torch.int64)
            semantic_right_down = torch.tensor(semantic_right_down, dtype=torch.int64)
            semantic_left_down = torch.tensor(semantic_left_down, dtype=torch.int64)

            # annotations
            target = {}
            target["edges"] = edges
            target["file_name"] = img_file_name
            target["image_id"] = image_id
            target["size"] = torch.as_tensor([img.size[1], img.size[0]])

            target["semantic_left_up"] = semantic_left_up
            target["semantic_right_up"] = semantic_right_up
            target["semantic_right_down"] = semantic_right_down
            target["semantic_left_down"] = semantic_left_down

            # get image
            img = F.to_tensor(img)
            img = F.normalize(img, mean=mean, std=std)
            target['unnormalized_points'] = points
            # normalize
            points = points / torch.tensor([img.shape[2], img.shape[1]], dtype=torch.float32)
            target["points"] = points
            target['layer_indices'] = torch.tensor(layer_indices)

            target['graph'] = graph_to_tensor(new_graph)

            return img, target


    def __len__(self):
        return len(self.ids)


class MyDataset2(Dataset):
    def __init__(self, img_path, annot_path, extract_roi, disable_sem_info=False):
        self.disable_sem_info = disable_sem_info
        self.img_path = img_path
        self.quadtree_path = '/'.join(img_path.split('/')[:-1]) + '/annotations_npy/' + img_path.split('/')[-1]
        self.edgecode_path = '/'.join(img_path.split('/')[:-1]) + '/annotations_edge/' + img_path.split('/')[-1]
        self.mode = img_path.split('/')[-1]

        available_ids = {int(x.replace('.npy', '')) for x in os.listdir(self.quadtree_path)}
        
        # load annotation
        with open(annot_path, 'r') as f:
            dataset = json.load(f)
        # images
        self.imgs = {}
        for img in dataset['images']:
            if img['id'] not in available_ids:
                continue
            self.imgs[img['id']] = img
        self.imgToAnns = defaultdict(list)
        for ann in dataset['annotations']:
            if ann['image_id'] not in available_ids:
                continue
            self.imgToAnns[ann['image_id']].append(ann)
        self.ids = list(sorted(self.imgs.keys())) # [os.path.basename(x).split('.')[0] for x in list(sorted(glob(f"{self.img_path}/*.png")))]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_file_name = self.imgs[int(img_id)]['file_name'] # self.imgs[int(img_id)]['id'] + '.png'
        img = Image.open(os.path.join(self.img_path, img_file_name)).convert('RGB')

        if 1:
            # get structure annotations
            anns = self.imgToAnns[int(img_id)]

            # new_anns = []
            # image_points = []
            # for ann in anns:
            #     new_ann = copy.deepcopy(ann)
            #     points = np.array(ann['segmentation']).reshape(-1, 2)
            #     new_ann['point'] = [[p[0], p[1]] for p in points]
            #     new_anns.append(new_ann)
            #     image_points.extend(list(zip(points[:,0], points[:,1])))


            data = np.load(os.path.join(self.quadtree_path,
                                                 img_file_name[:-4] + '.npy'), allow_pickle=True).item()
            orig_quadtree = data['quadtree']
            orig_graph = data['graph']
            image_points = data['points']

            new_anns = []
            for pt in image_points:
                new_ann = {
                    'point': [int(pt[0]), int(pt[1])],
                }
                new_anns.append(new_ann)
            target = {'image_id': img_id, 'annotations': new_anns}

            # orig_quadtree = np.load(os.path.join(self.quadtree_path,
            #                                      img_file_name[:-4] + '.npy'), allow_pickle=True).item()['quatree'][0]
            quadtree = {}
            for k, v in orig_quadtree.items():
                new_k = k
                new_v = []
                for pos in v:
                    new_pos = (int(pos[0]), int(pos[1]))
                    new_v.append(new_pos)
                quadtree[new_k] = new_v

            # orig_graph = np.load(os.path.join(self.quadtree_path,
            #                                   img_file_name[:-4] + '.npy'), allow_pickle=True).item()
            # del orig_graph['quatree']
            new_graph = {}
            for k, v in orig_graph.items():
                new_k = (int(k[0]), int(k[1]))
                new_v = []
                for adj in v:
                    if adj == (-1, -1):
                        new_v.append((-1, -1))
                    else:
                        new_v.append((int(adj[0]), int(adj[1])))
                new_graph[new_k] = new_v
            
            target_layers = []
            for layer, layer_points in quadtree.items():
                target_layer = []
                for layer_point in layer_points:
                    for target_i in target['annotations']:
                        if l1_dist(target_i['point'], list(layer_point)) <= 2:
                            target_layer.append(target_i)
                            break
                target_layers.extend(target_layer)
            layer_indices = []
            count = 0
            for k, v in quadtree.items():
                if k == 0:
                    layer_indices.append(0)
                else:
                    layer_indices.append(count)
                count += len(v)

            # image_id = torch.tensor([d[img_id]])
            image_id = torch.tensor([int(img_id)])

            points = [obj['point'] for obj in target_layers]

            # edges = [obj['edge_code'] for obj in target_layers]
            # edges = torch.tensor(edges, dtype=torch.int64)

            with open(os.path.join(self.edgecode_path,
                                                 img_file_name[:-4] + '.json'), 'r') as f:
                edge2code = json.load(f)
                edge2code = {
                tuple(map(lambda x: int(float(x)), key.strip("()").split(", "))): value
                for key, value in edge2code.items()
            }
            
            edges = [edge2code[(int(pt[0]), int(pt[1]))] for pt in points]
            points = torch.as_tensor(points, dtype=torch.int64).reshape(-1, 2)
            edges = torch.tensor(edges, dtype=torch.int64)

            # annotations
            target = {}
            target["edges"] = edges
            target["image_id"] = image_id
            target["file_name"] = img_file_name
            target["size"] = torch.as_tensor([img.size[1], img.size[0]])

            # get semantic annotations
            if not self.disable_sem_info:
                semantic_left_up = [semantics_dict[obj['semantic'][0]] for obj in target_layers]
                semantic_right_up = [semantics_dict[obj['semantic'][1]] for obj in target_layers]
                semantic_right_down = [semantics_dict[obj['semantic'][2]] for obj in target_layers]
                semantic_left_down = [semantics_dict[obj['semantic'][3]] for obj in target_layers]
                semantic_left_up = torch.tensor(semantic_left_up, dtype=torch.int64)
                semantic_right_up = torch.tensor(semantic_right_up, dtype=torch.int64)
                semantic_right_down = torch.tensor(semantic_right_down, dtype=torch.int64)
                semantic_left_down = torch.tensor(semantic_left_down, dtype=torch.int64)

                target["semantic_left_up"] = semantic_left_up
                target["semantic_right_up"] = semantic_right_up
                target["semantic_right_down"] = semantic_right_down
                target["semantic_left_down"] = semantic_left_down

            # get image
            img = F.to_tensor(img)
            img = F.normalize(img, mean=mean, std=std)
            target['unnormalized_points'] = points
            # normalize
            points = points / torch.tensor([img.shape[2], img.shape[1]], dtype=torch.float32)
            target["points"] = points
            target['layer_indices'] = torch.tensor(layer_indices)

            # padding (-1,-1) if not enough 4 neighbors
            for pt, neighbors in new_graph.items():
                if len(neighbors) < 4:
                    new_graph[pt].extend([(-1, -1)] * (4 - len(neighbors)))
                elif len(neighbors) > 4:
                    new_graph[pt] = neighbors[:4]
            target['graph'] = graph_to_tensor(new_graph)

            return img, target


    def __len__(self):
        return len(self.ids)
