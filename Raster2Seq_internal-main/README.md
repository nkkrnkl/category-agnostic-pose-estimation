<div align="center">
<h1 align="center">Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction</h1>
  <a href="https://hao-pt.github.io/" target="_blank">Hao&nbsp;Phung</a> &emsp;
  <a href="https://www.hadarelor.com/" target="_blank">Hadar&nbsp;Elor</a> &emsp;
  <br>
  Cornell University
  <br> <br>
</div>

<img src="./imgs/teaser.png" width=100% height=80%>

**TLDR:** We reformulate Raster2Vector conversion as a seq2seq polygon generation.

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#method">Method</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-structure">Data Structure</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#training">Training</a>
    </li>
    <li>
      <a href="#acknowledgment">Acknowledgment</a>
    </li>
  </ol>
</details>


## Abstract

Reconstructing a structured vector-graphics representation from a rasterized floorplan image is typically an important prerequisite for computational tasks involving floorplans such as automated understanding or CAD workflows. However, existing techniques struggle in faithfully generating the structure and semantics conveyed by complex floorplans that depict large indoor spaces with many rooms and a varying numbers of polygon corners. To this end, we propose Raster2Seq, framing floorplan reconstruction as a sequence-to-sequence task, where each room is represented as a polygon sequence---labeled with the room's semantics. Our approach introduces an autoregressive decoder that learns to predict the next corner conditioned on image features and previously generated corners using guidance from learnable anchors. These anchors represent spatial coordinates in image space, hence allowing for effectively directing the attention mechanism to focus on informative image regions. By embracing the autoregressive mechanism, our method offers flexibility in the output format, enabling for efficiently handling complex floorplans with numerous rooms and diverse polygon structures. Our method achieves state-of-the-art performance on standard benchmarks such as Structure3D and CubiCasa5K, while also demonstrating strong generalization to more challenging datasets like WAFFLE, which contain diverse room structures and complex geometric variations.

## Method
 ![space-1.jpg](./imgs/overview.png) 

Given a rasterized floorplan image (left), our approach converts it into vectorized format, represented as a labeled polygon sequence, separated using special <SEP> tokens. The main architectural component of our framework is an anchor-based autoregressive decoder, which predicts the next token given image features ($f_{img}$), learnable anchors ($v_{anc}$) and the previously generated tokens. Above, we visualize the first two labeled polygons predicted (colored in orange and pink, respectively). 

## Installation
* The code has been tested on Linux with python 3.10.13, pytorch 2.3.1  and cuda 11.8
* Create an environment:
  ```shell
  conda create -n raster2seq python=3.10
  conda activate raster2seq
  ```
* Install pytorch and required libraries:
  ```shell
  # adjust the cuda version accordingly
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```
* Compile the deformable-attention modules (from [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)) and the differentiable rasterization module (from [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)):
  ```shell
  cd models/ops
  sh make.sh

  # unit test for deformable-attention modules (should see all checking is True)
  # python test.py

  cd ../../diff_ras
  python setup.py build develop
  ```


## Data Structure

We use COCO-style formatting for all experiments. Data preprocessing are detailed in [data_preprocess](data_preprocess/README.md). Simply put, input data is RGB images and output is the 2D coordinate vectors of room regions which are represented as close-loop segmentation.

The data tree structure of Structured3D, for instance, is as follows:
```
code_root/
└── data/
    └── stru3d/
        ├── train/
        ├── val/
        ├── test/
        └── annotations/
            ├── train.json
            ├── val.json
            └── test.json

```

In this code, we experiments with 3 datasets: Structured3D, CubiCasa5K, and Raster2Graph.

### Checkpoints

Our model checkpoints can be found in table below:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Checkpoint Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Structured3D</td>
      <td>https://drive.google.com/file/d/1NLWnEkEgWN5ibOQlk3FVr4w2Nrhnjsmo/view?usp=sharing</td>
    </tr>
    <tr>
      <td>CubiCasa5K</td>
      <td>https://drive.google.com/file/d/1_kEbWvmQjS8Xr6lIZox2vMyTi6C6kTpj/view?usp=sharing</td>
    </tr>
    <tr>
      <td>Raster2Graph</td>
      <td>https://drive.google.com/file/d/16wkbt096DkvBT_lDK-qVSGT6kWan15I5/view?usp=sharing</td>
    </tr>
  </tbody>
</table>

To download, use `gdown --fuzzy path`.


## Inference
Use [predict.py](predict.py). To run inference on WAFFLE floorplan images, for instance, run `bash tools/predict_cc5k.sh`


## Evaluation

Evaluate model in  pretraing stage (only structural outputs, no semantic room prediction)

| Dataset       | Bash Script                  |
|---------------|----------------------------------|
| Structured3D  | `tools/eval_s3d_rgb_pretrain.sh`      |
| CubiCasa5K    | `tools/eval_cc5k_pretrain.sh`   |
| Raster2Graph  | `tools/eval_r2g_pretrain.sh`    |

Evaluate model in finetuning stage (structural outputs, along with semantic room prediction)

| Dataset       | Bash Script                  |
|---------------|----------------------------------|
| Structured3D  | `tools/eval_s3d_rgb_finetune.sh`      |
| CubiCasa5K    | `tools/eval_cc5k_finetune.sh`   |
| Raster2Graph  | `tools/eval_r2g_finetune.sh`    |


## Training
Raster2Seq involves two training stages: (1) Pretraining without semantic room class and (2) Finetuning with semantic room class.

### Pretraining:

| Dataset       | Bash Script                  |
|---------------|----------------------------------|
| Structured3D  | `tools/pretrain_s3d_rgb.sh`      |
| CubiCasa5K    | `tools/pretrain_cc5k.sh`   |
| Raster2Graph  | `tools/pretrain_r2g.sh`    |


### Finetuning:

| Dataset       | Bash Script                  |
|---------------|----------------------------------|
| Structured3D  | `tools/finetune_s3d_rgb.sh`      |
| CubiCasa5K    | `tools/finetune_cc5k.sh`   |
| Raster2Graph  | `tools/finetune_r2g.sh`    |



## Acknowledgment

We gratefully acknowledge the authors of [RoomFormer](https://github.com/ywyue/RoomFormer), [HEAT](https://github.com/woodfrog/heat), [Raster2Graph](https://github.com/SizheHu/Raster-to-Graph) and [MonteFloor](https://github.com/vevenom/MonteScene) for releasing their code and datasets. 
Our approach builds upon [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) for the architecture design and draws inspiration from [PolyFormer](https://github.com/amazon-science/polygon-transformer) for the seq2seq framework.
