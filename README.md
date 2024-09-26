# GVC-Seg: Training Free 3D Instance Segmentation via  Geometric Visual Correspondence

[[project page](https://liangxu123.github.io/GVC/)] [[paper](https://arxiv.org/pdf/)]

GVC-Seg(Geometric Visual Correspondence), a novel training-free 3D instance segmentation framework. By combining 3D geometric and 2D visual cues in a training-free manner, GVC-Seg prompts more reliable proposal generation and selection, which alleviates the confidence bias of multi-scale, multi-source proposals.

## Installation 
We do not introduct any new python packages, so please follow  [Mask3D](https://github.com/JonasSchult/Mask3D) and [Open3DIS](https://github.com/VinAIResearch/Open3DIS) for installation.

## Data preparation 
Please prepare the dataset Scannet200, Scannetpp and Replica as described [Open3DIS](https://github.com/VinAIResearch/Open3DIS/blob/main/docs/DATA.md).

For pretrained models, we use [Mask3D](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt) and [ISBNet](https://drive.google.com/file/d/1ZEZgQeT6dIakljSTx4s5YZM0n2rwC3Kw/view?usp=share_link) pretrained on Scannet200 as baseline.

## Scene split 
Suppose you have download the Scannet200, Scannetpp and Replica datasets, and put them under `data` directory, and run:
```
cd tools/
python3 two_crops.py
```
## Two-branch proposal generation
### Generate masks using Mask3D
Follow [HERE](https://github.com/JonasSchult/Mask3D), and run:
```
cd tools/
python3 extract_mask3d_proposal.py
```
### Generate masks using ISBNet
Follow [HERE](https://github.com/VinAIResearch/Open3DIS/blob/main/docs/DATA.md), and run:
```
cd ISBNet/
OR you may also go to ./Open3DIS/segmenter3d/ISBNet and then run:
python3 tools/test.py configs/scannet200/isbnet_scannet200.yaml ../pretrains/scannet200/head_scannetv2_200_val.pth --out ./
```

## Geometric Visual Correspondence calculation
### Extract 2D SAM result from all frames in the dataset
Follow [HERE](https://github.com/VinAIResearch/Open3DIS/blob/main/docs/RUN.md), and run:
```
cp tools/scannet200gvc.yaml ./Open3DIS/configs
cd ./Open3DIS && sh scripts/grounding_2d.sh configs/scannet200gvc.yaml
```
Follow [HERE](https://github.com/VinAIResearch/Open3DIS), suppose you have installed Open3DIS and put it to current directory, run:
```
cp tools/GVC.sh ./Open3DIS/scripts
cp tools/GVC_*.py ./Open3DIS/tools
cd ./Open3DIS && sh scripts/GVC.sh
```
and then you get the GVC results for each proposal, you may fuse them with NMS.
## Evaluation
### Class-agnostic Evaluation
 Evaluate the Class-agnostic results like [this](https://github.com/VinAIResearch/Open3DIS/blob/4b05043095aff1dcbc9882799d25e0fb6f4c86a9/docs/RUN.md?plain=1#L53).
### OV Evaluation
For open-vocabulary semantic understanding, you may refer to [openmask3d](https://github.com/OpenMask3D/openmask3d) for CLIP feature calculation.

We do not provide specific evaluation code here for simplicity.
## Citation

You can cite our work as follows.
```
@article{
      author    = {Liang Xu†, Fangjing Wang†, Jinyu Yang, Feng Zheng},
      title     = {GVC-Seg: Training Free 3D Instance Segmentation via  Geometric Visual Correspondence.},
      journal   = {},
      year      = {2024},
}
```
