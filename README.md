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
python3 tools/test.py configs/scannet200/isbnet_scannet200.yaml ../pretrains/scannet200/head_scannetv2_200_val.pth --out ./
```

## Geometric Visual Correspondence calculation
Follow [HERE](https://github.com/VinAIResearch/Open3DIS), suppose you have installed Open3DIS and put it to current directory, run:
```
cp tools/GVC.sh ./Open3DIS/scripts
cp tools/GVC_*.py ./Open3DIS/tools
cp tools/scannet200gvc.yaml ./Open3DIS/configs
cd ./Open3DIS && sh scripts/GVC.sh
```
and then you get the GVC results for each proposal, you may fuse them with NMS.
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
