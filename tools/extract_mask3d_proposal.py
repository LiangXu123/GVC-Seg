import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import torch,os
from glob import glob
import numpy as np
import sys
mask3d_path = '../Mask3D/mask3d'
sys.path.append(mask3d_path)
from mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh 

import random
random.seed(0)
import glob
from tqdm import tqdm
from utils import rle_encode_gpu_batch

# load model
model = get_model('../pretrains/scannet200/scannet200_benchmark.ckpt')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load input data
# you need to extract proposals for both left and right part, as well as the original full point data
files_right = sorted(glob.glob("../data/Scannet200_crop_ply/*.points.right*.ply"))

# output dir
mask3d_clsagnostic_scannet200 = '../data/mask3d_clsagnostic_feat_scannet200_right'
os.makedirs(mask3d_clsagnostic_scannet200, exist_ok=True)

mask3d_dc_feat_scannet200 = '../data/mask3d_dc_feat_scannet200_right'
os.makedirs(mask3d_dc_feat_scannet200, exist_ok=True)

with open('../ISBNet/dataset/scannetv2/scannetv2_val.txt', "r") as file:
    scene_ids = sorted([line.rstrip("\n") for line in file])
files_right = [i for i in files_right if str(i).split('/')[-2] in scene_ids]

Name_pattern_with_clean_str = False

for i in tqdm(range(len(files_right))):
    pointcloud_file = files_right[i]

    mesh = load_mesh(pointcloud_file)

    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

    # run model
    with torch.no_grad():
        if Name_pattern_with_clean_str:
            out_file_cls = os.path.join(mask3d_clsagnostic_scannet200, pointcloud_file[-16:-4] + ".pth")
            out_file_dc = os.path.join(mask3d_dc_feat_scannet200, pointcloud_file[-16:-4] + ".pth")
        else:
            scene_id = str(pointcloud_file).split('/')[-2]
            out_file_cls = os.path.join(mask3d_clsagnostic_scannet200, scene_id + ".pth")
            out_file_dc = os.path.join(mask3d_dc_feat_scannet200, scene_id + ".pth")        
        if os.path.isfile(out_file_cls) and os.path.isfile(out_file_dc):continue
        model.eval()
        outputs = model(data, raw_coordinates=features)
        
        # parse predictions
        logits = outputs["pred_logits"][0]
        masks = outputs["pred_masks"][0]
        # backbone_features = outputs["backbone_features"]   
        mask_features = outputs["mask_features_decomposed"][0]
        mask_features = mask_features[inverse_map, :].cpu()
        print('mask_features.shape:', mask_features.shape)
        print('old masks num:', masks.shape[-1])
        # mask_features.shape: torch.Size([92618, 128]) ISBEnet-->[feats.shape: torch.Size([136769, 32])]

        # labels = []
        confidences = []
        # keep_masks = []
        masks_binary = []
        confidence_threshold = 0.0
        for i in range(len(logits)):
            p_labels = torch.softmax(logits[i], dim=-1)
            p_masks = torch.sigmoid(masks[:, i])
            l = torch.argmax(p_labels, dim=-1)
            c_label = torch.max(p_labels)
            m = p_masks > 0.5
            c_m = p_masks[m].sum() / (m.sum() + 1e-8)
            c = c_label * c_m
            if l < 200 and c > confidence_threshold:
            # if  c > confidence_threshold:
                # labels.append(l.item())
                confidences.append(c.item())
                # keep_masks.append(masks[i, :])
                masks_binary.append(
                    m[inverse_map])  # mapping the mask back to the original point cloud            
                
        # print('len(confidences):', len(confidences))
        # print('len(labels):', len(labels))
        keep_masks_tensor = torch.stack(masks_binary)
        print('mask 3D proposal number:', keep_masks_tensor.shape[0])
        saved_masks = rle_encode_gpu_batch(keep_masks_tensor)
        saved_confs = confidences
        
        torch.save(
            {"ins": saved_masks, "conf": saved_confs},
            out_file_cls,
        )
        
        torch.save(mask_features, out_file_dc)
        print('saved to:', out_file_dc)
        print('saved to:', out_file_cls)