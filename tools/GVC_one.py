import argparse
import json
import os
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
import clip
import torch
import pycocotools
import copy
import yaml
from munch import Munch
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.dataset.scannet_loader import scaling_mapping
from open3dis.dataset import build_dataset
from open3dis.src.clustering.clustering import process_hierarchical_agglomerative
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm, trange
import tracemalloc
from torchvision.ops import masks_to_boxes
import pickle

def rle_encode_gpu(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    # mask = np.concatenate([[0], mask, [0]])
    zeros_tensor = torch.tensor([0], dtype=torch.bool, device=mask.device)
    mask = torch.cat([zeros_tensor, mask, zeros_tensor])

    runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1
    runs[1::2] -= runs[::2]
    # runs = np.where(mask[1:] != mask[:-1])[0] + 1
    # runs[1::2] -= runs[::2]
    # counts = " ".join(str(x) for x in runs)
    counts = runs.cpu().numpy()
    # breakpoint()
    rle = dict(length=length, counts=counts)
    return rle

def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles


def rle_decode(rle):
    """
    Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def show_mask(mask, ax, random_color=False):
    """
    Mask visualization
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
   
def match_SAM(current_instanc_box, boxes, iou = 0.3) :
    '''boxes.size() torch.Size([65, 4])  # (xmin, ymin, xmax, ymax)        
    '''
    
    dets = boxes
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    area_current = (current_instanc_box[2] - current_instanc_box[0] + 1) * (current_instanc_box[3] - current_instanc_box[1] + 1)
    #计算窗口i与其他所有窗口的交叠部分的面积
    xx1 = np.maximum(current_instanc_box[0], x1)  
    yy1 = np.maximum(current_instanc_box[1], y1)  
    xx2 = np.minimum(current_instanc_box[2], x2)  
    yy2 = np.minimum(current_instanc_box[3], y2)  

    w = np.maximum(0.0, xx2 - xx1 + 1)  
    h = np.maximum(0.0, yy2 - yy1 + 1)  
    inter = w * h  
    #交/并得到iou值  
    ovr = inter / (area_current + areas - inter)  
    ioa = inter / areas
    #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
    inds = np.where(ovr > iou)[0]  
    if len(inds) > 1:
        # choose the one with bigger area
        if False:
            idx = np.argmax(areas[inds])
            inds = [inds[idx]]
        # choose the one with bigger ioa
        idx = np.argmax(ioa[inds])
        inds = [inds[idx]]        
    return np.array(inds), ovr[inds].cpu().numpy()
    
def refined_3d_proposal_with_SAM(
    scene_id, cfg):
    """
    refined_3d_proposal_with_SAM
    return refined_masks, refined_confs
    {"ins": refined_masks, "conf": refined_confs}
    """
    refine_score = cfg.exp.refine_score
    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
    save_mapping_dir = os.path.join(
        cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mapping_dir
    )
    os.makedirs(save_mapping_dir, exist_ok=True)
    save_mapping_path = os.path.join(save_mapping_dir, f"{scene_id}.pkl")
    
    ### Set up dataloader
    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    loader = build_dataset(root_path=scene_dir, cfg=cfg)

    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
    )

    ########### Proposal branch selection ###########
    agnostic3d_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path, f"{scene_id}.pth")
    agnostic3d_data = torch.load(agnostic3d_path)
    instance_3d_encoded_original = agnostic3d_data["ins"]
    instance_3d_encoded = np.array(instance_3d_encoded_original)
    confidence_3d = torch.tensor(agnostic3d_data["conf"])

    n_instance_3d = instance_3d_encoded.shape[0]

    if isinstance(instance_3d_encoded[0], dict):
        instance_3d = torch.stack(
            [torch.from_numpy(rle_decode(in3d)) for in3d in instance_3d_encoded], dim=0
        )
    else:
        instance_3d = torch.stack([torch.tensor(in3d) for in3d in instance_3d_encoded], dim=0)


    instance = torch.cat([instance_3d], dim=0)

    ########### ########### ########### ###########
    n_instance = instance.shape[0]

    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    # H, W = 968, 1296
    interval = cfg.data.img_interval
    mappings = []
    images_paths = []
    images_ids = []
    
    # DEBUG HERE, HOW TO USE TOK VIEWS IMAGES BETTER ?
    DEBUG_MODE = False  #   True    False
    # view_topk = 10
    
    if cfg.data.dataset_name == 's3dis':
        target_frame = 300
        interval = max(interval, len(loader) // target_frame)
        

    for i in trange(0, len(loader), interval):
        frame = loader[i]
        frame_id = frame["frame_id"]  # str

        pose = loader.read_pose(frame["pose_path"])
        depth = loader.read_depth(frame["depth_path"])
        rgb_img = loader.read_image(frame["image_path"])
        rgb_img_dim = rgb_img.shape[:2]

        if "scannetpp" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
            depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
            mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic = frame["translated_intrinsics"])
        elif "scannet200" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic = frame["scannet_depth_intrinsic"])
            new_mapping = scaling_mapping(
                torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
            )
            mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)
        elif "replica" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
        elif "s3dis" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=frame["intrinsics"])
        else:
            raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

        if mapping[:, 3].sum() < 100:  # no points corresponds to this image, skip sure
            continue

        mappings.append(mapping.cpu())
        images_paths.append(frame["image_path"])
        images_ids.append(frame_id)
            
    mappings = torch.stack(mappings, dim=0)
    n_views = len(mappings)

    instance_id = -1
    refined_masks, refined_confs = [], []
    
    save_dir_mask = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)
    mask_pth = os.path.join(save_dir_mask, scene_id + ".pth")
    mask_SAM = torch.load(mask_pth)
    
    instance_changed_score_cnt = 0
    instance_changed_score_cnt_become_bigger = 0
    for inst in trange(n_instance): # topk = 5
        # Obtaining top-k views
        matched_once = False
        instance_id += 1
        view_topk = cfg.refine_grounding.top_k
        current_instance =instance[inst]
        current_conf = confidence_3d[inst]        
        
        
        # DEBUG here
        if DEBUG_MODE:
            pass
            # if instance_id not in [12, 31]:
            #     refined_masks.append(current_instance)
            #     refined_confs.append(current_conf)                
            #     continue
            if instance_id not in [4, 11]:
                refined_masks.append(current_instance)
                refined_confs.append(current_conf)                
                continue            
        
        refined_confs.append(current_conf)
        
        conds = (mappings[..., 3] == 1) & (instance[inst] == 1)[None].expand(n_views, -1)  # n_view, n_points
        count_views = conds.sum(1)
        valid_count_views = count_views > 20
        valid_inds = torch.nonzero(valid_count_views).view(-1)
        if len(valid_inds) == 0:
            continue
        topk_counts, topk_views = torch.topk(
            count_views[valid_inds], k=min(view_topk, len(valid_inds)), largest=True
        )
        topk_views = valid_inds[topk_views]
        # print('type : current_instance:', type(current_instance))
        # print('current_instance:', current_instance)
        # print('current_instance:', current_instance.shape)
        # print('points:', points.shape)
        # print('current_instance_gt_cnt:', len(current_instance_gt_cnt))
        # instance[inst]: torch.Size([149982])
        # points: torch.Size([149982, 3])     
        # current_instance_gt_cnt: 891           
        # Multiscale image crop from topk views
        view_id = -1
        instance_views_score = []
        for v in topk_views:
            view_id += 1
            point_inds_ = torch.nonzero((mappings[v][:, 3] == 1) & (current_instance == 1)).view(-1)
            projected_points = torch.tensor(mappings[v][point_inds_][:, [1, 2]]).cuda()
            # NOTE This is different from OpenMask3D paper method:'''
            # A straightforward approach to perform the cropping would be to project all of the visible 3D points belonging 
            # to the mask onto the 2D image, and fit a 2D bounding box around these points. '''
            # Calculate the bounding rectangle
            mi = torch.min(projected_points, axis=0)
            ma = torch.max(projected_points, axis=0)
            x1, y1 = mi[0][0].item(), mi[0][1].item()
            x2, y2 = ma[0][0].item(), ma[0][1].item()

            if x2 - x1 == 0 or y2 - y1 == 0:
                continue

            # current_img = copy.copy(images[v])
            current_img_path = images_paths[v]
            current_frameid = str(images_ids[v])                
            # print('save_dir_mask:', save_dir_mask)
            # print('current_img_path:', current_img_path)
            # print('current_frameid:', current_frameid)
            # print('mask_pth:', mask_pth)
            # print('mask_SAM.keys():', mask_SAM.keys())
            if int(current_frameid) in mask_SAM.keys() or str(current_frameid) in mask_SAM.keys():
                try:
                    current_img_SAM = mask_SAM[str(current_frameid)]
                except:
                    current_img_SAM = mask_SAM[int(current_frameid)]
                # grounded_data_dict[frame_id] = {
                #     "masks": masks_to_rle(masks), # GSAM masks
                #     "img_feat": image_features.cpu(), # CLIP(BG_blur(img)) mask/bbox wise
                #     "conf": confs_filt.cpu(),
                # }
                encoded_masks = current_img_SAM["masks"]
                current_img_SAM_img_feat = current_img_SAM["img_feat"]
                current_img_SAM_conf = current_img_SAM["conf"]
                
                # this is 2D mask
                masks = []
                if encoded_masks is not None:
                    for mask in encoded_masks:
                        masks.append(torch.tensor(pycocotools.mask.decode(mask)))
                    masks = torch.stack(masks, dim=0)
                    current_instanc_box = np.array([y1, x1, y2, x2])
                    try:
                        boxes = masks_to_boxes(masks)
                        match_idx, match_iou = match_SAM(current_instanc_box, boxes, iou = 0.5)
                        # print('match_idx:', match_idx)
                        if len(match_idx):
                            pass
                        else:
                            continue                
                    except:
                        print('masks_to_boxes failed! ', masks.shape)
                        continue
                    # boxes = masks_to_boxes(masks)
                    # current_img: (480, 640, 3)
                    # masks: torch.Size([65, 480, 640])
                    # boxes.size() torch.Size([65, 4])  # (xmin, ymin, xmax, ymax)                    
                    # print('current_img:', current_img.shape)
                    # print('masks:', masks.shape)
                    
                    # print('match_idx:', match_idx)
                    # print('match_iou:', match_iou)
                    if len(match_idx):
                        matched_once = True
                        mask = masks[match_idx][0]
                        box = boxes[match_idx][0]
                        xmin, ymin, xmax, ymax = box
                        # current_img: (480, 640, 3)
                        # masks: torch.Size([65, 480, 640])
                        
                        idx_vis_scene = torch.where(mappings[v][:, 3] == 1)[0]
                        idx_vis_instance = torch.where((mappings[v][:, 3] == 1) & (current_instance == 1))[0]
                        # print('idx_vis_scene:', len(idx_vis_scene))
                        # print('idx_vis_instance:', len(idx_vis_instance))
                        # idx_vis_scene: 9835
                        # idx_vis_instance: 577
                        in_mask_cnt = 0.
                        for p_idx in range(len(current_instance)): # for all points
                            # current_instance[p_idx] = 1 # for gt
                            # current_instance[p_idx] = 0 # for bg
                            current_point = mappings[v][p_idx]
                            if current_point[3].item() == 1: #for visable points, 9535
                                y, x = current_point[1].item(), current_point[2].item()
                                # if x >= xmin and x<=xmax and y >= ymin and y<=ymax:
                                #     # if point located within the bbox
                                #     # current_instance_matched_cnt[p_idx] += 0.5
                                #     pass
                                if mask[y, x] == 1 and current_instance[p_idx] == 1:
                                    # if point located within the mask
                                    in_mask_cnt += 1
                                # if current_instance[p_idx] == 1:
                                #     # if point located within the proposal
                                #     # RECORE all the index in current instance, because only topK=5 view points is not all the points in it
                                #     current_instance_topk_vis_idx_all.append(p_idx)
                            else:
                                # for invisibla points, 149982-9835
                                pass
                        cnt_ioa = in_mask_cnt / len(idx_vis_instance)
                        cnt_ioa = np.minimum(cnt_ioa, 1.0)
                        if refine_score == 'mul':
                            current_view_score = cnt_ioa * match_iou
                        elif refine_score == 'bbox': 
                            current_view_score = 1 * match_iou
                        elif refine_score == 'mask':
                            current_view_score = cnt_ioa * 1                                                        
                        instance_views_score.append(current_view_score)
                        # print('cnt_ioa:', cnt_ioa)
                        # print('current_view_score:', current_view_score)
        
        if matched_once:
            # match scuuessfully, rescore this proposal
            new_conf = np.mean(instance_views_score)
            # print('instance_views_score:', instance_views_score)
            print('before current_conf:', current_conf)
            print('after new_conf:', new_conf)            
            refined_confs[-1] = new_conf
            instance_changed_score_cnt += 1
            if new_conf > current_conf:
                instance_changed_score_cnt_become_bigger += 1
        else:
            # did not match scuuessfully, keep the same
            pass
    print('video: {}, number of proposal:{}'.format(scene_id, n_instance))
    print('instance_changed_score_cnt: {}, bigger cnt:{}'.format(instance_changed_score_cnt, instance_changed_score_cnt_become_bigger))
    return instance_3d_encoded_original, refined_confs


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration GVC")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--scene_id",type=str,required = True,help="choose a scene_id to run")
    return parser

if __name__ == "__main__":
    # Multiprocess logger
    # if os.path.exists("tracker_refine.txt") == False:
    #     with open("tracker_refine.txt", "w") as file:
    #         file.write("Processed Scenes .\n")

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])


    # Directory Init
    save_dir_refined_3D_proposal = os.path.join(
        cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.refined_3D_proposal
    )
    os.makedirs(save_dir_refined_3D_proposal, exist_ok=True)
    
    tracemalloc.start()
    
    choose_scene_id = str(args.scene_id)
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            if choose_scene_id != scene_id:
                continue
            print("refined_3D_proposal :", scene_id)            

            refined_masks, refined_confs = refined_3d_proposal_with_SAM(
                scene_id,
                cfg
            )
            
            # Saving refined features
            save_path = os.path.join(save_dir_refined_3D_proposal, f"{scene_id}.pth")
            torch.save({"ins": refined_masks, "conf": refined_confs}, save_path)
            print("save_path:", save_path)
            
            torch.cuda.empty_cache()
            del refined_masks, refined_confs
