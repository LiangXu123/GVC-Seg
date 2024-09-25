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
from torch.nn import functional as F
from tqdm import tqdm, trange
import tracemalloc
from torchvision.ops import masks_to_boxes

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

def match_bbox_with_SAM(current_instanc_box, boxes, iou = 0.3) :
    '''boxes.size() torch.Size([65, 4])  # (xmin, ymin, xmax, ymax)        
    '''
    
    dets = boxes
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    area_current = (current_instanc_box[2] - current_instanc_box[0] + 1) * (current_instanc_box[3] - current_instanc_box[1] + 1)
    xx1 = np.maximum(current_instanc_box[0], x1)  
    yy1 = np.maximum(current_instanc_box[1], y1)  
    xx2 = np.minimum(current_instanc_box[2], x2)  
    yy2 = np.minimum(current_instanc_box[3], y2)  

    w = np.maximum(0.0, xx2 - xx1 + 1)  
    h = np.maximum(0.0, yy2 - yy1 + 1)  
    inter = w * h  
    #iou
    ovr = inter / (area_current + areas - inter)  
    ioa = inter / areas

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
    
def GVC(
    scene_id, cfg):
    """
    GVC
    return proposal_masks, GVC_score
    """
    gvc_method = cfg.exp.gvc_method

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
    GVC_score = []
    
    # load 2D SAM results first
    save_dir_mask = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)
    mask_pth = os.path.join(save_dir_mask, scene_id + ".pth")
    SAM_mask_2D = torch.load(mask_pth)
    
    instance_changed_score_cnt = 0
    instance_changed_score_cnt_become_bigger = 0
    for inst in trange(n_instance): # topk = 5
        # Obtaining top-k views
        matched_once = False
        instance_id += 1
        view_topk = cfg.refine_grounding.top_k
        current_instance =instance[inst]
        current_conf = confidence_3d[inst]        
        
        GVC_score.append(current_conf)
        
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
        view_id = -1
        gvc_score_lst = []
        # for each top-k view, match it's 2D SAM bbox, and then calculate the GVC score
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
            Proj_2D_box = np.array([y1, x1, y2, x2])
            current_frameid = str(images_ids[v])                
            # find current frame SAM results
            if int(current_frameid) in SAM_mask_2D.keys() or str(current_frameid) in SAM_mask_2D.keys():
                try:
                    current_img_SAM = SAM_mask_2D[str(current_frameid)]
                except:
                    current_img_SAM = SAM_mask_2D[int(current_frameid)]
                encoded_masks = current_img_SAM["masks"]

                masks = []
                if encoded_masks is not None:
                    for mask in encoded_masks:
                        masks.append(torch.tensor(pycocotools.mask.decode(mask)))
                    masks = torch.stack(masks, dim=0)
                    
                    try:
                        boxes = masks_to_boxes(masks)
                        match_idx, box_match_iou = match_bbox_with_SAM(Proj_2D_box, boxes, iou = 0.5)
                        # current instance has matched a bbox in 2D image
                        if len(match_idx):
                            pass
                        else:
                            continue                
                    except:
                        print('masks_to_boxes failed! ', masks.shape)
                        continue

                    if len(match_idx):
                        matched_once = True
                        mask = masks[match_idx][0]
                        box = boxes[match_idx][0]
                        vis_instance_points = torch.where((mappings[v][:, 3] == 1) & (current_instance == 1))[0]
                        in_mask_cnt = 0.
                        for p_idx in range(len(current_instance)): # for all points
                            # current_instance[p_idx] = 1 # for gt
                            # current_instance[p_idx] = 0 # for bg
                            current_point = mappings[v][p_idx]
                            if current_point[3].item() == 1: #for visable points, 9535
                                y, x = current_point[1].item(), current_point[2].item()
                                if mask[y, x] == 1 and current_instance[p_idx] == 1:
                                    # if point located within the mask
                                    in_mask_cnt += 1
                            else:
                                pass
                        points_match_ioa = in_mask_cnt / len(vis_instance_points)
                        points_match_ioa = np.minimum(points_match_ioa, 1.0)
                        if gvc_method == 'mul':
                            gvc_score = points_match_ioa * box_match_iou
                        elif gvc_method == 'bbox': 
                            gvc_score = 1 * box_match_iou
                        elif gvc_method == 'mask':
                            gvc_score = points_match_ioa * 1                                                        
                        gvc_score_lst.append(gvc_score)
        
        if matched_once:
            # match scuuessfully, rescore this proposal
            new_gvc_score = np.mean(gvc_score_lst)
            print('before current_conf:', current_conf)
            print('after new_conf:', new_gvc_score)            
            GVC_score[-1] = new_gvc_score
            instance_changed_score_cnt += 1
            if new_gvc_score > current_conf:
                instance_changed_score_cnt_become_bigger += 1
        else:
            # did not match scuuessfully, keep the same
            pass
    print('video: {}, number of proposal:{}'.format(scene_id, n_instance))
    print('instance_changed_score_cnt: {}, bigger cnt:{}'.format(instance_changed_score_cnt, instance_changed_score_cnt_become_bigger))
    return instance_3d_encoded_original, GVC_score


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration GVC")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--scene_id",type=str,required = True,help="choose a scene_id to run")
    return parser

if __name__ == "__main__":
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
            print("Geometric Visual Correspondence calculation:", scene_id) 
            proposal_masks, GVC_score = GVC(
                scene_id,
                cfg
            )
            # Saving un-biased scores and masks
            save_path = os.path.join(save_dir_refined_3D_proposal, f"{scene_id}.pth")
            torch.save({"ins": proposal_masks, "conf": GVC_score}, save_path)
            print("save_path:", save_path)
            
            torch.cuda.empty_cache()
            del proposal_masks, GVC_score
