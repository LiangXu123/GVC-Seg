import warnings
warnings.filterwarnings("ignore")
import os,torch
import numpy as np
import open3d as o3d

def load_mesh(pcl_file):
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh

def save_ply(data_numpy, save_name="./data.ply"):
    # Pass data_numpy to Open3D.o3d.geometry.PointCloud and visualize
    # data_numpy [N, 3]
    # is_list=True list of [N,3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_numpy[:,:3])
    # (rgb/255-mean)/std=y
    # rgb/255=y*std+mean
    # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]".    
    # this rgb value has been normailzed to [-1,1], now we should revert it to [0,1]
    reverse_rgb = False
    if reverse_rgb:
        pcd.colors = o3d.utility.Vector3dVector(data_numpy[:,3:][:,::-1])
    else:
        pcd.colors = o3d.utility.Vector3dVector(data_numpy[:,3:])
    o3d.io.write_point_cloud(save_name, pcd)

def rle_encode_gpu_batch(masks):
    """Encode RLE (Run-length-encode) from 1D binary mask.

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