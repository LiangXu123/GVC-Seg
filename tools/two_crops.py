import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os,glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing.pool import Pool
from progressbar import ProgressBar,Bar,ETA
from utils import load_mesh, save_ply


CLOUD_FILE_PREFIX = ""

# load input data
out_dir = '../data/Scannet200_crop_ply'
os.makedirs(out_dir, exist_ok=True)

def crop_mesh(scene_path):
    scene_id = scene_path.split("/")[-1]
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PREFIX}.ply")
    mesh_path = scene_path
    mesh = load_mesh(mesh_path)
    
    out_path = os.path.join(out_dir, scene_id)
    os.makedirs(out_path, exist_ok=True)
    
    new_name = str(os.path.basename(mesh_path)).replace('.ply','.points.ply')
    new_file =  os.path.join(out_path, new_name)
    new_file = os.path.abspath(new_file)      

    # find median axis
    original_points = np.asarray(mesh.vertices)
    original_colors = np.asarray(mesh.vertex_colors)
    x_min = np.min(original_points[:, 0])
    x_max = np.max(original_points[:, 0])
    y_min = np.min(original_points[:, 1])
    y_max = np.max(original_points[:, 1])
    x_span = int(x_max - x_min)
    y_span = int(y_max - y_min)
    # split from the longest axis
    if x_span >= y_span:
        x_mean = np.median(original_points[:, 0])
        left_result = np.where(original_points[:,0] < x_mean)[0]
        right_result = np.where(original_points[:,0] >= x_mean)[0]
    else:
        y_mean = np.median(original_points[:, 1])
        left_result = np.where(original_points[:,1] < y_mean)[0]
        right_result = np.where(original_points[:,1] >= y_mean)[0]                
    crop_idx_np = np.zeros(len(original_points))
    crop_idx_np[right_result] = 1
    np.save(str(new_file).replace('.points.ply', '.index.ply'), crop_idx_np)            
    filter_points = original_points[left_result]
    filter_colors = original_colors[left_result]      
    data_np = np.concatenate((filter_points, filter_colors), 1)            
    save_ply(data_np, save_name=str(new_file).replace('.points', '.points.left'))     
    filter_points = original_points[right_result]
    filter_colors = original_colors[right_result]       
    data_np = np.concatenate((filter_points, filter_colors), 1)            
    save_ply(data_np, save_name=str(new_file).replace('.points', '.points.right'))                 

scene_paths = sorted(glob.glob("../data/Scannet200/" + "*ply"))

with Pool(processes=64) as p:
    widgets = [Bar(left="run crop 2 views"),ETA()]
    pbar = ProgressBar(widgets=widgets,maxval=len(scene_paths))
    res = list(pbar(p.imap(crop_mesh,scene_paths)))