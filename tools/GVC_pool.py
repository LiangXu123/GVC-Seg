import argparse
import os
import time
import numpy as np
from tqdm import tqdm, trange
from multiprocessing.pool import Pool
from progressbar import ProgressBar,Bar,ETA
import yaml,random
from munch import Munch
from glob import glob

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration GVC")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--GPU_ID",type=str,required = True,help="GPU ID to use,start from 0")
    parser.add_argument("--workers",type=int,required = True,help="each GPU worker number")
    return parser

def run_one(cmd):
    os.system(cmd)       

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    config = args.config
    GPU_ID = args.GPU_ID
    workers = args.workers 
    GPU_ID = str(GPU_ID).split(',')
    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])
    
    # Prepare directories
    save_dir_feat = os.path.join(
        cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.refined_3D_proposal
    )
    
    exist_pths = glob(os.path.join(save_dir_feat, "*.pth"))
    exist_pths = [str(os.path.basename(i)).replace('.pth', '') for i in exist_pths]
    # NOTE filter exist scene ids
    print('total scene ids: {}'.format(len(scene_ids)))
    scene_ids = [i for i in scene_ids if i not in exist_pths]
    print('exists scene ids: {}'.format(len(exist_pths)))
    print('remain scene ids: {}'.format(len(scene_ids)))
    # print('exist_pths: {}'.format(exist_pths))
    # print('scene_ids: {}'.format(scene_ids))
    batch_size = workers * len(GPU_ID)
    gpu_idx = []
    random.shuffle(GPU_ID)
    for i in GPU_ID:
        gpu_idx.extend([i] * workers)
    gpu_idx = gpu_idx * 1000000 
    gpu_idx = gpu_idx[:len(scene_ids)]
    cmds = []
    for gpu, scene in zip(gpu_idx, scene_ids):
        cmd = 'PYTHONPATH=./:$PYTHONPATH && export PYTHONPATH && CUDA_VISIBLE_DEVICES={} python3 tools/GVC_one.py --config {} --scene_id {}'.format(gpu, config, scene)
        cmds.append(cmd)
    # print(cmds)
with Pool(processes=batch_size) as p:
    widgets = [Bar(left="run python3 tools/GVC_pool.py"),ETA()]
    pbar = ProgressBar(widgets=widgets,maxval=len(cmds))
    res = list(pbar(p.imap(run_one,cmds)))