import os
import sys
import torch
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath('../../dataset/paris_orleans.py'))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torch.utils.data import DataLoader
from spconv.pytorch.utils import PointToVoxel
from second.dataset.paris_orleans import PAI_ORLEANS_Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    
    batch_size = 1
    
    # Load the data
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", help="train set directory", required=True)
    parser.add_argument("--annot_dir", help="train annotations directory", required=True)
    parser.add_argument("--voxel_dir", help="train voxels directory", required=True)
    
    args = parser.parse_args()
    
    data = PAI_ORLEANS_Dataset(
        data_dir=args.data_dir,
        annotation_dir=args.annot_dir
    )
    
    data_loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    preprocessor = PointToVoxel(
        vsize_xyz=[2, 2, 8],
        coors_range_xyz=[0, 0, 0, 2000, 2000, 320],
        num_point_features=4,
        max_num_voxels=20000,
        max_num_points_per_voxel=35,
        device=device
    )
    
    i = 1
    for cloud, label in data_loader:
        cloud = cloud.to(device)
        cloud = cloud.squeeze(0).to(torch.float32)
        voxels, coors, num_points = preprocessor(cloud)
        
        path = args.voxel_dir + '/BL' + f'{i}'.rjust(6, '0')
        os.makedirs(path, exist_ok=True)
        
        np.save(os.path.join(path, 'voxels.npy'), voxels.detach().cpu().numpy())
        np.save(os.path.join(path, 'coors.npy'), coors.detach().cpu().numpy())
        np.save(os.path.join(path, 'num_points.npy'), num_points.detach().cpu().numpy())
        
        i += 1
