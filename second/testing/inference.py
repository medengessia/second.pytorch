import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from second.model.second import SECOND
from second.utils.config import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize(new_size, voxels, coors):
    voxels = F.pad(voxels, (0, 0, 0, 0, 0, new_size - voxels.shape[0]))
    coors = F.pad(coors, (0, 0, 0, new_size - coors.shape[0]))
    return voxels, coors

def decode(anchors, deltas):
    cas, cts = [], []
    anchors = anchors.view(-1, 7)
    deltas = deltas.view(-1, 7)[:680]
    box_ndim = anchors.shape[-1]
    if box_ndim > 7:
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
    else:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

# Function to perform inference
def detect_objects(model, voxels, coords):
    with torch.no_grad():
        _, regression, _ = model(voxels, coords)
    return regression

# Function to visualize the 3D bounding boxes
def visualize_boxes(coords, boxes):
    coords_np = coords.squeeze(0).cpu().numpy().astype(np.float64)[:, :3]
    boxes_np = boxes.cpu().numpy().astype(np.float64)
    
    # Visualize the point cloud data
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords_np)
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for points
    
    # Visualize the point cloud and bounding boxes
    geometries = [point_cloud]

    # Draw bounding boxes
    for box in boxes_np:
        # box should have 7 elements: [center_x, center_y, center_z, extent_x, extent_y, extent_z, rotation_angle]
        center = box[:3]
        extent = box[3:6]
        
        # Use identity matrix for rotation if not provided
        R = np.eye(3)
        
        # Convert to the correct shapes for Open3D
        center_np = np.reshape(center, (3, 1))  # (3, 1)
        extent_np = np.reshape(extent, (3, 1))  # (3, 1)

        # Create an OrientedBoundingBox
        bbox = o3d.geometry.OrientedBoundingBox(center=center_np, R=R, extent=extent_np)
        bbox.color = (1, 0, 0)  # Red color for the bounding box
        
        geometries.append(bbox)

    # Draw the geometries (point cloud and bounding boxes)
    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    
    max_voxels = 20000
    num_classes = 4
    num_regression_offsets = 7
    num_directions = 2
    in_features = 4
    
    anchors = torch.from_numpy(cfg.anchors).to(device)
    
    model = SECOND(
        in_features=in_features,
        num_classes=num_classes,
        num_regression_offsets=num_regression_offsets,
        num_directions=num_directions,
    )
    
    checkpoint = torch.load('C:/Users/mmedeng/Documents/repositories/second/evaluation/weights/best_model.pth')
    state_dict = checkpoint['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel_file", help="voxel file directory", default= "C:/Users/mmedeng/Documents/repositories/NUAGES_DE_POINTS/PAI_ORLEANS/valid/voxels/BL000008/voxels.npy")
    parser.add_argument("--coord_file", help="coord file directory", default= "C:/Users/mmedeng/Documents/repositories/NUAGES_DE_POINTS/PAI_ORLEANS/valid/voxels/BL000008/coors.npy")
    args = parser.parse_args()
    
    voxels, coors = torch.from_numpy(np.load(args.voxel_file)), torch.from_numpy(np.load(args.coord_file))
    
    if voxels.shape[0] < max_voxels:
        voxels, coors = resize(max_voxels, voxels, coors)
    
    coors = torch.hstack((torch.zeros(max_voxels, 1), coors))
    
    voxels = voxels.unsqueeze(0)
    coors = coors.unsqueeze(0)
    
    voxels = voxels.to(device)
    coors = coors.to(device)
    
    # Run the detection
    regression = detect_objects(model, voxels, coors)
    boxes = decode(anchors, regression)
    
    # Visualize the output
    visualize_boxes(coors, boxes)