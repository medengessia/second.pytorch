import math
import glob
import numpy as np
from sklearn.cluster import DBSCAN


class config:
    # classes
    class_list = ["Electrical Cabinet", "Technical Room", "GSM-R Antenna", "Track Telephone"]

    # batch size
    N = 1

    # maximum number of points per voxel
    T = 35

    # voxel size
    vd = 0.8
    vh = 0.2
    vw = 0.2

    # # points cloud range
    xrange = (0, 1600)
    yrange = (0, 1600)
    zrange = (0, 32)

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)

    x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // 2)
    y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // 2)

    cx, cy = np.meshgrid(x, y)
    cz = np.ones_like(cx) * -2.0

    # iou threshold
    pos_threshold = 0.5
    neg_threshold = 0.35

    num_offsets = 7
    num_directions = 2
    anchors_per_position = 4

    # non-maximum suppression
    nms_threshold = 0.1
    score_threshold = 0.96
    
    box_dir = "C:/Users/mmedeng/Documents/repositories/NUAGES_DE_POINTS/ANNOTATIONS/np_data"
    boxes = glob.glob(box_dir + '/*.npy')
    boxes.sort()
    coors = []
    mask = np.zeros((1,3))

    for i in range(len(boxes)):
        coors.append(np.load(boxes[i])[:, 1:4])
        coors[i] = coors[i][coors[i] != mask]
        coors[i] = coors[i].reshape(coors[i].shape[0]//3, 3)
    coors = np.vstack(coors)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(min_samples=3).fit(coors)
    labels = dbscan.labels_

    # Extract cluster centers
    unique_labels = set(labels)
    cluster_centers = []
    for label in unique_labels:
        points_in_cluster = coors[labels == label]
        cluster_center = points_in_cluster.mean(axis=0)
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)

    def generate_anchors(scales, centers, aspect_ratios):
        anchors = []
        for scale in scales:
            for center in centers:
                for aspect_ratio in aspect_ratios:
                    class_, x, y, z, width, length, height, yaw = (
                        aspect_ratio[0],
                        center[0],
                        center[1],
                        center[2],
                        scale * aspect_ratio[1],
                        scale * aspect_ratio[2],
                        scale * aspect_ratio[3],
                        aspect_ratio[4],
                    )
                    anchor = [class_, x, y, z, width, length, height, yaw]
                    anchors.append(anchor)
        return anchors

    # Define scales and aspect ratios
    scales = [1, 1.25, 1.5, 1.75, 2]
    aspect_ratios = [
        (0, 1.75, 1.75, 2.6, 0),
        (1, 6.75, 2.75, 3.5, 0),
        (2, 9.2, 4.0, 22.5, 0),
        (3, 1.4, 1.0, 2.0, 0),
        (1, 6.75, 2.75, 3.5, np.pi / 2),
        (2, 9.2, 4.0, 22.5, np.pi / 2),
        (3, 1.4, 1.0, 2.0, np.pi / 2),
    ]  # Class ratios

    # Generate anchors around each cluster center
    anchors = generate_anchors(scales, cluster_centers, aspect_ratios)
    anchors = np.array(anchors)
