import glob
import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from second.utils.config import config as cfg


class PAI_ORLEANS_Dataset(Dataset):
    """
    Implements a loader of point cloud data from the set of PAI_ORLEANS.

    Args:
        data_dir (str): The directory of point cloud data.
        annotation_dir (str): The directory of their corresponding bounding boxes.

        loaded_arrays_dir (str, optional): The directory to store point cloud data under NPY format.
        Defaults to None.
        loaded_annot_arrays_dir (str, optional): The directory to store their bounding boxes under NPY format.
        Defaults to None.

        max_number_of_points (int, optional): The maximal number of points to reach. Defaults to None.
        max_number_of_objects (int, optional): The maximal number of objects to reach. Defaults to None.

        load_raw_clouds (bool, optional): Triggers the storage of data into npy format for faster loading.
        Defaults to False.
    """

    def __init__(
        self,
        data_dir: str,
        annotation_dir: str,
        loaded_arrays_dir: str = None,
        loaded_annot_arrays_dir: str = None,
        max_number_of_points: int = None,
        max_number_of_objects: int = None,
        load_raw_clouds: bool = False,
    ) -> None:
        # Constructor
        self.data = []
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.loaded_arrays_dir = loaded_arrays_dir
        self.loaded_annot_arrays_dir = loaded_annot_arrays_dir
        self.max_number_of_points = max_number_of_points
        self.max_number_of_objects = max_number_of_objects
        self.load_raw_clouds = load_raw_clouds

        if self.load_raw_clouds:
            assert (
                self.max_number_of_points
                and self.max_number_of_objects
                and self.loaded_arrays_dir
                and self.loaded_annot_arrays_dir
            )
            self.data = self.load_points_with_labels()
        else:
            self.data = self.get_data_from_dirs()

    def load_points_with_labels(self) -> list:
        """
        Loads the point cloud data and their boxes in a list after saving them into npy format.

        Returns:
            list: The list of point cloud data coupled with their labels.
        """

        point_clouds = glob.glob(self.data_dir + "/*.bin")
        point_clouds.sort()

        bounding_boxes = glob.glob(self.annotation_dir + "/*.json")
        bounding_boxes.sort()

        assert len(point_clouds) == len(bounding_boxes)

        for i in tqdm(range(len(point_clouds)), desc="saving data into npy format..."):
            cloud = np.fromfile(point_clouds[i], dtype=np.float32)
            nb_points = cloud.shape[0] // 4
            cloud = cloud.reshape(nb_points, 4)
            cloud = self.resize(self.max_number_of_points, cloud)

            path_cloud = os.path.join(self.loaded_arrays_dir, point_clouds[i][-12:-4] + ".npy")
            np.save(path_cloud, cloud)

            with open(bounding_boxes[i], "r") as json_file:
                dictionary = json.load(json_file)

            if dictionary["bounding boxes"]:
                object_ids = [
                    np.array([int(dictionary["bounding boxes"][i]["object_id"])])
                    for i in range(len(dictionary["bounding boxes"]))
                ]
                boxes = [
                    np.array(
                        [
                            dictionary["bounding boxes"][i]["center"]["x"],
                            dictionary["bounding boxes"][i]["center"]["y"],
                            dictionary["bounding boxes"][i]["center"]["z"],
                            dictionary["bounding boxes"][i]["width"],
                            dictionary["bounding boxes"][i]["length"],
                            dictionary["bounding boxes"][i]["height"],
                        ]
                    )
                    for i in range(len(dictionary["bounding boxes"]))
                ]
                directions = [
                    np.array([dictionary["bounding boxes"][i]["angle"]])
                    for i in range(len(dictionary["bounding boxes"]))
                ]

                object_ids = np.vstack(tuple(object_ids))
                boxes = np.vstack(tuple(boxes))
                directions = np.vstack(tuple(directions))

                labels = np.hstack((object_ids, boxes, directions))
                number_of_objects = labels.shape[0]

                if number_of_objects < self.max_number_of_objects:
                    labels = np.vstack(
                        (labels, np.zeros((self.max_number_of_objects - number_of_objects, labels.shape[1])))
                    )

                if number_of_objects > self.max_number_of_objects:
                    raise (KeyError("Please update your maximal number of objects to 40."))

            else:
                labels = np.zeros((self.max_number_of_objects, 8))

            path_box = os.path.join(self.loaded_annot_arrays_dir, bounding_boxes[i][-20:-5] + ".npy")
            np.save(path_box, labels)

            self.data.append([cloud, labels])

        return self.data

    def get_data_from_dirs(self) -> list:
        """
        Loads the point cloud data and their boxes in a list.

        Returns:
            list: The list of point cloud data coupled with their labels.
        """
        point_clouds = glob.glob(self.data_dir + "/*.npy")
        point_clouds.sort()

        bounding_boxes = glob.glob(self.annotation_dir + "/*.npy")
        bounding_boxes.sort()

        assert len(point_clouds) == len(bounding_boxes)

        for i in tqdm(range(len(point_clouds)), desc="loading data..."):
            cloud = np.load(point_clouds[i])
            labels = np.load(bounding_boxes[i])
            self.data.append([cloud, labels])

        return self.data

    def __len__(self) -> int:
        """
        Returns the length of the data list.

        Returns:
            int: The length of the data list.
        """
        return len(self.data)

    def normalize_point_cloud(
        self, point_cloud: ndarray, x_range: int = 1600, y_range: int = 1600, z_range: int = 32
    ) -> ndarray:
        """
        Normalize the point cloud so that x and y coordinates are in the range [0, 1600]
        and z coordinates are in the range [0, 32].

        Args:
            point_cloud (ndarray): An (N, 4) array where N is the number of points.
                                    Each point is represented by (x, y, z).
            x_range (int, optional): Target range for x-axis normalization. Default is 1600.
            y_range (int, optional): Target range for y-axis normalization. Default is 1600.
            z_range (int, optional): Target range for z-axis normalization. Default is 32.

        Returns:
            ndarray: Normalized point cloud with the same shape as the input.
        """

        # Find min and max values for each axis
        x_min, y_min, z_min = min(point_cloud[:, 0]), min(point_cloud[:, 1]), min(point_cloud[:, 2])
        x_max, y_max, z_max = max(point_cloud[:, 0]), max(point_cloud[:, 1]), max(point_cloud[:, 2])

        # Normalize each axis
        point_cloud[:, 0] = (point_cloud[:, 0] - x_min) / (x_max - x_min) * x_range
        point_cloud[:, 1] = (point_cloud[:, 1] - y_min) / (y_max - y_min) * y_range
        point_cloud[:, 2] = (point_cloud[:, 2] - z_min) / (z_max - z_min) * z_range

        return point_cloud

    def resize(self, new_size: int, cloud: ndarray) -> ndarray:
        """
        Resizes the point cloud to have the desired size for all of them.

        Args:
            new_size (int): The new size to apply to the cloud.
            cloud (ndarray): The cloud to be resized.

        Returns:
            ndarray: The resized point cloud.
        """
        nb_points = cloud.shape[0]

        # Filter the points
        cloud = cloud[(np.abs(cloud) <= 1e10).all(axis=1)]
        cloud = self.normalize_point_cloud(cloud)

        if nb_points < new_size:
            cloud = np.vstack((cloud, np.zeros((new_size - nb_points, cloud.shape[1]))))

        elif nb_points == new_size:
            return cloud

        else:
            factor = nb_points // new_size
            cloud = cloud[::factor]
            nb_points = cloud.shape[0]
            remainder = nb_points - new_size

            if remainder:
                rows_to_modify = cloud[: remainder * 2]
                modified = rows_to_modify[::2]
                remaining = cloud[remainder * 2 :]
                cloud = np.vstack((modified, remaining))[:new_size]

        return cloud

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """
        Returns a point cloud and its associated bounding box according to their index.

        Args:
            index (int): The index of the point cloud and its associated bounding boxes.

        Returns:
            Tuple[Tensor]: The point cloud and its associated bounding boxes.
        """
        cloud, bounding_boxes = self.data[index]

        cloud = torch.from_numpy(cloud)
        bounding_boxes = torch.from_numpy(bounding_boxes)

        return cloud, bounding_boxes


class ORLEANS_VOXELS_Dataset(Dataset):
    """
    Implements a loader of voxel data from the set of PAI_ORLEANS.

    Args:
        data_dir (str): The directory of voxel data.
        annotation_dir (str): The directory of their corresponding bounding boxes.
        max_voxels (int): maximal number of voxels allowed.
        cfg (module): A python module containing the hyperparameters for regression.
    """

    def __init__(self, data_dir: str, annotation_dir: str, max_voxels: int, cfg=cfg):
        # Constructor
        self.data = []
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.max_voxels = max_voxels

        self.num_offsets = cfg.num_offsets
        self.num_directions = cfg.num_directions
        self.anchors = cfg.anchors
        self.anchors_per_position = cfg.anchors_per_position
        self.feature_map_shape = (self.anchors.shape[0], (self.anchors_per_position + 2) * 10)
        self.pos_threshold = cfg.pos_threshold
        self.neg_threshold = cfg.neg_threshold

        self.data = self.get_data_from_dirs()

    def get_data_from_dirs(self) -> list:
        """
        Loads the voxels, their centroid coordinates and their boxes in a list.

        Returns:
            list: The list of voxels coupled with their labels.
        """
        voxelized = os.listdir(self.data_dir)
        voxelized.sort()

        bounding_boxes = glob.glob(self.annotation_dir + "/*.npy")
        bounding_boxes.sort()

        assert len(voxelized) == len(bounding_boxes)

        for i in tqdm(range(len(voxelized)), desc="loading data..."):
            voxels, coors = (
                np.load(self.data_dir + "/" + voxelized[i] + "/voxels.npy"),
                np.load(self.data_dir + "/" + voxelized[i] + "/coors.npy"),
            )
            labels = np.load(bounding_boxes[i])
            self.data.append([voxels, coors, labels])

        return self.data

    def resize(self, new_size: int, voxels: Tensor, coors: Tensor) -> Tuple[Tensor]:
        """
        Resizes the set of voxels and its centroid coordinates to have the desired size for all of them.

        Args:
            new_size (int): The new size to apply to the voxels.
            voxels (Tensor): The set of voxels to pad.
            coors (Tensor): The coordinates to pad.

        Returns:
            Tuple[Tensor]: The resized voxels and coordinates.
        """
        voxels = F.pad(voxels, (0, 0, 0, 0, 0, new_size - voxels.shape[0]))
        coors = F.pad(coors, (0, 0, 0, new_size - coors.shape[0]))
        return voxels, coors

    def encode_targets(self, gt_boxes: ndarray) -> Tuple[ndarray]:
        """
        Encodes the regression offsets into features that the model will learn.

        Args:
            gt_boxes (ndarray): The ground truth boxes.

        Returns:
            Tuple[ndarray]: The position arrays of objects, the encoded targets and the directions.
        """
        anchors = self.anchors
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        class_pos = np.zeros((self.feature_map_shape[0], self.feature_map_shape[1], 1), dtype=np.float32)
        targets = np.zeros(
            (self.feature_map_shape[0], self.feature_map_shape[1], self.anchors_per_position, self.num_offsets),
            dtype=np.float32,
        )
        directions = np.zeros(
            (self.feature_map_shape[0], self.feature_map_shape[1], self.anchors_per_position, self.num_directions),
            dtype=np.float32,
        )

        ious = self.compute_ious(anchors[:, 1:], gt_boxes)

        for i in range(num_anchors):
            for j in range(num_gt):
                anchor_class = int(anchors[i, 0])

                iou = ious[i, j]

                if iou >= self.pos_threshold:
                    class_pos[i, j, 0] = anchor_class
                    targets[i, j, anchor_class, :] = self.compute_targets(anchors[i, 1:], gt_boxes[j])
                    directions[i, j, anchor_class, 0], directions[i, j, anchor_class, 1] = (
                        float(gt_boxes[j, -1] > 0),
                        float(gt_boxes[j, -1] <= 0),
                    )

        class_pos = class_pos.reshape(self.feature_map_shape[0] * self.feature_map_shape[1], 1)[:200*176].reshape(200, 176, 1)
        targets = targets.reshape(self.feature_map_shape[0] * self.feature_map_shape[1], self.anchors_per_position, self.num_offsets)[:200*176].reshape(200, 176, self.anchors_per_position, self.num_offsets)
        directions = directions.reshape(self.feature_map_shape[0] * self.feature_map_shape[1], self.anchors_per_position, self.num_directions)[:200*176].reshape(200, 176, self.anchors_per_position, self.num_directions)

        return class_pos, targets, directions

    def compute_ious(self, anchors: ndarray, gt_boxes: ndarray) -> ndarray:
        """
        Computes the IoU between each anchor and ground truth boxes.

        Args:
            anchors (ndarray): The anchor boxes.
            gt_boxes (ndarray): The ground truth boxes.

        Returns:
            ndarray: The array of IoUs of each pair (anchor, gt_box).
        """
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        ious = np.zeros((num_anchors, num_gt), dtype=np.float32)

        for i in range(num_anchors):
            for j in range(num_gt):
                ious[i, j] = self.iou_3d(anchors[i], gt_boxes[j])

        return ious

    def iou_3d(self, box1: ndarray, box2: ndarray) -> float:
        """
        Computes the Intersection over Union (IoU) between two 3D bounding boxes.

        Args:
            box1 (ndarray): Bounding box in format [x, y, z, w, l, h, r].
            box2 (ndarray): Bounding box in format [x, y, z, w, l, h, r].

        Returns:
            float: IoU between the two bounding boxes.
        """
        # Extract the parameters for clarity
        x1, y1, z1, w1, l1, h1, _ = box1
        x2, y2, z2, w2, l2, h2, _ = box2

        # Convert box centers to corners
        box1_min = np.array([x1 - w1 / 2, y1 - l1 / 2, z1 - h1 / 2])
        box1_max = np.array([x1 + w1 / 2, y1 + l1 / 2, z1 + h1 / 2])
        box2_min = np.array([x2 - w2 / 2, y2 - l2 / 2, z2 - h2 / 2])
        box2_max = np.array([x2 + w2 / 2, y2 + l2 / 2, z2 + h2 / 2])

        # Coordinates of the intersection box
        ix_min = np.maximum(box1_min, box2_min)
        ix_max = np.minimum(box1_max, box2_max)

        # Intersection dimensions
        i_length = np.maximum(ix_max[0] - ix_min[0], 0)
        i_width = np.maximum(ix_max[1] - ix_min[1], 0)
        i_height = np.maximum(ix_max[2] - ix_min[2], 0)

        # Volume of intersection
        intersection_volume = i_length * i_width * i_height

        # Volumes of the bounding boxes
        box1_volume = w1 * l1 * h1
        box2_volume = w2 * l2 * h2

        # Volume of union
        union_volume = box1_volume + box2_volume - intersection_volume

        # IoU calculation
        iou = intersection_volume / union_volume

        return iou

    def compute_targets(self, anchor: ndarray, gt_box: ndarray) -> ndarray:
        """
        Computes the encoded targets between an anchor and a ground truth according to
        SECOND paper's formulas for anchors and targets.

        Args:
            anchor (ndarray): An anchor box.
            gt_box (ndarray): A ground truth box.

        Returns:
            ndarray: The encoded targets.
        """
        da = np.sqrt(anchor[3] ** 2 + anchor[4] ** 2)
        xt = (gt_box[0] - anchor[0]) / da
        yt = (gt_box[1] - anchor[1]) / da
        zt = (gt_box[2] - anchor[2]) / anchor[5]
        wt = np.log(gt_box[3] / anchor[3])
        lt = np.log(gt_box[4] / anchor[4])
        ht = np.log(gt_box[5] / anchor[5])
        theta_t = gt_box[6] - anchor[6]
        return np.array([xt, yt, zt, wt, lt, ht, theta_t], dtype=np.float32)

    def __len__(self) -> int:
        """
        Returns the length of the data list.

        Returns:
            int: The length of the data list.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """
        Returns a set of voxels encoding a point cloud, its centroids' coordinates and
        the associated bounding box according to their index.

        Args:
            index (int): The index of the set of voxels, the centroids and the associated bounding boxes.

        Returns:
            Tuple[Tensor]: The voxels, their coordinates, their encoded targets and their coefficients of presence.
        """
        voxels, coors, bounding_boxes = self.data[index]

        coors = torch.from_numpy(coors)
        voxels = torch.from_numpy(voxels)

        if voxels.shape[0] < self.max_voxels:
            voxels, coors = self.resize(self.max_voxels, voxels, coors)

        # bounding-box encoding
        class_pos, targets, directions = self.encode_targets(bounding_boxes[:, 1:])

        class_pos = torch.from_numpy(class_pos)
        targets = torch.from_numpy(targets)
        coors = torch.hstack((torch.zeros(self.max_voxels, 1), coors))

        return voxels, coors, class_pos, targets, directions
