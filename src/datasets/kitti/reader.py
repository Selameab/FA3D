import copy
import csv
import os
import pickle

import cv2
import imagesize
import numpy as np
from PIL import Image
from datasets.kitti.utils import fov_filter, box_filter, count_points_accurate
from tqdm import tqdm

from datasets.kitti.boxes import Box2D, Box3D, CORNER_CORNER, get_box_difficulty
from datasets.kitti.transforms_3D import transform

# Constants
KITTI_COLUMN_NAMES = ['Type', 'Truncated', 'Occluded', 'Alpha', 'X1', 'Y1', 'X2', 'Y2', '3D_H', '3D_W', '3D_L', '3D_X', '3D_Y', '3D_Z', 'Rot_Y', 'Score']

# Dataset path
KITTI_DIR = 'D:/Datasets/KITTI/training' if os.name == 'nt' else os.path.expanduser('~/datasets/KITTI/training/')
CARS_ONLY = {'Car': ['Car']}


class Reader:
    def __init__(self, class_dict, ds_dir=KITTI_DIR, invalidate_cache=False):
        self.ds_dir = ds_dir
        self.img2_dir = os.path.join(ds_dir, 'image_2')
        self.label_dir = os.path.join(ds_dir, 'label_2')
        self.velo_dir = os.path.join(ds_dir, 'velodyne')
        self.velo_reduced_dir = os.path.join(ds_dir, 'velodyne_reduced')
        self.calib_dir = os.path.join(ds_dir, 'calib')
        self.depth2_dir = os.path.join(ds_dir, 'depth_2_grayscale')
        self.cache_path = os.path.join(ds_dir, 'label_2.pkl')
        self.masks_dir = os.path.join(ds_dir, 'mask')

        # For grouping similar objects into one class (Eg. Cars and Vans)
        self.class_to_group = {}
        for group, classes in class_dict.items():
            for cls in classes:
                self.class_to_group[cls] = group

        if invalidate_cache or (not os.path.isfile(self.cache_path)):
            self._create_cache()

        self._load_cache()
        self._filter_cache()

    def _create_cache(self):
        print("Creating cache...")
        cache = {}
        for t in tqdm(os.listdir(self.img2_dir)):
            t = t.replace(".png", "")
            # Paths
            img_path = os.path.join(self.img2_dir, t + ".png")
            txt_path = os.path.join(self.label_dir, t + ".txt")
            calib_path = os.path.join(self.calib_dir, t + '.txt')

            w, h = imagesize.get(img_path)

            pts = self.get_velo_reduced(t)

            txt_lbl = _read_txt_file(txt_path, is_pred=False)
            boxes_2D, boxes_3D = [], []
            for row in txt_lbl:
                # Convert strings to numbers
                for param in ['Truncated', 'Alpha', 'X1', 'Y1', 'X2', 'Y2', '3D_H', '3D_W', '3D_L', '3D_X', '3D_Y', '3D_Z', 'Rot_Y']:
                    row[param] = np.round(float(row[param]), 2)

                box2D = Box2D((row['X1'], row['Y1'], row['X2'], row['Y2']), mode=CORNER_CORNER, cls=row['Type'])
                boxes_2D += [box2D]

                box3D = Box3D(row['3D_H'], row['3D_W'], row['3D_L'], row['3D_X'], row['3D_Y'], row['3D_Z'],
                              row['Rot_Y'], alpha=row['Alpha'], cls=row['Type'])
                box3D.pt_count = count_points_accurate(pts, box3D)
                box3D.truncated = row['Truncated']
                box3D.occluded = int(row['Occluded'])
                box3D.difficulty = get_box_difficulty(box2D=box2D, box3D=box3D)
                boxes_3D += [box3D]

            cache[t] = {'image_size': (h, w),
                        'boxes_2D': boxes_2D,
                        'boxes_3D': boxes_3D,
                        'calib': _get_calib(calib_path)}

        with open(self.cache_path, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cache(self):
        with open(self.cache_path, 'rb') as handle:
            print(f"Loading cache: {self.cache_path}")
            self.cache = pickle.load(handle)

    def _filter_cache(self):
        print("Filtering cache...")
        for t in self.cache:
            # Remove boxes not in class_dict
            self.cache[t]['boxes_2D'] = list(filter(lambda box: box.cls in self.class_to_group, self.cache[t]['boxes_2D']))
            self.cache[t]['boxes_3D'] = list(filter(lambda box: box.cls in self.class_to_group, self.cache[t]['boxes_3D']))

            # Replace class with group
            for i in range(len(self.cache[t]['boxes_2D'])):
                self.cache[t]['boxes_2D'][i].cls = self.class_to_group[self.cache[t]['boxes_2D'][i].cls]
                self.cache[t]['boxes_3D'][i].cls = self.class_to_group[self.cache[t]['boxes_3D'][i].cls]

    def get_image(self, t):
        return _get_image(os.path.join(self.img2_dir, t + ".png"))

    def get_PIL(self, t):
        return _get_PIL(os.path.join(self.img2_dir, t + ".png"))

    def get_velo(self, t, workspace_lim=((-40, 40), (-1, 3), (0, 70.4)), use_fov_filter=True):
        return _get_velo(path=os.path.join(self.velo_dir, t + '.bin'),
                         calib=self.cache[t]['calib'],
                         workspace_lim=workspace_lim,
                         use_fov_filter=use_fov_filter,
                         img_size=self.cache[t]['image_size'])

    def get_velo_reduced(self, t):
        return np.fromfile(os.path.join(self.velo_reduced_dir, t + '.bin'), dtype=np.float32).reshape((3, -1))

    def get_depth_map(self, t):
        return _get_image(os.path.join(self.depth2_dir, t + ".png"))[..., 0:1]  # Returns H x W x 1

    def get_mask(self, t, algorithm, dilate=False, dilation_iterations=5, dilation_kernel=5):
        mask = _get_image(os.path.join(self.masks_dir, algorithm, t + ".png"))  # Returns H x W

        if dilate:
            mask = cv2.dilate(mask, np.ones((dilation_kernel, dilation_kernel), np.uint8), iterations=dilation_iterations)

        return mask

    def get_boxes_2D(self, t):
        return copy.deepcopy(self.cache[t]['boxes_2D'])

    def get_boxes_3D(self, t):
        return copy.deepcopy(self.cache[t]['boxes_3D'])

    def get_boxes_3D_as_array(self, t):
        return [b.x for b in self.cache[t]['boxes_3D']]

    def get_calib(self, t):
        return copy.deepcopy(self.cache[t]['calib'])

    def get_ids(self, subset):
        # assert subset in ['train', 'val', 'micro', 'trainval']
        return [line.rstrip('\n') for line in open(os.path.join(self.ds_dir, 'subsets', subset + '.txt'))]

    def _id_contains(self, t, classes):
        for box in self.get_boxes_2D(t):
            if box.cls in classes:
                return True
        return False

    def get_ids_containing(self, subset, classes):
        return list(filter(lambda t: self._id_contains(t, classes), self.get_ids(subset)))


def _get_image(path):
    img = Image.open(path)
    return np.asarray(img, dtype=np.float32) / 255.0


def _get_PIL(path):
    return Image.open(path)


# Returns a list of dict; dict contains parameters for one object
def _read_txt_file(path, is_pred=False):
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=KITTI_COLUMN_NAMES if is_pred else KITTI_COLUMN_NAMES[:-1])
        txt_lbl = [row for row in reader]
    return txt_lbl


# Returns velo pts and reflectance in rectified camera coordinates
def _get_velo(path, calib, workspace_lim=((-40, 40), (-1, 3), (0, 70.4)), use_fov_filter=True, img_size=None):
    velo = np.fromfile(path, dtype=np.float32).reshape((-1, 4)).T
    pts = velo[0:3]
    reflectance = velo[3:]

    # Transform points from velo coordinates to rectified camera coordinates
    V2C, R0, P2 = calib
    pts = transform(np.dot(R0, V2C), pts)

    # Remove points out of workspace
    if workspace_lim is not None:
        pts, reflectance = box_filter(pts, workspace_lim, decorations=reflectance)

    # Remove points not projecting onto the image plane
    if use_fov_filter:
        pts, reflectance = fov_filter(pts, P=P2, img_size=img_size, decorations=reflectance)

    return pts, reflectance


def _get_calib(path):
    # Read file
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            # Skip if line is empty
            if len(line) == 0:
                continue
            # Load required matrices only
            matrix_name = line[0][:-1]
            if matrix_name == 'Tr_velo_to_cam':
                V2C = np.array([float(i) for i in line[1:]], dtype=np.float32).reshape(3, 4)  # Read from file
                V2C = np.insert(V2C, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row
            elif matrix_name == 'R0_rect':
                R0 = np.array([float(i) for i in line[1:]], dtype=np.float32).reshape(3, 3)  # Read from file
                R0 = np.insert(R0, 3, values=0, axis=1)  # Pad with zeros on the right
                R0 = np.insert(R0, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row
            elif matrix_name == 'P2':
                P2 = np.array([float(i) for i in line[1:]], dtype=np.float32).reshape(3, 4)
                P2 = np.insert(P2, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row

    return V2C, R0, P2


