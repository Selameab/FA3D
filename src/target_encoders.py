import cv2
import numpy as np
import tensorflow as tf
from datasets.kitti.boxes import Box3D, get_corners_3D


# Physical to Target Axis Mapping
# X -> Y      Y -> Z      Z -> X
class ResTargets:
    def __init__(self, shape, keep_factor, x_range=(-40, 40), z_range=(0, 70.4), anchor_hwly=(1.5, 1.6, 3.9, 1.75), default_class='Car', confidence_thresh=0.05):
        self.shape = shape
        self.x_range = x_range
        self.z_range = z_range
        self.default_class = default_class
        self.keep_factor = keep_factor  # Keep = 1 -> no target subsampling
        self.confidence_thresh = confidence_thresh

        self.ha, self.wa, self.la, self.ya = anchor_hwly
        self.da = np.sqrt(self.wa ** 2 + self.la ** 2)
        self.height, self.width = shape  # Target map shape

        self.delta_x = (x_range[1] - x_range[0]) / shape[0]
        self.delta_z = (z_range[1] - z_range[0]) / shape[1]

        # Anchor centers in physical space
        self.zas = np.linspace(z_range[0], z_range[1], shape[1] + 1)[:-1]
        self.xas = np.linspace(x_range[0], x_range[1], shape[0] + 1)[:-1]

        self.Tout = (tf.float32, tf.float32, tf.float32, tf.float32)  # For tf.data
        self.output_shapes = [(self.height, self.width, 1),
                              (self.height, self.width, 4),
                              (self.height, self.width, 4),
                              (self.height, self.width, 3)]

    def encode(self, boxes):
        cls = np.zeros(shape=(self.height, self.width, 1), dtype=np.float32)
        hwl = np.zeros(shape=(self.height, self.width, 4), dtype=np.float32)
        xyz = np.zeros(shape=(self.height, self.width, 4), dtype=np.float32)
        angle = np.zeros(shape=(self.height, self.width, 3), dtype=np.float32)
        for box in boxes:
            if self.keep_factor < 1:
                smaller_box = Box3D(h=box.h, w=box.w * self.keep_factor, l=box.l * self.keep_factor, x=box.x, y=box.y, z=box.z, yaw=box.yaw, cls=box.cls)
            else:
                smaller_box = box
            corners = get_corners_3D(smaller_box)[[2, 0], :4]  # Shape: 2 (#coordinates/z,x), 4 (#corners)
            corners[1] -= self.x_range[0]
            corners[0] /= self.delta_z
            corners[1] /= self.delta_x

            mask = np.zeros(shape=(self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, [corners.T.astype(np.int32)], (1, 1, 1))
            mask = mask.astype(np.bool)

            cls[mask, 0] = 1

            hwl[mask] = np.log(box.h / self.ha), np.log(box.w / self.wa), np.log(box.l / self.la), 1

            xyz[mask, 0] = np.repeat([(box.x - self.xas) / self.da], self.width, axis=0).T[mask]
            xyz[mask, 1] = (box.y - self.ya) / self.ha
            xyz[mask, 2] = np.repeat([(box.z - self.zas) / self.da], self.height, axis=0)[mask]
            xyz[mask, 3] = 1

            angle[mask] = np.math.cos(box.yaw), np.math.sin(box.yaw), 1

        return [cls, hwl, xyz, angle]

    def decode_raw(self, cls, hwl, xyz, angle):
        mask = np.squeeze(cls) >= self.confidence_thresh

        # Points on target map
        centers = np.array(np.where(mask), dtype=np.float32)
        centers = centers.T * np.array([self.delta_x, self.delta_z])  # centers.shape = K, 2(physical_x, physical_z)
        centers[:, 0] += self.x_range[0]

        cls = cls[mask]  # K, 1
        hwl = hwl[mask]  # K,3
        xyz = xyz[mask]  # K, 3
        angle = angle[mask]  # K, 2

        hwl = np.exp(hwl) * np.array([self.ha, self.wa, self.la])

        xyz[:, 0] = xyz[:, 0] * self.da + centers[:, 0]
        xyz[:, 1] = xyz[:, 1] * self.ha + self.ya
        xyz[:, 2] = xyz[:, 2] * self.da + centers[:, 1]

        yaws = np.arctan2(angle[:, 1], angle[:, 0])[:, np.newaxis]

        return np.concatenate([cls, hwl, xyz, yaws], axis=1)

    def decode(self, cls, hwl, xyz, angle):
        boxes = []
        for conf, h, w, l, x, y, z, yaw in self.decode_raw(cls, hwl, xyz, angle):
            boxes += [Box3D(h=h, w=w, l=l, x=x, y=y, z=z, yaw=yaw, confidence=conf, cls=self.default_class)]

        return boxes

    def decode_batch(self, y_pred):
        cls, hwl, xyz, angle = y_pred  # Split
        boxes_3D_batch = []
        for i in range(len(cls)):
            boxes_3D_batch += [self.decode(cls[i], hwl[i], xyz[i], angle[i])]

        return boxes_3D_batch
