import numpy as np
from datasets.kitti.kitti_utils import compute_mask_accurate, standardize_angle, compute_mask_estimate, compute_mask_accurate_v2, collision_test
import random
from datasets.kitti.transforms_3D import transform, rot_y_matrix, translation_matrix
import copy


def per_box_dropout(dropout_ratio=0.1):
    def _per_box_dropout(pts, boxes):
        ids_to_remove = []
        for box in boxes:
            ids = compute_mask_accurate_v2(pts, box)
            if len(ids) > 10:
                random.shuffle(ids)
                ids_to_remove += list(ids[:int(len(ids) * dropout_ratio)])
        pts = np.delete(pts, ids_to_remove, axis=1)
        return pts, boxes

    return _per_box_dropout


# trans_range - Tuple of tuple
def per_box_rotation_translation(rot_range, trans_range):
    def _per_box_rotation_translation(pts, boxes):
        for box in boxes:
            # Create random translation and rotation matrices
            alpha = np.random.uniform(rot_range[0], rot_range[1])
            R = rot_y_matrix(alpha)[:3, :3]
            trans_x = np.random.uniform(trans_range[0][0], trans_range[0][1])
            trans_y = np.random.uniform(trans_range[1][0], trans_range[1][1])
            trans_z = np.random.uniform(trans_range[2][0], trans_range[2][1])

            ids = compute_mask_accurate_v2(pts, box)

            center = np.array([[box.x, box.y, box.z]], dtype=np.float32).T
            pts[:, ids] = (R @ (pts[:, ids] - center)) + center + np.array([[trans_x, trans_y, trans_z]]).T

            # Transform box
            box.x += trans_x
            box.y += trans_y
            box.z += trans_z
            box.yaw = standardize_angle(box.yaw + alpha)

        return pts, boxes

    return _per_box_rotation_translation


def per_box_noise(rot_range, trans_std):
    def _per_box_noise(pts, boxes):
        for box in boxes:
            # Create random translation and rotation matrices
            alpha = np.random.uniform(-rot_range, rot_range)
            R = rot_y_matrix(alpha)[:3, :3]
            trans = np.random.normal(scale=trans_std, size=(3,)).astype(np.float32)

            ids = compute_mask_accurate_v2(pts, box)

            center = np.array([[box.x, box.y, box.z]], dtype=np.float32).T
            pts[:, ids] = (R @ (pts[:, ids] - center)) + center + trans[:, np.newaxis]

            # Transform box
            box.x += trans[0]
            box.y += trans[1]
            box.z += trans[2]
            box.yaw = standardize_angle(box.yaw + alpha)

        return pts, boxes

    return _per_box_noise


def per_box_noise_with_collision_test(rot_range, trans_std, attempts=10):
    def _per_box_noise_with_collision_test(pts, boxes):
        final_boxes = []
        for box in boxes:
            attempt = 0
            while attempt < attempts:
                attempt += 1

                # Create random translation and rotation matrices
                alpha = np.random.uniform(-rot_range, rot_range)
                trans = np.random.normal(scale=trans_std, size=(3,)).astype(np.float32)

                new_box = copy.deepcopy(box)
                new_box.x += trans[0]
                new_box.y += trans[1]
                new_box.z += trans[2]
                new_box.yaw = standardize_angle(new_box.yaw + alpha)

                if not collision_test(new_box, final_boxes):
                    ids = compute_mask_accurate_v2(pts, box)
                    center = np.array([[box.x, box.y, box.z]], dtype=np.float32).T
                    R = rot_y_matrix(alpha)[:3, :3]
                    pts[:, ids] = (R @ (pts[:, ids] - center)) + center + trans[:, np.newaxis]
                    final_boxes += [new_box]
                    break

        return pts, final_boxes

    return _per_box_noise_with_collision_test


def flip_along_y():
    def _flip_along_y(pts, boxes):
        pts[0] = -pts[0]

        for b in boxes:
            b.x *= -1
            if 0 <= b.yaw < np.pi / 2:
                b.yaw = np.pi - b.yaw
            elif -np.pi / 2 <= b.yaw < 0:
                b.yaw = -(b.yaw + np.pi)
            elif np.pi / 2 <= b.yaw < np.pi:
                b.yaw = np.pi - b.yaw
            elif -np.pi <= b.yaw < -np.pi / 2:
                b.yaw = -(b.yaw + np.pi)

            b.yaw = standardize_angle(b.yaw)

        return pts, boxes

    return _flip_along_y


def global_scale(scale_range):
    def _global_scale(pts, boxes):
        s = np.random.uniform(scale_range[0], scale_range[1])
        pts = pts * s
        for b in boxes:
            b.x *= s
            b.y *= s
            b.z *= s
            b.w *= s
            b.l *= s
            b.h *= s
        return pts, boxes

    return _global_scale


def global_rot(rot_range):
    def _global_rot(pts, boxes):
        alpha = np.random.uniform(rot_range[0], rot_range[1])
        R = rot_y_matrix(alpha)[:3, :3]
        # pts = transform(R, pts)
        pts = R @ pts
        for box in boxes:
            box.x, box.y, box.z = (R @ np.array([[box.x, box.y, box.z]]).T)[:, 0]
            box.yaw = standardize_angle(box.yaw + alpha)

        return pts, boxes

    return _global_rot


def global_trans(trans_range):
    def _global_trans(pts, boxes):
        trans_x = np.random.uniform(trans_range[0][0], trans_range[0][1])
        trans_y = np.random.uniform(trans_range[1][0], trans_range[1][1])
        trans_z = np.random.uniform(trans_range[2][0], trans_range[2][1])

        # T = translation_matrix(trans_x, trans_y, trans_z)
        # pts = transform(T, pts)
        pts = pts + np.array([[trans_x, trans_y, trans_z]], np.float32).T
        for box in boxes:
            box.x += trans_x
            box.y += trans_y
            box.z += trans_z
            # box.x, box.y, box.z = transform(T, np.array([[box.x, box.y, box.z]]).T)[:, 0]

        return pts, boxes

    return _global_trans
