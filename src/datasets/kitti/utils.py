import numpy as np
from numba import njit
from shapely.geometry import Polygon

from datasets.kitti import transforms_3D
from datasets.kitti.boxes import get_corners_3D, get_corners_bev


def box_filter(pts, box_lim, decorations=None):
    x_range, y_range, z_range = box_lim
    mask = ((pts[0] >= x_range[0]) & (pts[0] <= x_range[1]) &
            (pts[1] >= y_range[0]) & (pts[1] <= y_range[1]) &
            (pts[2] >= z_range[0]) & (pts[2] <= z_range[1]))
    pts = pts[:, mask]
    if decorations is None:
        return pts
    else:
        return pts, decorations[:, mask]


# img_size = (img_height, img_width)
def fov_filter(pts, P, img_size, decorations=None):
    pts_projected = transforms_3D.project(P, pts)
    mask = ((pts_projected[0] >= 0) & (pts_projected[0] <= img_size[1]) &
            (pts_projected[1] >= 0) & (pts_projected[1] <= img_size[0]))
    pts = pts[:, mask]
    return pts if decorations is None else pts, decorations[:, mask]


def compute_iou_bev(box_1, box_2):
    box_1 = Polygon(get_corners_3D(box_1)[[2, 0], :4].T)
    box_2 = Polygon(get_corners_3D(box_2)[[2, 0], :4].T)
    return box_1.intersection(box_2).area / box_1.union(box_2).area


# Faster version (5.5 times) - avoids heavy rotated iou computation for far apart boxes
def compute_iou_bev_v2(box_1, box_2):
    if - dist_bev(box_1, box_2) > 7:
        return 0
    box_1 = Polygon(get_corners_3D(box_1)[[2, 0], :4].T)
    box_2 = Polygon(get_corners_3D(box_2)[[2, 0], :4].T)
    return box_1.intersection(box_2).area / box_1.union(box_2).area


def dist_bev(box_1, box_2):
    return - np.sqrt((box_1.x - box_2.x) ** 2 + (box_1.z - box_2.z) ** 2)


def nms_bev(thresh, kernel=dist_bev, max_boxes=50, min_hit=5):  # ~ 20ms
    def _nms_bev(boxes):
        boxes.sort(key=lambda box: box.confidence, reverse=True)

        filtered_boxes = []
        while len(boxes) > 0 and len(filtered_boxes) < max_boxes:
            top_box = boxes[0]
            # boxes = np.delete(boxes, 0)  # Remove top box from main list
            boxes = boxes[1:]

            # Remove all other boxes overlapping with selected box
            boxes_to_remove = []
            hits = 0
            for box_id in range(len(boxes)):
                iou = kernel(boxes[box_id], top_box)
                if iou > thresh:
                    boxes_to_remove += [box_id]
                    hits += 1
            boxes = np.delete(boxes, boxes_to_remove)

            # Add box with highest confidence to output list
            if hits >= min_hit:
                filtered_boxes += [top_box]

        return filtered_boxes

    return _nms_bev


def compute_mask_accurate(pts, box):
    corners = get_corners_3D(box).T
    v0P = pts.T - corners[0]
    v01 = corners[1] - corners[0]
    v03 = corners[3] - corners[0]
    v04 = corners[4] - corners[0]

    p1 = np.dot(v0P, v01)
    p3 = np.dot(v0P, v03)
    p4 = np.dot(v0P, v04)

    mask = (0 <= p1) & (p1 <= np.dot(v01, v01)) & \
           (0 <= p3) & (p3 <= np.dot(v03, v03)) & \
           (0 <= p4) & (p4 <= np.dot(v04, v04))

    return mask


# Faster than v1 (Removes most points using approximate method)
def compute_mask_accurate_v2(pts, box):
    corners = get_corners_3D(box)
    sub_ids = np.argwhere(compute_mask_estimate(pts, box))[:, 0]
    pts_subset = pts[:, sub_ids]

    v0P = pts_subset - corners[:, 0, np.newaxis]
    v01 = corners[:, 1] - corners[:, 0]
    v03 = corners[:, 3] - corners[:, 0]
    v04 = corners[:, 4] - corners[:, 0]

    p1 = v01 @ v0P
    p3 = v03 @ v0P
    p4 = v04 @ v0P

    mask = (0 <= p1) & (p1 <= (v01 @ v01)) & (0 <= p3) & (p3 <= (v03 @ v03)) & (0 <= p4) & (p4 <= (v04 @ v04))

    return sub_ids[mask]


# Uses the smallest non-oriented box to segment
def compute_mask_estimate(pts, box_3D):
    corners = get_corners_3D(box_3D)
    mins = np.min(corners, axis=1)
    maxes = np.max(corners, axis=1)
    mask = (mins[0] <= pts[0]) & (pts[0] <= maxes[0]) & \
           (mins[1] <= pts[1]) & (pts[1] <= maxes[1]) & \
           (mins[2] <= pts[2]) & (pts[2] <= maxes[2])
    return mask


def compute_mask_cylinder(pts, x, z, r):
    dist = np.square(pts[0] - x) + np.square(pts[2] - z)
    return dist <= r ** 2


def count_points_accurate(pts, box):
    return len(compute_mask_accurate_v2(pts, box))


# KITTI Angles = [-pi, pi]
def standardize_angle(angle):
    while angle > np.pi:
        angle = angle - 2 * np.pi
    while angle < -np.pi:
        angle = angle + 2 * np.pi
    return angle


def compute_alpha(box):
    return box.yaw - np.arctan2(box.x, box.z)


def compute_yaw(alpha, x, z):
    return np.round(standardize_angle(alpha + np.arctan2(x, z)), 2)


def pt_in_boxes_2D(pt, boxes_2D):  # pt.shape = (2, )
    for box_2D in boxes_2D:
        if (box_2D.x1 < pt[0] < box_2D.x2) and (box_2D.y1 < pt[1] < box_2D.y2):
            return True
    return False


def pt_in_boxes_3D(pt, boxes_3D):
    for box in boxes_3D:
        if compute_mask_accurate(pt[:, np.newaxis], box):
            return True
    return False


# Returns id of nearest box. If nearest box is above threshold, returns none
def get_nearest_box(needle, haystack, threshold=1000):
    if haystack is None or len(haystack) == 0 or needle is None:
        return None

    distances = [(needle.x - box.x) ** 2 + (needle.z - box.z) ** 2 for box in haystack]
    return None if np.min(distances) > threshold ** 2 else np.argmin(distances)


# Returns axis aligned rectangle (in bev) = (x1, y1, x2, y2)
def get_aa_rect(box):
    dw, dl = box.w / 2, box.l / 2
    return box.z - dw, box.x - dl, box.z + dw, box.x + dl


# Source: https://gitlab.com/haghdam/3d_object_localization/-/blob/Development/nn/nms.py
def compute_rect_ious(ref_rect, rects):
    """
    Returns the iou of ref_rect against all other rectangles
    """
    x1 = np.maximum(ref_rect[0], rects[:, 0])
    y1 = np.maximum(ref_rect[1], rects[:, 1])
    x2 = np.minimum(ref_rect[2], rects[:, 2])
    y2 = np.minimum(ref_rect[3], rects[:, 3])

    intersection = np.where(np.logical_or(x1 > x2, y1 > y2), 0, (x2 - x1) * (y2 - y1))

    union = ((rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1]) + (ref_rect[2] - ref_rect[0]) * (ref_rect[3] - ref_rect[1]) - intersection)
    iou = intersection / union

    # if not np.all(np.logical_and(0 <= iou, iou <= 1)):
    #     raise ValueError()
    return iou


# Uses axis aligned IOU for matching
def nms_bev_v2(thresh, pre_count=2500, post_count=50, min_hit=3):  # ~ 2.9ms
    def _nms_bev_v2(boxes):
        boxes.sort(key=lambda box: box.confidence, reverse=True)
        boxes = np.array(boxes[:pre_count])

        rects = np.array([get_aa_rect(b) for b in boxes], dtype=np.float32)

        filtered_boxes = []
        while len(boxes) > 0 and len(filtered_boxes) < post_count:
            top_box, top_rect = boxes[0], rects[0]
            boxes, rects = boxes[1:], rects[1:]

            ids_to_keep = np.where(compute_rect_ious(top_rect, rects) < thresh)[0]
            if len(rects) - len(ids_to_keep) >= min_hit:  # min_hit - box has at least x other overlapping detections (remove false positives)
                filtered_boxes += [top_box]
            boxes, rects = boxes[ids_to_keep], rects[ids_to_keep]

        return filtered_boxes

    return _nms_bev_v2


@njit
def ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


@njit
def intersects(set1, set2):
    result = np.zeros(shape=(len(set1), len(set2)))

    for i in range(len(set1)):
        ax, ay, bx, by = set1[i]
        for j in range(len(set2)):
            cx, cy, dx, dy = set2[j]
            if (ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)) and \
                    (ccw(cx, cy, dx, dy, ax, ay) != ccw(cx, cy, dx, dy, bx, by)):
                result[i, j] = True

    return result


def pts_to_segments(pts):
    return np.array([
        [pts[0, 0], pts[1, 0], pts[0, 1], pts[1, 1]],
        [pts[0, 1], pts[1, 1], pts[0, 2], pts[1, 2]],
        [pts[0, 2], pts[1, 2], pts[0, 3], pts[1, 3]],
        [pts[0, 3], pts[1, 3], pts[0, 0], pts[1, 0]]
    ], dtype=np.float32)


def collision_test(new_box, existing_boxes):
    if len(existing_boxes) == 0:
        return False

    existing_lines = np.concatenate([pts_to_segments(get_corners_bev(b)) for b in existing_boxes], axis=0)
    new_lines = pts_to_segments(get_corners_bev(new_box))

    return np.any(intersects(existing_lines, new_lines))



# Faster version of nmv_bev (distance based)
def nms_bev_v3(thresh, pre_count=2500, post_count=50, min_hit=3):
    thresh = thresh * thresh

    def _nms_bev_v3(boxes):
        boxes.sort(key=lambda box: box.confidence, reverse=True)
        boxes = np.array(boxes[:pre_count])

        centers = np.array([(box.x, box.z) for box in boxes])

        filtered_boxes = []
        while len(boxes) > 0 and len(filtered_boxes) < post_count:
            top_box, top_center = boxes[0], centers[0]
            boxes, centers = boxes[1:], centers[1:]

            # Remove all other boxes overlapping with selected box
            distances = centers - top_center
            distances = (distances[:, 0] * distances[:, 0]) + (distances[:, 1] * distances[:, 1])
            ids_to_keep = np.where(distances > thresh)[0]
            if len(centers) - len(ids_to_keep) >= min_hit:  # min_hit - box has at least x other overlapping detections (remove false positives)
                filtered_boxes += [top_box]
            boxes, centers = boxes[ids_to_keep], centers[ids_to_keep]

        return filtered_boxes

    return _nms_bev_v3


def nms_rotated_bev(iou_thresh, pre_count=5000, post_count=200, min_hit=3):
    def _nms_bev_v3(boxes):
        boxes.sort(key=lambda box: box.confidence, reverse=True)
        boxes = np.array(boxes[:pre_count])

        filtered_boxes = []
        while len(boxes) > 0 and len(filtered_boxes) < post_count:
            top_box = boxes[0]
            boxes = boxes[1:]

            # Remove all other boxes overlapping with selected box
            ids_to_keep = np.where(np.array([compute_iou_bev(top_box, b) for b in boxes]) < iou_thresh)[0]
            if len(boxes) - len(ids_to_keep) >= min_hit:  # min_hit - box has at least x other overlapping detections (remove false positives)
                filtered_boxes += [top_box]
            boxes = boxes[ids_to_keep]

        return filtered_boxes

    return _nms_bev_v3
