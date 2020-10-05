import numpy as np
from datasets.kitti import transforms_3D

# 2D Box declaration modes
CORNER_CORNER = 0
CORNER_DIM = 1
CENTER_DIM = 2


class DIFFICULTY:
    EASY = 1
    MODERATE = 2
    HARD = 3
    UNK = 4


class Box2D:
    def __init__(self, values, mode, cls=None, confidence=None, text=None):
        self.cls = cls
        self.confidence = confidence
        self.text = text
        if mode == CORNER_CORNER:
            self.x1, self.y1, self.x2, self.y2 = values
            self.cx, self.cy, self.w, self.h = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.x2 - self.x1, self.y2 - self.y1
        elif mode == CENTER_DIM:
            self.cx, self.cy, self.w, self.h = values
            self.x1, self.y1, self.x2, self.y2 = self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2
        elif mode == CORNER_DIM:
            self.x1, self.y1, self.w, self.h = values
            self.cx, self.cy, self.x2, self.y2 = self.x1 + self.w / 2, self.y1 + self.h / 2, self.x1 + self.w, self.y1 + self.h


class Box3D:
    def __init__(self, h, w, l, x, y, z, yaw, alpha=-10, cls=None, confidence=None, text=None):
        # Copy params
        self.h, self.w, self.l = h, w, l
        self.x, self.y, self.z = x, y, z
        self.yaw = yaw
        self.alpha = alpha
        self.cls = cls
        self.confidence = confidence
        self.text = text


class Cylinder:
    def __init__(self, h, r, x, y, z, cls=None, text=None, solid=False):
        # Copy params
        self.h, self.r = h, r
        self.x, self.y, self.z = x, y, z
        self.cls = cls
        self.text = text
        self.solid = solid


class Sphere:
    def __init__(self, r, x, y, z, cls=None, text=None):
        # Copy params
        self.r = r
        self.x, self.y, self.z = x, y, z
        self.cls = cls
        self.text = text


def box_to_string(box):
    s = ""
    if isinstance(box, Box2D):
        s += "Center: (%.3f, %.3f)   H,W: (%.3f, %.3f)   (X1Y1X2Y2): (%.3f, %.3f, %.3f, %.3f)" % (box.cx, box.cy, box.h, box.w, box.x1, box.y1, box.x2, box.y2)
    elif isinstance(box, Box3D):
        s += "Center: (%.3f, %.3f, %.3f)   HWL: (%.3f, %.3f, %.3f)  YAW: %.3f" % (box.x, box.y, box.z, box.h, box.w, box.l, box.yaw)

    s += "" if box.cls is None else ("   Cls: " + box.cls)
    s += "" if box.confidence is None else ("   Conf: " + str(box.confidence))
    s += "" if not hasattr(box, 'alpha') or box.alpha is None else ("   Alpha: " + str(box.alpha))
    return s


#         |
# (2,6)___|___(1,5)
#    |    |   |
#    |    |   |           (bottom, top)
#    |        |
#    |        |
# (3,7)------(0,4)

def get_corners_3D(box):
    corners = np.array([[-box.l / 2, box.l / 2, box.l / 2, -box.l / 2, -box.l / 2, box.l / 2, box.l / 2, -box.l / 2],
                        [0, 0, 0, 0, -box.h, -box.h, -box.h, -box.h],
                        [-box.w / 2, -box.w / 2, box.w / 2, box.w / 2, -box.w / 2, -box.w / 2, box.w / 2, box.w / 2]], dtype=np.float32)
    H = np.dot(transforms_3D.translation_matrix(box.x, box.y, box.z), transforms_3D.rot_y_matrix(box.yaw))
    return transforms_3D.transform(H, corners)


def get_corners_bev(box):
    corners = np.array([[-box.l / 2, box.l / 2, box.l / 2, -box.l / 2],
                        [-box.w / 2, -box.w / 2, box.w / 2, box.w / 2]], dtype=np.float32)
    H = np.array([[np.math.cos(box.yaw), np.math.sin(box.yaw)],
                  [-np.math.sin(box.yaw), np.math.cos(box.yaw)]], dtype=np.float32)

    return (H @ corners + np.array([[box.x], [box.z]], dtype=np.float32))[[1, 0]]


def project_box_3D(P, box: Box3D) -> Box2D:
    corners = get_corners_3D(box)
    corners = corners[:, corners[2] > 0.5]  # Remove points behind the image plane
    corners = transforms_3D.project(P, corners)
    x1, y1 = np.min(corners, axis=1)
    x2, y2 = np.max(corners, axis=1)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, 1241), min(y2, 374)  # Bound to image
    # x1, x2 = np.clip([x1, x2], 0, 1241)
    # y1, y2 = np.clip([y1, y2], 0, 374)
    return Box2D((x1, y1, x2, y2), mode=CORNER_CORNER, cls=box.cls, confidence=box.confidence, text=box.text)


def get_box_difficulty(box2D, box3D):
    if box2D.h >= 40 and box3D.occluded == 0 and box3D.truncated <= 0.15:
        return DIFFICULTY.EASY
    elif box2D.h >= 25 and box3D.occluded <= 1 and box3D.truncated <= 0.3:
        return DIFFICULTY.MODERATE
    elif box2D.h >= 25 and box3D.occluded <= 3 and box3D.truncated <= 0.5:
        return DIFFICULTY.HARD
    return DIFFICULTY.UNK
