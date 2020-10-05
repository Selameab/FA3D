import numpy as np


# Transformation functions

def rot_y_matrix(angle):
    return np.array([[np.math.cos(angle), 0, np.math.sin(angle), 0],
                     [0, 1, 0, 0],
                     [-np.math.sin(angle), 0, np.math.cos(angle), 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=np.float32)


def C2H(pts):
    return np.insert(pts, 3, values=1, axis=0)


def H2C(pts):
    return pts[:3, :] / pts[3:, :]


def transform(H, pts):
    return H2C(np.dot(H, C2H(pts)))


def project(P, pts):
    pts = transform(P, pts)
    pts = pts[:2, :] / pts[2, :]
    return pts
