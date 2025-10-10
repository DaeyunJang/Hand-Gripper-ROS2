#!/usr/bin/env python3
"""
algebra_utils.py
- Collection of commonly used linear algebra and robotics coordinate system utilities
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any, List, Optional, Tuple  # compatible with Python < 3.9

# ---------------------- Utilities ----------------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector.

    Note:
        `normalize` and `norm` are different.
        This rescales the vector to unit length.

    Args:
        v (np.ndarray): input vector
        eps (float): epsilon to avoid division by zero

    Returns:
        np.ndarray: normalized vector (or original if norm < eps)
    """
    n = np.linalg.norm(v)
    return v if n < eps else (v / n)

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray,
                          normal: np.ndarray,
                          degrees: bool = False) -> float:
    """
    Signed angle between two vectors, defined with respect to a reference normal.

    Args:
        v1 (np.ndarray): first vector
        v2 (np.ndarray): second vector
        normal (np.ndarray): reference normal (to determine sign)
        degrees (bool): if True returns angle in degrees, otherwise radians

    Returns:
        float: signed angle
    """
    v1u = normalize(v1)
    v2u = normalize(v2)
    dot_prod = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
    cross_prod = np.cross(v1u, v2u)
    theta_abs = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
    sign_dir = np.sign(np.dot(normalize(normal), cross_prod))
    return float(sign_dir * (np.rad2deg(theta_abs) if degrees else theta_abs))

def make_homogeneous(rotm: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Construct a 4x4 homogeneous transformation matrix.

    Args:
        rotm (np.ndarray): 3x3 rotation matrix
        trans (np.ndarray): 3-element translation vector

    Returns:
        np.ndarray: 4x4 homogeneous transform
    """
    T = np.eye(4)
    T[:3, :3] = rotm
    T[:3, 3] = trans.reshape(3)
    return T

def make_inv_homogeneous(T: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 homogeneous transformation.

    Args:
        T (np.ndarray): 4x4 homogeneous transform

    Returns:
        np.ndarray: inverse transform
    """
    rotm = T[:3, :3]
    trans = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = rotm.T
    Ti[:3, 3] = -rotm.T @ trans
    return Ti

# ---------------------- Rotation / Quaternion ----------------------
def rotm_to_quat(R: np.ndarray, eps=1e-12) -> Tuple[float, float, float, float]:
    """
    Convert a rotation matrix into a quaternion.

    Args:
        R (np.ndarray): 3x3 or 4x4 rotation matrix
        eps (float): epsilon to avoid division by zero

    Returns:
        tuple: quaternion (qx, qy, qz, qw)
    """
    R = np.asarray(R, dtype=float)
    if R.shape == (4, 4):
        R = R[:3, :3]
    elif R.shape != (3, 3):
        raise ValueError("R must be 3x3 or 4x4 matrix")

    qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw + eps)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw + eps)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw + eps)
    return (qx, qy, qz, qw)

def rotm_to_quat_v2(R: np.ndarray, eps=1e-12) -> Tuple[float, float, float, float]:
    """
    Convert a rotation matrix into a quaternion (qx, qy, qz, qw).

    - Handles both 3x3 or 4x4 (upper 3x3 block) matrices
    - Uses trace-based method with branching for numerical stability
    - Normalizes the quaternion before returning

    Args:
        R (np.ndarray): 3x3 or 4x4 rotation matrix
        eps (float): epsilon threshold for normalization

    Returns:
        tuple: quaternion (qx, qy, qz, qw)
    """
    R = np.asarray(R, dtype=float)
    if R.shape == (4, 4):
        R = R[:3, :3]
    elif R.shape != (3, 3):
        raise ValueError("R must be 3x3 or 4x4 rotation matrix")

    t = R[0, 0] + R[1, 1] + R[2, 2]

    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n < eps:   # prevent division by zero
        return (0.0, 0.0, 0.0, 1.0)
    q /= n
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

def quat_to_rotm(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) into a 3x3 rotation matrix.

    Args:
        qx, qy, qz, qw (float): quaternion components

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)],
    ], dtype=float)

def scale_translation(T: np.ndarray, scale: float | Tuple[float, float, float]) -> np.ndarray:
    """
    Scale only the translation part of a homogeneous transform.

    Args:
        T (np.ndarray): 4x4 homogeneous transform
        scale (float or tuple): scalar or (sx, sy, sz) vector

    Returns:
        np.ndarray: transformed 4x4 matrix with scaled translation
    """
    M = T.copy()
    M[:3, 3] = M[:3, 3] * scale
    return M

def rot_y_180_mat() -> np.ndarray:
    """
    Generate a homogeneous transform representing a 180° rotation around Y-axis.

    Returns:
        np.ndarray: 4x4 homogeneous transform (rotation + zero translation)
    """
    Ry = R.from_euler('y', 180.0, degrees=True).as_matrix()
    return make_homogeneous(Ry, np.zeros(3))
