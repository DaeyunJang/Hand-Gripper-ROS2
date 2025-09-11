#!/usr/bin/env python3
"""
algebra_utils.py
- 로봇 좌표계/선형대수 관련 자주 쓰이는 함수 모음
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple # compatible python < 3.9

# ---------------------- 유틸 ----------------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normlaize vector
    @note
    normalize & norm are different!!!!

    Returns:
        np.ndarray: return nomalized vector
    """
    n = np.linalg.norm(v)
    return v if n < eps else (v / n)

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray,
                          normal: np.ndarray,
                          degrees: bool = False) -> float:
    """두 벡터 사이의 부호 있는 각도 (normal 방향 기준)"""
    v1u = normalize(v1)
    v2u = normalize(v2)
    dot_prod = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
    cross_prod = np.cross(v1u, v2u)
    theta_abs = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
    sign_dir = np.sign(np.dot(normalize(normal), cross_prod))
    return float(sign_dir * (np.rad2deg(theta_abs) if degrees else theta_abs))

def make_homogeneous(rotm: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """회전행렬 + 평행이동 → 4x4 동차행렬"""
    T = np.eye(4)
    T[:3, :3] = rotm
    T[:3,  3] = trans.reshape(3)
    return T

def make_inv_homogeneous(T: np.ndarray) -> np.ndarray:
    """4x4 동차행렬 역행렬"""
    rotm = T[:3, :3]
    trans = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3,:3] = rotm.T
    Ti[:3,3] = -rotm.T @ trans
    return Ti

# ---------------------- 회전/쿼터니언 ----------------------
def rotm_to_quat(R: np.ndarray, eps=1e-12) -> Tuple[float, float, float, float]:
        # 안전한 회전행렬 → 쿼터니언
        # epsilon: prevent zero division
        
        # check size
        R = np.asarray(R, dtype=float)
        if R.shape == (4, 4):
            R = R[:3, :3]
        elif R.shape != (3, 3):
            raise ValueError("R must be 3x3 or 4x4 matrix")
        
        qw = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
        qx = (R[2,1] - R[1,2]) / (4*qw + eps)
        qy = (R[0,2] - R[2,0]) / (4*qw + eps)
        qz = (R[1,0] - R[0,1]) / (4*qw + eps)
        return (qx,qy,qz,qw)


def rotm_to_quat_v2(R: np.ndarray, eps=1e-12) -> Tuple[float, float, float, float]:
    """
    회전행렬(3x3 또는 4x4 상위 3x3) -> 쿼터니언 (qx, qy, qz, qw)
    - trace>0 공식 + 대각원소 최대 분기 방식으로 수치 안정성 강화
    - 최종 쿼터니언 정규화
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
    if n < eps:   # prevent zero division
        return (0.0, 0.0, 0.0, 1.0)
    q /= n
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

def quat_to_rotm(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """쿼터니언(x,y,z,w) -> 회전행렬(3x3)"""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)],
    ], dtype=float)