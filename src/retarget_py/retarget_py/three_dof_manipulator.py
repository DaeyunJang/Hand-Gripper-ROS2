#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal, Optional
import numpy as np

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

@dataclass
class ThreeDoFManipulator:
    """
    3-자유도 R-R-R 매니퓰레이터(베이스 Rz, 나머지 Ry, Ry) 포지셔닝 IK.
    - 링크는 각 pitch 조인트 뒤에 +x 방향으로 뻗는다고 가정 (L1, L2)
    - tip 프레임의 길이/오프셋은 tool_offset_local 로 보정
    - 3DoF라 '자세'는 추종하지 않고 '위치'로 각도 계산

    Parameters
    ----------
    link1 : float
        어깨→팔꿈치 링크 길이 (L1)
    link2 : float
        팔꿈치→손목(툴 기준점) 링크 길이 (L2)
    base_z_offset : float
        베이스 프레임 z 오프셋 (필요시)
    tool_offset_local : np.ndarray, shape (3,)
        tip 로컬 좌표계에서의 툴 오프셋 [tx, ty, tz]
        (예: 그리퍼 길이 L_tool 이면 [L_tool, 0, 0])
    angle_unit : "rad" | "deg"
        반환 각도 단위
    elbow : "up" | "down"
        2링크 평면 IK 해 선택
    """
    link1: float
    link2: float
    base_z_offset: float = 0.0
    tool_offset_local: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=float)
    angle_unit: Literal["rad", "deg"] = "rad"
    elbow: Literal["up", "down"] = "up"

    def ik(self, T_base_to_tip: np.ndarray) -> Tuple[float, float, float]:
        """
        IK 계산: 4x4 동차행렬 입력 → (theta1, theta2, theta3) 출력

        T_base_to_tip: ^base T_tip
        """
        assert T_base_to_tip.shape == (4, 4), "T must be 4x4"

        R = T_base_to_tip[:3, :3]
        p = T_base_to_tip[:3, 3].astype(float)

        # tool 오프셋 보정: tip 위치에서 로컬 오프셋을 회전시켜 빼면 '손목' 위치
        p_wrist = p - R @ self.tool_offset_local

        px = float(p_wrist[0])
        py = float(p_wrist[1])
        pz = float(p_wrist[2] - self.base_z_offset)

        # theta1: 베이스 yaw(z)
        theta1 = np.arctan2(py, px)

        # 평면화 (theta1로 정한 평면에서 2-링크 IK)
        r = np.hypot(px, py)       # XY 반경
        z = pz

        L1, L2 = self.link1, self.link2
        dist2 = r*r + z*z

        # reachability clamp
        cos_th3 = _clamp((dist2 - L1*L1 - L2*L2) / (2.0*L1*L2), -1.0, 1.0)
        sin_th3_pos = np.sqrt(max(0.0, 1.0 - cos_th3*cos_th3))
        if self.elbow == "up":
            theta3 = np.arctan2(sin_th3_pos, cos_th3)
        else:
            theta3 = np.arctan2(-sin_th3_pos, cos_th3)

        theta2 = np.arctan2(z, r) - np.arctan2(L2*np.sin(theta3), L1 + L2*np.cos(theta3))

        # normalize
        theta1 = _wrap_pi(theta1)
        theta2 = _wrap_pi(theta2)
        theta3 = _wrap_pi(theta3)

        if self.angle_unit == "deg":
            th = np.degrees([theta1, theta2, theta3])
            return float(th[0]), float(th[1]), float(th[2])
        return theta1, theta2, theta3

    # (옵션) FK 검증용
    def fk(self, theta1: float, theta2: float, theta3: float, angle_unit: Optional[str] = None) -> np.ndarray:
        unit = angle_unit or self.angle_unit
        if unit == "deg":
            theta1, theta2, theta3 = np.radians([theta1, theta2, theta3])

        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c23, s23 = np.cos(theta2 + theta3), np.sin(theta2 + theta3)

        x_plane = self.link1 * c2 + self.link2 * c23
        z_plane = self.link1 * s2 + self.link2 * s23
        px = c1 * x_plane
        py = s1 * x_plane
        pz = z_plane + self.base_z_offset

        Rz = np.array([[ c1,-s1,0],
                       [ s1, c1,0],
                       [  0,  0,1]])
        Ry2 = np.array([[ c2,0, s2],
                        [  0,1,  0],
                        [-s2,0, c2]])
        Ry3 = np.array([[ np.cos(theta3),0, np.sin(theta3)],
                        [ 0,              1, 0             ],
                        [-np.sin(theta3),0, np.cos(theta3)]])
        R = Rz @ Ry2 @ Ry3

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = np.array([px, py, pz], dtype=float) + R @ self.tool_offset_local
        return T
