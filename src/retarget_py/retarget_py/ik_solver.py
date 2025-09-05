#!/usr/bin/env python3
from typing import Optional
import numpy as np
from retarget_py.algebra_utils import *

# 필요 시 algebra_utils의 make_homogeneous, make_inv_homogeneous 등을 사용
# from algebra_utils import make_homogeneous, make_inv_homogeneous

class IKSolver:
    """
    IK 래퍼 스켈레톤.
    - 실제 IK 라이브러리(예: TRAC-IK/IKFast/Pinocchio/Drake)를 여기서 래핑해 사용한다.
    - solve(pose_6d): 6D 포즈( x,y,z, roll,pitch,yaw ) -> joint vector
    """
    def __init__(self, urdf_path: Optional[str] = None, base_link: str = 'base_link', tip_link: str = 'tool0'):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.tip_link = tip_link
        # TODO: 실제 IK 초기화 (TRAC-IK or Pinocchio 등)

    def solve(self, pose_6d: np.ndarray, q_seed: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        입력: pose_6d = [x, y, z, roll, pitch, yaw]
        출력: 조인트 각도 배열 (라디안). 실패 시 None
        """
        # TODO: 실제 IK 호출. 여기서는 더미 반환(None)
        return None
