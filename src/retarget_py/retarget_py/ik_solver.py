#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ik_solver.py
# standalone module
# 3-DoF DH manipulator (roll, pan, pan) — position-only IK (mask) + animation
# Requires: roboticstoolbox-python, spatialmath-python, matplotlib, numpy

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath import base as sm
from math import pi


@dataclass
class IKResult:
    success: bool
    reason: str
    q_final: np.ndarray
    iterations: Optional[int]
    residual: Optional[float]
    elapsed_ms: float
    pos_err: np.ndarray
    pos_err_norm: float
    ang_err_rad: Optional[float]
    T_final: SE3


class ThreeDoFManipulator:
    """
    3-DoF DH manipulator (roll, pan, pan).
    Link layout (default):
      - L1: d = l1 (z offset), a = 0,    alpha = +pi/2
      - L2: d = 0,  a = l2 (x link),     alpha = 0
      - L3: d = 0,  a = l3 (x link),     alpha = 0
    L2, L3 limits default to [-pi/2, +pi/2].
    """

    def __init__(
        self,
        l1: float = 0.00,
        l2: float = 0.020,
        l3: float = 0.020,
        qlim1: Tuple[float, float] = (-pi, pi),
        qlim2: Tuple[float, float] = (-pi/2, pi/2),
        qlim3: Tuple[float, float] = (-pi/2, pi/2),
        name: str = "3DOF_Manipulator",
    ):
        self.l1, self.l2, self.l3 = l1, l2, l3
        L1 = rtb.DHLink(d=l1, a=0.0,   alpha=pi/2, offset=0, qlim=list(qlim1))
        L2 = rtb.DHLink(d=0.0, a=l2,   alpha=0.0,  offset=0, qlim=list(qlim2))
        L3 = rtb.DHLink(d=0.0, a=l3,   alpha=0.0,  offset=0, qlim=list(qlim3))
        self.robot = rtb.DHRobot([L1, L2, L3], name=name)

    # ---------- Kinematics ----------
    def fkine(self, q: np.ndarray) -> SE3:
        return self.robot.fkine(q)

    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        """Return joint/world positions [base, j1, j2, ee] as (4,3)."""
        T = SE3()
        pts = [T.t]  # base origin
        for link, qi in zip(self.robot.links, q):
            T = T * link.A(qi)
            pts.append(T.t)
        return np.vstack(pts)

    # ---------- IK solve ----------
    def solve(
        self,
        T_target: np.ndarray | SE3,
        q0: Optional[np.ndarray] = None,
        position_only: bool = True,
        ilimit: int = 1000,
        tol: float = 1e-6,
    ) -> IKResult:
        """
        DLS(LM) 기반 역기구학.
        - position_only=True 이면 mask=[1,1,1, 0,0,0] 로 위치만 추적.
        - False면 풀 포즈 추적(mask=[1,1,1, 1,1,1]).
        """
        if isinstance(T_target, np.ndarray):
            if T_target.shape != (4, 4):
                raise ValueError("T_target must be 4x4 homogeneous matrix or spatialmath.SE3.")
            T_goal = SE3(T_target)
        elif isinstance(T_target, SE3):
            T_goal = T_target
        else:
            raise TypeError("T_target must be np.ndarray (4x4) or SE3.")

        mask = np.r_[1, 1, 1, 0, 0, 0] if position_only else np.r_[1, 1, 1, 1, 1, 1]
        if q0 is None:
            q0 = np.zeros(3)

        t0 = time.perf_counter()
        sol = self.robot.ikine_LM(T_goal, q0=q0, mask=mask, ilimit=ilimit, tol=tol)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        T_final = self.fkine(sol.q)
        p_goal, p_final = T_goal.t, T_final.t
        pos_err = p_goal - p_final
        pos_err_norm = float(np.linalg.norm(pos_err))

        ang_err_rad = None
        if not position_only:
            R_err = T_goal.R @ T_final.R.T
            ang_err_rad, _ = sm.tr2angvec(R_err)

        return IKResult(
            success=bool(sol.success),
            reason=getattr(sol, "reason", ""),
            q_final=np.asarray(sol.q).copy(),
            iterations=getattr(sol, "iterations", None),
            residual=getattr(sol, "residual", None),
            elapsed_ms=float(elapsed_ms),
            pos_err=pos_err,
            pos_err_norm=pos_err_norm,
            ang_err_rad=ang_err_rad,
            T_final=T_final,
        )

    # ---------- Animation ----------
    def animate(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        target_pos: Optional[np.ndarray] = None,
        frames: int = 180,
        interval: int = 20,
        show: bool = True,
    ):
        """
        q_start -> q_end 조인트공간 선형 보간을 애니메이션으로 표시.
        """
        q_traj = np.linspace(q_start, q_end, frames)
        ee_path = np.zeros((frames, 3))
        for i, q in enumerate(q_traj):
            ee_path[i] = self.fkine(q).t

        reach_xy = self.l2 + self.l3 + 0.05
        zmax = self.l1 + 0.05
        zmin = -0.05

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([-reach_xy, reach_xy])
        ax.set_ylim([-reach_xy, reach_xy])
        ax.set_zlim([zmin, zmax])
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("3-DoF DH Manipulator — Motion")

        if target_pos is not None:
            ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], s=40)

        link_line, = ax.plot([], [], [], linewidth=3)
        path_line, = ax.plot([], [], [], linewidth=1)

        def init():
            link_line.set_data([], [])
            link_line.set_3d_properties([])
            path_line.set_data([], [])
            path_line.set_3d_properties([])
            return link_line, path_line

        def update(i):
            q = q_traj[i]
            pts = self.joint_positions(q)  # (4,3): base→j1→j2→ee
            link_line.set_data(pts[:, 0], pts[:, 1])
            link_line.set_3d_properties(pts[:, 2])

            trail = ee_path[:i + 1]
            path_line.set_data(trail[:, 0], trail[:, 1])
            path_line.set_3d_properties(trail[:, 2])
            return link_line, path_line

        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                      blit=True, interval=interval)
        if show:
            plt.show()
        return ani


# 편의용 팩토리 (요청하신 이름을 대신하는 역할)
def get_manipulator(**kwargs) -> ThreeDoFManipulator:
    """
    Example:
        from ik_solver import get_manipulator
        manip = get_manipulator(l1=0.20, l2=0.20, l3=0.15)
    """
    return ThreeDoFManipulator(**kwargs)


# 샘플 실행
if __name__ == "__main__":
    # 기존 예시와 동일한 목표/초기값
    l1, l2, l3 = 0.20, 0.20, 0.15
    manip = ThreeDoFManipulator(l1, l2, l3)

    T_target = np.array([
        [1.0, 0.0, 0.0, 0.22],
        [0.0, 1.0, 0.0, 0.20],
        [0.0, 0.0, 1.0, 0.10],
        [0.0, 0.0, 0.0, 1.0]
    ])
    q0 = np.deg2rad([90.0, 90.0, -90.0])

    # 위치만 추적
    # (mask=[1,1,1,0,0,0])
    res = manip.solve(T_target, q0=q0, position_only=True, ilimit=1000, tol=1e-6)
    actual_T = res.T_final
    # 상세 리포트
    print("=== IK (LM/DLS) — Detailed Report ===")
    print(f"Success           : {res.success}  | Reason: {res.reason}")
    print(f"Iterations        : {res.iterations}  | Residual: {res.residual}")
    print(f"Elapsed time      : {res.elapsed_ms:.3f} ms")
    print("----- Joints -----")
    print(f"q0     [deg]      : {np.rad2deg(q0)}")
    print(f"q_final[deg]      : {np.rad2deg(res.q_final)}")
    print("----- Pose -----")
    print(f"Target T          : \n{T_target}")
    print(f"Final T   [m]     : \n{res.T_final}")
    print(f"pos err  [m]      : {res.pos_err}  ||e_p|| = {res.pos_err_norm:.6e}")
    if res.ang_err_rad is not None:
        print(f"ang err [rad/deg] : {res.ang_err_rad:.6e} rad / {np.rad2deg(res.ang_err_rad):.6e} deg")
    else:
        print("ang err [rad/deg] : (orientation masked)")

    # 애니메이션 (q0 -> q_final)
    manip.animate(q_start=q0, q_end=res.q_final, target_pos=T_target[:3, 3], frames=180, interval=20, show=True)


