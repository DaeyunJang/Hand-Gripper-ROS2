#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import math

import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState

from ament_index_python.packages import get_package_share_directory


class RetargetDirectNode(Node):
    """
    손의 world_abs 기준 TF를 직접 사용하여 관절각 산출:
      - j2 := yaw_z(T_mcp_to_pip)
      - j3 := yaw_z(T_pip_to_dip) + yaw_z(T_dip_to_tip)
      - j1 : (TODO) 나중에 구현

    출력: JointState(name=[finger*_j2, finger*_j3, ...], position=[...])
    """

    def __init__(self) -> None:
        super().__init__("retarget_direct_node")

        # ---------------- Parameters ----------------
        # 퍼블리시 토픽
        self.declare_parameter("output_topic", "/hand_gripper/joint_states_direct")
        self.output_topic: str = self.get_parameter("output_topic").get_parameter_value().string_value

        # 손 프레임 접미사 (기본 world_abs)
        self.declare_parameter("target_suffix", "world_abs")
        self.target_suffix: str = self.get_parameter("target_suffix").get_parameter_value().string_value

        # 퍼블리시 주기(Hz)
        self.declare_parameter("rate_hz", 30.0)
        self.rate_hz: float = float(self.get_parameter("rate_hz").value)

        # gripper 설정 파일 경로 (없으면 패키지/로컬 기본값 탐색)
        self.declare_parameter("gripper_config_path", "")
        self.config_path: str = self.get_parameter("gripper_config_path").get_parameter_value().string_value

        # 파라미터 변경 콜백
        self.add_on_set_parameters_callback(self._on_params)

        # ---------------- Load Config ----------------
        cfg = self._load_gripper_config(self.config_path)
        self.fingers_cfg: List[Dict[str, Any]] = cfg.get("fingers") or []
        if not self.fingers_cfg:
            self.get_logger().warn("[retarget_direct_node] No fingers in config. Nothing to publish.")

        # 조인트 이름 빌드 (각 finger에 대해 j2, j3만 사용)
        self.joint_names: List[str] = []
        for fdef in self.fingers_cfg:
            fname = str(fdef["name"])
            self.joint_names += [f"{fname}_j2", f"{fname}_j3"]   # j1은 나중에 구현 예정

        # ---------------- ROS IO ----------------
        self.pub = self.create_publisher(JointState, self.output_topic, 1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # 타이머
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"[retarget_direct_node] output_topic='{self.output_topic}', "
            f"target_suffix='{self.target_suffix}', rate={self.rate_hz:.1f}Hz, "
            f"fingers={len(self.fingers_cfg)}"
        )

    # ---------------- Param change ----------------
    def _on_params(self, params):
        for p in params:
            if p.name == "output_topic" and p.type_ == Parameter.Type.STRING:
                new_topic = p.value
                if new_topic != self.output_topic:
                    self.output_topic = new_topic
                    try:
                        self.destroy_publisher(self.pub)
                    except Exception:
                        pass
                    self.pub = self.create_publisher(JointState, self.output_topic, 1)
                    self.get_logger().info(f"[retarget_direct_node] output_topic -> {self.output_topic}")
            elif p.name == "rate_hz" and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                val = float(p.value)
                self.rate_hz = max(1.0, val)
                try:
                    self.destroy_timer(self.timer)
                except Exception:
                    pass
                self.timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)
                self.get_logger().info(f"[retarget_direct_node] rate_hz -> {self.rate_hz}")
            elif p.name == "gripper_config_path" and p.type_ == Parameter.Type.STRING:
                self.get_logger().warn("[retarget_direct_node] gripper_config_path changed; restart recommended.")
            elif p.name == "target_suffix" and p.type_ == Parameter.Type.STRING:
                self.target_suffix = p.value
                self.get_logger().info(f"[retarget_direct_node] target_suffix -> {self.target_suffix}")
        return SetParametersResult(successful=True)

    # ---------------- Config loader ----------------
    def _load_gripper_config(self, user_path: str) -> Dict[str, Any]:
        """
        config 형식 (필요 필드만 사용):
        gripper:
          fingers:
            - name: finger1
              target_finger: {label: left,  finger: index}
            - name: finger2
              target_finger: {label: left,  finger: thumb}
            - name: finger3
              target_finger: {label: right, finger: index}
            - name: finger4
              target_finger: {label: right, finger: thumb}
        """
        candidates: List[Path] = []
        if user_path:
            candidates.append(Path(user_path).expanduser().resolve())
        env_p = os.environ.get("HAND_CONFIG_PATH")
        if env_p:
            candidates.append(Path(env_p).expanduser().resolve())
        try:
            pkg_share = Path(get_package_share_directory("retarget_py"))
            candidates.append((pkg_share / "config" / "gripper_config.yaml").resolve())
        except Exception:
            pass
        candidates.append((Path(__file__).resolve().parent / "config" / "gripper_config.yaml").resolve())

        cfg_path = next((c for c in candidates if c.is_file()), None)
        if not cfg_path:
            self.get_logger().warning("No gripper_config.yaml found. Using defaults.")
            return {"fingers": []}

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            g = data.get("gripper") or {}
            fingers: List[Dict[str, Any]] = []
            for i, fdef in enumerate(g.get("fingers") or [], start=1):
                name = str(fdef.get("name") or f"finger{i}")
                tfing = fdef.get("target_finger") or {}
                label = str(tfing.get("label") or "")
                finger = str(tfing.get("finger") or "")
                if not label or not finger:
                    self.get_logger().warning(f"[config] {name}: target_finger.label/finger required. Skipping.")
                    continue
                fingers.append({"name": name, "target_finger": {"label": label, "finger": finger}})
            out = {"fingers": fingers}
            self.get_logger().info(f"Loaded gripper config: {cfg_path} (fingers={len(fingers)})")
            return out
        except Exception as ex:
            self.get_logger().warning(f"Failed to load gripper config '{cfg_path}': {ex}. Using defaults.")
            return {"fingers": []}

    # ---------------- Helpers ----------------
    def _hand_frame_name(self, label: str, finger: str, joint: str) -> str:
        # 예) hand_left_index_mcp_world_abs
        return f"hand_{label}_{finger}_{joint}_{self.target_suffix}"

    @staticmethod
    def _yaw_from_matrix(T: np.ndarray) -> float:
        """4x4 또는 3x3 행렬에서 Z축 yaw(rad) 추출, [-pi, pi]로 정규화"""
        if T.shape == (4, 4):
            Rm = T[:3, :3]
        else:
            Rm = T
        yaw, pitch, roll = R.from_matrix(Rm).as_euler('ZYX', degrees=False)
        # return math.atan2(math.sin(yaw), math.cos(yaw))  # wrap to [-pi, pi]
        return yaw  # wrap to [-pi, pi]

    def _lookup_transform_matrix(self, parent: str, child: str) -> Optional[np.ndarray]:
        """TF (parent <- child) 4x4 행렬 반환. 실패 시 None"""
        try:
            Tmsg: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=parent, source_frame=child, time=rclpy.time.Time())
            t = Tmsg.transform.translation
            q = Tmsg.transform.rotation
            Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            M = np.eye(4, dtype=float)
            M[:3, :3] = Rm
            M[0, 3], M[1, 3], M[2, 3] = float(t.x), float(t.y), float(t.z)
            return M
        except Exception:
            return None

    # ---------------- Main timer ----------------
    def _on_timer(self) -> None:
        if not self.fingers_cfg:
            return

        names: List[str] = []
        pos:   List[float] = []

        for fdef in self.fingers_cfg:
            fname = str(fdef["name"])                    # ex) "finger1"
            target = fdef["target_finger"]               # {"label": "...", "finger": "..."}
            label = str(target["label"]).lower()         # ex) "left"
            tfing = str(target["finger"]).lower()        # ex) "index"

            # world_abs 기준 프레임들
            mcp = self._hand_frame_name(label, tfing, "mcp")
            pip = self._hand_frame_name(label, tfing, "pip")
            dip = self._hand_frame_name(label, tfing, "dip")
            tip = self._hand_frame_name(label, tfing, "tip")

            # 상대변환
            T_mcp_to_pip = self._lookup_transform_matrix(mcp, pip)
            T_pip_to_dip = self._lookup_transform_matrix(pip, dip)
            T_dip_to_tip = self._lookup_transform_matrix(dip, tip)

            if T_mcp_to_pip is None or T_pip_to_dip is None or T_dip_to_tip is None:
                # 일부만 없어도 불안정 → 이 손가락은 건너뜀 (필요 시 캐시/보간 가능)
                continue
            
            ## j0_linear: MIDAS gripper 전용 - MCP 위치 기반 j0 선형값 산출
            # T_center_to_mcp = self._lookup_transform_matrix("camera_color_optical_frame", f"hand_{label}_{tfing}_world_abs")
            T_center_to_mcp = self._lookup_transform_matrix("camera_color_optical_frame", mcp)
            if T_center_to_mcp is None:
                self.get_logger().warning(f'[{fname}] T_center_to_mcp not found, skip j0_linear')
                continue
            tx = T_center_to_mcp[0,3]
            if fname in ["finger1", "finger2"]:   # left index
                tx = tx
            elif fname in ["finger3", "finger4"]: # right index
                tx = -tx
            tx = max(0.1, min(tx, 0.2))
            tx = (tx-0.1) / 2
            names.append(f"{fname}_j0_linear")
            pos.append(tx)
            
            # j1 은 나중에 구현 (TODO)
            # j1_name = f"{fname}_j1";
            # j1_val = 0.0  # TODO
            # j1 = 0.0
            names.append(f"{fname}_j1")
            pos.append(0.0)
            
            # yaw 추출 및 합성append
            yaw_mcp_pip = self._yaw_from_matrix(T_mcp_to_pip)      # -> j2
            yaw_pip_dip = self._yaw_from_matrix(T_pip_to_dip)
            yaw_dip_tip = self._yaw_from_matrix(T_dip_to_tip)
            
            # self.get_logger().info(f'==========================================================')
            # self.get_logger().info(f'========================= {fname} ========================')
            # self.get_logger().info(f'yaw_mcp_pip: rad = {yaw_mcp_pip} | deg = {np.rad2deg(yaw_mcp_pip)}')
            # self.get_logger().info(f'yaw_pip_dip: rad = {yaw_pip_dip} | deg = {np.rad2deg(yaw_pip_dip)}')
            # self.get_logger().info(f'yaw_dip_tip: rad = {yaw_dip_tip} | deg = {np.rad2deg(yaw_dip_tip)}')
            
            yaw_j3 = yaw_pip_dip + yaw_dip_tip                      # -> j3

            # 필요하면 부호/범위 조정 (환경에 따라 플립될 수 있음)
            yaw_mcp_pip = float(np.clip(yaw_mcp_pip, -1.57, 1.57))
            yaw_j3      = float(np.clip(yaw_j3, -1.57, 1.57))

            

            names += [f"{fname}_j2", f"{fname}_j3"]
            pos   += [yaw_mcp_pip,   yaw_j3]

        if not names:
            return

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = names
        js.position = pos
        js.velocity = []   # 비워도 OK
        js.effort   = []   # 비워도 OK

        self.pub.publish(js)
        # self.get_logger().info(f'= zPUBLISH !! =')
        


def main() -> None:
    rclpy.init()
    node = RetargetDirectNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
