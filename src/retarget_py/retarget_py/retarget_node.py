#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os

import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
import tf2_ros
from tf2_ros import TransformException, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory

from retarget_py.algebra_utils import make_homogeneous, make_inv_homogeneous
from retarget_py.handpose_tf_listener import mat_to_tf


# ---------- 작은 유틸 ----------

def rot_y_180_mat() -> np.ndarray:
    """source -> gripper (Ry 180°)"""
    Ry = R.from_euler('y', 180.0, degrees=True).as_matrix()
    return make_homogeneous(Ry, np.zeros(3))

def trans_xyz_mat(x: float, y: float, z: float = 0.0) -> np.ndarray:
    """parent -> child: pure translation"""
    return make_homogeneous(np.eye(3), np.array([x, y, z], dtype=float))

def add_suffix(frame_name: str, suffix: str = "gripper_scale") -> str:
    tail = f"_{suffix}"
    return frame_name if frame_name.endswith(tail) else (frame_name + tail)


# ---------- 메인 노드 ----------

class RetargetNode(Node):
    """
    순서
    (A) source_root 기준 모든 hand 프레임을 먼저 스케일 적용하여 '<원래이름>_gripper_scale'로 퍼블리시 (parent=source_root)
    (B) 손가락 i:
        - finger{i}_base_frame: gripper에서 (offset_x, offset_y) 평행이동 (정적)
        - finger{i}_mcp: finger_base에서 '스케일된 target_base_frame'을 본 변환의 x만 반영
        - finger{i}_tip: finger{i}_mcp에 '스케일된 (target_base -> target_eef)' 변환을 그대로 곱함
    """

    def __init__(self) -> None:
        super().__init__("retarget_node")

        # --- Params ---
        self.source_root: str = self.declare_parameter(
            "source_root", "camera_color_optical_frame"
        ).get_parameter_value().string_value

        self.gripper_base_frame: str = self.declare_parameter(
            "gripper_base_frame", "gripper_base_frame"
        ).get_parameter_value().string_value

        # scale: param 우선, 없으면 YAML
        self.declare_parameter("scale", 0.0)
        self.scale_scalar: float = 1.0
        self.scale_xyz: Optional[np.ndarray] = None  # [sx,sy,sz] 지정 시 우선

        # suffix for scaled hand frames
        self.scale_suffix: str = self.declare_parameter(
            "scale_suffix", "gripper_scale"
        ).get_parameter_value().string_value

        self.rate_hz: float = float(self.declare_parameter("rate_hz", 100.0).value)

        # YAML path
        self.config_path: str = self.declare_parameter(
            "gripper_config_path", ""
        ).get_parameter_value().string_value

        # --- TF IO ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)

        # --- Load Config ---
        cfg = self._load_gripper_config(self.config_path)
        # fingers: [{name, base_offset_x, base_offset_y, target_base_frame, target_eef_frame}]
        self.fingers: List[Dict[str, Any]] = cfg["fingers"]
        # (선택) 모든 손 프레임 리스트 (없으면 fingers에 등장하는 것들만 스케일)
        self.hand_frames_to_scale: List[str] = cfg.get("hand_frames", [])

        # scale 우선순위
        scale_param = float(self.get_parameter("scale").value or 0.0)
        if scale_param > 0.0:
            self.scale_scalar = scale_param
            self.scale_xyz = None
            scale_src = "param"
        else:
            if cfg.get("scale_xyz") is not None:
                self.scale_xyz = np.array(cfg["scale_xyz"], dtype=float)
                self.scale_scalar = 1.0
                scale_src = "yaml(scale_xyz)"
            else:
                self.scale_scalar = float(cfg.get("scale", 1.0))
                self.scale_xyz = None
                scale_src = "yaml(scale)"

        # --- Static frames: source->gripper, gripper->finger_base ---
        self._finger_base_names: List[str] = []
        self._gripper_to_fingerBase: List[np.ndarray] = []
        self._publish_static_frames()

        # --- Timer ---
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        scale_repr = list(self.scale_xyz) if self.scale_xyz is not None else self.scale_scalar
        self.get_logger().info(
            f"[retarget] source='{self.source_root}', gripper='{self.gripper_base_frame}', "
            f"scale={scale_repr} ({scale_src}), fingers={len(self.fingers)}, rate={self.rate_hz:.1f}Hz"
        )

        self._last_warn: Dict[str, float] = {}
        self._warn_interval: float = 2.0

    # ---------- Config ----------

    def _load_gripper_config(self, user_path: str) -> Dict[str, Any]:
        """
        YAML 예:
        gripper:
          scale: 0.06
          # scale_xyz: [sx, sy, sz]
          # (선택) hand_frames: [ hand_left_*_canon ... ]   # 여기 넣으면 이 목록 전체를 스케일해서 퍼블리시
          fingers:
            - name: finger1
              base_offset_x: -0.06
              base_offset_y: -0.010
              target_base_frame: hand_left_index_mcp_canon
              target_eef_frame:  hand_left_index_tip_canon
            - ...
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
            self.get_logger().warning("No gripper_config.yaml found. Using default config.")
            return {
                "scale": 1.0,
                "scale_xyz": None,
                "hand_frames": [],
                "fingers": [{
                    "name": "finger1",
                    "base_offset_x": 0.0, "base_offset_y": 0.0,
                    "target_base_frame": "hand_left_index_mcp_canon",
                    "target_eef_frame":  "hand_left_index_tip_canon",
                }],
            }

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            g = data.get("gripper") or {}

            fingers: List[Dict[str, Any]] = []
            for i, fdef in enumerate(g.get("fingers") or [], start=1):
                name = str(fdef.get("name") or f"finger{i}")
                x = float(fdef.get("base_offset_x") or 0.0)
                y = float(fdef.get("base_offset_y") or 0.0)

                base = fdef.get("target_base_frame")
                eef  = fdef.get("target_eef_frame")
                if not base or not eef:
                    self.get_logger().warning(f"[config] finger '{name}' requires target_base_frame & target_eef_frame. Skipping.")
                    continue

                fingers.append({
                    "name": name,
                    "base_offset_x": x, "base_offset_y": y,
                    "target_base_frame": str(base),
                    "target_eef_frame":  str(eef),
                })

            if not fingers:
                fingers = [{
                    "name": "finger1",
                    "base_offset_x": 0.0, "base_offset_y": 0.0,
                    "target_base_frame": "hand_left_index_mcp_canon",
                    "target_eef_frame":  "hand_left_index_tip_canon",
                }]

            scale = float(g.get("scale") or 1.0)
            sxyz = g.get("scale_xyz")
            if sxyz is not None and (not isinstance(sxyz, (list, tuple)) or len(sxyz) != 3):
                self.get_logger().warning("gripper.scale_xyz must be [sx, sy, sz]; ignoring.")
                sxyz = None

            hand_frames = g.get("hand_frames") or []
            if hand_frames and not isinstance(hand_frames, list):
                self.get_logger().warning("gripper.hand_frames must be a list of strings; ignoring.")
                hand_frames = []

            self.get_logger().info(f"Loaded gripper config: {cfg_path} (fingers={len(fingers)})")
            return {"scale": scale, "scale_xyz": sxyz, "fingers": fingers, "hand_frames": hand_frames}

        except Exception as ex:
            self.get_logger().warning(f"Failed to load gripper config '{cfg_path}': {ex}. Using defaults.")
            return {
                "scale": 1.0,
                "scale_xyz": None,
                "hand_frames": [],
                "fingers": [{
                    "name": "finger1",
                    "base_offset_x": 0.0, "base_offset_y": 0.0,
                    "target_base_frame": "hand_left_index_mcp_canon",
                    "target_eef_frame":  "hand_left_index_tip_canon",
                }],
            }

    # ---------- 정적 프레임 퍼블리시 ----------

    def _publish_static_frames(self) -> None:
        """(정적) source -> gripper, gripper -> finger{i}_base"""
        static_tfs: List[TransformStamped] = []
        stamp = Time().to_msg()

        # source -> gripper
        source_to_gripper = rot_y_180_mat()
        static_tfs.append(mat_to_tf(source_to_gripper,
                                    parent=self.source_root,
                                    child=self.gripper_base_frame,
                                    stamp=stamp))

        # gripper -> finger_base
        self._finger_base_names.clear()
        self._gripper_to_fingerBase.clear()
        for fdef in self.fingers:
            finger_base = f"{fdef['name']}_base_frame"
            self._finger_base_names.append(finger_base)

            gripper_to_finger = trans_xyz_mat(fdef["base_offset_x"], fdef["base_offset_y"], 0.0)
            self._gripper_to_fingerBase.append(gripper_to_finger)

            static_tfs.append(mat_to_tf(gripper_to_finger,
                                        parent=self.gripper_base_frame,
                                        child=finger_base,
                                        stamp=stamp))

        self.static_br.sendTransform(static_tfs)

    # ---------- 주기 콜백 ----------

    def _on_timer(self) -> None:
        """
        A. 먼저 source 기준으로 hand 프레임들 스케일 적용해 '<name>_gripper_scale'로 퍼블리시 (parent=source)
        B. 손가락별로 finger_mcp, finger_tip 생성
        """
        stamp = self.get_clock().now().to_msg()
        source = self.source_root
        gripper = self.gripper_base_frame

        # source <-> gripper (정적)
        source_to_gripper = rot_y_180_mat()
        gripper_to_source = make_inv_homogeneous(source_to_gripper)

        # ---- A) 스케일된 hand 프레임들 퍼블리시 ----
        frames_to_scale: List[str] = list(self.hand_frames_to_scale)
        if not frames_to_scale:
            # 최소한 fingers에 등장하는 타깃 프레임들은 스케일해서 퍼블리시
            s = set()
            for f in self.fingers:
                s.add(f["target_base_frame"])
                s.add(f["target_eef_frame"])
            frames_to_scale = sorted(s)
            self.get_logger().warn(
                "gripper.hand_frames가 비어있습니다. fingers에 등장한 프레임들만 스케일합니다."
            )

        # source -> (각 hand_frame)
        source_to_scaled: Dict[str, np.ndarray] = {}
        for hf in frames_to_scale:
            H = self._lookup_mat_latest(source, hf)  # source -> hf
            if H is None:
                continue
            H_scaled = self._scale_translation(H)    # 병진만 스케일
            source_to_scaled[hf] = H_scaled
            self.br.sendTransform(
                mat_to_tf(H_scaled, parent=source, child=add_suffix(hf, self.scale_suffix), stamp=stamp)
            )

        # ---- B) 손가락별 mcp / tip 생성 ----
        for finger_base, gripper_to_finger, fdef in zip(
            self._finger_base_names, self._gripper_to_fingerBase, self.fingers
        ):
            base_name = fdef["target_base_frame"]
            eef_name  = fdef["target_eef_frame"]

            # 스케일된 source->base, source->eef 가 있어야 진행
            Hb = source_to_scaled.get(base_name)
            He = source_to_scaled.get(eef_name)
            if Hb is None or He is None:
                # 해당 프레임이 손 전체 리스트에 없으면 fingers 등장분만으로도 만들어지니 여기선 거의 안걸림
                continue

            # gripper 기준으로도 필요: gripper -> base_scaled, gripper -> eef_scaled
            gripper_to_base_scaled = gripper_to_source @ Hb
            gripper_to_eef_scaled  = gripper_to_source @ He

            # (1) finger_base 기준으로 'base_scaled'를 본 변환: finger_base -> base_scaled
            finger_to_base_scaled = make_inv_homogeneous(gripper_to_finger) @ gripper_to_base_scaled

            # x 성분만 취해 finger{i}_mcp 생성 (parent = finger_base)
            x_only = float(finger_to_base_scaled[0, 3])
            finger_mcp_name = f"{finger_base.replace('_base_frame', '')}_mcp"
            finger_to_mcp = trans_xyz_mat(x_only, 0.0, 0.0)
            self.br.sendTransform(mat_to_tf(finger_to_mcp, parent=finger_base, child=finger_mcp_name, stamp=stamp))

            # (2) 스케일된 (base -> eef) 변환: base_scaled^-1 @ eef_scaled
            base_to_eef_scaled = make_inv_homogeneous(gripper_to_base_scaled) @ gripper_to_eef_scaled

            # finger{i}_mcp 를 parent로 finger{i}_tip 생성 (mcp->tip = base->eef_scaled)
            finger_tip_name = f"{finger_base.replace('_base_frame', '')}_tip"
            self.br.sendTransform(mat_to_tf(base_to_eef_scaled, parent=finger_mcp_name, child=finger_tip_name, stamp=stamp))

    # ---------- 헬퍼 ----------

    def _lookup_mat_latest(self, target: str, source: str) -> Optional[np.ndarray]:
        """(target -> source) 최신 TF (timeout=0)"""
        try:
            T: TransformStamped = self.tf_buffer.lookup_transform(
                target, source, Time(), Duration(seconds=0.0)
            )
            t = T.transform.translation
            q = T.transform.rotation
            Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            return make_homogeneous(Rm, np.array([t.x, t.y, t.z], dtype=float))
        except TransformException as ex:
            self._warn_throttled(f"lookup failed: {target} <- {source} ({ex})")
            return None
        except Exception as ex:
            self._warn_throttled(f"lookup failed: {target} <- {source} ({ex})")
            return None

    def _scale_translation(self, T: np.ndarray) -> np.ndarray:
        """병진만 스케일. 회전은 그대로."""
        M = T.copy()
        if self.scale_xyz is None:
            M[:3, 3] = M[:3, 3] * self.scale_scalar
        else:
            M[:3, 3] = M[:3, 3] * self.scale_xyz
        return M

    def _warn_throttled(self, text: str) -> None:
        from time import monotonic
        now = monotonic()
        last = self._last_warn.get(text, 0.0)
        if now - last >= self._warn_interval:
            self.get_logger().warning(text)
            self._last_warn[text] = now


# ---------- 엔트리 ----------

def main() -> None:
    rclpy.init()
    node = RetargetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
