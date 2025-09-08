#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import re

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

# --- 작은 유틸 -------------------------------------------------------------

def make_homogeneous(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rm
    M[:3, 3] = t
    return M

def make_inv_homogeneous(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti

def mat_to_tf(T: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = float(T[0, 3])
    tf.transform.translation.y = float(T[1, 3])
    tf.transform.translation.z = float(T[2, 3])
    q = R.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
    tf.transform.rotation.x = float(q[0])
    tf.transform.rotation.y = float(q[1])
    tf.transform.rotation.z = float(q[2])
    tf.transform.rotation.w = float(q[3])
    return tf

def rot_y_180_mat() -> np.ndarray:
    """source_root -> gripper_base_frame (Ry 180°)"""
    Ry = R.from_euler('y', 180.0, degrees=True).as_matrix()
    return make_homogeneous(Ry, np.zeros(3))

def trans_xyz_mat(x: float, y: float, z: float = 0.0) -> np.ndarray:
    """parent -> child: pure translation"""
    return make_homogeneous(np.eye(3), np.array([x, y, z], dtype=float))

def add_suffix(frame_name: str, suffix: str = "gripper_scale") -> str:
    tail = f"_{suffix}"
    return frame_name if frame_name.endswith(tail) else (frame_name + tail)


# --- 메인 노드 -------------------------------------------------------------

class RetargetNode(Node):
    """
    (A) source_root 기준으로, 이름이 hand_* 이고 suffix가 target_suffix 인 모든 프레임을 찾아
        병진(x,y,z)에 스케일(scale 또는 scale_xyz)을 곱한 좌표계를
        '<원래이름>_{scale_suffix}'로 퍼블리시(parent=source_root).
    (B) 정적 프레임:
        - (정적) source_root --Ry180--> gripper_base_frame
        - (정적) gripper_base_frame --(base_offset_x,y)--> gripper_finger{i}_base_frame
    (C) 각 finger i에 대해:
        - gripper_base_frame 기준 hand_{label}_{finger}_mcp_{target_suffix}_{scale_suffix} 좌표까지의
          변환에서 병진 성분만 추출하여,
          parent=gripper_base_frame, child=gripper_finger{i}_mcp_frame 으로 퍼블리시(회전은 gripper와 동일).
    """

    def __init__(self) -> None:
        super().__init__("retarget_node")

        # --- Parameters ---
        self.source_root: str = self.declare_parameter(
            "source_root", "camera_color_optical_frame"
        ).get_parameter_value().string_value

        self.gripper_base_frame: str = self.declare_parameter(
            "gripper_base_frame", "gripper_base_frame"
        ).get_parameter_value().string_value

        self.target_suffix: str = self.declare_parameter(
            "target_suffix", "canon_norm"
        ).get_parameter_value().string_value

        self.scale_suffix: str = self.declare_parameter(
            "scale_suffix", "gripper_scale"
        ).get_parameter_value().string_value

        self.rate_hz: float = float(self.declare_parameter("rate_hz", 100.0).value)

        self.config_path: str = self.declare_parameter(
            "gripper_config_path", ""
        ).get_parameter_value().string_value

        # 스케일 파라미터(선언)
        self.declare_parameter("scale", 0.0)
        self.declare_parameter("scale_xyz", [])

        # --- TF IO ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)

        # --- Load Config ---
        cfg = self._load_gripper_config(self.config_path)

        # 스케일: param > yaml(scale_xyz or scale) > default
        # 요청: 이번 단계에서는 선호에 따라 scalar/xyz 둘 다 지원
        scale_param = float(self.get_parameter("scale").value or 0.0)
        scale_xyz_param = self.get_parameter("scale_xyz").value or []
        self.scale_xyz: Optional[np.ndarray] = None
        self.scale_scalar: float = 1.0
        if scale_param > 0.0:
            self.scale_scalar = scale_param
            src_scale = "param:scale"
        elif scale_xyz_param:
            v = np.array(scale_xyz_param, dtype=float).reshape(-1)
            if v.size == 3:
                self.scale_xyz = v
                self.scale_scalar = 1.0
                src_scale = "param:scale_xyz"
            else:
                self.get_logger().warning("param scale_xyz must have 3 elements. Fallback to YAML/default.")
                self._pick_yaml_scale(cfg)
                src_scale = "yaml"
        else:
            self._pick_yaml_scale(cfg)
            src_scale = "yaml"

        # target_suffix는 param 우선, 없으면 yaml
        if self.get_parameter("target_suffix").get_parameter_value().string_value:
            self.target_suffix = self.get_parameter("target_suffix").value
        else:
            self.target_suffix = str(cfg.get("target_suffix") or self.target_suffix)

        # fingers 설정 로드
        self.fingers_cfg = cfg.get("fingers") or []  # [{name, base_offset_x, base_offset_y, target_finger:{label,finger}}]

        # --- Static frames: (B) source->gripper, gripper->finger_base ---
        self._finger_base_names: List[str] = []
        self._publish_static_frames()

        # --- Timer ---
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        scale_repr = list(self.scale_xyz) if self.scale_xyz is not None else self.scale_scalar
        self.get_logger().info(
            f"[retarget] source='{self.source_root}', gripper='{self.gripper_base_frame}', "
            f"target_suffix='{self.target_suffix}', scale={scale_repr} ({src_scale}), "
            f"scale_suffix='{self.scale_suffix}', fingers={len(self.fingers_cfg)}, rate={self.rate_hz:.1f}Hz"
        )

        self._last_warn: Dict[str, float] = {}
        self._warn_interval: float = 2.0

    # --- config 로더 --------------------------------------------------------

    def _pick_yaml_scale(self, cfg: Dict[str, Any]) -> None:
        sxyz = cfg.get("scale_xyz")
        if sxyz is not None and isinstance(sxyz, (list, tuple)) and len(sxyz) == 3:
            self.scale_xyz = np.array(sxyz, dtype=float)
            self.scale_scalar = 1.0
        else:
            self.scale_scalar = float(cfg.get("scale", 1.0))
            self.scale_xyz = None

    def _load_gripper_config(self, user_path: str) -> Dict[str, Any]:
        """
        YAML 예:
        gripper:
          scale: 0.24
          # scale_xyz: [0.12,0.12,0.12]
          target_suffix: canon_norm
          fingers:
            - name: finger1
              base_offset_x: -0.12
              base_offset_y: -0.010
              target_finger: { label: left, finger: index }
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
            self.get_logger().warning("No gripper_config.yaml found. Using defaults.")
            return {"scale": 1.0, "target_suffix": "canon_norm", "fingers": []}

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            g = data.get("gripper") or {}

            # sanitize fingers
            fingers: List[Dict[str, Any]] = []
            for i, fdef in enumerate(g.get("fingers") or [], start=1):
                name = str(fdef.get("name") or f"finger{i}")
                x = float(fdef.get("base_offset_x") or 0.0)
                y = float(fdef.get("base_offset_y") or 0.0)
                tfing = fdef.get("target_finger") or {}
                label = str(tfing.get("label") or "")
                finger = str(tfing.get("finger") or "")
                if not label or not finger:
                    self.get_logger().warning(f"[config] {name}: target_finger.label/finger required. Skipping.")
                    continue
                fingers.append({
                    "name": name,
                    "base_offset_x": x, "base_offset_y": y,
                    "target_finger": {"label": label, "finger": finger},
                })

            out = {
                "scale": float(g.get("scale") or 1.0),
                "scale_xyz": g.get("scale_xyz"),
                "target_suffix": str(g.get("target_suffix") or "canon_norm"),
                "fingers": fingers,
            }
            self.get_logger().info(f"Loaded gripper config: {cfg_path} (fingers={len(fingers)})")
            return out
        except Exception as ex:
            self.get_logger().warning(f"Failed to load gripper config '{cfg_path}': {ex}. Using defaults.")
            return {"scale": 1.0, "target_suffix": "canon_norm", "fingers": []}

    # --- 프레임 이름 조합기 --------------------------------------------------

    def _hand_frame_name(self, label: str, finger: str, joint: str) -> str:
        """hand_{label}_{finger}_{joint}_{suffix}"""
        return f"hand_{label}_{finger}_{joint}_{self.target_suffix}"

    def _scaled_name(self, frame: str) -> str:
        """'<원래이름>_{scale_suffix}'"""
        return add_suffix(frame, self.scale_suffix)

    # (이전 이슈를 피하기 위해) suffix 기반 가능한 모든 hand 프레임을 생성
    def _list_all_hand_frames_by_suffix(self) -> List[str]:
        labels  = ["left", "right"]
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        joints  = ["mcp", "pip", "dip", "tip"]
        frames: List[str] = []
        for lb in labels:
            for fg in fingers:
                for jt in joints:
                    frames.append(self._hand_frame_name(lb, fg, jt))
        return frames

    # --- 정적 프레임 퍼블리시 (B) -------------------------------------------

    def _publish_static_frames(self) -> None:
        """(정적) source_root -> gripper_base_frame (Ry180), gripper -> finger_base(offset)"""
        static_tfs: List[TransformStamped] = []
        stamp = Time().to_msg()

        # source_root -> gripper_base_frame (Ry180)
        H = rot_y_180_mat()
        static_tfs.append(mat_to_tf(H, parent=self.source_root, child=self.gripper_base_frame, stamp=stamp))

        # gripper_base_frame -> gripper_finger{i}_base_frame (offset)
        self._finger_base_names.clear()
        for fdef in self.fingers_cfg:
            fname = str(fdef["name"])
            fx, fy = float(fdef["base_offset_x"]), float(fdef["base_offset_y"])
            finger_base = f"gripper_{fname}_base_frame"
            self._finger_base_names.append(finger_base)
            Hfb = trans_xyz_mat(fx, fy, 0.0)
            static_tfs.append(mat_to_tf(Hfb, parent=self.gripper_base_frame, child=finger_base, stamp=stamp))

        if static_tfs:
            self.static_br.sendTransform(static_tfs)

    # --- 헬퍼 ---------------------------------------------------------------

    def _lookup_mat_latest(self, target: str, source: str) -> Optional[np.ndarray]:
        """(target -> source) 최신 TF (timeout=0)"""
        try:
            T: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=target, source_frame=source, time=Time(), timeout=Duration(seconds=0.0)
            )
            t = T.transform.translation
            q = T.transform.rotation
            Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            return make_homogeneous(Rm, np.array([t.x, t.y, t.z], dtype=float))
        except TransformException as ex:
            # self._warn_throttled(f"lookup failed: {target} <- {source} ({ex})")
            return None
        except Exception as ex:
            # self._warn_throttled(f"lookup failed: {target} <- {source} ({ex})")
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

    # --- 주기 콜백 ----------------------------------------------------------

    def _on_timer(self) -> None:
        """
        (A) 스케일 퍼블리시: source_root -> hand_*_{target_suffix} 조회 → (x,y,z) 스케일 → *_<scale_suffix> 으로 퍼블리시
        (C) 각 finger의 MCP 위치에, 회전은 gripper와 동일하고 병진만 반영한 gripper_finger{i}_mcp_frame 퍼블리시
        """
        stamp = self.get_clock().now().to_msg()
        source = self.source_root
        gripper = self.gripper_base_frame

        # source <-> gripper (행렬)
        source_to_gripper = rot_y_180_mat()
        gripper_to_source = make_inv_homogeneous(source_to_gripper)

        # ---- (A) suffix 기반 hand 프레임들 스케일 퍼블리시 ----
        source_to_scaled: Dict[str, np.ndarray] = {}
        for hf in self._list_all_hand_frames_by_suffix():
            H = self._lookup_mat_latest(source, hf)  # source -> hand_frame
            if H is None:
                continue
            Hs = self._scale_translation(H)         # 병진만 스케일
            source_to_scaled[hf] = Hs
            self.br.sendTransform(mat_to_tf(Hs, parent=source, child=self._scaled_name(hf), stamp=stamp))
        
        # self.get_logger().info(f'source_to_scaled: {source_to_scaled}')
        
        # ---- (C) finger별 gripper_finger{i}_mcp_frame 퍼블리시 ----
        # 규칙: YAML의 target_finger = {label, finger}에서 MCP 프레임 이름 생성 후,
        #       그 스케일된 프레임(source->..._gripper_scale)을 찾아 gripper 기준 병진만 추출
        for fdef in self.fingers_cfg:
            name = str(fdef["name"])
            label = fdef["target_finger"]["label"]
            finger = fdef["target_finger"]["finger"]

            # 1) 원본 MCP 이름과 스케일된 프레임 이름
            mcp_name = self._hand_frame_name(label, finger, "mcp")  # hand_{label}_{finger}_mcp_{target_suffix}
            mcp_scaled_name = self._scaled_name(mcp_name)            # ..._{target_suffix}_{scale_suffix}

            # 2) source_root -> mcp_scaled 행렬 확보
            H_src_to_mcp_scaled = source_to_scaled.get(mcp_name)
            if H_src_to_mcp_scaled is None:
                # 스케일 퍼블리시가 아직 없거나 lookup 실패
                H_src_to_mcp_scaled = self._lookup_mat_latest(source, mcp_scaled_name)
                if H_src_to_mcp_scaled is None:
                    self._warn_throttled(f"skip MCP for {name}: no TF '{mcp_name}' or '{mcp_scaled_name}'")
                    continue

            # 3) gripper 기준으로 변환: gripper -> mcp_scaled
            H_grip_to_mcp_scaled = gripper_to_source @ H_src_to_mcp_scaled

            # 4) 병진만 추출(회전은 gripper와 동일) → pure translation
            tx, ty, tz = H_grip_to_mcp_scaled[0, 3], H_grip_to_mcp_scaled[1, 3], H_grip_to_mcp_scaled[2, 3]
            H_grip_to_mcp_pure = trans_xyz_mat(float(tx), float(ty), float(tz))

            # 5) 퍼블리시: parent=gripper_base_frame, child=gripper_finger{i}_mcp_frame
            child_name = f"gripper_{name}_mcp_frame"
            self.br.sendTransform(mat_to_tf(H_grip_to_mcp_pure, parent=gripper, child=child_name, stamp=stamp))
            
            
            # ---------- NEW: MCP(parent) -> TIP(child) 퍼블리시 ----------
            # tip 프레임 이름 구성: hand_{label}_{finger}_tip_{target_suffix} 및 그 스케일 버전
            tip_name = self._hand_frame_name(label, finger, "tip")
            tip_scaled_name = self._scaled_name(tip_name)

            # source_root -> tip_scaled 행렬 확보(가능하면 캐시 사용)
            H_src_to_tip_scaled = source_to_scaled.get(tip_name)
            if H_src_to_tip_scaled is None:
                H_src_to_tip_scaled = self._lookup_mat_latest(source, tip_scaled_name)
                if H_src_to_tip_scaled is None:
                    self._warn_throttled(f"skip TIP for {name}: no TF '{tip_name}' or '{tip_scaled_name}'")
                    continue

            # gripper -> tip_scaled
            H_grip_to_tip_scaled = gripper_to_source @ H_src_to_tip_scaled

            # mcp(parent) -> tip(child) = (gripper->mcp_pure)^(-1) @ (gripper->tip_scaled)
            H_mcp_to_tip = make_inv_homogeneous(H_grip_to_mcp_pure) @ H_grip_to_tip_scaled

            # 퍼블리시: parent=gripper_{name}_mcp_frame, child=gripper_{name}_tip_frame
            tip_child = f"gripper_{name}_tip_frame"
            self.br.sendTransform(mat_to_tf(H_mcp_to_tip, parent=child_name, child=tip_child, stamp=stamp))
            # ---------- /NEW ----------


# --- 엔트리 -----------------------------------------------------------------

def main() -> None:
    rclpy.init()
    node = RetargetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
