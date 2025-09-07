#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R


# ---------- 유틸 ----------

def make_homogeneous(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rm
    M[:3, 3] = t
    return M

def mat_to_tf(T: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = float(T[0, 3])
    tf.transform.translation.y = float(T[1, 3])
    tf.transform.translation.z = float(T[2, 3])
    q = R.from_matrix(T[:3, :3]).as_quat()
    tf.transform.rotation.x = float(q[0])
    tf.transform.rotation.y = float(q[1])
    tf.transform.rotation.z = float(q[2])
    tf.transform.rotation.w = float(q[3])
    return tf

def add_suffix(frame_name: str, suffix: str) -> str:
    tail = f"_{suffix}"
    return frame_name if frame_name.endswith(tail) else (frame_name + tail)


# ---------- 메인 노드 ----------

class NormalizeRetargetNode(Node):
    """
    목적:
      - 라벨별(예: left/right)로 index_mcp ↔ thumb_mcp 사이 거리 d_hand를 매 주기 측정
      - 모든 hand_*_{target_suffix} 프레임의 병진에 factor = scale_after_normalize / d_hand 를 곱해 스케일
      - 회전은 원본 유지, parent는 source_root 유지
      - child 이름은 '<원래이름>_{scale_suffix}'
    """

    def __init__(self) -> None:
        super().__init__("retarget_normalize_node")

        # ---- Params ----
        self.source_root: str = self.declare_parameter(
            "source_root", "camera_color_optical_frame"
        ).get_parameter_value().string_value

        # 입력 프레임 접미사(정규화 전): 보통 "canon_norm"
        self.target_suffix: str = self.declare_parameter(
            "target_suffix", "canon_norm"
        ).get_parameter_value().string_value

        # 출력 프레임 접미사(정규화+스케일 후)
        self.scale_suffix: str = self.declare_parameter(
            "scale_suffix", "norm_scaled"
        ).get_parameter_value().string_value

        # 정규화 후 적용할 절대 스케일 [m per normalized_unit]
        # 예: 0.24면 index_mcp–thumb_mcp 간격이 항상 0.24m가 되도록 전체 스케일링
        self.scale_after_normalize: float = float(self.declare_parameter(
            "scale_after_normalize", 0.24
        ).value)

        # 퍼블리시 주기
        self.rate_hz: float = float(self.declare_parameter("rate_hz", 100.0).value)

        # 어떤 라벨들을 처리할지(기본: left/right)
        self.labels: List[str] = list(self.declare_parameter(
            "labels", ["left", "right"]
        ).value or ["left", "right"])

        # fingers/joints 목록(필요 시 확장 가능)
        self.fingers = ["thumb", "index", "middle", "ring", "pinky"]
        self.joints  = ["mcp", "pip", "dip", "tip"]

        # TF IO
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)

        # Timer
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        self._last_warn: Dict[str, float] = {}
        self._warn_interval: float = 1.5

        self.get_logger().info(
            f"[normalize] source='{self.source_root}', target_suffix='{self.target_suffix}', "
            f"scale_after_normalize={self.scale_after_normalize}m, out_suffix='{self.scale_suffix}', "
            f"labels={self.labels}, rate={self.rate_hz:.1f}Hz"
        )

    # ---------- 이름 유틸 ----------

    def _hand_name(self, label: str, finger: str, joint: str) -> str:
        return f"hand_{label}_{finger}_{joint}_{self.target_suffix}"

    # ---------- 헬퍼 ----------

    def _lookup_mat_latest(self, target: str, source: str) -> Optional[np.ndarray]:
        """(source -> target) 최신 TF 조회 (timeout=0)"""
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

    def _warn_throttled(self, text: str) -> None:
        from time import monotonic
        now = monotonic()
        last = self._last_warn.get(text, 0.0)
        if now - last >= self._warn_interval:
            self.get_logger().warning(text)
            self._last_warn[text] = now

    # ---------- 메인 루프 ----------

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()
        src = self.source_root

        # 라벨별로 정규화 스케일 계산 후 적용
        for label in self.labels:
            # 1) 기준 거리: |index_mcp - thumb_mcp|
            idx_mcp = self._hand_name(label, "index", "mcp")
            thb_mcp = self._hand_name(label, "thumb", "mcp")

            H_idx = self._lookup_mat_latest(src, idx_mcp)
            H_thb = self._lookup_mat_latest(src, thb_mcp)
            if H_idx is None or H_thb is None:
                self._warn_throttled(f"[{label}] missing MCP frames: '{idx_mcp}' or '{thb_mcp}'")
                continue

            p_idx = H_idx[:3, 3]
            p_thb = H_thb[:3, 3]
            d = float(np.linalg.norm(p_idx - p_thb))
            if d <= 1e-9:
                self._warn_throttled(f"[{label}] MCP distance too small: {d:.3e}")
                continue

            factor = float(self.scale_after_normalize) / d  # ← 핵심: 정규화 후 원하는 미터 스케일

            # 2) 이 label의 모든 관절 프레임에 factor 적용
            for finger in self.fingers:
                for joint in self.joints:
                    name = self._hand_name(label, finger, joint)
                    H = self._lookup_mat_latest(src, name)
                    if H is None:
                        continue

                    # 회전은 유지, 병진만 factor 배
                    M = H.copy()
                    M[:3, 3] = M[:3, 3] * factor

                    out_name = add_suffix(name, self.scale_suffix)
                    self.br.sendTransform(mat_to_tf(M, parent=src, child=out_name, stamp=stamp))


def main() -> None:
    rclpy.init()
    node = NormalizeRetargetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
