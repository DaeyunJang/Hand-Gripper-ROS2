#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState

from retarget_py.three_dof_manipulator import ThreeDoFManipulator

def quat_to_rotm(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion(x,y,z,w) → 3x3 rotation matrix"""
    # normalized 가정(ROS의 TF는 정규화되어 옴)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=float)

def tf_to_matrix(tfmsg: TransformStamped) -> np.ndarray:
    t = tfmsg.transform.translation
    q = tfmsg.transform.rotation
    Rm = quat_to_rotm(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rm
    T[:3,  3] = [t.x, t.y, t.z]
    return T

class ThreeDoFIkNode(Node):
    """
    TF: (base_frame <- eef_frame) 조회 → ^base T_tip 4x4 변환행렬 → 3DoF IK → JointState 퍼블리시
    """
    def __init__(self) -> None:
        super().__init__("three_dof_ik_node")

        # ---- Parameters ----
        self.declare_parameter("base_frame", "gripper_finger1_mcp")
        self.declare_parameter("eef_frame",  "gripper_finger1_tip")
        self.declare_parameter("rate_hz", 100.0)

        self.declare_parameter("link1", 0.1)
        self.declare_parameter("link2", 0.1)
        self.declare_parameter("base_z_offset", 0.0)
        self.declare_parameter("tool_offset_local", [0.0, 0.0, 0.0])
        self.declare_parameter("elbow", "up")     # "up" / "down"
        self.declare_parameter("angle_unit", "rad")  # 내부 계산/리턴 단위 (JointState는 rad가 일반적)

        self.declare_parameter("joint_names", ["joint1", "joint2", "joint3"])
        self.declare_parameter("output_topic", "ik_joint_states")

        self.base_frame: str = self.get_parameter("base_frame").get_parameter_value().string_value
        self.eef_frame:  str = self.get_parameter("eef_frame").get_parameter_value().string_value
        self.rate_hz: float = float(self.get_parameter("rate_hz").value)

        link1 = float(self.get_parameter("link1").value)
        link2 = float(self.get_parameter("link2").value)
        base_z_offset = float(self.get_parameter("base_z_offset").value)

        tool_offset_local_param = self.get_parameter("tool_offset_local").value
        tool_offset_local = np.array(tool_offset_local_param, dtype=float).reshape(3)

        elbow = str(self.get_parameter("elbow").value).lower()
        angle_unit = str(self.get_parameter("angle_unit").value).lower()

        self.joint_names: List[str] = list(self.get_parameter("joint_names").value)
        self.pub_topic: str = self.get_parameter("output_topic").get_parameter_value().string_value

        # ---- IK Solver ----
        self.ik = ThreeDoFManipulator(
            link1=link1,
            link2=link2,
            base_z_offset=base_z_offset,
            tool_offset_local=tool_offset_local,
            angle_unit="rad" if angle_unit not in ("rad", "deg") else angle_unit,  # 안전
            elbow="up" if elbow != "down" else "down",
        )

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Publisher ----
        self.pub_js = self.create_publisher(JointState, self.pub_topic, 10)

        # ---- Timer ----
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"[IK] base='{self.base_frame}', tip='{self.eef_frame}', rate={self.rate_hz:.1f}Hz, "
            f"links=({link1},{link2}), tool={tool_offset_local.tolist()}, elbow={self.ik.elbow}, unit={self.ik.angle_unit}"
        )

        # 간단한 경고 스로틀용
        self._last_warn_sec = 0.0
        self._warn_period_sec = 1.0  # 1초에 한 번만 경고

    def _warn_throttled(self, msg: str) -> None:
        from time import monotonic
        now = monotonic()
        if now - self._last_warn_sec >= self._warn_period_sec:
            self.get_logger().warning(msg)
            self._last_warn_sec = now

    def _on_timer(self) -> None:
        # ^base T_tip 얻기 위해 target=base, source=tip 로 조회
        try:
            tfmsg = self.tf_buffer.lookup_transform(
                target_frame=self.base_frame,
                source_frame=self.eef_frame,
                time=Time(),
                timeout=Duration(seconds=0.0)
            )
        except TransformException as e:
            # self._warn_throttled(f"TF lookup failed: {self.base_frame} -> {self.eef_frame} ({e})")
            return

        T = tf_to_matrix(tfmsg)

        # IK (필요시 도달불가 케이스는 내부 clamp로 근사)
        theta1, theta2, theta3 = self.ik.ik(T)
        # JointState는 rad가 일반적. self.ik.angle_unit이 'deg'라면 rad로 변환.
        if self.ik.angle_unit == "deg":
            theta1, theta2, theta3 = np.radians([theta1, theta2, theta3])

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = [float(theta1), float(theta2), float(theta3)]
        # self.get_logger().info(f'joint target: {js.position}')
        # self.get_logger().info(f'joint target: {np.asarray(js.position)*180/np.pi}')
        self.pub_js.publish(js)

def main() -> None:
    rclpy.init()
    node = ThreeDoFIkNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
