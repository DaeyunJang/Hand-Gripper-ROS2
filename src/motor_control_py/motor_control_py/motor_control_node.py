#!/usr/bin/env python3
"""
조인트 타깃을 구독하여 Dynamixel을 구동하는 노드.
- 구독: sensor_msgs/JointState (name[], position[])
- 매핑: joint_name → motor_id, scale/offset/sign/clamp
- 모터: DynamixelInterface 사용
"""
from __future__ import annotations
from typing import Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .dxl_interface import DynamixelInterface

class DXLActuatorNode(Node):
    def __init__(self):
        super().__init__("hand_dxl_actuator")

        # 구독 토픽
        in_topic = self.declare_parameter("in_topic", "/hand_gripper/joint_targets").get_parameter_value().string_value
        self.sub = self.create_subscription(JointState, in_topic, self._on_joint_targets, 10)

        # 매핑 파라미터 로드: joint_name -> {motor_id, scale, offset, sign, clamp_deg}
        self.mapping: Dict[str, Dict] = {}  # TODO: 파라미터에서 채우기

        # Dynamixel
        dxl_enable = bool(self.declare_parameter("dxl.enable", False).value)
        if dxl_enable:
            # TODO: port/baud/ids 등 파라미터 획득
            # self.dxl = DynamixelInterface(port, baud, ids, ...)
            # self.dxl.open()
            self.dxl = None
        else:
            self.dxl = None

    def _on_joint_targets(self, msg: JointState):
        """JointState → 모터 목표각(도) 계산 → DXL 쓰기."""
        # TODO:
        # 1) joint_name 순회하며 매핑 찾기
        # 2) joint_angle(rad/deg) → (필요 시 deg로 변환)
        # 3) motor_deg = offset + sign*scale*joint_angle
        # 4) clamp
        # 5) {motor_id: deg} dict 구성
        # 6) if self.dxl: self.dxl.write_positions_deg(targets)
        pass

    def destroy_node(self):
        # TODO: if self.dxl: self.dxl.close()
        return super().destroy_node()

def main():
    rclpy.init()
    node = DXLActuatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
