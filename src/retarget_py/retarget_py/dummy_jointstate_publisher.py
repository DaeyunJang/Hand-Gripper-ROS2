#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time

class DummyJointStatePublisher(Node):
    def __init__(self):
        super().__init__('dummy_joint_state_publisher')
        self.pub = self.create_publisher(JointState, '/dummy/joint_states', 1)
        self.timer = self.create_timer(0.05, self.publish_state)  # 20Hz
        self.joint_names = [
            "finger1_j1", "finger1_j2", "finger1_j3",
            "finger2_j1", "finger2_j2", "finger2_j3",
            "finger3_j1", "finger3_j2", "finger3_j3",
            "finger4_j1", "finger4_j2", "finger4_j3",
        ]
        self.position = [0.0] * len(self.joint_names)

    def publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.position
        if "j2" in self.joint_names:
            msg.position = 1.5708   # initial pose
        msg.velocity = [0.1] * len(self.joint_names)
        msg.effort = [0.1] * len(self.joint_names)
        self.pub.publish(msg)
        # 간단히 움직임 시뮬레이션도 가능
        self.position = [p + 0.1 for p in self.position]  

def main():
    rclpy.init()
    node = DummyJointStatePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
