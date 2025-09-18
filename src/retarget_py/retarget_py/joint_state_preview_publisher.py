#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePreviewPublihser(Node):
    def __init__(self):
        super().__init__('joint_state_preview_node')
        self.declare_parameter('cmd_topic', '/hand_gripper/joint_states_cmd')
        self.declare_parameter('out_topic', '/hand_gripper/joint_states_preview')

        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('out_topic').get_parameter_value().string_value

        self.sub = self.create_subscription(JointState, cmd_topic, self.cb, 3)
        self.pub = self.create_publisher(JointState, out_topic, 3)

        self.get_logger().info(f'Bridging: {cmd_topic}  ->  {out_topic}')

    def cb(self, msg: JointState):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(msg.name)
        out.position = list(msg.position)
        self.pub.publish(out)
        # self.get_logger().info(f'out: {out}')

def main():
    rclpy.init()
    node = JointStatePreviewPublihser()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
