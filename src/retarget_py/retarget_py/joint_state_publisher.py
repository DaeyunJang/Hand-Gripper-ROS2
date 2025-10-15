#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublishNode(Node):
    def __init__(self):
        super().__init__('joint_state_pulish_node')
        self.declare_parameter('cmd_topic', '/hand_gripper/joint_states_cmd')
        # self.declare_parameter('cmd_topic', '/hand_gripper/joint_states_direct')
        self.declare_parameter('out_topic', '/joint_command')
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('out_topic').get_parameter_value().string_value
        
        self.sub = self.create_subscription(JointState, cmd_topic, self.cb, 1)
        self.pub = self.create_publisher(JointState, out_topic, 1)

    def cb(self, msg: JointState):
        filtered = JointState()
        filtered.header = msg.header

        for i, name in enumerate(msg.name):
            if name.endswith('_linear'):
                continue
            
            ## ik based
            # if name.endswith('_j2'):
            #     filtered.name.append(name)
            #     filtered.position.append(msg.position[i]+1.5708)
            #     continue
            
            ## direct based
            if name.endswith(('finger1_j1', 'finger3_j1')):
                filtered.name.append(name)
                filtered.position.append(0.3491)    # 0.3491 = 20 deg
                continue
            if name.endswith(('finger2_j1', 'finger4_j1')):
                filtered.name.append(name)
                filtered.position.append(-0.3491)    # -0.3491 = -20 deg
                continue
            
            filtered.name.append(name)
            filtered.position.append(msg.position[i])
            filtered.velocity.append(0.1)

        # velocity/effort는 비워둬도 OK
        # filtered.velocity = []
        # filtered.effort = []

        self.pub.publish(filtered)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublishNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
