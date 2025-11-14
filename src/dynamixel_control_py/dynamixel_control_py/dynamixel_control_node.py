#!/usr/bin/env python3
"""Node subscribing to /hand_gripper/joint_states_cmd and forwarding to Dynamixel motors.

This node uses the existing `DynamixelController` class from `hand_control.py` when available.
It maps incoming JointState messages to the controller's motor id ordering and issues position
commands using `set_goal_positions` + `apply_motion`.
"""
from __future__ import annotations
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading
from typing import List, Optional
import math
import re

try:
    # Import both the controller and its loaded config
    from dynamixel_control_py.dynamixel_controller import DynamixelController, config as controller_config
    CONTROLLER_AVAILABLE = True
except Exception as e:
    print(f"Failed to import DynamixelController: {e}")
    DynamixelController = None
    controller_config = {} # provide an empty config on failure
    CONTROLLER_AVAILABLE = False


class DynamixelControlNode(Node):
    def __init__(self):
        super().__init__('hand_control_node')
        # runtime options
        self.declare_parameter('verbose', True)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('state_poll_hz', 1.0)
        # conversion ranges for incoming commands
        self.declare_parameter('joint0_min_m', 0.0)
        self.declare_parameter('joint0_max_m', 0.05)
        self.declare_parameter('angular_min_rad', -math.pi)
        self.declare_parameter('angular_max_rad', math.pi)

        self.verbose = bool(self.get_parameter('verbose').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.joint0_min_m = float(self.get_parameter('joint0_min_m').value)
        self.joint0_max_m = float(self.get_parameter('joint0_max_m').value)
        self.angular_min_rad = float(self.get_parameter('angular_min_rad').value)
        self.angular_max_rad = float(self.get_parameter('angular_max_rad').value)

        self.lock = threading.RLock()

        self.get_logger().info(f"DynamixelControlNode starting. controller available: {CONTROLLER_AVAILABLE}")

        # Initialize controller if available and not dry-run
        self.controller: Optional[object] = None
        if CONTROLLER_AVAILABLE and not self.dry_run:
            try:
                self.controller = DynamixelController(verbose=self.verbose)
                self.get_logger().info('DynamixelController initialized')
            except Exception as e:
                self.get_logger().error(f'Failed to initialize DynamixelController: {e}')
                self.controller = None

        # Subscribe to joint command topic
        # self.sub = self.create_subscription(JointState, '/hand_gripper/joint_states_cmd', self._on_joint_cmd, 1)
        self.sub = self.create_subscription(JointState, '/hand_gripper/joint_states_direct', self._on_joint_cmd, 1)

        # Publisher for current joint states
        self.state_pub = self.create_publisher(JointState, '/hand_gripper/joint_states', 10)

        # Timer: poll motor states periodically and print/log them
        poll_hz = float(self.get_parameter('state_poll_hz').value or 1.0)
        period = 1.0 / max(0.01, poll_hz)
        self.state_timer = self.create_timer(period, self._publish_states)

    def _map_positions_for_controller(self, msg: JointState) -> Optional[List[int]]:
        """
        [개별 매핑 버전] JointState의 position 배열을 각 모터 ID에 맞게 개별적으로 변환합니다.
        """
        if not (self.controller and hasattr(self.controller, 'motor_ids')):
            self.get_logger().warning("Controller or motor_ids not found, skipping mapping.")
            return None

        positions = msg.position
        motor_ids = self.controller.motor_ids

        if len(positions) != len(motor_ids):
            self.get_logger().warning(
                f"Position count ({len(positions)}) does not match motor count ({len(motor_ids)}). Skipping."
            )
            return None

        goal_positions = []
        for i, motor_id in enumerate(motor_ids):
            value = positions[i]

            motor_details = controller_config.get('motor_details', {}).get(str(motor_id), {})
            min_encoder_value = motor_details.get('min_position', 0)
            max_encoder_value = motor_details.get('max_position', 4095)
            encoder_range = max_encoder_value - min_encoder_value
            
            goal_position = 0

            # --- 모터 ID별 개별 매핑 ---
            if motor_id == 10:  # joint0_max_m = 0.05 m
                goal_position = (1- value / 0.05) * 3300
            elif motor_id == 20:  # joint0_max_m = 0.05 m
                goal_position = (1- value / 0.05) * 3300
            elif motor_id == 30:  # joint0_max_m = 0.05 m
                goal_position = (1- value / 0.05) * 3300
            elif motor_id == 40:  # joint0_max_m = 0.05 m
                goal_position = (1- value / 0.05) * 3300
                
            elif motor_id == 11:
                # goal_position = (value / math.pi) * 2048 + 2048
                goal_position = 3072
            elif motor_id == 12:
                goal_position = -(value / math.pi) * 2048 + 2500
            elif motor_id == 13:
                goal_position = -(value / math.pi) * 2048 + 2048
            elif motor_id == 21:
                # goal_position = (value / math.pi) * 2048 + 2048
                goal_position = 1024
            elif motor_id == 22:
                goal_position = -(value / math.pi) * 2048 + 2048
            elif motor_id == 23:
                goal_position = -(value / math.pi) * 2048 + 1800
            elif motor_id == 31:
                # goal_position = (value / math.pi) * 2048 + 2048
                goal_position = 1024
            elif motor_id == 32:
                goal_position = -(value / math.pi) * 2048 + 2500
            elif motor_id == 33:
                goal_position = -(value / math.pi) * 2048 + 2048
            elif motor_id == 41:
                # goal_position = (value / math.pi) * 2048 + 2048
                goal_position = 3072
            elif motor_id == 42:
                goal_position = -(value / math.pi) * 2048 + 2048
            elif motor_id == 43:
                goal_position = -(value / math.pi) * 2048 + 1800
            
            # # --- 모터 ID별 개별 매핑 ---
            # if motor_id == 10:  # joint0_max_m = 0.05 m
            #     goal_position = (value / 0.05) * 2048
            # elif motor_id == 20:  # joint0_max_m = 0.05 m
            #     goal_position = (value / 0.05) * 2048
            # elif motor_id == 30:  # joint0_max_m = 0.05 m
            #     goal_position = (1- value / 0.05) * 2048
            # elif motor_id == 40:  # joint0_max_m = 0.05 m
            #     goal_position = (1- value / 0.05) * 2048
                
            # elif motor_id == 11:
            #     # goal_position = (value / math.pi) * 2048 + 2048
            #     goal_position = 3072
            # elif motor_id == 12:
            #     goal_position = (value / math.pi) * 2048 + 1024
            # elif motor_id == 13:
            #     goal_position = (value / math.pi) * 2048 + 2048
            # elif motor_id == 21:
            #     # goal_position = (value / math.pi) * 2048 + 2048
            #     goal_position = 1024
            # elif motor_id == 22:
            #     goal_position = (value / math.pi) * 2048 + 1024
            # elif motor_id == 23:
            #     goal_position = (value / math.pi) * 2048 + 2048
            # elif motor_id == 31:
            #     # goal_position = (value / math.pi) * 2048 + 2048
            #     goal_position = 1024
            # elif motor_id == 32:
            #     goal_position = ((math.pi - value) / math.pi) * 2048 + 1024
            # elif motor_id == 33:
            #     goal_position = -(value / math.pi) * 2048 + 2048
            # elif motor_id == 41:
            #     # goal_position = (value / math.pi) * 2048 + 2048
            #     goal_position = 3072
            # elif motor_id == 42:
            #     goal_position = ((math.pi - value) / math.pi) * 2048 + 1024
            # elif motor_id == 43:
            #     goal_position = -(value / math.pi) * 2048 + 2048
                

            else: # 알려지지 않은 ID는 중간 값으로 설정
                goal_position = min_encoder_value + encoder_range / 2

            # 최종 목표 엔코더 값이 모터의 물리적 한계를 벗어나지 않도록 조정합니다.
            goal_position = max(min_encoder_value, min(goal_position, max_encoder_value))
            goal_positions.append(int(round(goal_position)))

        return goal_positions

    def _on_joint_cmd(self, msg: JointState):
        with self.lock:
            mapped = self._map_positions_for_controller(msg)
            if mapped is None:
                self.get_logger().warning('Could not map JointState to motor positions; ignoring')
                return

            if self.verbose:
                self.get_logger().info(f'Received command, msg            : {msg.position}')
                self.get_logger().info(f'Received command, mapped to ticks: {mapped}')

            if self.controller and not self.dry_run:
                self.controller.set_goal_positions(mapped, limit_clamping=True)
                self.controller.apply_motion()
                if self.verbose:
                    self.get_logger().info(f'Executed motor command.')
            else:
                if self.verbose:
                    self.get_logger().info(f'Dry-run mode or controller unavailable. Command not sent.')

    def _publish_states(self):
        """주기적으로 컨트롤러에서 현재 모터 상태를 읽어와 JointState 메시지로 게시합니다."""
        if not self.controller:
            return

        try:
            self.controller.get_states(verbose=False)
            state_data = getattr(self.controller, 'state_data', None)
            if state_data:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                
                names = []
                positions = []
                velocities = []
                currents = []

                motor_ids = sorted(state_data.keys())
                for mid in motor_ids:
                    st = state_data[mid]
                    pos_ticks = st.get('present_position')
                    vel_ticks = st.get('present_velocity')
                    cur_ticks = st.get('present_current')

                    if pos_ticks is not None:
                        # Generate a more descriptive joint name
                        finger_idx = mid // 10
                        joint_idx = mid % 10
                        names.append(f'finger{finger_idx}_joint{joint_idx}')
                        
                        # Convert ticks to SI units for publishing
                        # positions.append(self._ticks_to_si(pos_ticks, mid))
                        
                        # Note: Velocity and Current conversion need specific factors from datasheet.
                        # For now, publishing raw values, assuming downstream nodes know how to interpret.
                        velocities.append(float(vel_ticks if vel_ticks is not None else 0.0))
                        currents.append(float(cur_ticks if cur_ticks is not None else 0.0))

                msg.name = names
                msg.position = positions
                msg.velocity = velocities
                msg.effort = currents # 'effort' is the standard field for current/torque
                
                self.state_pub.publish(msg)
                if self.verbose:
                    self.get_logger().info(f'Published joint states for {len(names)} motors.')

            else:
                if self.verbose:
                    self.get_logger().info('Controller present but no state_data available')
        except Exception as e:
            self.get_logger().error(f'Error polling and publishing motor states: {e}', exc_info=True)

    def destroy_node(self):
        # ensure controller cleanup
        if self.controller:
            try:
                self.controller.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DynamixelControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()