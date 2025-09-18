from __future__ import annotations
import threading
import time
from typing import Dict, List

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from motor_control_py.utils import rad_to_tick_linear

from sensor_msgs.msg import JointState
from motor_control_py.motor_driver import DynamixelBus, OP_MODE_CURRENT_POS, ADDR_PRESENT_POSITION, ADDR_CURRENT_LIMIT, ADDR_GOAL_CURRENT
from motor_control_py.utils import rad_to_tick_linear, tick_to_rad_linear


class DXLControlNode(Node):
    """
    /hand_gripper/joint_trajectory (JointTrajectory) 구독
     - full_joint_names: 저장용 전체 조인트 목록(예: 12개)
     - actuated_names/actuated_ids: 실제 구동 서브셋(지금은 finger1_j1~j3)
     - 각 조인트별 라디안/틱 범위로 선형 맵핑하여 SyncWrite
     - 메시지에 일부 이름만 와도 last_cmd 누적 저장, actuated 서브셋만 구동
    """

    def __init__(self):
        super().__init__("dxl_control_node")

        # ---- 통신 파라미터 ----
        self.declare_parameter("port_name", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 57600)
        self.declare_parameter("protocol", 2.0)

        # ---- 이름 목록 ----
        # self.declare_parameter("full_joint_names", [], ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY))
        # self.declare_parameter("actuated_names", [], ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY))
        # self.declare_parameter("actuated_ids", [], ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY))
        
        ## From config/params.yaml
        self.declare_parameter("full_joint_names", None)
        self.declare_parameter("actuated_names", None)
        self.declare_parameter("actuated_ids", None)
        

        # ---- 각 조인트별 라디안/틱 범위 ----
        self.declare_parameter("min_pos_rad", 0.0)
        self.declare_parameter("max_pos_rad", 3.14159)
        self.declare_parameter("tick_min", 0)
        self.declare_parameter("tick_max", 4095)
        
        # ✅ 전류 제한(원시 값) 파라미터 추가: 기본 40
        self.declare_parameter("current_limit_raw", 200)
        self.declare_parameter("goal_current_raw", 20)

        # ---- 파라미터 로드 ----
        port = self.get_parameter("port_name").get_parameter_value().string_value
        baud = self.get_parameter("baudrate").get_parameter_value().integer_value
        proto = self.get_parameter("protocol").get_parameter_value().double_value
        # 범위 값은 스칼라 혹은 배열 둘 다 허용
        rmin_val = self.get_parameter("min_pos_rad").value   # float or [float]
        rmax_val = self.get_parameter("max_pos_rad").value
        tmin_val = self.get_parameter("tick_min").value      # int or [int]
        tmax_val = self.get_parameter("tick_max").value
        current_limit = int(self.get_parameter("current_limit_raw").value)
        goal_current = int(self.get_parameter("goal_current_raw").value)
        
        full_names = list(self.get_parameter("full_joint_names").get_parameter_value().string_array_value)
        act_names  = list(self.get_parameter("actuated_names").get_parameter_value().string_array_value)
        act_ids    = list(self.get_parameter("actuated_ids").get_parameter_value().integer_array_value)
        
        self.get_logger().info(f"port: {port}")
        self.get_logger().info(f"baud: {baud}")
        self.get_logger().info(f"proto: {proto}")
        self.get_logger().info(f"full_names: {full_names}")
        self.get_logger().info(f"act_names: {act_names}")
        self.get_logger().info(f"act_ids: {act_ids}")
        
        if len(full_names) == 0:
            raise RuntimeError("full_joint_names 가 비어있습니다.")
        if len(act_names) == 0 or len(act_names) != len(act_ids):
            raise RuntimeError("actuated_names 와 actuated_ids 길이는 같고 0보다 커야 합니다.")

        # --- 값이 스칼라이면 브로드캐스트, 리스트면 길이 맞는지 확인해서 매핑
        def build_map(names: List[str], v, cast):
            # v가 리스트/튜플이면 그대로, 아니면 [v] 로
            arr = list(v) if isinstance(v, (list, tuple)) else [v]
            if len(arr) == 1:
                val = cast(arr[0])
                return {nm: val for nm in names}
            if len(arr) != len(names):
                raise RuntimeError(f"파라미터 길이 불일치: {len(names)}개 조인트에 대해 {len(arr)}개 값")
            return {nm: cast(x) for nm, x in zip(names, arr)}

        rmin_map = build_map(full_names, rmin_val, float)
        rmax_map = build_map(full_names, rmax_val, float)
        tmin_map = build_map(full_names, tmin_val, int)
        tmax_map = build_map(full_names, tmax_val, int)

        # --- 로그로 한번 확인
        self.get_logger().info(f"port={port}, baud={baud}, proto={proto}")
        self.get_logger().info(f"full_names({len(full_names)}): {full_names}")
        self.get_logger().info(f"actuated: {list(zip(act_names, act_ids))}")

        # --- 상태/맵 저장
        self.full_names = full_names
        self.act_names = act_names
        self.name_to_id: Dict[str, int] = dict(zip(act_names, act_ids))
        self.rmin_map, self.rmax_map = rmin_map, rmax_map
        self.tmin_map, self.tmax_map = tmin_map, tmax_map
        self.last_cmd: Dict[str, float] = {nm: 0.0 for nm in full_names}

        # --- 버스 초기화(서브셋만 활성화)
        self.bus = DynamixelBus(port, baudrate=baud, protocol=proto, log=self.get_logger().info)
        for nm in self.act_names:
            dxl_id = self.name_to_id[nm]
            self.bus.set_operating_mode(dxl_id, OP_MODE_CURRENT_POS)
            self.bus.torque_enable(dxl_id, True)
            
            # ✅ 전류 제한 쓰기 (토크 OFF 상태에서)
            try:
                self.bus.write2(dxl_id, ADDR_CURRENT_LIMIT, current_limit)
                self.get_logger().info(f"Set current limit: id={dxl_id} ← {current_limit}")
            except Exception as e:
                self.get_logger().warn(f"Set current limit failed id={dxl_id}: {e}")
                
            # ✅ Goal Current 설정
            try:
                self.bus.write_goal_current(dxl_id, goal_current)
                self.get_logger().info(f"Set goal current: id={dxl_id} ← {goal_current}")
            except Exception as e:
                self.get_logger().warn(f"Set goal current failed id={dxl_id}: {e}")



        # --- 구독 및 실행 스레드 세팅
        self.exec_lock = threading.Lock()
        self.exec_thread: threading.Thread | None = None
        self.stop_flag = False

        # ... 기존 파라미터 로드/버스 초기화 끝난 뒤
        # 상태 퍼블리시 주기 파라미터(기본 50 Hz)
        self.declare_parameter("state_rate_hz", 50.0)
        rate_hz = float(self.get_parameter("state_rate_hz").value)
        period = 1.0 / max(1e-3, rate_hz)
        # JointState 퍼블리셔 (지금은 actuated 3축만 송출)
        self.js_pub = self.create_publisher(JointState, "/hand_gripper/joint_states", 1)
        self.state_timer = self.create_timer(period, self._publish_joint_states)

        # ✅ JointState 기반 "목표" 커맨드 구독 (retarget_node 가 퍼블리시하는 토픽)
        #    retarget_node 에서 f"/{gripper_name}/joint_states_cmd" 로 보낸다고 했으니 동일하게 맞춤
        self.cmd_sub = self.create_subscription(JointState, "/hand_gripper/joint_states_cmd", self.on_joint_states_cmd, 1)

        
        self.sub = self.create_subscription(
            JointTrajectory,
            "/hand_gripper/joint_trajectory",
            self.on_trajectory,
            10
        )
        # __init__
        self.get_logger().info("motor_control_py ready (actuated: %s)" % ", ".join(self.act_names))

        
    def on_joint_states_cmd(self, msg: JointState):
        """
        retarget_node 가 퍼블리시하는 JointState(목표값)를 읽어서,
        full_joint_names에 누적 저장하고, actuated_names 서브셋만 모터로 전송.
        """
        if not msg.name or not msg.position:
            self.get_logger().warn("joint_states_cmd: empty message")
            return

        # 이름→인덱스
        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(msg.name)}

        # 1) 전체(last_cmd) 갱신: 명시된 조인트만 라디안 업데이트
        updated = []
        for jn in self.full_names:
            j = name_to_idx.get(jn, None)
            if j is not None and j < len(msg.position):
                self.last_cmd[jn] = float(msg.position[j])
                updated.append(jn)

        if not updated:
            # 이름 불일치 혹은 빈 메시지면 그냥 리턴
            self.get_logger().warn("joint_states_cmd: no matching joint names; nothing updated")
            return

        # 2) 서브셋만 라디안→틱 변환해서 전송
        id_to_goal_tick: Dict[int, int] = {}
        debug = []
        for jn in self.act_names:
            dxl_id = self.name_to_id[jn]
            cmd_rad = self.last_cmd[jn]
            rmin = self.rmin_map[jn]; rmax = self.rmax_map[jn]
            tmin = self.tmin_map[jn]; tmax = self.tmax_map[jn]

            goal_tick = rad_to_tick_linear(cmd_rad, rmin, rmax, tmin, tmax)
            id_to_goal_tick[dxl_id] = goal_tick
            debug.append(f"{jn}: rad={cmd_rad:.5f} → tick={goal_tick}")

        try:
            # 전송
            # self.bus.sync_write_goal_positions(id_to_goal_tick)
            self.get_logger().info("CMD → " + " | ".join(debug))
        except Exception as e:
            self.get_logger().error(f"joint_states_cmd: sync_write failed: {e}")

        
    def on_trajectory(self, msg: JointTrajectory):
        

        if not msg.points or not msg.joint_names:
            self.get_logger().warn("Received empty trajectory.")
            return
        
        # Timer
        # self.active = {"names": msg.joint_names, "points": msg.points, "t0": time.monotonic(), "i": 0}
        
        # Thread
        # with self.exec_lock:
        #     self.stop_flag = True
        #     if self.exec_thread and self.exec_thread.is_alive():
        #         self.exec_thread.join()
        #         # self.exec_thread.join(timeout=0.5)
        #     self.stop_flag = False
        #     self.exec_thread = threading.Thread(
        #         target=self._execute_trajectory,
        #         args=(msg.joint_names, msg.points),
        #         daemon=True
        #     )
        #     self.exec_thread.start()

    def _execute_trajectory(self, rx_names: List[str], points: List[JointTrajectoryPoint]):
        idx_of: Dict[str, int] = {nm: i for i, nm in enumerate(rx_names)}
        t0 = time.monotonic()

        for i, pt in enumerate(points):
            if self.stop_flag:
                self.get_logger().info("Trajectory execution stopped.")
                return

            due = t0 + pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            now = time.monotonic()
            if due > now:
                time.sleep(due - now)

            # 전체 값 저장
            for nm in self.full_names:
                j = idx_of.get(nm)
                if j is not None and j < len(pt.positions):
                    self.last_cmd[nm] = float(pt.positions[j])

            # 서브셋만 전송
            id_to_pos: Dict[int, int] = {}
            for nm in self.act_names:
                dxl_id = self.name_to_id[nm]
                rad = self.last_cmd[nm]
                tick = rad_to_tick_linear(rad, self.rmin_map[nm], self.rmax_map[nm], self.tmin_map[nm], self.tmax_map[nm])
                id_to_pos[dxl_id] = tick
                self.get_logger().info(f'rad: {rad}')
                
                
            self.get_logger().info(f'self.act_names: {self.act_names}')
            self.get_logger().info(f'id_to_pos: {id_to_pos}')

            # try:
            #     self.bus.sync_write_goal_positions(id_to_pos)
            # except Exception as e:
            #     self.get_logger().error(f"SyncWrite failed at point {i}: {e}")
            #     return

        self.get_logger().info("Trajectory execution done.")


    def _publish_joint_states(self):
        """
        actuated_names에 해당하는 모터들의 현재 위치를 읽어
        /hand_gripper/joint_states (sensor_msgs/JointState)로 퍼블리시.
        positions 단위는 라디안.
        """
        msg = JointState()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now

        joint_names = []
        positions = []

        if self.bus.lock:
            for joint_name in self.act_names:
                dxl_id = self.name_to_id[joint_name]
                try:
                    tick = self.bus.read4(dxl_id, ADDR_PRESENT_POSITION)  # 0~4095
                    rad = tick_to_rad_linear(
                        tick,
                        self.rmin_map[joint_name], self.rmax_map[joint_name],
                        self.tmin_map[joint_name], self.tmax_map[joint_name]
                    )
                    joint_names.append(joint_name)
                    positions.append(rad)
                except Exception as e:
                    # 읽기 실패 시 해당 조인트는 스킵 (원하면 NaN을 넣어도 됨)
                    self.get_logger().warn(f"read pos fail id={dxl_id} ({joint_name}): {e}")

        if joint_names:
            msg.name = joint_names
            msg.position = positions
            # velocity/effort은 비워둠
            self.js_pub.publish(msg)


    def destroy_node(self):
        try:
            for nm in self.act_names:
                dxl_id = self.name_to_id[nm]
                try:
                    self.bus.torque_enable(dxl_id, False)
                except Exception:
                    pass
            self.bus.close()
        finally:
            super().destroy_node()

def main():
    rclpy.init()
    node = DXLControlNode() 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()