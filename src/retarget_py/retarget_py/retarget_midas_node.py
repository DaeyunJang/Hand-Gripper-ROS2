#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os, time

import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

import tf2_ros
from tf2_ros import TransformException, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

from ament_index_python.packages import get_package_share_directory

from retarget_py.algebra_utils import *
from retarget_py.ik_solver import ThreeDoFManipulator

# ---------- 유틸 ----------
def add_suffix(frame_name: str, suffix: str = "gripper_scale") -> str:
    tail = f"_{suffix}"
    return frame_name if frame_name.endswith(tail) else (frame_name + tail)

# ---------- 메인 노드 ----------

class RetargetNode(Node):
    """
    왼손/오른손 각각에 대해 Wrist→(선택 손가락)MCP 3D 거리 L_obs(label)을 계산하고,
    동일한 L_real(파라미터)로 정규화 스케일 s(label)=L_real/L_obs(label)을 만든다(EMA 스무딩 옵션).
    그 스케일을 해당 손의 모든 hand_* 프레임 '병진'에 적용하여 *_{scale_suffix}를 퍼블리시한다.
    또한 gripper 기준 MCP/TIP 파생 프레임도 동일 스케일이 반영된 결과로 퍼블리시한다.
    """

    def __init__(self) -> None:
        super().__init__("retarget_node")
        
        # Dummy joint_state publihser
        self.declare_parameter("state_topic", "/hand_gripper/joint_states")  # 기본은 실제
        # self.declare_parameter("state_topic", "/dummy/joint_states")  # 기본은 실제
        self._state_topic = self.get_parameter("state_topic").get_parameter_value().string_value
        self._state_sub = None
        self._subscribe_joint_state(self._state_topic)

        ################################################################################################################################
        ## Configuration First ##
        # config (yaml) 경로
        self.config_path: str = self.declare_parameter(
            "gripper_config_path", "MIDAS_gripper.yaml"
        ).get_parameter_value().string_value
        # (legacy) 기존 스케일 파라미터
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("scale_xyz", [])
        
        # -------- Load Config --------
        cfg = self._load_gripper_config(self.config_path)
        # (legacy) yaml/param 스케일 로딩
        scale_param = float(self.get_parameter("scale").value or 1.0)
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
        self.fingers_cfg = cfg.get("fingers") or []  # [{name, base_offset_x, base_offset_y, target_finger:{label,finger}}]
        self.finger_joint_names_map = self._make_finger_joint_names_map()
        # 전체 조인트 이름(컨트롤러 yaml의 joints 순서와 반드시 일치해야 함)
        self.all_joint_names: List[str] = []
        for fdef in self.fingers_cfg:
            self.all_joint_names.extend(self.finger_joint_names_map.get(fdef["name"], []))

        ################################################################################################################################
        
        # -------- Parameters --------
        
        # Robot name
        self.gripper_name: str = self.declare_parameter(
            "gripper_name", "hand_gripper"
        ).get_parameter_value().string_value
        
        ## Hand Pose ##
        self.source_frame_name: str = self.declare_parameter(
            "source_root", "camera_color_optical_frame"
        ).get_parameter_value().string_value

        self.gripper_base_frame_name: str = self.declare_parameter(
            "gripper_base_frame", "gripper_base_frame"
        ).get_parameter_value().string_value

        self.target_suffix: str = self.declare_parameter(
            "target_suffix", "world_abs"
        ).get_parameter_value().string_value

        self.scale_suffix: str = self.declare_parameter(
            "scale_suffix", "gripper_scale"
        ).get_parameter_value().string_value
        
        ## In this code file, do not over 60.0 Hz (if 100 Hz, timer will overrun --> throttle)
        self.rate_hz: float = float(self.declare_parameter("rate_hz", 30.0).value)

        # ---- 정규화 관련 파라미터 ----
        # Wrist -> MCP 실제 기준 길이(양손 동일하게 사용). 단위: meter
        self.wrist_to_mcp_real_length: float = float(
            self.declare_parameter("wrist_to_mcp_real_length", 0.06).value  # 예: 60mm
        )
        # 어떤 손가락 MCP를 기준으로 할지 (예: index, middle, ring, pinky, thumb)
        self.scale_finger_name: str = str(
            self.declare_parameter("scale_finger_name", "index").value
        ).lower()

        self.enable_finger_scale: bool = self.declare_parameter("finger_scale.enable", False).get_parameter_value().bool_value
        self.finger_scale: float = float(self.declare_parameter("finger_scale.scale", 1.5).value)
        # 수치 안정
        self.norm_eps: float = float(self.declare_parameter("norm_eps", 1e-6).value)
        # 스케일 EMA
        self.scale_smooth_alpha: float = float(self.declare_parameter("scale_smooth_alpha", 0.3).value)
        # legacy scale과 곱할지 여부
        self.multiply_with_config_scale: bool = bool(self.declare_parameter("multiply_with_config_scale", False).value)
        
        
        self.filter_alpha: float = float(self.declare_parameter("filter.alpha", 0.5).value)
        
        # Declare parameters - END #############################
        
        # Gripper configuration
        # __init__ 끝부분에: finger별 로봇 인스턴스 생성
        # ✅ finger별로 gripper(= ThreeDoFManipulator) 인스턴스를 만든다.
        self.grippers: Dict[str, ThreeDoFManipulator] = {}  # Dict[str, ThreeDoFManipulator]
        for fdef in self.fingers_cfg:
            fname = str(fdef["name"])
            # self.get_logger().info(f"[gripper init] fingers_cfg={fdef}: fname={fname}")
            dh_cfg = (fdef.get("dh") or {})
            try:
                self.grippers[fname] = self._build_gripper_from_dh(fname, dh_cfg)
                self.get_logger().info(f"[gripper init] {fname} -> {dh_cfg}: built from DH")
            except Exception as ex:
                self.get_logger().warning(f"[gripper init] {fname} -> {dh_cfg}: DH build failed: {ex}")
        self.get_logger().info(f'self.grippers: {self.grippers}')
        
        # Publisher
        # JointTrajectory publisher (ros2_control joint_trajectory_controller)
        self.joint_states_cmd_pub = self.create_publisher(JointState, f"/{self.gripper_name}/joint_states_cmd", 1)
        self.joint_states_cmd_prev = None
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f"/{self.gripper_name}/joint_trajectory",   # 컨트롤러 이름에 맞게 바꾸기
            1
        )
        self._joint_state_map: Dict[str, float] = {}  # {joint_name: position}
        self.state_sub = self.create_subscription(JointState, "/joint_states", self._on_joint_state, 1)
        
        # -------- TF IO --------
        self.T_src_to_grip = rot_y_180_mat() # transform matrix of gripper base frame
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # 정적 프레임 퍼블리시
        self._publish_static_frames()

        # 타이머
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        # 내부 상태
        self._last_warn: Dict[str, float] = {}
        self._warn_interval: float = 2.0
        # 손별 EMA 저장소
        self._s_prev_map: Dict[str, Optional[float]] = {"left": None, "right": None}

        scale_repr = list(self.scale_xyz) if self.scale_xyz is not None else self.scale_scalar
        self.get_logger().info(
            f"[retarget] source='{self.source_frame_name}', gripper='{self.gripper_base_frame_name}', "
            f"target_suffix='{self.target_suffix}', legacy_scale={scale_repr} ({src_scale}), "
            f"normalize_by_wrist_to_mcp (per hand): finger='{self.scale_finger_name}', "
            f"L_real={self.wrist_to_mcp_real_length:.4f} m, smooth_alpha={self.scale_smooth_alpha}, "
            f"multiply_with_config_scale={self.multiply_with_config_scale}, rate={self.rate_hz:.1f}Hz"
        )
        
        
        # 파라미터 변경 감지 → 핫스왑
        self.add_on_set_parameters_callback(self._on_params)

    # ---------- Config 로더 ----------
    def _pick_yaml_scale(self, cfg: Dict[str, Any]) -> None:
        sxyz = cfg.get("scale_xyz")
        if sxyz is not None and isinstance(sxyz, (list, tuple)) and len(sxyz) == 3:
            self.scale_xyz = np.array(sxyz, dtype=float)
            self.scale_scalar = 1.0
        else:
            self.scale_scalar = float(cfg.get("scale", 1.0))
            self.scale_xyz = None

    def _load_gripper_config(self, user_path: str) -> Dict[str, Any]:
        # Treat user_path as a filename to be searched under package share's config folder.
        # If user_path is empty, default to "gripper_config.yaml".
        filename = Path(user_path).name if user_path else "gripper_config.yaml"
        try:
            pkg_share = Path(get_package_share_directory("retarget_py"))
            cfg_path = (pkg_share / "config" / "midas" / filename).resolve()
        except Exception:
            cfg_path = None

        if not cfg_path or not cfg_path.is_file():
            raise FileNotFoundError(
                f"Config file '{filename}' not found in retarget_py package share (config/). "
                f"Set -p gripper_config_path:=<filename-in-config> or ensure the file exists."
            )

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            g = data.get("gripper") or {}

            fingers: List[Dict[str, Any]] = []
            for i, fdef in enumerate(g.get("fingers") or [], start=1):
                name = str(fdef.get("name") or f"finger{i}")
                x = float(fdef.get("base_offset_x") or 0.0)
                y = float(fdef.get("base_offset_y") or 0.0)
                tfing = fdef.get("target_finger") or {}
                label = str(tfing.get("label") or "")
                finger = str(tfing.get("finger") or "")
                finger_length = fdef.get("finger_length") or {}
                dh = fdef.get("dh") or {}
                
                if not label or not finger:
                    self.get_logger().warning(f"[config] {name}: target_finger.label/finger required. Skipping.")
                    continue
                
                fingers.append({
                    "name": name,
                    "base_offset_x": x, "base_offset_y": y,
                    "target_finger": {"label": label, "finger": finger},
                    "finger_length": finger_length,
                    "dh": dh            # ✅ 추가: DH 파라미터까지 보관
                })

            out = {
                "scale": float(g.get("scale") or 1.0),
                "scale_xyz": g.get("scale_xyz"),
                "target_suffix": str(g.get("target_suffix")),
                "fingers": fingers,
            }
            self.get_logger().info(f"Loaded gripper config: {cfg_path} (fingers={len(fingers)})")
            return out
        except Exception as ex:
            self.get_logger().warning(f"Failed to load gripper config '{cfg_path}': {ex}. Using defaults.")
            return {"scale": 0.1, "target_suffix": "world_abs", "fingers": []}
        
    def _make_finger_joint_names_map(self) -> Dict[str, List[str]]:
        """
        예: {"finger1": ["finger1_j0_linear", "finger1_j1","finger1_j2","finger1_j3"], ...}
        """
        out: Dict[str, List[str]] = {}
        for fdef in self.fingers_cfg:
            fname = str(fdef["name"])
            joints = (fdef.get("dh") or {}).get("joints") or []
            
            # Prismatic joint는 수동 추가.
            names: List[str] = []
            # names: List[str] = [f"{fname}_j0_linear"]
            for j in joints:
                jn = str(j.get("name") or "")
                if not jn:
                    continue
                names.append(f"{fname}_{jn}")   # ✅ 동적 이름 규칙
            out[fname] = names
        return out
    
    def _build_gripper_from_dh(self, fname: str, dh_cfg: dict) -> ThreeDoFManipulator:
        joints = (dh_cfg or {}).get("joints") or []
        # if len(joints) < 3:
        #     raise ValueError(f"{fname}: need 3 joints in dh.joints")

        j0, j1, j2, j3 = joints

        # ThreeDoFManipulator 가정:
        #  - L1: d = l1
        #  - L2: a = l2
        #  - L3: a = l3
        l1 = float(j1.get("d", 0.0))
        l2 = float(j2.get("a", 0.0))
        l3 = float(j3.get("a", 0.0))

        # 선택: qlim_deg → rad (없으면 클래스 기본값 사용)
        def qlim_deg_to_rad(j, default):
            qd = j.get("qlim_deg")
            if isinstance(qd, (list, tuple)) and len(qd) == 2:
                lo, hi = float(qd[0]), float(qd[1])
                return (np.deg2rad(lo), np.deg2rad(hi))
            return default

        qlim1 = qlim_deg_to_rad(j1, (-np.pi, np.pi))
        qlim2 = qlim_deg_to_rad(j2, (-np.pi/2, np.pi/2))
        qlim3 = qlim_deg_to_rad(j3, (-np.pi/2, np.pi/2))

        return ThreeDoFManipulator(
            l1=l1, l2=l2, l3=l3,
            qlim1=qlim1, qlim2=qlim2, qlim3=qlim3,
            name=f"{fname}_3DoFManipulator"
        )
        
    
    # ---------- 프레임 이름/탐색 ----------

    def _hand_frame_name(self, label: str, finger: str, joint: str, suffix: str) -> str:
        return f"hand_{label}_{finger}_{joint}_{suffix}"

    def _scaled_name(self, frame: str) -> str:
        return add_suffix(frame, self.scale_suffix)

    def _list_all_hand_frames_by_suffix(self, label: str) -> List[str]:
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        joints  = ["mcp", "pip", "dip", "tip"]
        frames: List[str] = []
        for fg in fingers:
            for jt in joints:
                frames.append(self._hand_frame_name(label, fg, jt, self.target_suffix))
        return frames

    def _candidate_wrist_frames(self, label: str) -> List[str]:
        # 프로젝트 네이밍에 맞춰 필요시 보완
        return [
            f"hand_{label}_wrist_{self.target_suffix}",
            f"hand_{label}_root_{self.target_suffix}",
            f"hand_{label}_base_{self.target_suffix}",
        ]

    ## ROS2 callback function       
    def _collect_current_positions(self, joint_names: List[str]) -> List[float]:
        return [self._joint_state_map.get(n, 0.0) for n in joint_names]
    
    def _subscribe_joint_state(self, topic: str):
        if self._state_sub:
            self.destroy_subscription(self._state_sub)
        self._state_sub = self.create_subscription(JointState, topic, self._on_joint_state, 1)
        self.get_logger().info(f"[retarget] subscribing joint state: {topic}")
    
    def _on_joint_state(self, msg: JointState) -> None:
        # 이름->값 매핑 캐시
        for n, p in zip(msg.name, msg.position):
            self._joint_state_map[n] = p
    
    def _on_params(self, params):
        for p in params:
            if p.name == "state_topic" and p.type_ == Parameter.Type.STRING:
                new_topic = p.value
                if new_topic != self._state_topic:
                    self._state_topic = new_topic
                    self._subscribe_joint_state(new_topic)
        return SetParametersResult(successful=True)
    
    # ---------- TF 생성/전송 ----------

    def make_tf_from_matrix(self, parent: str, child: str, transform_matrix: np.ndarray, stamp) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        rotm = transform_matrix[:3, :3]
        qx, qy, qz, qw = rotm_to_quat(rotm)
        t.transform.translation.x = float(transform_matrix[0, 3])
        t.transform.translation.y = float(transform_matrix[1, 3])
        t.transform.translation.z = float(transform_matrix[2, 3])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        return t

    def _send_tf_from_matrix(self, parent: str, child: str, transform_matrix: np.ndarray, stamp) -> bool:
        t = self.make_tf_from_matrix(parent=parent, child=child, transform_matrix=transform_matrix, stamp=stamp)
        self.tf_broadcaster.sendTransform(t)
        return True

    # ---------- 정적 프레임 ----------

    def _publish_static_frames(self) -> None:
        stamp = self.get_clock().now().to_msg()

        # source_root -> gripper_base_frame (Ry180)
        T = rot_y_180_mat()
        tfmsg = self.make_tf_from_matrix(
            parent=self.source_frame_name,
            child=self.gripper_base_frame_name,
            transform_matrix=T,
            stamp=stamp
        )
        self.static_broadcaster.sendTransform(tfmsg)

        # gripper_base_frame -> gripper_{i}_base_frame (offset)
        for fdef in self.fingers_cfg:
            fname = str(fdef["name"])
            fx, fy = float(fdef["base_offset_x"]), float(fdef["base_offset_y"])
            finger_base_name = f"gripper_{fname}_base"
            Tfb = make_homogeneous(np.eye(3), np.array([fx, fy, 0.0], dtype=float))
            tfmsg = self.make_tf_from_matrix(
                parent=self.gripper_base_frame_name,
                child=finger_base_name,
                transform_matrix=Tfb,
                stamp=stamp
            )
            self.static_broadcaster.sendTransform(tfmsg)

    # ---------- TF 헬퍼 ----------

    def _lookup_transform_matrix(self, target: str, source: str) -> Optional[np.ndarray]:
        """make tf to 4x4 matrix (ROS2 lookup_transform is opposite definition)

        Args:
            target (str): to
            source (str): from

        Returns:
            Optional[np.ndarray]: I don't know why definition of lookup_transform function is opposite input variable target<->source
            
            It means, tf2_echo is correctly operate {from}->{to} w.r.t {source}->{target}.
            But lookup_transform operate {from}->{to} w.r.t. {target}->{source}
        """
        try:
            T: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=source, source_frame=target, time=Time(), timeout=Duration(seconds=0.0)
            )
            t = T.transform.translation
            q = T.transform.rotation
            Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            return make_homogeneous(Rm, np.array([t.x, t.y, t.z], dtype=float))
        except Exception:
            return None

    # ---------- 스케일 적용/거리 ----------

    def _apply_translation_scale(self, T: np.ndarray, scalar: float, xyz: Optional[np.ndarray] = None) -> np.ndarray:
        """회전은 보존, 병진만 스케일"""
        M = T.copy()
        if xyz is None:
            M[:3, 3] = M[:3, 3] * scalar
        else:
            M[:3, 3] = M[:3, 3] * (xyz * scalar)
        return M

    def _dist3(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    
    # retarget process
    def _retarget_pinch_motion_from_hand_to_gripper(
        self,
        index_primary: int,
        index_secondary: int,
        gripper_finger_primary_number: int,
        gripper_finger_secondary_number: int,
        stamp
    ) -> None:
        """Generate gripper coordinate system
        @author DY
        @date 2025-9-11
        
        1) Pre-process
        A transformation matrix is computed from the {gripper_base_frame} to the position of each {hand coordinate system mcp},
        applying translation only and ignoring the rotation component. This transformation is then published via TF.
        In other words, it is defined in advance to enable the subsequent creation of {gripper_finger{i}_mcp} in section 2) and the transformation from {gripper_finger{i}_mcp} to {gripper_finger{i}_tip} in section 3).
        
        2) Define the frame {gripper_finger_mcp} from frame {gripper_finger_base}
            - A new frame {gripper_finger{i}_mcp} is generated by applying only the translational component along the x-axis from the transformation between {gripper_finger_base} and {hand_{label}_{index or thumb}_mcp_world_abs}.
                {gripper_finger{i}_base} is defined `@function _publish_static_frames()`
            * TBD: software limit have to be generated. -> linear guide limit
            
        3) Convert hand finger scale to gripper finger scale
        The scale between the hand coordinate system (with the suffix 'world_abs') of the desired finger (e.g., index finger) and the predefined gripper finger length is computed.
        The finger length of the desired finger is obtained by summing the norms of the translational components from the transformation matrices of mcp→pip, pip→dip, and dip→tip.
        The actual hardware design parameter "finger_length"(refer to config/gripper_config.yaml) is then used to adjust the scale factors for the x, y, and z components.
        After obtaining scale value, make new transform matrices {gripper_finger{i}_mcp}->{gripper_finger{i}_tip}.
        This matrices are calculated like code below
            T_gripper_finger{i}_mcp_to_tip = T_{finger(index)}_mcp_to_tip
            T_gripper_finger{i}_mcp_to_tip[:3, 3] = T_gripper_finger{i}_mcp_to_tip[:3, 3] * scale
            
        In our setup, finger1 and finger2 cannot move independently.
        Therefore, if the scale mapping is performed using finger1,
        the {gripper_finger2_tip} must be represented (re-anchored) with respect to {gripper_finger1_mcp} before applying the scaling.
        
        ** In each step of '1)', '2)'and '3)', tf must be published. **
        """
        
        """
        한 손(두 개의 결합된 finger)을 처리한다.
        - primary finger로 스케일 계산
        - secondary tip은 스케일 적용 전, primary MCP에 대해 re-anchoring
        """
        if not (0 <= index_primary < len(self.fingers_cfg)) or not (0 <= index_secondary < len(self.fingers_cfg)):
            self.get_logger().warning(
                f"_retarget_pinch_motion_from_hand_to_gripper: bad indices (primary={index_primary}, secondary={index_secondary}, len={len(self.fingers_cfg)})"
            )
            return

        target_finger_primary = self.fingers_cfg[index_primary]["target_finger"]      # {label, finger}
        target_finger_secondary = self.fingers_cfg[index_secondary]["target_finger"]  # {label, finger}

        # --- hand frame names ---
        frame_name_primary_mcp = self._hand_frame_name(target_finger_primary["label"], target_finger_primary["finger"], "mcp", self.target_suffix)
        frame_name_primary_pip = self._hand_frame_name(target_finger_primary["label"], target_finger_primary["finger"], "pip", self.target_suffix)
        frame_name_primary_dip = self._hand_frame_name(target_finger_primary["label"], target_finger_primary["finger"], "dip", self.target_suffix)
        frame_name_primary_tip = self._hand_frame_name(target_finger_primary["label"], target_finger_primary["finger"], "tip", self.target_suffix)

        frame_name_secondary_mcp = self._hand_frame_name(target_finger_secondary["label"], target_finger_secondary["finger"], "mcp", self.target_suffix)
        frame_name_secondary_tip = self._hand_frame_name(target_finger_secondary["label"], target_finger_secondary["finger"], "tip", self.target_suffix)

        # --- 1) align MCPs to gripper_base (translation-only) ---
        T_gripper_base_to_primary_mcp_world_abs = self._lookup_transform_matrix(frame_name_primary_mcp, self.gripper_base_frame_name)
        T_gripper_base_to_secondary_mcp_world_abs = self._lookup_transform_matrix(frame_name_secondary_mcp, self.gripper_base_frame_name)

        frame_name_primary_mcp_aligned = self._hand_frame_name(target_finger_primary["label"], target_finger_primary["finger"], "mcp", suffix="gripper_aligned")
        frame_name_secondary_mcp_aligned = self._hand_frame_name(target_finger_secondary["label"], target_finger_secondary["finger"], "mcp", suffix="gripper_aligned")

        # make mcp coordinate which has same orientation as gripper base frame
        if T_gripper_base_to_primary_mcp_world_abs is not None:
            trans = T_gripper_base_to_primary_mcp_world_abs[:3, 3]
            self._send_tf_from_matrix(self.gripper_base_frame_name, frame_name_primary_mcp_aligned, make_homogeneous(np.eye(3), trans), stamp=stamp)
        # else:
        #     self.get_logger().warning("Primary MCP alignment skipped: T_gripper_base_to_primary_mcp_world_abs is None.")

        if T_gripper_base_to_secondary_mcp_world_abs is not None:
            trans = T_gripper_base_to_secondary_mcp_world_abs[:3, 3]
            self._send_tf_from_matrix(self.gripper_base_frame_name, frame_name_secondary_mcp_aligned, make_homogeneous(np.eye(3), trans), stamp=stamp)
        # else:
        #     self.get_logger().warning("Secondary MCP alignment skipped: T_gripper_base_to_secondary_mcp_world_abs is None.")

        # --- 2) define gripper_finger{N}_mcp from gripper_finger{N}_base (x-translation only) ---
        gripper_finger_primary_base = f"gripper_finger{gripper_finger_primary_number}_base"
        gripper_finger_secondary_base = f"gripper_finger{gripper_finger_secondary_number}_base"
        gripper_finger_primary_mcp  = f"gripper_finger{gripper_finger_primary_number}_mcp"
        gripper_finger_secondary_mcp = f"gripper_finger{gripper_finger_secondary_number}_mcp"
        gripper_finger_primary_tip  = f"gripper_finger{gripper_finger_primary_number}_tip"
        gripper_finger_secondary_tip = f"gripper_finger{gripper_finger_secondary_number}_tip"

        T_base_to_primary_mcp_aligned = self._lookup_transform_matrix(frame_name_primary_mcp_aligned, gripper_finger_primary_base)
        if T_base_to_primary_mcp_aligned is not None:
            tx = float(T_base_to_primary_mcp_aligned[0, 3])
            self._send_tf_from_matrix(gripper_finger_primary_base, gripper_finger_primary_mcp, make_homogeneous(np.eye(3), np.array([tx, 0.0, 0.0], dtype=float)), stamp=stamp)

        T_base_to_secondary_mcp_aligned = self._lookup_transform_matrix(frame_name_secondary_mcp_aligned, gripper_finger_secondary_base)
        if T_base_to_secondary_mcp_aligned is not None:
            tx = float(T_base_to_secondary_mcp_aligned[0, 3])
            self._send_tf_from_matrix(gripper_finger_secondary_base, gripper_finger_secondary_mcp, make_homogeneous(np.eye(3), np.array([tx, 0.0, 0.0], dtype=float)), stamp=stamp)

        # --- 3) scale from primary chain length ---
        T_primary_mcp_to_pip = self._lookup_transform_matrix(frame_name_primary_mcp, frame_name_primary_pip)
        T_primary_pip_to_dip = self._lookup_transform_matrix(frame_name_primary_pip, frame_name_primary_dip)
        T_primary_dip_to_tip = self._lookup_transform_matrix(frame_name_primary_dip, frame_name_primary_tip)

        finger_length_design = float(self.fingers_cfg[index_primary]["finger_length"])
        if T_primary_mcp_to_pip is not None and T_primary_pip_to_dip is not None and T_primary_dip_to_tip is not None:
            finger_length_observed = float(
                np.linalg.norm(T_primary_mcp_to_pip[:3, 3]) +
                np.linalg.norm(T_primary_pip_to_dip[:3, 3]) +
                np.linalg.norm(T_primary_dip_to_tip[:3, 3])
            )
        else:
            # self.get_logger().warning("Scale: missing transforms; using design finger_length as fallback.")
            finger_length_observed = finger_length_design

        ##############################################################################################################################
        ### DY ###
        # Hand landmark의 스케일 조절 중 링크 길이가 들쑥날쑥하는데, 이거 매핑하는 꼼수임.
        # 정규화된 손 (world_abs) 프레임 기준으로 검지 관절들의 길이는 0.04~0.06 정도로 측정됨
        # y=ymin​+(ymax​−ymin​)⋅(x - xmin)/(xma​x−xmin​​)
        # 아래 예시는 0.4 ~ 0.6 input, 1.0~0.5 output
        def linear_map(x, in_min=0.04, in_max=0.06, out_min=0.8, out_max=0.5):
            x = min(max(x, 0.04), 0.06)
            return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)
        scale_stabilizer = linear_map(finger_length_observed)
        ##############################################################################################################################
        
        scale_length = (scale_stabilizer)* finger_length_design / max(1e-9, finger_length_observed)
        # self.get_logger().info(
        #     f"[{target_finger_primary['label']}/{target_finger_primary['finger']}] "
        #     f"scale={scale_length} (design={finger_length_design}, observed={finger_length_observed}, clip={scale_stabilizer})"
        # )

        # --- 4) primary TIP (aligned MCP -> TIP, scaled) ---
        T_primary_mcp_aligned_to_tip = self._lookup_transform_matrix(frame_name_primary_tip, frame_name_primary_mcp_aligned)
        if T_primary_mcp_aligned_to_tip is not None:
            # translation values are multipled as much as scale.
            self._send_tf_from_matrix(gripper_finger_primary_mcp, gripper_finger_primary_tip, scale_translation(T_primary_mcp_aligned_to_tip, scale_length), stamp=stamp)

            # check length
            T_scale = scale_translation(T_primary_mcp_aligned_to_tip, scale_length)
            trans = T_scale[:3, 3]
            l = np.linalg.norm(trans)
            # self.get_logger().info(f"scaled mcp->tip length: {l} m")
        
        # --- 5) secondary TIP (re-anchored to primary MCP, then scaled) ---
        T_intermediate_from_primary_mcp_to_secondary_tip_scaled = np.eye(4)
        T_primary_mcp_aligned_to_secondary_tip = self._lookup_transform_matrix(frame_name_secondary_tip, frame_name_primary_mcp_aligned)
        if T_primary_mcp_aligned_to_secondary_tip is not None:
            T_intermediate_from_primary_mcp_to_secondary_tip_scaled = scale_translation(T_primary_mcp_aligned_to_secondary_tip, scale_length)
            
            ## z방향이 좀 엄지가 항상 낮게 추정되는 경향이 있어서 임의로 스칼라 값을 더해줌;
            # world_abs 기준 한 1cm 엄지가 낮게 나오는듯?
            T_intermediate_from_primary_mcp_to_secondary_tip_scaled[2,3] += scale_length*0.012

        T_primary_mcp_to_secondary_mcp = self._lookup_transform_matrix(gripper_finger_secondary_mcp, gripper_finger_primary_mcp)
        if T_primary_mcp_to_secondary_mcp is not None:
            T_secondary_mcp_aligned_to_secondary_tip = make_inv_homogeneous(T_primary_mcp_to_secondary_mcp) @ T_intermediate_from_primary_mcp_to_secondary_tip_scaled
            self._send_tf_from_matrix(gripper_finger_secondary_mcp, gripper_finger_secondary_tip, T_secondary_mcp_aligned_to_secondary_tip, stamp=stamp)

    
    
    def get_target_joints_from_ik(self, finger_index: int, parent_frame: str, tip_frame: str) -> Optional[Tuple[List[str], List[float]]]:
        """
        finger_index: self.fingers_cfg의 인덱스(finger1=0, finger2=1, ...)
        반환: (joint_names_for_that_finger, q_goal)
        """
        if not (0 <= finger_index < len(self.fingers_cfg)):
            self.get_logger().warn(f'finger index not match: 0 ~ {len(self.fingers_cfg)}')
            return None

        T = self._lookup_transform_matrix(tip_frame, parent_frame)
        
        if T is None:
            return None
        
        # check the limit of link's lengthes is less than workspace
        if np.linalg.norm(T[:3, 3]) >= self.fingers_cfg[finger_index]["finger_length"]:
            # self.get_logger().warn(f'[get_target_joints_from_ik()] Transform matrix is out of workspace (link length): {np.linalg.norm(T[:3, 3])}')
            return None
        
        # # ✅ IK 호출
        fname = self.fingers_cfg[finger_index]["name"]
        gripper = self.grippers.get(fname)
        if gripper is None:
            # self.get_logger().warn(f'No {fname} gripper: {gripper}')
            return None
        
        ik_joint_names = [f"{fname}_j1", f"{fname}_j2", f"{fname}_j3"]
        
        # initial pose is 0, pi/2, 0 in 3 dof manipulator
        q0 = np.array([
            1.5708 if 'j2' in n else self._joint_state_map.get(n, 0.0)
            for n in ik_joint_names
            ], dtype=float)
        
        res = gripper.solve(T_target=T, q0=q0, position_only=True, ilimit=20, tol=1e-5)
        if not res.success:
            return None
        
        # self.get_logger().info(f"finger_index: {finger_index}")
        # self.get_logger().info(f"parent_frame: {parent_frame}")
        # self.get_logger().info(f"tip_frame: {tip_frame}")
        # self.get_logger().info(f"fname: {fname}")
        # self.get_logger().info(f"gripper: {gripper}")
        # self.get_logger().info(f"T:\n{T}")
        # self.get_logger().info(f"ik_joint_names: {ik_joint_names}\n res.q_final.tolist(): {res.q_final.tolist()}")
        
        return ik_joint_names, res.q_final.tolist()

    def publish_target_joint_and_trajectory_step_all_fingers(self, finger_solutions: list[tuple[list[str], list[float]]], duration_s: float = 0.1) -> None:
        jt = JointTrajectory()
        jt.joint_names = list(self.all_joint_names)
        # self.get_logger().info(f'jt.joint_names: \n{jt.joint_names}')

        # p0: 현재값
        p0 = JointTrajectoryPoint()
        p0.positions = self._collect_current_positions(jt.joint_names)
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = 0
        
        
        goal_map: Dict[str, float] = {}
        
        ### Joint 0: prismatic joint 먼저 채우기 ###############3
        for fdef in self.fingers_cfg:
            fname = str(fdef["name"])  # "finger1" 등
            j0_name = f"{fname}_j0_linear"

            gripper_base = f"gripper_{fname}_base"
            gripper_mcp = f"gripper_{fname}_mcp"
            T = self._lookup_transform_matrix(gripper_mcp, gripper_base)
            if T is not None:
                tx = T[0, 3]
                if fname in ('finger1', 'finger2'):
                    goal_map[j0_name] = float(max(0, min(tx, 0.05)))
                elif fname in ('finger3', 'finger4'):
                    goal_map[j0_name] = float(max(0, min(tx-0.1, 0.05)))

        # p1: 목표(q_goal들을 전체 순서에 맞게 병합)
        for names, qs in finger_solutions:
            for n, q in zip(names, qs):
                # self.get_logger().info(f'n={n}, q={q}')
                goal_map[n] = float(q)


            
        p1 = JointTrajectoryPoint()
        p1.positions = [goal_map.get(n, p0.positions[i]) for i, n in enumerate(jt.joint_names)]
        p1.time_from_start.sec = int(duration_s)
        p1.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1e9)

        jt.points = [p0, p1]
        self.traj_pub.publish(jt)
        
        # 첫 실행이면 그냥 현재값 저장
        if self.joint_states_cmd_prev is None:
            filtered_positions = np.array(p1.positions, dtype=float)
        else:
            prev = np.array(self.joint_states_cmd_prev.position, dtype=float)
            curr = np.array(p1.positions, dtype=float)
            filtered_positions = self.filter_alpha * prev + (1.0 - self.filter_alpha) * curr


        # ✅ JointState 목표 퍼블리시
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = jt.joint_names
        # js.position = p1.positions
        js.position = filtered_positions.tolist()
        self.joint_states_cmd_pub.publish(js)
        
        self.joint_states_cmd_prev = js

        # self.get_logger().info(f'p1.positions: {p1.positions}')

    # ---------- 메인 루프 ----------

    def _on_timer(self) -> None:
        
        
        stamp = self.get_clock().now().to_msg()
        
        t0 = time.perf_counter()
        self._publish_static_frames()
        t1 = time.perf_counter()
        # 왼손: finger1 / finger2
        self._retarget_pinch_motion_from_hand_to_gripper(index_primary=0, index_secondary=1, gripper_finger_primary_number=1, gripper_finger_secondary_number=2, stamp=stamp)
        t2 = time.perf_counter()
        # 오른손: finger3 / finger4
        self._retarget_pinch_motion_from_hand_to_gripper(index_primary=2, index_secondary=3, gripper_finger_primary_number=3, gripper_finger_secondary_number=4, stamp=stamp)

        # IK
        solutions = []
        s1 = self.get_target_joints_from_ik(0, "gripper_finger1_mcp", "gripper_finger1_tip")
        if s1: solutions.append(s1)
        else:
            return
        s2 = self.get_target_joints_from_ik(1, "gripper_finger2_mcp", "gripper_finger2_tip")
        if s2: solutions.append(s2)
        else:
            return
        s3 = self.get_target_joints_from_ik(2, "gripper_finger3_mcp", "gripper_finger3_tip")
        if s3: solutions.append(s3)
        else:
            return
        s4 = self.get_target_joints_from_ik(3, "gripper_finger4_mcp", "gripper_finger4_tip")
        if s4: solutions.append(s4)
        else:
            return

        if solutions:
            self.publish_target_joint_and_trajectory_step_all_fingers(solutions, duration_s=0.01)  # 주기에 맞춰 0.05~0.3s 조절
            # self.get_logger().info(f'solutions: {solutions}')
            # self.get_logger().info(f'time: t0-t1: {t1-t0} / t1-t2: {t2-t1} / t2-t3: {t3-t2}')
            
        t3 = time.perf_counter()
        
        
        
    # ---------- 경고 스로틀 ----------
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

if __name__ == "__main__":
    main()
