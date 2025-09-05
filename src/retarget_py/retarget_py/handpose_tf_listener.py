#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
from scipy.spatial.transform import Rotation as R

from retarget_py.algebra_utils import rotm_to_quat, make_homogeneous  # rotm->quat, 4x4 조립

def tf_to_mat(T: TransformStamped) -> np.ndarray:
    """TF 메시지 -> 4x4 동차행렬"""
    t = T.transform.translation
    q = T.transform.rotation  # (x,y,z,w)
    Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return make_homogeneous(Rm, np.array([t.x, t.y, t.z], dtype=float))

def mat_to_tf(M: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    """4x4 동차행렬 -> TF 메시지"""
    Rm = M[:3, :3]
    p = M[:3, 3]
    qx, qy, qz, qw = rotm_to_quat(Rm)

    T = TransformStamped()
    T.header.stamp = stamp
    T.header.frame_id = parent
    T.child_frame_id = child
    T.transform.translation.x = float(p[0])
    T.transform.translation.y = float(p[1])
    T.transform.translation.z = float(p[2])
    T.transform.rotation.x = float(qx)
    T.transform.rotation.y = float(qy)
    T.transform.rotation.z = float(qz)
    T.transform.rotation.w = float(qw)
    return T

class HandPoseTFListener(Node):
    """ TF 버퍼/리스너 래퍼. lookup_mat(target<-source) : 4x4 변환행렬 반환 """
    def __init__(self):
        super().__init__('handpose_tf_listener')
        self.buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.listener = tf2_ros.TransformListener(self.buffer, self)

    def lookup_mat(self, target: str, source: str, timeout_sec: float = 0.2):
        try:
            T: TransformStamped = self.buffer.lookup_transform(
                target, source, rclpy.time.Time(), rclpy.duration.Duration(seconds=timeout_sec)
            )
            return tf_to_mat(T)
        except Exception as e:
            self.get_logger().debug(f'lookup_mat({target} <- {source}) 실패: {e}')
            return None
