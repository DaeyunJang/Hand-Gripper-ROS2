from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share_dir = get_package_share_directory('retarget_py')
    rviz_config = os.path.join(share_dir, 'config', 'rviz_config.rviz')
    
    return LaunchDescription([
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         [get_package_share_directory('realsense2_camera'), '/launch/rs_launch.py']),
        #         launch_arguments={
        #         'rgb_camera.color_profile': f'{image_width},{image_height},{cameara_fps}',
        #         'depth_module.depth_profile': f'{image_width},{image_height},{cameara_fps}',
        #         # default
        #         # 'rgb_camera.color_profile': '1280,720,30',
        #         # 'depth_module.depth_profile': '1280,720,30',
                
        #         # 'rgb_camera.enable_auto_exposure': 'false',
        #         # 'rgb_camera.exposure': '80',
        #         # 'rgb_camera.profile': '640,480,30',
        #         # 'depth_module.profile': '640,480,30',
        #         }.items()
        # ),
        Node(
            package='retarget_py',
            executable='retarget_node',
            name='retarget_node',
            output='screen',
            parameters=[{
                "source_root": "camera_color_optical_frame",
                "scale": 0.6
            }]
        ),
        # Node(
        #     package='handpose_ros',
        #     executable='handpose_tf_broadcaster',
        #     name='handpose_tf_broadcaster',
        #     output='screen',
        #     parameters=[{
        #         'hands_topic': 'hands/detections',
        #         'camera_info_topic': '/camera/camera/color/camera_info',
        #         'camera_frame': 'camera_color_optical_frame',   # For TF name
        #         'tf.norm.enable': False,
        #         'tf.canonical.enable': True,
        #         'tf.canonical.scale': scale # 1/1280
        #         # Not use
        #         # 'use_depth': True,  # True면 아래 두 토픽도 필요
        #         # 'depth_topic': '/camera/aligned_depth_to_color/image_raw',
        #         # 'camera_info_topic': '/camera/color/camera_info',
        #     }]
        # ),
        # === RViz2 추가 ===
        Node(
            package='rviz2',
            executable='rviz2',
            name='handpose_rviz',
            output='screen',
            arguments=['-d', rviz_config],
            # env={'LD_LIBRARY_PATH': ''},
        ),
    ])
