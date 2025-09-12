from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # ---- Launch args (필요시 override) ----
    with_handpose   = LaunchConfiguration('with_handpose')
    image_width     = LaunchConfiguration('image_width')
    image_height    = LaunchConfiguration('image_height')
    camera_fps      = LaunchConfiguration('camera_fps')

    # ---- handpose_pkg (handpose_ros) 런치 포함 ----
    # 파일명은 너가 만든 handpose 런치 파일명으로 맞춰줘 (예: handpose_bringup.launch.py)
    handpose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('handpose_ros'),
                'launch',
                'handpose_launch.py',  # <-- 여기를 실제 파일명으로 수정
            ])
        ),
        launch_arguments={
            'image_width':  image_width,
            'image_height': image_height,
            'camera_fps':   camera_fps,
        }.items(),
        condition=IfCondition(with_handpose)
    )

    # ---- RViz config (retarget_py) ----
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('retarget_py'),
        'config',
        'rviz_config.rviz'
    ])

    retarget_node = Node(
        package='retarget_py',
        executable='retarget_node',
        name='retarget_node',
        output='screen',
        parameters=[{
            "source_root": "camera_color_optical_frame",
            # "scale": 0.48,
            "warning_log_flag": False,
        }]
    )
    
    # ik_solver_node = Node(
    #     package='retarget_py',
    #     executable='ik_solver_node',
    #     name='ik_solver_node',
    #     output='screen',
    #     parameters=[{
    #         # "scale": 0.48,
    #         "base_frame": "gripper_finger1_mcp",
    #         "eef_frame": "gripper_finger1_tip",
    #     }]
    # )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='handpose_rviz',
        output='screen',
        arguments=['-d', rviz_config_path],
    )

    return LaunchDescription([
        # 기본값: handpose 함께 실행 + 1280x720@30
        DeclareLaunchArgument('with_handpose', default_value='true'),
        DeclareLaunchArgument('image_width',   default_value='1280'),
        DeclareLaunchArgument('image_height',  default_value='720'),
        DeclareLaunchArgument('camera_fps',    default_value='30'),

        handpose_launch,
        retarget_node,
        # ik_solver_node,
        rviz_node,
    ])
