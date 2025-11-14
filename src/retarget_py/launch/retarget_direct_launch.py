from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os
import xacro

def generate_launch_description():
    # ---- Launch args (필요시 override) ----
    with_handpose   = LaunchConfiguration('with_handpose')
    image_width     = LaunchConfiguration('image_width')
    image_height    = LaunchConfiguration('image_height')
    camera_fps      = LaunchConfiguration('camera_fps')

    pkg = get_package_share_directory('retarget_py')
    xacro_path = os.path.join(pkg, 'urdf', 'hand_gripper.urdf.xacro')
    robot_description = xacro.process_file(xacro_path).toxml()
    
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
    rviz_config_sideview_path = PathJoinSubstitution([
        FindPackageShare('retarget_py'),
        'config',
        'rviz_config_sideview.rviz'
    ])

    retarget_node = Node(
        package='retarget_py',
        executable='retarget_node',
        name='retarget_node',
        output='screen',
        parameters=[{
            "gripper_name": "hand_gripper",
            "source_root": "camera_color_optical_frame",
            # "scale": 0.48,
            "warning_log_flag": False,
        }],
        remappings=[("/joint_states", "/isaacsim_joint_states")],
    )
    
    retarget_direct_node = Node(
        package='retarget_py',
        executable='retarget_direct_node',
        name='retarget_direct_node',
        output='screen',
        parameters=[{
            # "source_root": "camera_color_optical_frame",
            # "scale": 0.48,
            # "warning_log_flag": False,
        }],
        # remappings=[("/joint_states", "/isaacsim_joint_states")],
    )
    
    joint_state_preview_node = Node(
        package='retarget_py',
        executable='joint_state_preview_publisher',
        name='joint_state_preview_publisher',
        output='screen',
        parameters=[{
            'cmd_topic': '/hand_gripper/joint_states_cmd',
            'out_topic': '/hand_gripper/joint_states_preview'
        }]
    )
    
    joint_state_node = Node(
        package='retarget_py',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'cmd_topic': '/hand_gripper/joint_states_cmd',
            # 'cmd_topic': '/hand_gripper/joint_states_direct',
            'out_topic': '/joint_command'
        }]
    )
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='gripper_robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        # ✅ joint_states 구독을 프리뷰 토픽으로 리맵
        remappings=[('joint_states', '/hand_gripper/joint_states_preview')],
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
    rviz_sideview_node = Node(
        package='rviz2',
        executable='rviz2',
        name='handpose_rviz_sideview',
        output='screen',
        arguments=['-d', rviz_config_sideview_path],
    )

    return LaunchDescription([
        # 기본값: handpose 함께 실행 + 1280x720@30
        DeclareLaunchArgument('with_handpose', default_value='true'),
        DeclareLaunchArgument('image_width',   default_value='1280'),
        DeclareLaunchArgument('image_height',  default_value='720'),
        DeclareLaunchArgument('camera_fps',    default_value='30'),

        handpose_launch,
        # retarget_node,
        retarget_direct_node,
        
        # ik_solver_node,
        rviz_node,
        rviz_sideview_node,
        joint_state_preview_node,
        joint_state_node,
        # midas_joint_state_node,
        robot_state_publisher,
    ])
