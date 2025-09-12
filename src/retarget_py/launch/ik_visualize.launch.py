# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
# from launch_ros.actions import Node
# from launch_ros.substitutions import FindPackageShare
# from launch_ros.parameter_descriptions import ParameterValue

# def generate_launch_description():
#     L1 = LaunchConfiguration('L1')
#     L2 = LaunchConfiguration('L2')
#     base_frame = LaunchConfiguration('base_frame')
#     eef_frame  = LaunchConfiguration('eef_frame')  # tip_frame 대체

#     xacro_file = PathJoinSubstitution([
#         FindPackageShare('retarget_py'),
#         'urdf',
#         'three_dof_manipulator.urdf.xacro',
#     ])

    
#     robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

#     rsp = Node(
#         package='robot_state_publisher',
#         executable='robot_state_publisher',
#         parameters=[robot_description],
#         output='screen',
#     )

#     ik_node = Node(
#         package='retarget_py',
#         executable='ik_solver_node',    # 요청 반영
#         name='ik_solver_node',          # 요청 반영
#         output='screen',
#         parameters=[{
#             'base_frame': base_frame,
#             'tip_frame':  eef_frame,    # 노드 파라메터 tip_frame에 eef_frame 값 매핑
#             'link1': L1,
#             'link2': L2,
#             'tool_offset_local': [0.0, 0.0, 0.0],
#             'joint_names': ['joint1','joint2','joint3'],
#             'output_topic': 'joint_states',
#             'rate_hz': 100.0,
#         }]
#     )

#     static_tf = Node(
#         package='tf2_ros',
#         executable='static_transform_publisher',
#         name='arm_base_anchor',
#         arguments=[
#             '0','0','0','0','0','0',    # xyz rpy (humble에서 rpy/quat 둘 다 지원)
#             base_frame, 'arm_base'
#         ]
#     )

#     rviz = Node(
#         package='rviz2',
#         executable='rviz2',
#         arguments=['-d', PathJoinSubstitution([FindPackageShare('retarget_py'), 'config', 'rviz_config.rviz'])],
#         output='screen'
#     )

#     return LaunchDescription([
#         DeclareLaunchArgument('L1', default_value='0.05'),
#         DeclareLaunchArgument('L2', default_value='0.05'),
#         DeclareLaunchArgument('base_frame', default_value='gripper_finger1_mcp'),
#         DeclareLaunchArgument('eef_frame',  default_value='gripper_finger1_tip'),

#         rsp,
#         ik_node,
#         static_tf,   # arm_base == base_frame이면 제거 가능
#         rviz,
#     ])




from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # --- args ---
    ns         = LaunchConfiguration('ns')          # ★ 네임스페이스
    L1         = LaunchConfiguration('L1')
    L2         = LaunchConfiguration('L2')
    base_frame = LaunchConfiguration('base_frame')
    eef_frame  = LaunchConfiguration('eef_frame')

    xacro_file = PathJoinSubstitution([
        FindPackageShare('retarget_py'),
        'urdf', 'three_dof_manipulator.urdf.xacro',
    ])

    # # ★ 공백 토큰 없이 각 항목을 분리해서 전달
    # robot_description_content = Command([
    #     'xacro', xacro_file,
    #     'L1:=', L1,
    #     'L2:=', L2,
    # ])
    
    # ✅ 공백 포함해서 전달 (자동으로 안 넣어줌)
    robot_description_content = Command([
        'xacro', ' ',              # 끝에 공백
        xacro_file,            # 경로
        ' ', 'L1:=', L1,           # 앞에 공백
        ' ', 'L2:=', L2,           # 앞에 공백
    ])
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    # 네임스페이스 그룹
    grouped = GroupAction([
        PushRosNamespace(ns),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[robot_description],
            # IK에서 'joint_states'로 퍼블리시하므로 리맵 불필요 (같은 ns 안에서 자동 매칭)
            output='screen',
        ),

        Node(
            package='retarget_py',
            executable='ik_solver_node',
            name='ik_solver_node',
            output='screen',
            parameters=[{
                'base_frame': base_frame,
                'tip_frame':  eef_frame,      # eef_frame 값을 tip_frame으로 전달
                'link1': L1,
                'link2': L2,
                'tool_offset_local': [0.0, 0.0, 0.0],
                'joint_names': ['joint1','joint2','joint3'],
                'output_topic': 'joint_states',  # 상대 토픽 → ns가 자동으로 붙음
                'rate_hz': 100.0,
            }],
        ),

        # world(또는 원하는 상위 프레임) ← arm_base 고정 부착
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='arm_base_anchor',
            arguments=['0','0','0','0','0','0', base_frame, 'arm_base'],
        ),
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='arm_base_anchor',
        #     arguments=['0','0','0','0','0','0', 'world', 'arm_base'],
        # ),

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', PathJoinSubstitution([FindPackageShare('retarget_py'), 'config', 'rviz_config.rviz'])],
            output='screen',
        ),
    ])

    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='finger_3dof'),                     # ★ ns 인자
        DeclareLaunchArgument('L1', default_value='0.05'),
        DeclareLaunchArgument('L2', default_value='0.05'),
        DeclareLaunchArgument('base_frame', default_value='gripper_finger1_mcp'),
        DeclareLaunchArgument('eef_frame',  default_value='gripper_finger1_tip'),
        grouped,
    ])
