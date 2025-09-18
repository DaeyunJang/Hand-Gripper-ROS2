from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('motor_control_py')
    # params = os.path.join(pkg_share, 'config', 'params.yaml')
    params = os.path.join(
        get_package_share_directory('motor_control_py'),
        'config',
        'params.yaml'
        )
    
    dxl_control_node = Node(
        package='motor_control_py',
        executable='dxl_control_node',
        name='dxl_control_node',
        output='screen',
        parameters=[params]
    )
    
    return LaunchDescription([
        dxl_control_node
    ])
