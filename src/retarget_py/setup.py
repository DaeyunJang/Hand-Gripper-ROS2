from setuptools import find_packages, setup
import os, glob
package_name = 'retarget_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일 설치 (있다면)
        # ('share/' + package_name + '/launch', ['launch/hand_to_gripper_launch.py']),
        (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # configuration files
        ('share/' + package_name + '/config', ['config/rviz_config.rviz']),
        (os.path.join('share', package_name, 'config'), glob.glob(os.path.join(f'{package_name}', 'config', 'gripper_config.yaml'))),
        (os.path.join('share', package_name, 'urdf'), glob.glob(os.path.join('urdf', 'three_dof_manipulator.urdf.xacro'))),
        # (os.path.join('share', package_name, 'config'), [f'/gripper_config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DaeyunJang',
    maintainer_email='bigyun9375@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'retarget_node = retarget_py.retarget_node:main',
            'dummy_joint_state_publisher = retarget_py.dummy_jointstate_publisher:main',
            'joint_state_preview_publisher = retarget_py.joint_state_preview_publisher:main',
        ],
    },
)
