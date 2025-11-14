from setuptools import find_packages, setup
import os, glob

package_name = 'dynamixel_control_py'

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
        (os.path.join('share', package_name, 'config'), ['config/control_table.json']),
        (os.path.join('share', package_name, 'config', 'midas'), glob.glob(os.path.join('config', 'midas', '*'))),    # rviz setting & gripper config
        (os.path.join('share', package_name, 'config', 'patrol'), glob.glob(os.path.join('config', 'patrol', '*'))),    # rviz setting & gripper config
        # install config files so package can find them when imported from installed location
        # (os.path.join('share', package_name, 'config'), [
        #     'src/dynamixel_control_py/config/control_table.json']),
        # (os.path.join('share', package_name, 'config', 'midas'), [
        #     'src/dynamixel_control_py/config/midas/config.json',
        #     'src/dynamixel_control_py/config/midas/motion_table.json',
        # ]),
        # (os.path.join('share', package_name, 'config', 'patrol'), [
        #     'src/dynamixel_control_py/config/patrol/motion_table.json',
        # ]),
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
            'dynamixel_control_node = dynamixel_control_py.dynamixel_control_node:main',
        ],
    },
)
