from setuptools import find_packages, setup

package_name = 'motor_control_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
        ('share/' + package_name + '/launch', ['launch/motor_control.launch.py']),
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
            'dxl_control_node = motor_control_py.dxl_control_node:main',
        ],
    },
)
