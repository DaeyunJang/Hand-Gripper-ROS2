# Hand-Gripper-ROS2
Custom hand type gripper control using ROS2 (also, standalone)
Teleoperation system for using RGB camera(realsense).

## DEMO
https://github.com/user-attachments/assets/28e03374-c446-4642-b8eb-b5e198755ad9

- Hand landmark is normalized from mediapipe.
- Using TF ROS system for make target transfrom matrix for each gripper finger.
- Retarget(Remap) scale of length of fingers between human and gripper.
- Inverse kinematics for finger manipulator
- Dyanmixel motor is used.


---
## Index
1. [Prerequisite](#prerequisite)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Execute](#execute)

## Prerequisite

#### OS
Ubuntu 22.04.6 LTS

#### Software
- [ROS2 humble](https://docs.ros.org/en/humble/index.html)
- [Hand landmarks detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

## System Requirements
- Ubuntu 22.04 (ROS2 humble)
- Realsense camera (D415)

## Installation
#### virtual environment
[Using Python Packages with ROS 2](https://docs.ros.org/en/humble/How-To-Guides/Using-Python-Packages.html#installing-via-a-virtual-environment)
```bash
source {PATH_OF_YOUR_VIRTUAL_ENV}/bin/activate
```
#### python modules
```bash
  $ pip install -r requirements.txt
```
(site-packages) If you want to check the ROS2 packages, check the `requirements_site-packages.txt`.

## Execute
#### Clone repositories
`HandPose` packages is required for run this project.
```bash
cd ~
git clone https://github.com/DaeyunJang/HandPose.git
git clone https://github.com/DaeyunJang/Hand-Gripper-ROS2.git
```

#### Build and source
```bash
  # activate python environment
  source {PATH_OF_YOUR_VIRTUAL_ENV}/bin/activate

  # build dependent package
  cd ~/HandPose
  colcon build
  ./install/setup.bash # must be enter this line befor build 'Hand-Gripper-ROS2' package.

  # build package
  cd ~/Hand-Gripper-ROS2
  colcon build
  ./install/setup.bash
```

#### Launch
```bash
  ros2 launch retarget_py retarget_launch.py
```

