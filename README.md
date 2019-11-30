# End_to_end_USV
End_to_end learning to control autonomous ship(ROS/Gazebo)

## Introduction

In this term project, we applied End-to-end learning method to control autonomous ship. We used monocular camera image in the ship simulation software to predict left and right thruster values. This is an end-to-end approach to apply to autonomous driving.


## Prerequisite

We used Python and Pytorch. To implement this application, other resources and libraries are required.
(* Environment : Ubuntu 18.04, python 3.6.9, pytorch 1.3.1, CUDA 10.1)

1. Robot Operating System [ROS](http://wiki.ros.org/melodic/Installation/Ubuntu)
2. Gazebo and VRx(Virtual Robot X challenge) libraries, developed by ONR(Office of Naval Research). You can install whole program and libraries through [VRx tutorials](https://bitbucket.org/osrf/vrx/wiki/tutorials/SystemSetupInstall)
3. In repo ROS folder, you need to copy 3 files to ROS package in your computer location.
    - Simulator launch file : Copy 'e2e.launch' file to 'launch' folder in 'VRx_gazebo' ROS package. 
    - Simulator world file : Copy 'test.world' file to 'world' folder in 'VRx_gazebo' ROS package.
    - WAM-V model file : Copy 'cs470.urdf.xacro' file to 'urdf' folder in 'wamv_gazebo' ROS package.

## Dataset(다운로드?!)

Consists of image folder and 'data.csv'. In 'data.csv', there is the list of dataset. 
- Data list : Root of image, left_thruster value, right_thruster value

## Usage

Download this repo. It is made for ROS package. You should download in your ROS workspace. In common case, the name of ROS work space is 'catkin_ws'

```
git clone https://github.com/hongsj235/End_to_end_USV.git

cd catkin_ws/src/e2e_usv/script
```

### Training

In training code, you should change your data root if you git clone to other location, not a ROS work space. And you can tune hyper-parameters in training codes.

Training usage is shown as follows:
```
python3 train.py
```

### Evaluation Test

> You can evaluate this model in a two ways, 

Training usage is shown as follows:
```
python3 train.py
```

