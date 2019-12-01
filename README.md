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

## Usage

Download this repo. It is made for ROS package. You should download in your ROS workspace. In common case, the name of ROS work space is 'catkin_ws'.

```
git clone https://github.com/hongsj235/End_to_end_USV.git

cd catkin_ws/src/e2e_usv/script
```

You can implement this application using 'drive.py'
At first, you need to modify Directories('weight_path', 'image_root') and result file root(line 64) to your roots. In the case of Google Colab, you need to mount data.

```
python drive.py
```

### Training

In training code, you should change your data root if you git clone to other location, not a ROS work space. And you can tune hyper-parameters in training codes.

Training usage is shown as follows:
```
python3 train.py
```

### Evaluation Test

> After training process, you can evaluate trained model in a two ways. First, the real-time simulation by using ROS Gazebo. Second, validation from the valid data. Compare the thruster value from the model with the value I got from joystick. 

#### 1. Real-time simulation
```
~/catkin_ws$ catkin_make                              # Build ROS packages
~/catkin_ws$ roslaunch vrx_gazebo e2e.launch          # Operate WAM-V simulator
~/catkin_ws$ roslaunch e2e_usv drive_ros.launch       # Operate End-to-end learning
```

#### 2. Validation data

You need to change weights_path and image_root of your PC.
After change the roots, enter the follows:
```
python drive.py
```
Then, you can get 'result.txt' which contains training loss and validation loss every epoch.

#### 3. Plot the loss graph
```
python plot.py
```

## Future work

- Make the dataset more robust and reduce training errors created from ships characteristic
- Get more training dataset to train the model in various environment
- Try another structure to operate thrusters by adding velocity and angular node(Control part)
- If it works well in simulator, try in a real world

## References

[1] Nvidia research, [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)  
[2] Self-driving car simulator developed by [Udacity](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) with Unity  
