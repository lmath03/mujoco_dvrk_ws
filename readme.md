# dVRK & Franka Panda: Vision and RL Project

This repository contains the source code for my university internship project. It focuses on Robotic Vision (Point Cloud processing and Bounding Box tracking) for the dVRK surgical robot, and Reinforcement Learning (RL) applied to collaborative robots like the Franka Panda.

## How To Build

The workspace is built using `catkin build`. To compile the project on your local machine, open a terminal at the root of your workspace and run:

```bash
cd ~/dvrk_ws
catkin build
source devel/setup.bash
```
# Part 1: dVRK Vision & Peg-Ring Task (CoppeliaSim)
This section controls the Patient Side Manipulator 1 (PSM1) to perform a peg-and-ring task while simultaneously processing a point cloud to track the robot's bounding box during manipulation.

## 1. ROS & dVRK Initialization
Open a terminal and launch the dVRK ROS interface for the PSM1 arm:
``` source devel/setup.bash ```
roslaunch dvrk_robot dvrk_arm_rviz.launch arm:=PSM1
Note: In the dVRK GUI that appears, click "Power On" and then "Home" to initialize the robot.
## 2. Simulator Setup
 - Open CoppeliaSim.
 - Load the scene: latest_dvrk_simple_pegring_scene.ttt.
 - Press Play to start the simulation.

## 3. Execution (Parallel Tasks)
You will need to run the following components in separate terminals:
 - Task A: Peg and Ring Manipulation
Executes the pick-and-place task of the first 3 right rings using the right arm (PSM1).
```Bash
python3 src/simple_peg_ring/script/simple_peg_ring.py
```
 - Task B: Point Cloud & Bounding Box
Applies filters to the point cloud and visualizes the bounding box tracking the PSM1 during motion.

```Bash
roslaunch plumake_vision_module arm_seg2.launch
```
<img width="639" height="395" alt="c16e2665-086c-4803-9e9e-802fbcfd66cd" src="https://github.com/user-attachments/assets/ca33d44c-8425-4810-8911-8a0815f12c9a" />
