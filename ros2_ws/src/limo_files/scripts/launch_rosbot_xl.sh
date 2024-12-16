#! /usr/bin/env bash

# We remove a folder that otherwise gives issues in ROS2 launches
sudo rm -r /home/user/.ros

# Check if the first argument is 'debug'
if [ "$1" = "debug" ]; then
    export ROS2_WS_PATH=/home/user/ros2_ws
else
    export ROS2_WS_PATH=/home/simulations/ros2_sims_ws
fi

# We set up the environment for ROS2
. /usr/share/gazebo/setup.sh
. ${ROS2_WS_PATH}/install/setup.bash

cp ${ROS2_WS_PATH}/src/rosbot_xl_simulation/rosbot_xl_ros/rosbot_xl_description/rviz/rosbot_laser_scan.rviz /home/user
export GAZEBO_RESOURCE_PATH=${ROS2_WS_PATH}/src/rosbot_xl_simulation/rosbot_xl_ros/rosbot_xl_description:${GAZEBO_RESOURCE_PATH}
export GAZEBO_MODEL_PATH=${ROS2_WS_PATH}/src/rosbot_xl_simulation/rosbot_xl_ros/rosbot_xl_description:${GAZEBO_MODEL_PATH}
export GAZEBO_MODEL_DATABASE_URI=""
ros2 launch rosbot_xl_description rosbot_xl_basic.launch.xml