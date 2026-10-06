#!/bin/bash
cd /home/satyajit/PPO_nav

colcon build --packages-select rl_env
source install/setup.bash

export NAV_EVAL_WORLD="/home/satyajit/PPO_nav/evaluation/maps/map_0.world"
export NAV_EVAL_MAP="/home/satyajit/PPO_nav/evaluation/maps/map_0.npy"
export EVAL_MAP_IDX="0"

pkill -f gzserver 2>/dev/null || true
ros2 launch rl_robot_gazebo evaluate.launch.py > /dev/null 2>&1 &
GAZ_PID=$!

sleep 15
python3 scripts/evaluate_model.py

kill -INT $GAZ_PID
sleep 2
pkill -f gzserver
