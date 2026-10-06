#!/bin/bash
set -e

cd /home/satyajit/PPO_nav

cleanup() {
    echo "Cleaning up before exit..."
    killall -9 gzserver gzclient robot_state_publisher 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Rebuild BOTH packages (rl_env has the NAV_EVAL_MAP fix, rl_robot_gazebo has the new launch file)
echo "Building packages..."
colcon build --packages-select rl_robot_gazebo rl_env
source install/setup.bash

    # Make sure old Gazebo instances are dead
    killall -9 gzserver gzclient robot_state_publisher 2>/dev/null || true
    sleep 3

mkdir -p evaluation/results

echo "============================================================"
echo "  TD3 Generalization Test — 10 Maps x 20 Episodes"
echo "============================================================"

for i in {0..9}; do
    echo ""
    echo "=========================================="
    echo ">>> Testing Map $i"
    echo "=========================================="

    export NAV_EVAL_WORLD="/home/satyajit/PPO_nav/evaluation/maps/map_${i}.world"
    export NAV_EVAL_MAP="/home/satyajit/PPO_nav/evaluation/maps/map_${i}.npy"
    export EVAL_MAP_IDX="$i"

    # Verify files exist
    if [ ! -f "$NAV_EVAL_WORLD" ]; then
        echo "ERROR: World file not found: $NAV_EVAL_WORLD"
        continue
    fi
    if [ ! -f "$NAV_EVAL_MAP" ]; then
        echo "ERROR: Map file not found: $NAV_EVAL_MAP"
        continue
    fi

    # Start Gazebo in the background (headless)
    echo "  Starting Gazebo with map_${i}.world..."
    ros2 launch rl_robot_gazebo evaluate.launch.py > evaluation/gazebo_output.log 2>&1 &
    GAZEBO_PID=$!

    # Wait for Gazebo + robot to be fully ready
    echo "  Waiting for simulation physics to initialize (15s)..."
    sleep 15

    # Run evaluation
    echo "  Running 20 episodes..."
    python3 scripts/evaluate_model.py || echo "  WARNING: evaluation exited with error"

    # Clean shutdown
    echo "  Shutting down simulation..."
    kill -INT $GAZEBO_PID 2>/dev/null || true
    sleep 2
    killall -9 gzserver gzclient robot_state_publisher 2>/dev/null || true
    sleep 4
done

echo ""
echo "============================================================"
echo "  All maps evaluated. Generating report..."
echo "============================================================"
echo ""
python3 scripts/generate_report.py
