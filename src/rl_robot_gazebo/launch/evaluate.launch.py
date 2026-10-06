import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot_desc = get_package_share_directory('rl_robot_description')
    
    # Read world path from environment
    world = os.environ.get('NAV_EVAL_WORLD', '')
    map_npy = os.environ.get('NAV_EVAL_MAP', '')
    xacro_file = os.path.join(pkg_robot_desc, 'urdf', 'robot.urdf.xacro')

    spawn_x, spawn_y = -4.5, -4.5  # default
    if map_npy and os.path.exists(map_npy):
        import numpy as np
        grid = np.load(map_npy)
        # Find a safe cell (all 8 neighbors are 0)
        found = False
        for i in range(1, 9):
            for j in range(1, 9):
                if grid[i, j] == 0:
                    safe = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if grid[i+di, j+dj] == 1:
                                safe = False
                    if safe:
                        spawn_x = (i - 5.0) + 0.5
                        spawn_y = (j - 5.0) + 0.5
                        found = True
                        break
            if found:
                break

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world, 'verbose': 'false'}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', xacro_file]), 'use_sim_time': True}]
    )

    # Spawn robot in the safe location
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'rl_robot', '-topic', 'robot_description', '-x', str(spawn_x), '-y', str(spawn_y), '-z', '0.01'],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
