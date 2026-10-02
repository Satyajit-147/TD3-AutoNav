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
    pkg_robot_gazebo = get_package_share_directory('rl_robot_gazebo')
    
    world = os.path.join(pkg_robot_gazebo, 'worlds', 'navigation_test.world')
    xacro_file = os.path.join(pkg_robot_desc, 'urdf', 'robot.urdf.xacro')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world, 'verbose': 'true'}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', xacro_file]), 'use_sim_time': True}]
    )

    # Spawn robot in a known free corner of the new map (-4.5, -4.5)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'rl_robot', '-topic', 'robot_description', '-x', '-4.5', '-y', '-4.5', '-z', '0.01'],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
