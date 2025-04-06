#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os 

def generate_launch_description():
    # Get the package share directory of rtabmap_ros
    pkg_share = get_package_share_directory('rtabmap_ros')
    # Default path to the parameters file for 2D lidar (make sure to create/modify this file as needed)
    default_params_file = os.path.join(pkg_share, 'config', 'rtabmap_2d.yaml')

    # Declare the parameter file launch argument
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS2 parameters file to use for rtabmap (2D lidar config)'
    )

    # rtabmap node configuration for 2D lidar
    rtabmap_node = Node(
        package='rtabmap_ros',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                # Enable subscription to 2D laser scans
                'subscribe_scan': True,
                # Disable 3D inputs
                'subscribe_depth': False,
                'subscribe_rgbd': False,
                'subscribe_stereo': False,
            }
        ],
        # Remap topics if necessary (e.g., if your laser publishes on a different topic)
        remappings=[
            ('/scan', '/scan'),
            ('/odom', '/vesc/odom')
        ]
    )

    # Create and return the launch description
    return LaunchDescription([
        declare_params_file_cmd,
        rtabmap_node,
    ])