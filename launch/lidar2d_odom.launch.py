#!/usr/bin/env python3

# Transpiled version of: https://github.com/introlab/rtabmap_ros/blob/master/rtabmap_demos/launch/demo_hector_mapping.launch
# Used by http://wiki.ros.org/rtabmap_ros/Tutorials/SetupOnYourRobot

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory


def generate_icp_odometry(context):
    # Read launch arguments (which are provided as strings)
    odom_guess = context.perform_substitution(LaunchConfiguration("odom_guess")).lower() == "true"
    p2n = context.perform_substitution(LaunchConfiguration("p2n")).lower() == "true"
    max_range = context.perform_substitution(LaunchConfiguration("max_range"))
    pm = context.perform_substitution(LaunchConfiguration("pm")).lower()

    params = {
        "frame_id": "base_footprint",
        "deskewing": "true",
        "Icp/VoxelSize": "0.05",
        "Icp/RangeMax": max_range,
        "Icp/Epsilon": "0.001",
        "Icp/MaxCorrespondenceDistance": "0.1",
        "Icp/PM": pm,
        "Icp/PMOutlierRatio": "0.85",
        "Odom/Strategy": "0",
        "Odom/GuessMotion": "true",
        "Odom/ResetCountdown": "0",
        "Odom/ScanKeyFrameThr": "0.75",
        "use_sim_time": True,
    }
    if odom_guess:
        params["odom_frame_id"] = "icp_odom"
        params["guess_frame_id"] = "odom"
    else:
        params["Icp/MaxTranslation"] = "0"

    if p2n:
        params["Icp/PointToPlane"] = "true"
        params["Icp/PointToPlaneK"] = "5"
        params["Icp/PointToPlaneRadius"] = "0.3"
    else:
        params["Icp/PointToPlane"] = "false"
        params["Icp/PointToPlaneK"] = "0"
        params["Icp/PointToPlaneRadius"] = "0"

    icp_odometry_node = Node(
        package="rtabmap_odom",
        executable="icp_odometry",
        name="icp_odometry",
        output="screen",
        parameters=[params],
        remappings=[
            ("scan", "/jn0/base_scan"),
            ("odom", "/scanmatch_odom"),
            ("odom_info", "/rtabmap/odom_info")
        ]
    )
    return [icp_odometry_node]


def generate_rtabmap_node(context):
    hector = context.perform_substitution(LaunchConfiguration("hector")).lower() == "true"
    camera = context.perform_substitution(LaunchConfiguration("camera")).lower() == "true"
    max_range = context.perform_substitution(LaunchConfiguration("max_range"))

    params = {
        "frame_id": "base_footprint",
        "subscribe_rgb": False,
        "subscribe_depth": False,
        "subscribe_rgbd": camera,
        "subscribe_scan": True,
        "Reg/Strategy": "1",         # 0=Visual, 1=ICP, 2=Visual+ICP
        "Reg/Force3DoF": "true",
        "RGBD/ProximityBySpace": "true",
        "Icp/CorrespondenceRatio": "0.2",
        "Icp/VoxelSize": "0.05",
        "Icp/RangeMax": max_range,
        "Grid/RangeMax": max_range,
        "use_sim_time": True,
    }
    remappings = [("scan", "/jn0/base_scan")]
    if hector:
        params["odom_frame_id"] = "hector_map"
        params["odom_tf_linear_variance"] = 0.0005
        params["odom_tf_angular_variance"] = 0.0005
    else:
        remappings.append(("odom", "/scanmatch_odom"))
        params["subscribe_odom_info"] = True

    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        arguments=["--delete_db_on_start"],
        parameters=[params],
        remappings=remappings
    )
    return [rtabmap_node]


def generate_rtabmap_viz_node(context):
    hector = context.perform_substitution(LaunchConfiguration("hector")).lower() == "true"
    camera = context.perform_substitution(LaunchConfiguration("camera")).lower() == "true"

    params = {
        "subscribe_rgbd": camera,
        "subscribe_laserScan": True,
        "frame_id": "base_footprint",
        "use_sim_time": True,
    }
    remappings = [("scan", "/jn0/base_scan")]
    if hector:
        params["odom_frame_id"] = "hector_map"
    else:
        remappings.append(("odom", "/scanmatch_odom"))
        params["subscribe_odom_info"] = True

    # Determine the configuration file path (assuming it is installed with the package)
    config_file = os.path.join(
        get_package_share_directory("rtabmap_demos"),
        "launch",
        "config",
        "rgbd_gui.ini"
    )
    rtabmap_viz_node = Node(
        package="rtabmap_viz",
        executable="rtabmap_viz",
        name="rtabmap_viz",
        output="screen",
        arguments=["-d", config_file],
        parameters=[params],
        remappings=remappings
    )
    return [rtabmap_viz_node]


def generate_launch_description():
    # Declare launch arguments
    arg_rviz = DeclareLaunchArgument("rviz", default_value="true", description="Launch RViz")
    arg_rtabmap_viz = DeclareLaunchArgument("rtabmap_viz", default_value="false", description="Launch RTAB-Map visualization")
    arg_hector = DeclareLaunchArgument("hector", default_value="true", description="Use Hector mapping for odometry")
    arg_odom_guess = DeclareLaunchArgument("odom_guess", default_value="false",
                                             description="Feed wheel odometry as guess to icp_odometry")
    arg_camera = DeclareLaunchArgument("camera", default_value="true", description="Use camera")
    arg_max_range = DeclareLaunchArgument("max_range", default_value="0",
                                          description="Limit lidar range if > 0 (0 means unlimited)")
    arg_p2n = DeclareLaunchArgument("p2n", default_value="true", description="Use Point-to-Plane ICP")
    arg_pm = DeclareLaunchArgument("pm", default_value="true", description="Use libpointmatcher for ICP")

    # Static transform publisher (only if using Hector mapping)
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="scanmatcher_to_base_footprint",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "/scanmatcher_frame", "/base_footprint", "100"],
        condition=IfCondition(LaunchConfiguration("hector"))
    )

    # Hector mapping node (if hector is true)
    hector_mapping_node = Node(
        package="hector_mapping",
        executable="hector_mapping",
        name="hector_mapping",
        output="screen",
        parameters=[{
            "map_frame": "hector_map",
            "base_frame": "base_footprint",
            "odom_frame": "odom",
            "pub_map_odom_transform": False,
            "pub_map_scanmatch_transform": True,
            "pub_odometry": True,
            "map_resolution": 0.050,
            "map_size": 2048,
            "map_multi_res_levels": 2,
            "map_update_angle_thresh": 0.06,
            "scan_topic": "/jn0/base_scan",
            "use_sim_time": True,
        }],
        condition=IfCondition(LaunchConfiguration("hector"))
    )

    # ICP odometry node (launched only if hector is false)
    icp_odometry_node = OpaqueFunction(
        function=generate_icp_odometry,
        condition=UnlessCondition(LaunchConfiguration("hector"))
    )

    # Group for RTAB-Map nodes in the "rtabmap" namespace
    rtabmap_group = GroupAction(
        actions=[
            PushRosNamespace("rtabmap"),
            # RGBD sync node (if camera is true)
            Node(
                package="nodelet",
                executable="nodelet",
                name="rgbd_sync",
                output="screen",
                arguments=["standalone", "rtabmap_sync/rgbd_sync"],
                remappings=[
                    ("rgb/image", "/data_throttled_image"),
                    ("depth/image", "/data_throttled_image_depth"),
                    ("rgb/camera_info", "/data_throttled_camera_info"),
                ],
                parameters=[{
                    "rgb/image_transport": "compressed",
                    "depth/image_transport": "compressedDepth",
                    "use_sim_time": True,
                }],
                condition=IfCondition(LaunchConfiguration("camera"))
            ),
            # RTAB-Map SLAM node (common to both Hector and ICP odometry cases)
            OpaqueFunction(function=generate_rtabmap_node),
            # RTAB-Map visualization node (if enabled)
            OpaqueFunction(
                function=generate_rtabmap_viz_node,
                condition=IfCondition(LaunchConfiguration("rtabmap_viz"))
            ),
        ]
    )

    # RViz node (if enabled)
    rviz_config = os.path.join(
        get_package_share_directory("rtabmap_demos"),
        "launch",
        "config",
        "demo_robot_mapping.rviz"
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("rviz"))
    )

    # Additional point cloud node for RGB-D (if camera is true)
    points_node = Node(
        package="nodelet",
        executable="nodelet",
        name="points_xyzrgb",
        output="screen",
        arguments=["standalone", "rtabmap_util/point_cloud_xyzrgb"],
        remappings=[
            ("rgbd_image", "/rtabmap/rgbd_image"),
            ("cloud", "voxel_cloud"),
        ],
        parameters=[{
            "voxel_size": 0.01,
            "use_sim_time": True,
        }],
        condition=IfCondition(LaunchConfiguration("camera"))
    )

    return LaunchDescription([
        arg_rviz,
        arg_rtabmap_viz,
        arg_hector,
        arg_odom_guess,
        arg_camera,
        arg_max_range,
        arg_p2n,
        arg_pm,
        static_tf,
        hector_mapping_node,
        icp_odometry_node,
        rtabmap_group,
        rviz_node,
        points_node,
    ])


if __name__ == "__main__":
    generate_launch_description()
