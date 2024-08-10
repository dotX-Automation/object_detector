"""
Object Detector Python launch file.

August 7, 2023
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()

    # Build config file path
    config = os.path.join(
        get_package_share_directory('object_detector'),
        'config',
        'object_detector_py.yaml'
    )

    # Declare launch arguments
    ns = LaunchConfiguration('namespace')
    cf = LaunchConfiguration('cf')
    ns_launch_arg = DeclareLaunchArgument(
        'namespace',
        default_value=''
    )
    cf_launch_arg = DeclareLaunchArgument(
        'cf',
        default_value=config
    )
    ld.add_action(ns_launch_arg)
    ld.add_action(cf_launch_arg)

    # Create node launch description
    node = Node(
        package='object_detector',
        executable='object_detector_py_app.py',
        namespace=ns,
        emulate_tty=True,
        output='both',
        log_cmd=True,
        parameters=[cf],
        remappings=[
            ('/image', '/zed/zed_driver/left/sd/image_rect_color'),
            ('/depth_distances', '/zed/zed_driver/depth_distances'),
            ('/detections', '/detections'),
            ('/detections_stream', '/detections_stream'),
        ]
    )

    ld.add_action(node)

    return ld
