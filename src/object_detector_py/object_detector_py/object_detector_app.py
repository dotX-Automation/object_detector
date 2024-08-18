#!/usr/bin/env python

import sys
import rclpy
from rclpy.executors import SingleThreadedExecutor
from object_detector_py.object_detector_node import ObjectDetectorNode

def main():
    # Initialize ROS 2 context and node
    rclpy.init(args=sys.argv)
    object_detector_node = ObjectDetectorNode()

    executor = SingleThreadedExecutor()
    executor.add_node(object_detector_node)

    # Run the node
    try:
        executor.spin()
    finally:
        executor.shutdown()
        object_detector_node.cleanup()
        object_detector_node.destroy_node()

if __name__ == '__main__':
    main()
