#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from detection_msgs.srv import BoundingBoxTo3D
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point, Quaternion, Vector3,
    PoseWithCovariance, TwistWithCovariance
)


class BoundingBoxTo3DClient(Node):
    def __init__(self):
        super().__init__('bounding_box_to_3d_client')
        self.client = self.create_client(BoundingBoxTo3D, '/okvis/bounding_box_to_3d')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
    
    def call_service(self, camera_id, cx, cy, width, height):
        request = BoundingBoxTo3D.Request()
        
        # Set header with specific timestamp
        request.header = Header()
        request.header.stamp = Time()
        request.header.stamp.sec = 1766413398
        request.header.stamp.nanosec = 548205000
        request.header.frame_id = 'world'
        
        # Set camera ID
        request.camera_id = camera_id
        
        # Set detection target
        request.detection_target.camera_id = camera_id
        request.detection_target.cx = float(cx)
        request.detection_target.cy = float(cy)
        request.detection_target.width = float(width)
        request.detection_target.height = float(height)
        
        # Set target pose (odometry)
        request.detection_target.target_pose = Odometry()
        request.detection_target.target_pose.header = Header()
        request.detection_target.target_pose.header.stamp = self.get_clock().now().to_msg()
        request.detection_target.target_pose.header.frame_id = 'camera'
        request.detection_target.target_pose.child_frame_id = 'base_link'
        
        # Set pose (position and orientation)
        request.detection_target.target_pose.pose = PoseWithCovariance()
        request.detection_target.target_pose.pose.pose.position = Point(x=40.0, y=-28.0, z=1.8)
        request.detection_target.target_pose.pose.pose.orientation = Quaternion(x=0.9, y=0.13, z=0.01, w=-0.0066)
        request.detection_target.target_pose.pose.covariance = [0.0] * 36
        
        # Set twist (velocity)
        request.detection_target.target_pose.twist = TwistWithCovariance()
        request.detection_target.target_pose.twist.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        request.detection_target.target_pose.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
        request.detection_target.target_pose.twist.covariance = [0.0] * 36
        
        # Call service
        self.get_logger().info(f'Calling service with camera_id={camera_id}, bbox=({cx}, {cy}, {width}, {height})')
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Service response:')
            self.get_logger().info(f'  Success: {response.success}')
            self.get_logger().info(f'  Position: ({response.position.x}, {response.position.y}, {response.position.z})')
            self.get_logger().info(f'  Distance: {response.distance}')
            if not response.success:
                self.get_logger().error(f'  Error: {response.error_message}')
            return response
        else:
            self.get_logger().error('Service call failed')
            return None


def main(args=None):
    rclpy.init(args=args)
    
    client = BoundingBoxTo3DClient()
    
    # Example call: camera_id=0, bounding box at (100, 100) with size 50x50
    response = client.call_service(
        camera_id=0,
        cx=600.0,
        cy=310.0,
        width=10.0,
        height=10.0
    )
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

