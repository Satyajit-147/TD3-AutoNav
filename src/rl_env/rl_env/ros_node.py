import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
import threading
import math
import numpy as np
import subprocess
import tempfile
import os

class RLEnvNode(Node):
    def __init__(self):
        super().__init__('rl_env_node')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # State variables
        self.latest_odom = None
        self.latest_scan = None
        self.latest_goal_x = None
        self.latest_goal_y = None
        
        # Services
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        self.set_entity_state_client = self.create_client(SetEntityState, '/set_entity_state')

    def teleport_entity(self, name, x, y, z=0.0, yaw=0.0):
        if self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            req = SetEntityState.Request()
            req.state.name = name
            req.state.pose.position.x = float(x)
            req.state.pose.position.y = float(y)
            req.state.pose.position.z = float(z)
            
            # Convert yaw to quaternion
            req.state.pose.orientation.z = math.sin(yaw / 2.0)
            req.state.pose.orientation.w = math.cos(yaw / 2.0)
            
            self.set_entity_state_client.call_async(req)
            return True
        return False
        
    def teleport_robot(self, x, y, yaw=0.0):
        return self.teleport_entity("rl_robot", x, y, 0.0, yaw)

    def spawn_goal_marker(self, x, y):
        sdf_xml = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='goal_marker'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <sphere>
            <radius>0.2</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0 1 0 1</ambient>
          <diffuse>0 1 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            f.write(sdf_xml)
            temp_path = f.name
        
        # Spawn the entity in the background
        cmd = ['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'goal_marker', '-file', temp_path, '-x', str(x), '-y', str(y), '-z', '0.0']
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.get_logger().info(f"Spawned visual goal_marker at ({x:.2f}, {y:.2f}) in Gazebo")

    def odom_callback(self, msg):
        self.latest_odom = msg

    def scan_callback(self, msg):
        self.latest_scan = msg

    def goal_callback(self, msg):
        self.latest_goal_x = msg.pose.position.x
        self.latest_goal_y = msg.pose.position.y
        self.get_logger().info(f"\n{'='*40}\n🎯 NEW GOAL SET FROM /goal_pose: ({self.latest_goal_x:.2f}, {self.latest_goal_y:.2f})\n{'='*40}\n")

    def get_state(self):
        """Returns a snapshot of the latest sensor data"""
        return {
            'odom': self.latest_odom,
            'scan': self.latest_scan
        }
        
    def publish_cmd_vel(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)

    def reset_simulation(self):
        """Calls the Gazebo reset simulation service"""
        # We try to use reset_world first
        if self.reset_world_client.wait_for_service(timeout_sec=1.0):
            req = Empty.Request()
            future = self.reset_world_client.call_async(req)
            # Node is spinning in background thread, so we just wait for the future
            return True
        return False


def start_ros_node():
    """Starts the ROS node in a separate thread and returns the node instance."""
    if not rclpy.ok():
        rclpy.init()
    
    node = RLEnvNode()
    
    # Run spin in a background thread so it doesn't block the gym env step
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    
    return node, thread
