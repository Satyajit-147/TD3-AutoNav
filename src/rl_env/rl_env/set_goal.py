import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

def main(args=None):
    if len(sys.argv) != 3:
        print("Usage: ros2 run rl_env set_goal <x> <y>")
        print("Example: ros2 run rl_env set_goal 4.0 -3.0")
        sys.exit(1)
        
    try:
        target_x = float(sys.argv[1])
        target_y = float(sys.argv[2])
    except ValueError:
        print("Error: Coordinates must be numbers.")
        sys.exit(1)

    rclpy.init(args=args)
    node = Node('set_goal_node')
    pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
    
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = target_x
    msg.pose.position.y = target_y
    msg.pose.position.z = 0.0
    msg.pose.orientation.w = 1.0 # Default upright
    
    # Wait a bit for the subscriber to connect
    import time
    time.sleep(0.5)
    
    pub.publish(msg)
    print(f"Goal ({target_x}, {target_y}) published to /goal_pose!")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
