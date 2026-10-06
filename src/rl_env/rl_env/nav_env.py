import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
import random
import os

from .ros_node import start_ros_node
from .maze_generator import MazeGenerator

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


# New test map layout
STATIC_MAP = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int32)


class NavEnv(gym.Env):
    """
    TD3 Navigation Environment — Static Map.
    
    The robot trains on a fixed maze layout. Each episode randomizes the
    start and goal positions among the 78 free cells. The BFS geodesic
    distance field (privileged, reward-only) is recomputed per episode
    based on the goal position.
    
    Observation (24-dim):
        [20 LiDAR beams (normalized 0-1)] + [goal_dist_norm, goal_angle_norm, linear_vel, angular_vel]
    
    Action (2-dim, continuous):
        [linear_velocity (-0.5 to 0.5), angular_velocity (-1.0 to 1.0)]
    """
    metadata = {'render_modes': ['human']}
    
    # LiDAR config
    NUM_LIDAR_BEAMS = 20
    LIDAR_MAX_RANGE = 10.0
    
    # Goal config
    GOAL_RADIUS = 0.5
    MAX_STEPS = 500
    
    # Collision config
    COLLISION_DIST = 0.18
    
    # Reward magnitudes (scaled for TD3 critic stability)
    REWARD_GOAL = 100.0
    REWARD_COLLISION = -100.0
    REWARD_STEP = -0.1
    REWARD_GEODESIC_SCALE = 5.0
    REWARD_MOVEMENT_SCALE = 0.5
    REWARD_SPIN_PENALTY = 0.4
    
    # Proximity Penalty (teaches robot to avoid walls BEFORE hitting them)
    SAFE_DIST = 0.4
    REWARD_PROXIMITY_SCALE = 8.0

    def __init__(self, interactive_mode=False):
        super(NavEnv, self).__init__()
        self.interactive_mode = interactive_mode
        self.node, self.ros_thread = start_ros_node()
        self.maze_gen = MazeGenerator(size=10)
        
        # Action space: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0]), 
            high=np.array([0.5, 1.0]), 
            dtype=np.float32
        )
        
        # Observation: 20 LiDAR (normalized) + goal_dist + goal_angle + lin_vel + ang_vel = 24
        self.observation_space = spaces.Box(
            low=np.array([0.0]*self.NUM_LIDAR_BEAMS + [0.0, -1.0, -0.5, -1.0]), 
            high=np.array([1.0]*self.NUM_LIDAR_BEAMS + [1.0,  1.0,  0.5,  1.0]), 
            dtype=np.float32
        )
        
        # Precompute free cells from the static map or loaded map
        eval_map_path = os.environ.get('NAV_EVAL_MAP')
        if eval_map_path and os.path.exists(eval_map_path):
            self.grid = np.load(eval_map_path)
            print(f"[NavEnv] Loaded custom evaluation map from {eval_map_path}")
        else:
            self.grid = STATIC_MAP
            
        self.free_cells = [
            (i, j) for i in range(10) for j in range(10) if self.grid[i, j] == 0
        ]
        
        # Goal state
        self.goal_x = 4.5
        self.goal_y = 4.5
        
        # Episode state
        self.current_step = 0
        self.current_distance = 0.0
        self.min_obstacle_dist = float('inf')
        self.goal_marker_spawned = False

        
        # Geodesic shaping state (privileged, reward-only)
        self.distance_field = None
        self.prev_geodesic_D = float('inf')
        
        # Stats
        self.total_episodes = 0
        self.total_successes = 0

    def _normalize_lidar(self, ranges):
        """Normalize LiDAR readings to [0, 1]. Inverted so 1.0 = immediate obstacle, 0.0 = free space."""
        ranges = np.array(ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), self.LIDAR_MAX_RANGE, ranges)
        ranges = np.where(np.isnan(ranges), self.LIDAR_MAX_RANGE, ranges)
        ranges = np.clip(ranges, 0.0, self.LIDAR_MAX_RANGE)
        
        # Invert the normalization! Neural networks learn much faster when the 
        # "presence" of an obstacle causes a high activation signal (closer to 1.0).
        return 1.0 - (ranges / self.LIDAR_MAX_RANGE)

    def get_observation(self):
        state = self.node.get_state()
        wait_start = time.time()
        while state['odom'] is None or state['scan'] is None:
            if time.time() - wait_start > 10.0:
                print("ERROR: Timeout waiting for /odom and /scan data from Gazebo!", flush=True)
                raise TimeoutError("Gazebo sensor data timeout")
            time.sleep(0.01)
            state = self.node.get_state()
            
        odom = state['odom']
        scan = state['scan']
        
        # Robot pose
        robot_x = odom.pose.pose.position.x
        robot_y = odom.pose.pose.position.y
        qx = odom.pose.pose.orientation.x
        qy = odom.pose.pose.orientation.y
        qz = odom.pose.pose.orientation.z
        qw = odom.pose.pose.orientation.w
        _, _, yaw = euler_from_quaternion(qx, qy, qz, qw)
        
        # Goal vector in robot frame
        dx = self.goal_x - robot_x
        dy = self.goal_y - robot_y
        self.current_distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        rel_angle = angle_to_goal - yaw
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
        
        # Velocities
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        self._last_linear_vel = linear_x
        self._last_angular_vel = angular_z
        
        # LiDAR — 20 beams, front-only 180°
        raw_ranges = np.array(scan.ranges)
        if len(raw_ranges) != self.NUM_LIDAR_BEAMS:
            indices = np.linspace(0, len(raw_ranges)-1, self.NUM_LIDAR_BEAMS, dtype=int)
            raw_ranges = raw_ranges[indices]
        
        lidar_norm = self._normalize_lidar(raw_ranges)
        self.min_obstacle_dist = np.min(np.where(np.isinf(scan.ranges), self.LIDAR_MAX_RANGE, scan.ranges))
        
        # Normalize goal distance (clip to [0,1] — 14m is diagonal of 10x10 arena)
        goal_dist_norm = np.clip(self.current_distance / 14.0, 0.0, 1.0)
        goal_angle_norm = rel_angle / np.pi  # [-1, 1]
        
        obs = np.concatenate((
            lidar_norm,
            [goal_dist_norm, goal_angle_norm, linear_x, angular_z]
        )).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0
        self.node.publish_cmd_vel(0.0, 0.0)
        
        if not self.interactive_mode:
            # Pick random spawn and goal from free cells
            while True:
                goal_cell = random.choice(self.free_cells)
                spawn_cell = random.choice(self.free_cells)
                
                if goal_cell == spawn_cell:
                    continue
                    
                # Ensure the spawn cell is "safe" (all 8 neighbors are free) to prevent spawning too close to blocks
                si, sj = spawn_cell
                is_safe = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = si + di, sj + dj
                        if 0 <= ni < 10 and 0 <= nj < 10:
                            if self.grid[ni, nj] == 1:
                                is_safe = False
                                break
                        else:
                            is_safe = False # boundary walls count as obstacles
                if not is_safe:
                    continue
                
                # Compute BFS distance field from goal (privileged, reward-only)
                self.distance_field = self.maze_gen.compute_distance_field(
                    self.grid, goal_cell[0], goal_cell[1]
                )
                
                # Must be reachable and at least 2 cells apart
                if (not np.isinf(self.distance_field[spawn_cell[0], spawn_cell[1]]) 
                    and self.distance_field[spawn_cell[0], spawn_cell[1]] > 2.0):
                    break
            
            self.goal_x, self.goal_y = self.maze_gen.get_physical_center(goal_cell[0], goal_cell[1])
            spawn_x, spawn_y = self.maze_gen.get_physical_center(spawn_cell[0], spawn_cell[1])
            spawn_yaw = random.uniform(-np.pi, np.pi)
            
            # Teleport robot to spawn
            self.node.teleport_robot(spawn_x, spawn_y, spawn_yaw)
            
            # Wait for physics to settle and clear stale sensor data
            time.sleep(0.5)
            self.node.latest_odom = None
            self.node.latest_scan = None
            # get_observation() will now block until fresh messages arrive

        else:
            # Interactive mode: use dynamic goal if available
            if self.node.latest_goal_x is not None:
                self.goal_x = self.node.latest_goal_x
                self.goal_y = self.node.latest_goal_y
            time.sleep(0.1)
            
        if not self.goal_marker_spawned:
            self.node.spawn_goal_marker(self.goal_x, self.goal_y)
            self.goal_marker_spawned = True
        else:
            self.node.teleport_entity("goal_marker", self.goal_x, self.goal_y)
            
        obs = self.get_observation()
        
        # Initialize geodesic watermark
        if not self.interactive_mode and self.distance_field is not None:
            robot_x = self.node.latest_odom.pose.pose.position.x
            robot_y = self.node.latest_odom.pose.pose.position.y
            curr_i, curr_j = self.maze_gen.get_cell_index(robot_x, robot_y)
            self.prev_geodesic_D = self.distance_field[curr_i, curr_j]
        else:
            self.prev_geodesic_D = self.current_distance
            
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        # Dynamic goal update in interactive mode
        if self.interactive_mode and self.node.latest_goal_x is not None:
            if self.goal_x != self.node.latest_goal_x or self.goal_y != self.node.latest_goal_y:
                self.goal_x = self.node.latest_goal_x
                self.goal_y = self.node.latest_goal_y
                _ = self.get_observation()
                self.prev_geodesic_D = self.current_distance
                print(f"\n---> Goal changed to ({self.goal_x:.2f}, {self.goal_y:.2f})", flush=True)
                if self.goal_marker_spawned:
                    self.node.teleport_entity("goal_marker", self.goal_x, self.goal_y)
                else:
                    self.node.spawn_goal_marker(self.goal_x, self.goal_y)
                    self.goal_marker_spawned = True

        # Apply action
        linear_x = float(np.clip(action[0], -0.5, 0.5))
        angular_z = float(np.clip(action[1], -1.0, 1.0))
        self.node.publish_cmd_vel(linear_x, angular_z)
        time.sleep(0.1)  # 10 Hz control rate
        
        obs = self.get_observation()
        
        # === REWARD COMPUTATION ===
        reward = self.REWARD_STEP  # -1.0 per step
        terminated = False
        truncated = False
        info = {'is_success': False, 'is_collision': False, 'timeout': False}
        
        # 1. Geodesic progress reward (privileged, reward-only)
        if not self.interactive_mode and self.distance_field is not None:
            robot_x = self.node.latest_odom.pose.pose.position.x
            robot_y = self.node.latest_odom.pose.pose.position.y
            curr_i, curr_j = self.maze_gen.get_cell_index(robot_x, robot_y)
            current_D = self.distance_field[curr_i, curr_j]
            
            if not np.isinf(current_D) and not np.isinf(self.prev_geodesic_D):
                geodesic_progress = self.prev_geodesic_D - current_D
                reward += self.REWARD_GEODESIC_SCALE * geodesic_progress
            
            self.prev_geodesic_D = current_D
        else:
            # Straight-line progress for interactive mode
            progress = self.prev_geodesic_D - self.current_distance
            reward += self.REWARD_GEODESIC_SCALE * progress
            self.prev_geodesic_D = self.current_distance
        
        # 2. Movement quality reward
        reward += self.REWARD_MOVEMENT_SCALE * (
            self._last_linear_vel - self.REWARD_SPIN_PENALTY * abs(self._last_angular_vel)
        )
        
        # 2.5 Obstacle Proximity Penalty (Explicit LiDAR feedback)
        # If the laser scan detects an obstacle closer than SAFE_DIST, apply a continuous 
        # penalty that gets worse the closer the robot gets. This forces it to steer away.
        if self.min_obstacle_dist < self.SAFE_DIST:
            proximity_penalty = (self.SAFE_DIST - self.min_obstacle_dist) * self.REWARD_PROXIMITY_SCALE
            reward -= proximity_penalty
            
        # 3. Goal reached
        if self.current_distance < self.GOAL_RADIUS:
            reward += self.REWARD_GOAL
            terminated = True
            info['is_success'] = True
            self.total_successes += 1
            self.total_episodes += 1
            rate = self.total_successes / self.total_episodes * 100
            print(f"\n========================================", flush=True)
            print(f"✅ GOAL REACHED! ({self.goal_x:.1f},{self.goal_y:.1f}) | Ep {self.total_episodes} | Success rate: {rate:.0f}%", flush=True)
            print(f"========================================\n", flush=True)
            
        # 4. Collision
        elif self.min_obstacle_dist < self.COLLISION_DIST:
            reward += self.REWARD_COLLISION
            terminated = True
            info['is_collision'] = True
            self.total_episodes += 1
            print(f"❌ Collision! Min obstacle dist: {self.min_obstacle_dist:.3f}m", flush=True)
            
        # 5. Timeout
        if self.current_step >= self.MAX_STEPS:
            truncated = True
            info['timeout'] = True
            if not terminated:
                self.total_episodes += 1
            
        return obs, reward, terminated, truncated, info

    def close(self):
        self.node.publish_cmd_vel(0.0, 0.0)
        self.node.destroy_node()
