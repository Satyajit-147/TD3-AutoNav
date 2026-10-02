import numpy as np
import random
from collections import deque

class MazeGenerator:
    def __init__(self, size=10):
        self.size = size # 10x10 grid
        # Map physical coordinates (-5 to 5) to grid indices (0 to 9)
        self.offset = size / 2.0 
        
    def generate_maze(self, difficulty=0.0):
        """
        Generates a 10x10 occupancy grid. 0 = Free, 1 = Wall.
        difficulty (0.0 to 1.0) controls obstacle density.
        At difficulty=0.0 the room is completely empty (just boundary walls).
        """
        grid = np.zeros((self.size, self.size), dtype=np.int32)
        
        if difficulty <= 0.01:
            # Completely empty room — robot learns goal-seeking first
            return grid
        
        # Obstacle density scales linearly with difficulty
        density = difficulty * 0.35
        
        for i in range(self.size):
            for j in range(self.size):
                # Don't place obstacles in the corners to ensure valid spawns
                if (i <= 1 and j <= 1) or (i >= self.size-2 and j >= self.size-2):
                    continue
                # Keep a clear ring around edges so robot can always navigate boundary
                if i == 0 or i == self.size-1 or j == 0 or j == self.size-1:
                    continue
                    
                if random.random() < density:
                    grid[i, j] = 1
                    
        return grid
        
    def get_cell_index(self, x, y):
        """Map continuous physical coordinates to discrete grid indices."""
        i = int(np.floor(x + self.offset))
        j = int(np.floor(y + self.offset))
        
        # Clamp to grid bounds
        i = max(0, min(self.size - 1, i))
        j = max(0, min(self.size - 1, j))
        return (i, j)
        
    def get_physical_center(self, i, j):
        """Map discrete grid indices to continuous physical coordinates (center of cell)."""
        x = (i - self.offset) + 0.5
        y = (j - self.offset) + 0.5
        return (x, y)
        
    def compute_distance_field(self, grid, goal_i, goal_j):
        """
        Runs BFS from the goal to compute the shortest path distance to all reachable free cells.
        Returns a 2D array where unreachable/wall cells are set to infinity.
        """
        distance_field = np.full((self.size, self.size), np.inf)
        
        if grid[goal_i, goal_j] == 1:
            return distance_field # Goal is inside a wall (should not happen)
            
        queue = deque([(goal_i, goal_j)])
        distance_field[goal_i, goal_j] = 0.0
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            curr_i, curr_j = queue.popleft()
            curr_dist = distance_field[curr_i, curr_j]
            
            for di, dj in directions:
                ni, nj = curr_i + di, curr_j + dj
                
                # Check bounds
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    # Check if free space and not visited
                    if grid[ni, nj] == 0 and np.isinf(distance_field[ni, nj]):
                        distance_field[ni, nj] = curr_dist + 1.0 # 1 meter per cell
                        queue.append((ni, nj))
                        
        return distance_field
