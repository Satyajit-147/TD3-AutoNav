#!/usr/bin/env python3
"""Generate 10 random valid test maps for generalization evaluation.

Each map is a 10x10 grid (0=free, 1=obstacle) with:
  - All border cells (row 0, row 9, col 0, col 9) kept free
  - Random interior obstacles with 15-35% density
  - Guaranteed full connectivity (all free cells reachable from each other)

Outputs:
  evaluation/maps/map_N.npy   — numpy array
  evaluation/maps/map_N.world — Gazebo SDF world file
"""

import numpy as np
import os
from collections import deque

SEEDS = [42, 137, 256, 314, 500, 617, 789, 888, 950, 1024]


def check_connectivity(grid):
    """Return (is_connected, num_free_cells) via BFS from the first free cell."""
    free_cells = [(i, j) for i in range(10) for j in range(10) if grid[i, j] == 0]
    if not free_cells:
        return False, 0

    visited = set()
    queue = deque([free_cells[0]])
    visited.add(free_cells[0])

    while queue:
        ci, cj = queue.popleft()
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < 10 and 0 <= nj < 10 and grid[ni, nj] == 0 and (ni, nj) not in visited:
                visited.add((ni, nj))
                queue.append((ni, nj))

    return len(visited) == len(free_cells), len(free_cells)


def generate_valid_map(seed):
    """Generate a random 10x10 map with guaranteed connectivity."""
    rng = np.random.RandomState(seed)
    attempt = 0

    while True:
        grid = np.zeros((10, 10), dtype=np.int32)

        # Random density between 15-35% for variety
        density = rng.uniform(0.15, 0.35)
        for i in range(1, 9):
            for j in range(1, 9):
                if rng.random() < density:
                    grid[i, j] = 1

        connected, num_free = check_connectivity(grid)

        if connected and num_free >= 50:
            return grid

        attempt += 1
        rng = np.random.RandomState(seed + attempt * 7919)


def grid_to_world(grid, filepath):
    """Convert an occupancy grid to a Gazebo .world file."""
    lines = [
        '<?xml version="1.0" ?>',
        '<sdf version="1.6">',
        '  <world name="default">',
        '    <include><uri>model://sun</uri></include>',
        '    <include><uri>model://ground_plane</uri></include>',
        '',
        '    <!-- Boundary walls (10x10m arena, -5 to 5) -->',
        '    <model name="bound_n"><pose>0 5.25 0.5 0 0 0</pose><static>true</static>'
        '<link name="link"><collision name="collision"><geometry><box><size>11 0.5 1</size></box></geometry></collision>'
        '<visual name="visual"><geometry><box><size>11 0.5 1</size></box></geometry>'
        '<material><ambient>0.5 0.5 0.5 1</ambient></material></visual></link></model>',

        '    <model name="bound_s"><pose>0 -5.25 0.5 0 0 0</pose><static>true</static>'
        '<link name="link"><collision name="collision"><geometry><box><size>11 0.5 1</size></box></geometry></collision>'
        '<visual name="visual"><geometry><box><size>11 0.5 1</size></box></geometry>'
        '<material><ambient>0.5 0.5 0.5 1</ambient></material></visual></link></model>',

        '    <model name="bound_e"><pose>5.25 0 0.5 0 0 0</pose><static>true</static>'
        '<link name="link"><collision name="collision"><geometry><box><size>0.5 10 1</size></box></geometry></collision>'
        '<visual name="visual"><geometry><box><size>0.5 10 1</size></box></geometry>'
        '<material><ambient>0.5 0.5 0.5 1</ambient></material></visual></link></model>',

        '    <model name="bound_w"><pose>-5.25 0 0.5 0 0 0</pose><static>true</static>'
        '<link name="link"><collision name="collision"><geometry><box><size>0.5 10 1</size></box></geometry></collision>'
        '<visual name="visual"><geometry><box><size>0.5 10 1</size></box></geometry>'
        '<material><ambient>0.5 0.5 0.5 1</ambient></material></visual></link></model>',
        '',
    ]

    obs_idx = 0
    for i in range(10):
        for j in range(10):
            if grid[i, j] == 1:
                x = (i - 5.0) + 0.5
                y = (j - 5.0) + 0.5
                lines.append(f'    <model name="obs_{obs_idx}">')
                lines.append(f'      <pose>{x} {y} 0.5 0 0 0</pose>')
                lines.append(f'      <static>true</static>')
                lines.append(f'      <link name="link">')
                lines.append(f'        <collision name="collision"><geometry><box><size>1 1 1</size></box></geometry></collision>')
                lines.append(f'        <visual name="visual"><geometry><box><size>1 1 1</size></box></geometry>')
                lines.append(f'          <material><ambient>0.8 0.2 0.1 1</ambient></material></visual>')
                lines.append(f'      </link>')
                lines.append(f'    </model>')
                obs_idx += 1

    lines.extend([
        '',
        '    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">',
        '      <ros><namespace>/</namespace></ros>',
        '      <update_rate>50</update_rate>',
        '    </plugin>',
        '  </world>',
        '</sdf>',
    ])

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return obs_idx


def print_map(grid, idx):
    """Print a visual representation of the map."""
    symbols = {0: '.', 1: '#'}
    print(f"  Map {idx}:")
    for row in grid:
        print("    " + " ".join(symbols[c] for c in row))
    print()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(script_dir)
    maps_dir = os.path.join(workspace, 'evaluation', 'maps')
    os.makedirs(maps_dir, exist_ok=True)

    print("=" * 60)
    print("  Generating 10 Test Maps for Generalization Evaluation")
    print("=" * 60)
    print()
    print(f"  {'Map':>4} | {'Seed':>6} | {'Obstacles':>9} | {'Free Cells':>10} | {'Density':>7}")
    print("  " + "-" * 55)

    for idx, seed in enumerate(SEEDS):
        grid = generate_valid_map(seed)
        num_obs = int(np.sum(grid))
        num_free = 100 - num_obs
        density = num_obs / 64 * 100  # 64 interior cells

        # Save numpy array
        np.save(os.path.join(maps_dir, f'map_{idx}.npy'), grid)

        # Save world file
        world_path = os.path.join(maps_dir, f'map_{idx}.world')
        grid_to_world(grid, world_path)

        print(f"  {idx:>4} | {seed:>6} | {num_obs:>9} | {num_free:>10} | {density:>6.1f}%")

    print()
    print(f"  Maps saved to: {maps_dir}")
    print()

    # Print visual layouts
    print("  Map Layouts (. = free, # = wall):")
    print()
    for idx, seed in enumerate(SEEDS):
        grid = np.load(os.path.join(maps_dir, f'map_{idx}.npy'))
        print_map(grid, idx)


if __name__ == '__main__':
    main()
