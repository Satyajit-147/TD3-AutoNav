#!/usr/bin/env python3
import os
import sys
import json
import rclpy
from stable_baselines3 import TD3
import warnings

# Suppress Gym warnings for cleaner output
warnings.filterwarnings("ignore")

from rl_env.nav_env import NavEnv

def evaluate():
    rclpy.init()
    
    map_idx = os.environ.get('EVAL_MAP_IDX', 'unknown')
    num_episodes = 20
    
    print(f"============================================================")
    print(f"  Evaluating Model on Map {map_idx} ({num_episodes} episodes)")
    print(f"============================================================")

    # Initialize Environment
    env = NavEnv(interactive_mode=False)
    
    # Load Model
    model_path = "./td3_models/td3_nav_final.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = TD3.load(model_path, env=env)
    
    results = {
        'map_idx': map_idx,
        'episodes': num_episodes,
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'total_steps': 0,
        'success_steps': []
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            
            if done:
                if info.get('is_success'):
                    results['success'] += 1
                    results['success_steps'].append(steps)
                    print(f"  Ep {ep+1:2d}/{num_episodes} | ✅ SUCCESS   | Steps: {steps:3d}")
                elif info.get('is_collision'):
                    results['collision'] += 1
                    print(f"  Ep {ep+1:2d}/{num_episodes} | ❌ COLLISION | Steps: {steps:3d}")
                else:
                    results['timeout'] += 1
                    print(f"  Ep {ep+1:2d}/{num_episodes} | ⏳ TIMEOUT   | Steps: {steps:3d}")
                    
        results['total_steps'] += steps
        
    # Save results
    results_dir = "evaluation/results"
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"results_map_{map_idx}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults for Map {map_idx}:")
    print(f"  Success:   {results['success']}/{num_episodes} ({(results['success']/num_episodes)*100:.1f}%)")
    print(f"  Collision: {results['collision']}/{num_episodes} ({(results['collision']/num_episodes)*100:.1f}%)")
    print(f"  Timeout:   {results['timeout']}/{num_episodes} ({(results['timeout']/num_episodes)*100:.1f}%)")
    
    env.close()
    try:
        rclpy.shutdown()
    except:
        pass

if __name__ == '__main__':
    evaluate()
