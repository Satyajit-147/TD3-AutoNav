import os
import glob
from stable_baselines3 import TD3
from rl_env import NavEnv

def main(args=None):
    print("=" * 60)
    print("  TD3 Navigation — Inference Mode")
    print("=" * 60)
    print("Initializing environment (continuous random-goal mode)...")
    env = NavEnv(interactive_mode=True)
    
    # Set an initial random goal
    import random
    goal_cell = random.choice(env.free_cells)
    gx, gy = env.maze_gen.get_physical_center(goal_cell[0], goal_cell[1])
    env.node.latest_goal_x = gx
    env.node.latest_goal_y = gy
    
    # Find the best available model
    model_dir = "./td3_models/"
    model_path = os.path.join(model_dir, "td3_nav_final.zip")
    
    if not os.path.exists(model_path):
        checkpoints = glob.glob(os.path.join(model_dir, "td3_nav_*_steps.zip"))
        if not checkpoints:
            print("ERROR: No saved TD3 models found in ./td3_models/")
            print("Please run the training script first: ros2 run rl_env train")
            env.close()
            return
        checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
        model_path = checkpoints[-1]
        
    print(f"Loading model: {model_path}")
    model = TD3.load(model_path)
    
    print(f"Default goal: ({env.goal_x:.1f}, {env.goal_y:.1f})")
    print("Publish to /goal_pose to set a new goal dynamically.")
    print()
    print("Starting inference loop...")
    
    obs, _ = env.reset()
    
    try:
        while True:
            # Deterministic action (no exploration noise)
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                if info.get('is_success', False):
                    print(f"\n{'='*40}", flush=True)
                    print(f"🎉 SUCCESS! Reached goal ({env.goal_x:.1f}, {env.goal_y:.1f})", flush=True)
                    print(f"{'='*40}\n", flush=True)
                    
                    # Pick a new random goal instead of teleporting the robot
                    goal_cell = random.choice(env.free_cells)
                    gx, gy = env.maze_gen.get_physical_center(goal_cell[0], goal_cell[1])
                    env.node.latest_goal_x = gx
                    env.node.latest_goal_y = gy
                    print(f"🎯 Assigned new random goal: ({gx:.1f}, {gy:.1f})", flush=True)
                    
                elif info.get('is_collision', False):
                    print(f"❌ Collision! Min obstacle dist: {env.min_obstacle_dist:.3f}m", flush=True)
                elif info.get('timeout', False):
                    print(f"⏳ Timeout after {env.MAX_STEPS} steps", flush=True)
                
                # Reset and continue
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nInference stopped.")
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
