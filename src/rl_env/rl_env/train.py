import os
import glob
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_env import NavEnv


def main(args=None):
    print("=" * 60)
    print("  TD3 Static-Map Navigation Training")
    print("=" * 60)
    print()
    
    # Instantiate the environment
    print("Initializing environment...")
    env = NavEnv()
    
    # Exploration noise: Gaussian, σ=0.1 of action range
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    # Check for existing TD3 checkpoints to resume
    model_dir = "./td3_models/"
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoints = glob.glob(os.path.join(model_dir, "td3_nav_*_steps.zip"))
    
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming training from: {latest_checkpoint}")
        model = TD3.load(
            latest_checkpoint, 
            env=env, 
            tensorboard_log="./td3_nav_tensorboard/",
            action_noise=action_noise,
        )
    else:
        print("Creating new TD3 model...")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Network: [800, 600]")
        print(f"  Buffer: 300k | Batch: 256 | Warmup: 5000")
        print(f"  LR: 3e-4 | γ: 0.99 | τ: 0.005")
        print()
        
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=5_000,       # 5k warmup steps (random actions)
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[800, 600]),
            tensorboard_log="./td3_nav_tensorboard/",
            verbose=1,
        )
    
    # Save a checkpoint every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=model_dir,
        name_prefix='td3_nav'
    )
    
    print("Starting TD3 training (target: 1,000,000 timesteps)...")
    print(f"  Static map: 22 obstacles, 78 free cells")
    print(f"  Randomized spawn/goal each episode")
    print()
    
    try:
        model.learn(
            total_timesteps=1_000_000, 
            callback=checkpoint_callback,
            tb_log_name="td3_run",
            reset_num_timesteps=False,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")
        
    print("Saving final model...")
    model.save(os.path.join(model_dir, "td3_nav_final"))
    
    print("Training complete.")
    env.close()
    
if __name__ == '__main__':
    main()
