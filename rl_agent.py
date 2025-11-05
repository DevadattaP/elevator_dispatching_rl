from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from elevator_env import ElevatorEnv

def train_rl_agent(verbose: int = 0):
    # Create environment with smaller parameters for testing
    env = ElevatorEnv(
        num_floors=10,           # Smaller for faster testing
        num_elevators=4,        # Fewer elevators
        episode_length=3600,     # 10 minute episodes for testing
        headless=True,
        passenger_generation_rate=5.0,
        verbose=verbose-1  # if want all logs from elevator-building-passangers, verbose=2, if only want rl_agent logs, verbose=1, no logs verbose=0
    )
    if verbose > 0:
        print("Checking environment...")
    try:
        check_env(env)
        if verbose > 0:
            print("Environment check passed!")
    except Exception as e:
        if verbose > 0:
            print(f"Environment check failed: {e}")
        # Continue anyway for debugging
    
    # Test reset
    if verbose > 0:
        print("Testing reset...")
    obs, info = env.reset()
    if verbose > 0:
        print(f"Initial observation shape: {obs.shape}")
    if verbose > 0:
        print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test step
    if verbose > 0:
        print("Testing step...")
    obs, reward, terminated, truncated, info = env.step(0)
    if verbose > 0:
        print(f"Step observation shape: {obs.shape}")
        print(f"Reward: {reward}, Terminated: {terminated}")
    
    # Create RL model
    if verbose > 0:
        print("Creating PPO model...")
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=3e-4,
        n_steps=512,           # Smaller for testing
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        verbose=1 if verbose > 0 else 0
    )
    
    # Train the model
    if verbose > 0:
        print("Starting training...")
    model.learn(total_timesteps=10000)  # Small for testing
    
    # Save the model
    model.save("elevator_ppo_test")
    if verbose > 0:
        print("Training completed and model saved!")
    
    return model, env

def evaluate_agent(model, env, num_episodes=3, verbose: int = 0):
    """Evaluate the trained agent"""
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        step_count = 0
        
        if verbose > 0:
            print(f"\n=== Evaluation Episode {episode + 1} ===")
        
        while not terminated and step_count < 100:  # Limit steps for evaluation
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 20 == 0:
                if verbose > 0:
                    print(f"Step {step_count}: Reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        if verbose > 0:
            print(f"Episode {episode + 1} completed. Total reward: {total_reward:.2f}")
            print(f"Final stats: {info}")

if __name__ == "__main__":
    # Train and evaluate
    model, env = train_rl_agent(verbose=1)
    
    # Evaluate
    print("\n=== Starting Evaluation ===")
    evaluate_agent(model=model, env=env, num_episodes=3, verbose=1)
    
    env.close()