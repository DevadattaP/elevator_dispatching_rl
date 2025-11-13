# train_elevator_rl.py
import numpy as np
from elevator_rl_env import ElevatorRLEnv
from elevator_dqn import ElevatorDQN, ElevatorDDQN, ElevatorTDQN
import matplotlib.pyplot as plt

def train_elevator_dqn():
    # Create environment
    env = ElevatorRLEnv(num_floors=10, num_elevators=4, capacity=8, max_steps=3600, speed_multiplier=10)
    
    # Train DQN
    print("=== Training DQN ===")
    dqn_agent = ElevatorDQN(
        env, 
        replay_size=10000, 
        batch_size=32,
        gamma=0.95,
        sync_after=100,
        lr=0.0005
    )
    dqn_rewards = dqn_agent.learn(timesteps=100000, save_path='./models/elevator_dqn.pth')
    
    print("\n=== Training Double DQN ===")
    env_ddqn = ElevatorRLEnv(num_floors=10, num_elevators=4, capacity=8, max_steps=3600, speed_multiplier=10)
    ddqn_agent = ElevatorDDQN(
        env_ddqn,
        replay_size=10000,
        batch_size=32,
        gamma=0.95,
        sync_after=100,
        lr=0.0005
    )
    ddqn_rewards = ddqn_agent.learn(timesteps=100000, save_path='./models/elevator_ddqn.pth')
    
    # Train TDQN
    print("\n=== Training Triple DQN ===")
    env_tdqn = ElevatorRLEnv(num_floors=10, num_elevators=4, capacity=8, max_steps=3600, speed_multiplier=10)
    tdqn_agent = ElevatorTDQN(
        env_tdqn,
        replay_size=10000,
        batch_size=32,
        gamma=0.95,
        sync_after1=100,  # More frequent sync for first target
        sync_after2=500,  # Less frequent sync for second target
        lr=0.0005,
        aggregator='min'  # Use min for more conservative estimates
    )
    tdqn_rewards = tdqn_agent.learn(timesteps=100000, save_path='./models/elevator_tdqn.pth')
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot individual rewards
    plt.subplot(2, 1, 1)
    if dqn_rewards:
        plt.plot(dqn_rewards, alpha=0.7, label='DQN')
    if ddqn_rewards:
        plt.plot(ddqn_rewards, alpha=0.7, label='Double DQN')
    if tdqn_rewards:
        plt.plot(tdqn_rewards, alpha=0.7, label='Triple DQN')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Elevator RL Algorithms - Training Progress')
    plt.legend()
    plt.grid(True)
    
    # Plot moving averages
    plt.subplot(2, 1, 2)
    window = 50
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if dqn_rewards and len(dqn_rewards) >= window:
        plt.plot(moving_average(dqn_rewards, window), label='DQN (MA)')
    if ddqn_rewards and len(ddqn_rewards) >= window:
        plt.plot(moving_average(ddqn_rewards, window), label='Double DQN (MA)')
    if tdqn_rewards and len(tdqn_rewards) >= window:
        plt.plot(moving_average(tdqn_rewards, window), label='Triple DQN (MA)')
    
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average Comparison (Window=50)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Training Completed ===")
    print("Models saved: elevator_dqn.pth, elevator_ddqn.pth, elevator_tdqn.pth")
    print("Comparison plot saved: algorithm_comparison.png")

def evaluate_agent(model, env, num_episodes=10):  # Increased episodes for better stats
    """Evaluate a trained RL agent with enhanced metrics."""
    all_stats = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        episode_rewards = []
        
        while not terminated and not truncated:
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            
        # Add episode reward to info
        info['episode_reward'] = sum(episode_rewards)
        info['episode_length'] = len(episode_rewards)
        all_stats.append(info)
        
    # Calculate comprehensive statistics
    avg_completed = np.mean([s['passengers_completed'] for s in all_stats])
    avg_wait = np.mean([s['average_wait_time'] for s in all_stats])
    avg_journey = np.mean([s['average_journey_time'] for s in all_stats])
    avg_reward = np.mean([s['episode_reward'] for s in all_stats])
    
    # Additional metrics from research papers
    max_waits = [s.get('max_wait_time', 0) for s in all_stats]
    fairness_metrics = [s.get('fairness_metric', 0) for s in all_stats]
    
    return {
        "avg_passengers_completed": avg_completed,
        "avg_wait_time": avg_wait,
        "avg_journey_time": avg_journey,
        "avg_episode_reward": avg_reward,
        "max_wait_time": np.mean(max_waits),
        "fairness_metric": np.mean(fairness_metrics),
        "evaluation_episodes": num_episodes
    }

if __name__ == "__main__":
    # train_elevator_dqn()
    dummy_env = ElevatorRLEnv(10, 4, 8, 3600, 10)
    
    for model_name in ['dqn', 'ddqn', 'tdqn']:
        if model_name == 'dqn':
            rl_agent = ElevatorDQN(env=dummy_env,resume=True)
        elif model_name == 'ddqn':
            rl_agent = ElevatorDDQN(env=dummy_env,resume=True)
        elif model_name == 'tdqn':
            rl_agent = ElevatorTDQN(env=dummy_env,resume=True)
        stats = evaluate_agent(rl_agent, dummy_env, num_episodes=10)
        print(f"Evaluation results for {model_name.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            