# train_elevator_rl.py
from agent.elevator_env import ElevatorEnv
from agent.elevator_dqn import ElevatorDQN, ElevatorDDQN, ElevatorTDQN
import matplotlib.pyplot as plt
import numpy as np

def create_environment():
    """Create your advanced elevator environment"""
    return ElevatorEnv(
        num_floors=10,
        num_elevators=4,
        lift_capacity=8,
        speed_multiplier=10.0,
        episode_length=3600,  # 1 hour simulation
        headless=True,
        passenger_generation_rate=1.0,
        observation_type='enhanced',  # Use your enhanced observation
        action_type='assignment',     # Or 'combinatorial' based on your preference
        reward_type='fairness',       # Use fairness reward like Crites & Barto
        use_smdp=True,               # Use SMDP for better performance
        traffic_pattern='all_in_one', # Mixed traffic patterns
        verbose=0
    )

def train_all_algorithms_advanced():
    """Train all algorithms with your advanced environment"""
    
    # Train DQN
    print("=== Training Advanced DQN ===")
    env_dqn = create_environment()
    dqn_agent = ElevatorDQN(
        env_dqn, 
        replay_size=50000, 
        batch_size=64,
        gamma=0.95,
        sync_after=1000,
        lr=0.0001
    )
    dqn_rewards = dqn_agent.learn(timesteps=50000, save_path='elevator_dqn_advanced.pth')
    
    # Train DDQN
    print("\n=== Training Advanced Double DQN ===")
    env_ddqn = create_environment()
    ddqn_agent = ElevatorDDQN(
        env_ddqn,
        replay_size=50000,
        batch_size=64,
        gamma=0.95,
        sync_after=1000,
        lr=0.0001
    )
    ddqn_rewards = ddqn_agent.learn(timesteps=50000, save_path='elevator_ddqn_advanced.pth')
    
    # Train TDQN
    print("\n=== Training Advanced Triple DQN ===")
    env_tdqn = create_environment()
    tdqn_agent = ElevatorTDQN(
        env_tdqn,
        replay_size=50000,
        batch_size=64,
        gamma=0.95,
        sync_after1=1000,
        sync_after2=5000,
        lr=0.0001,
        aggregator='min'  # More conservative estimates
    )
    tdqn_rewards = tdqn_agent.learn(timesteps=50000, save_path='elevator_tdqn_advanced.pth')
    
    # Plot comparison
    plot_training_comparison(dqn_rewards, ddqn_rewards, tdqn_rewards)
    
    print("\n=== Advanced Training Completed ===")
    print("Models saved: elevator_dqn_advanced.pth, elevator_ddqn_advanced.pth, elevator_tdqn_advanced.pth")

def plot_training_comparison(dqn_rewards, ddqn_rewards, tdqn_rewards):
    """Plot training comparison"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw rewards
    plt.subplot(2, 2, 1)
    if dqn_rewards:
        plt.plot(dqn_rewards, alpha=0.7, label='DQN', color='blue')
    if ddqn_rewards:
        plt.plot(ddqn_rewards, alpha=0.7, label='Double DQN', color='green')
    if tdqn_rewards:
        plt.plot(tdqn_rewards, alpha=0.7, label='Triple DQN', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress - Raw Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Moving averages
    plt.subplot(2, 2, 2)
    window = 20
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if dqn_rewards and len(dqn_rewards) >= window:
        plt.plot(moving_average(dqn_rewards, window), label='DQN (MA)', color='blue', linewidth=2)
    if ddqn_rewards and len(ddqn_rewards) >= window:
        plt.plot(moving_average(ddqn_rewards, window), label='Double DQN (MA)', color='green', linewidth=2)
    if tdqn_rewards and len(tdqn_rewards) >= window:
        plt.plot(moving_average(tdqn_rewards, window), label='Triple DQN (MA)', color='red', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title(f'Moving Average Comparison (Window={window})')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Final performance comparison
    plt.subplot(2, 2, 3)
    algorithms = ['DQN', 'Double DQN', 'Triple DQN']
    final_performance = []
    
    if dqn_rewards and len(dqn_rewards) >= 50:
        final_performance.append(np.mean(dqn_rewards[-50:]))
    else:
        final_performance.append(0)
        
    if ddqn_rewards and len(ddqn_rewards) >= 50:
        final_performance.append(np.mean(ddqn_rewards[-50:]))
    else:
        final_performance.append(0)
        
    if tdqn_rewards and len(tdqn_rewards) >= 50:
        final_performance.append(np.mean(tdqn_rewards[-50:]))
    else:
        final_performance.append(0)
    
    colors = ['blue', 'green', 'red']
    bars = plt.bar(algorithms, final_performance, color=colors, alpha=0.7)
    plt.ylabel('Average Final Reward (Last 50 episodes)')
    plt.title('Final Performance Comparison')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_performance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('advanced_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_all_algorithms_advanced()