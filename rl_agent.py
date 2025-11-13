import time
import traceback
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from elevator_env import (ElevatorEnv, D3QNWrapper, SMDPWrapper, TrafficAwareWrapper, 
                          DiscreteAssignmentWrapper, DiscreteCombinatorialWrapper, MultiDiscreteWrapper, FlattenMultiDiscreteWrapper)
import os
import json
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
np.random.seed(42)

# ===== RL Agent Training =====
def train_rl_agent(model_name: str, observation_type: str, reward_type: str,
                   action_type: str, env_wrapper: str = "default", 
                   traffic_pattern: str = "mixed", use_smdp: bool = False,
                   verbose: int = 0, log_evaluation: bool = False):
    """Creates the environment, trains the specified agent, and returns it."""
    
    # Choose environment wrapper
    wrapper_classes = {
        "default": ElevatorEnv,
        "d3qn": D3QNWrapper,
        "smdp": SMDPWrapper,
        "traffic_aware": TrafficAwareWrapper
    }
    
    env_class = wrapper_classes[env_wrapper]

    # Environment parameters based on research recommendations
    env_kwargs = {
        'num_floors': 10,
        'num_elevators': 4,
        'episode_length': 3600,    # 60 minute episodes
        'headless': True,
        'passenger_generation_rate': 1.0,
        'observation_type': observation_type,  # 'simple', 'detailed', 'enhanced'
        'reward_type': reward_type,            # 'simple', 'complex', 'fairness', 'squared'
        'action_type': action_type,            # 'discrete', 'continuous', 'combinatorial', 'assignment'
        'traffic_pattern': traffic_pattern,    # 'up_peak', 'down_peak', 'mixed', 'all_in_one'
        'use_smdp': use_smdp,                  # Semi-Markov Decision Process
        'verbose': verbose - 1
    }

    # Special handling for wrappers that override certain parameters
    if env_wrapper == "d3qn":
        # D3QN wrapper automatically sets observation_type='enhanced', reward_type='squared'
        env_kwargs.pop('observation_type', None)
        env_kwargs.pop('reward_type', None)
    elif env_wrapper == "smdp":
        env_kwargs['use_smdp'] = True
        env_kwargs['observation_type'] = 'enhanced'
    elif env_wrapper == "traffic_aware":
        env_kwargs['traffic_pattern'] = 'all_in_one'
        env_kwargs['observation_type'] = 'enhanced'

    env = env_class(**env_kwargs)
    
    # Apply action space wrappers for DQN compatibility
    if model_name == "DQN":
        if action_type == "combinatorial":
            env = DiscreteCombinatorialWrapper(env)
        elif action_type == "assignment":
            env = DiscreteAssignmentWrapper(env)
        elif action_type == "discrete":
            env = MultiDiscreteWrapper(env)
        # continuous action types are not supported by DQN

    if log_evaluation:
        eval_env = env_class(**env_kwargs)
        if model_name == "DQN":
            if action_type == "combinatorial":
                eval_env = DiscreteCombinatorialWrapper(eval_env)
            elif action_type == "assignment":
                eval_env = DiscreteAssignmentWrapper(eval_env)
            elif action_type == "discrete":
                eval_env = MultiDiscreteWrapper(eval_env)

    # --- Model Dictionary ---
    model_zoo = {
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN,
        "SAC": SAC,
        "TD3": TD3,
        "DDPG": DDPG,
    }

    model_class = model_zoo[model_name]

    # --- Enhanced Model Configurations based on research papers ---
    policy = "MlpPolicy"
    model_kwargs = {
        "policy": policy,
        "env": env,
        "verbose": 1 if verbose > 0 else 0,
        "device": "cpu",
        "seed": 42,
    }

    # Enhanced hyperparameters based on successful research configurations
    if model_name == "PPO":
        model_kwargs.update({
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
            "ent_coef": 0.01,  # Added for better exploration
        })
    elif model_name == "A2C":
        model_kwargs.update({
            "learning_rate": 3e-4,
            "n_steps": 10,
            "gamma": 0.99,
            "max_grad_norm": 0.5,
            "ent_coef": 0.01,
        })
    elif model_name == "DQN":
        # Enhanced DQN config for elevator problem (like D3QN papers)
        model_kwargs.update({
            "learning_rate": 1e-4,
            "buffer_size": 100000,  # Increased for better replay
            "learning_starts": 5000,  # More initial exploration
            "batch_size": 32,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.2,  # Longer exploration
            "exploration_final_eps": 0.02,  # Lower final epsilon
        })
        # Use Dueling DQN if available (better for elevator problem)
        if hasattr(model_kwargs, 'policy_kwargs'):
            model_kwargs["policy_kwargs"] = dict(net_arch=[256, 256])  # Deeper network
    elif model_name in ["SAC", "TD3", "DDPG"]:
        # Enhanced continuous action algorithms
        n_actions = env.action_space.shape[-1] if hasattr(env.action_space, 'shape') else 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model_kwargs.update({
            "action_noise": action_noise,
            "buffer_size": 200000,  # Larger buffer for continuous control
            "batch_size": 256,  # Larger batch size
            "gamma": 0.99,
            "learning_starts": 10000,  # More initial samples
            "train_freq": (1, "episode"),  # Train every episode
        })
        
        # SAC-specific enhancements
        if model_name == "SAC":
            model_kwargs.update({
                "tau": 0.005,  # Soft update coefficient
                "learning_rate": 3e-4,
            })

    model = model_class(**model_kwargs)

    # Create unique run ID with all parameters
    run_id = f"{model_name}_{env_wrapper}_{observation_type}_{reward_type}_{action_type}_{traffic_pattern}"
    if use_smdp:
        run_id += "_smdp"
        
    save_dir = f"models/{run_id}"
    os.makedirs(save_dir, exist_ok=True)

    if log_evaluation:
        log_dir = f"logs/{run_id}/"
        os.makedirs(log_dir, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=log_dir, 
            log_path=log_dir,
            eval_freq=5000,  # More frequent evaluation
            n_eval_episodes=3,  # Multiple episodes for better stats
            deterministic=True, 
            render=False
        )

    if verbose > 0:
        print(f"Starting training for {run_id}...")
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space}")

    # Increased training timesteps for better convergence
    total_timesteps = 100000  # Increased from 50k for better learning

    if log_evaluation:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    else:
        model.learn(total_timesteps=total_timesteps)

    model.save(f"{save_dir}/{run_id}_model.zip")

    if verbose > 0:
        print(f"Training completed and model for {run_id} saved!")

    return model, env


# ===== Evaluation Functions =====
def evaluate_agent(model, env, num_episodes=10):  # Increased episodes for better stats
    """Evaluate a trained RL agent with enhanced metrics."""
    all_stats = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        episode_rewards = []
        
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
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


def evaluate_rule_based(observation_type="simple", reward_type="simple", 
                       traffic_pattern="mixed", num_episodes=10, verbose: int = 0):
    """Evaluate rule-based elevator logic under given configuration."""
    env = ElevatorEnv(
        num_floors=10,
        num_elevators=4,
        episode_length=3600,
        headless=True,
        passenger_generation_rate=1.0,
        observation_type=observation_type,
        reward_type=reward_type,
        action_type="discrete",  # Rule-based uses discrete actions
        traffic_pattern=traffic_pattern,
        use_smdp=False,
        verbose=verbose - 1
    )

    all_stats = []
    for episode in range(num_episodes):
        env.reset()
        terminated, truncated = False, False
        episode_rewards = []
        
        # The rule-based system runs inside building.step()
        dummy_action = np.zeros(env.num_elevators, dtype=int)
        while not terminated and not truncated:
            _, reward, terminated, truncated, info = env.step(dummy_action)
            episode_rewards.append(reward)
            
        info['episode_reward'] = sum(episode_rewards)
        info['episode_length'] = len(episode_rewards)
        all_stats.append(info)

    env.close()

    # Calculate statistics
    avg_completed = np.mean([s['passengers_completed'] for s in all_stats])
    avg_wait = np.mean([s['average_wait_time'] for s in all_stats])
    avg_journey = np.mean([s['average_journey_time'] for s in all_stats])
    avg_reward = np.mean([s['episode_reward'] for s in all_stats])
    
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


# ===== Utility for Nested Dict Access =====
def nested_set(dic, keys, value):
    """Safely set a nested dictionary value."""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def nested_get(dic, keys):
    """Safely get a nested dictionary value if it exists."""
    for key in keys:
        if key not in dic:
            return None
        dic = dic[key]
    return dic


# ===== Main Training Logic =====
if __name__ == "__main__":
    VERBOSITY = 0   # 0 for silent, 1 for updates, 2 for detailed
    results_file = "evaluation_results.json"

    # Load existing results
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {results_file}")
    else:
        all_results = {}

    # Enhanced model compatibility based on new action types
    MODEL_COMPATIBILITY = {
        "PPO": ["discrete", "continuous", "combinatorial"],
        "A2C": ["discrete", "continuous", "combinatorial"], 
        "DQN": ["discrete", "combinatorial"],
        "SAC": ["continuous"],
        "TD3": ["continuous"],
        "DDPG": ["continuous"],
    }

    # Enhanced configurations based on research recommendations
    observation_types = ["simple", "detailed", "enhanced"]
    reward_types = ["simple", "complex", "fairness", "squared"]
    traffic_patterns = ["up_peak", "down_peak", "mixed", "all_in_one"]
    env_wrappers = ["default", "d3qn", "smdp", "traffic_aware"]

    # === Rule-based baseline evaluations ===
    print("Evaluating rule-based baselines...")
    for traffic in traffic_patterns:
        key = ["rule_based", "default", "simple", "simple", "discrete", traffic]
        if nested_get(all_results, key) is None:
            print(f"\nEvaluating rule_based for {traffic} pattern")
            start_time = time.time()
            stats = evaluate_rule_based(
                observation_type="simple", 
                reward_type="simple",
                traffic_pattern=traffic,
                num_episodes=10, 
                verbose=VERBOSITY
            )
            nested_set(all_results, key, stats)
            end_time = time.time()
            print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=4)

    # === RL Model combinations - Focused on best configurations ===
    print("\nStarting RL model training...")
    
    # Priority configurations based on research papers
    priority_configs = [
        # D3QN with squared rewards (like Crites & Barto)
        ("DQN", "d3qn", "enhanced", "squared", "combinatorial", "all_in_one", False),
        
        # Traffic-aware training (like Wan et al.)
        ("PPO", "traffic_aware", "enhanced", "fairness", "discrete", "all_in_one", False),
        
        # SMDP training (like Wan et al.)
        ("PPO", "smdp", "enhanced", "complex", "discrete", "mixed", True),
        
        # Standard configurations for comparison
        ("PPO", "default", "enhanced", "complex", "discrete", "mixed", False),
        ("A2C", "default", "detailed", "fairness", "discrete", "mixed", False),
        ("SAC", "default", "enhanced", "complex", "continuous", "mixed", False),
    ]

    for config in priority_configs:
        model_name, wrapper, obs_type, reward_type, action_type, traffic, use_smdp = config
        
        # Check if this configuration is compatible
        if action_type not in MODEL_COMPATIBILITY.get(model_name, []):
            continue
            
        keys = [model_name, wrapper, obs_type, reward_type, action_type, traffic]
        if use_smdp:
            keys.append("smdp")
            
        if nested_get(all_results, keys) is not None:
            print(f"\nSkipping already completed {keys}")
            continue

        print("\n" + "=" * 50)
        print(f"Training {model_name} | Wrapper:{wrapper} | Obs:{obs_type}")
        print(f"Reward:{reward_type} | Action:{action_type} | Traffic:{traffic} | SMDP:{use_smdp}")
        print("=" * 50)
        
        start_time = time.time()
        try:
            model, env = train_rl_agent(
                model_name=model_name,
                observation_type=obs_type,
                reward_type=reward_type,
                action_type=action_type,
                env_wrapper=wrapper,
                traffic_pattern=traffic,
                use_smdp=use_smdp,
                verbose=VERBOSITY,
                log_evaluation=False
            )
            
            stats = evaluate_agent(model, env, num_episodes=10)
            end_time = time.time()
            
            print(f"Training & evaluation completed in {end_time - start_time:.2f} seconds.")
            print(f"Results: {stats}")
            
            nested_set(all_results, keys, stats)
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=4)
                
        except Exception as e:
            print(f"Error in {keys}: {e}")
            traceback.print_exc()
            # Continue with other configurations
            continue

    # # === Additional comprehensive comparisons ===
    # print("\nRunning comprehensive comparisons...")
    # for model_name, supported_actions in MODEL_COMPATIBILITY.items():
    #     for wrapper in ["default", "d3qn"]:  # Focus on most promising wrappers
    #         for action_type in supported_actions:
    #             # Test best observation and reward combinations
    #             for obs_type in ["enhanced"]:  # Focus on enhanced observations
    #                 for reward_type in ["squared", "fairness"]:  # Best reward types from research
    #                     for traffic in ["mixed", "all_in_one"]:  # Most realistic patterns
                            
    #                         keys = [model_name, wrapper, obs_type, reward_type, action_type, traffic]
    #                         if nested_get(all_results, keys) is not None:
    #                             continue
                                
    #                         # Skip incompatible combinations
    #                         if wrapper == "d3qn" and model_name != "DQN":
    #                             continue
                                
    #                         print(f"\nTraining {keys}")
    #                         start_time = time.time()
    #                         try:
    #                             model, env = train_rl_agent(
    #                                 model_name=model_name,
    #                                 observation_type=obs_type,
    #                                 reward_type=reward_type,
    #                                 action_type=action_type,
    #                                 env_wrapper=wrapper,
    #                                 traffic_pattern=traffic,
    #                                 use_smdp=False,
    #                                 verbose=VERBOSITY
    #                             )
                                
    #                             stats = evaluate_agent(model, env, num_episodes=5)
    #                             nested_set(all_results, keys, stats)
    #                             with open(results_file, "w") as f:
    #                                 json.dump(all_results, f, indent=4)
                                    
    #                         except Exception as e:
    #                             print(f"Error in {keys}: {e}")
    #                             continue

    # === Final Summary and Analysis ===
    print("\n" + "=" * 60)
    # Save final results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll evaluation results saved to {results_file}!")
    print("Analysis complete!")