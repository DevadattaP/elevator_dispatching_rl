import time
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from stable_baselines3.common.env_checker import check_env
from elevator_env import ElevatorEnv
import os
import json
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium import spaces
np.random.seed(42)

class FlattenMultiDiscreteWrapper(ElevatorEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.action_space, spaces.MultiDiscrete)
        self.original_action_space = self.action_space
        self.action_space = spaces.Discrete(np.prod(self.original_action_space.nvec))

    def step(self, action):
        # Convert flat action index back to multidiscrete vector
        actions = np.unravel_index(action, self.original_action_space.nvec)
        return super().step(np.array(actions))


def train_rl_agent(model_name: str, action_type: str, verbose: int = 0, log_evaluation: bool = False):
    """Creates the environment, trains the specified agent, and returns it."""
    # Create environment with the correct action type
    if model_name == 'DQN':
        class_name = FlattenMultiDiscreteWrapper
    else:
        class_name = ElevatorEnv
    env = class_name(
        num_floors=10,
        num_elevators=4,
        episode_length=3600,     # 60 minute episodes
        headless=True,
        passenger_generation_rate=5.0,
        observation_type='detailed', # 'simple' or 'detailed'
        reward_type='complex',     # 'simple' or 'complex'
        action_type=action_type,    # 'discrete' or 'continuous'
        verbose=verbose-1
    )
    
    # Create a separate environment for evaluation during training
    if log_evaluation:
        eval_env = ElevatorEnv(
            num_floors=10,
            num_elevators=4,
            episode_length=3600,
            headless=True,
            passenger_generation_rate=5.0,
            observation_type='detailed',
            reward_type='complex',
            action_type=action_type,
            verbose=verbose-1
        )

    # --- Model Dictionary ---
    model_zoo = {
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN,
        "SAC": SAC,
        "TD3": TD3,
        "DDPG": DDPG,
    }
    
    if model_name not in model_zoo:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {list(model_zoo.keys())}")

    model_class = model_zoo[model_name]
    
    # --- Model Specific Configurations ---
    policy = "MlpPolicy"
    if model_name == "DQN":
        policy = "MlpPolicy" # DQN has its own MlpPolicy

    model_kwargs = {
        "policy": policy,
        "env": env,
        "verbose": 1 if verbose > 0 else 0,
        "device": "cpu"
    }

    if model_name == "PPO":
        model_kwargs.update({
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
        })
    elif model_name == "A2C":
        model_kwargs.update({
            "learning_rate": 3e-4,
            "n_steps": 5,
            "gamma": 0.99,
        })
    elif model_name == "DQN":
        model_kwargs.update({
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
        })
    elif model_name in ["SAC", "TD3", "DDPG"]:
        # Action noise is often useful for exploration in continuous action spaces
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model_kwargs.update({
            "action_noise": action_noise,
            "buffer_size": 100000,
            "batch_size": 100,
            "gamma": 0.99,
            "learning_starts": 1000,
        })

    if verbose > 0:
        print(f"Creating {model_name} model...")
    model = model_class(**model_kwargs)
    
    # Create log directory for the specific model
    if log_evaluation:
        log_dir = f"logs/{model_name}/"
        os.makedirs(log_dir, exist_ok=True)

        # Setup callback for saving learning curve
        eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                    log_path=log_dir, eval_freq=1024, # Evaluate every N steps
                                    deterministic=True, render=False)

    # Train the model
    if verbose > 0:
        print(f"Starting training for {model_name}...")
    if log_evaluation:
        model.learn(total_timesteps=50000, callback=eval_callback)
    else:
        model.learn(total_timesteps=50000)
    
    # Save the model
    model.save(f"{model_name}_elevator_model")
    if verbose > 0:
        print(f"Training completed and model for {model_name} saved!")
    
    return model, env

def evaluate_agent(model, env: ElevatorEnv | FlattenMultiDiscreteWrapper, num_episodes=5, verbose: int = 0):
    """Evaluate a trained agent."""
    all_stats = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        if verbose > 0:
            print(f"\n=== RL Evaluation Episode {episode + 1} ===")
        
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        
        all_stats.append(info)
        if verbose > 0:
            print(f"Episode {episode + 1} finished. Stats: {info}")
    
    # Calculate and return average statistics
    avg_completed = np.mean([s['passengers_completed'] for s in all_stats])
    avg_wait = np.mean([s['average_wait_time'] for s in all_stats])
    avg_journey = np.mean([s['average_journey_time'] for s in all_stats])
    
    return {
        "avg_passengers_completed": avg_completed,
        "avg_wait_time": avg_wait,
        "avg_journey_time": avg_journey
    }

def evaluate_rule_based(num_episodes=5, verbose: int = 0):
    """Evaluate the default rule-based system."""
    # Create a standard environment; action space doesn't matter here
    env = ElevatorEnv(
        num_floors=10,
        num_elevators=4,
        episode_length=1800,
        headless=True,
        passenger_generation_rate=5.0,
        verbose=verbose-1
    )
    
    all_stats = []
    for episode in range(num_episodes):
        env.reset()
        terminated = False
        truncated = False
        
        if verbose > 0:
            print(f"\n=== Rule-Based Evaluation Episode {episode + 1} ===")
            
        # The rule-based system runs inside building.step()
        # We provide a dummy action which will be ignored by idle elevators
        dummy_action = np.zeros(env.num_elevators, dtype=int)
        
        while not terminated and not truncated:
            # The environment's step function calls building.step()
            # which contains the rule-based logic.
            obs, reward, terminated, truncated, info = env.step(dummy_action)

        all_stats.append(info)
        if verbose > 0:
            print(f"Episode {episode + 1} finished. Stats: {info}")

    # Calculate and return average statistics
    avg_completed = np.mean([s['passengers_completed'] for s in all_stats])
    avg_wait = np.mean([s['average_wait_time'] for s in all_stats])
    avg_journey = np.mean([s['average_journey_time'] for s in all_stats])
    
    env.close()
    
    return {
        "avg_passengers_completed": avg_completed,
        "avg_wait_time": avg_wait,
        "avg_journey_time": avg_journey
    }


if __name__ == "__main__":
    VERBOSITY = 1 # 0 for silent, 1 for updates, 2 for detailed
    
    # === Load previous evaluation results if available ===
    results_file = "evaluation_results.json"
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                all_results = json.load(f)
            print(f"Loaded existing results from {results_file}")
        except Exception as e:
            print(f"Warning: Could not read {results_file}: {e}")
            all_results = {}
    else:
        all_results = {}


    # --- Define Models to Train ---
    # (Model_Name, Action_Type)
    models_to_run = [
        ("PPO", "discrete"),
        ("A2C", "discrete"),
        ("DQN", "discrete"),
        ("SAC", "continuous"),
        ("TD3", "continuous"),
        ("DDPG", "continuous"),
    ]
    
    # 1. Evaluate the rule-based system as a baseline
    print("\n" + "="*30)
    print("  Evaluating Rule-Based System")
    print("="*30)
    start_time = time.time()
    rule_based_stats = evaluate_rule_based(num_episodes=5, verbose=VERBOSITY)
    all_results['rule_based'] = rule_based_stats
    print(f"\nRule-based evaluation finished in {time.time() - start_time:.2f} seconds.")
    # Save initial base results
    with open('results_file', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # 2. Loop through, train, and evaluate each RL model
    for model_name, action_type in models_to_run:
        if model_name in all_results:
            print(f"\nSkipping {model_name}: already trained and evaluated.")
            continue
        
        print("\n" + "="*30)
        print(f"      Training {model_name}")
        print("="*30)
        start_time = time.time()
        
        try:
            model, train_env = train_rl_agent(model_name, action_type, verbose=VERBOSITY, log_evaluation=False)
            print(f"\n{model_name} training finished in {time.time() - start_time:.2f} seconds.")

            if model and train_env:
                print("\n" + "="*30)
                print(f"      Evaluating {model_name}")
                print("="*30)
                start_time = time.time()
                rl_agent_stats = evaluate_agent(model=model, env=train_env, num_episodes=5, verbose=VERBOSITY)
                all_results[model_name] = rl_agent_stats
                print(f"\n{model_name} evaluation finished in {time.time() - start_time:.2f} seconds.")
                train_env.close()
                # Save intermediate results
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=4)
        except Exception as e:
            print(f"\nAn error occurred while training/evaluating {model_name}: {e}")
            print(f"Skipping {model_name}.")

    # 3. Save and Compare all results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n" + "="*40)
    print(f"All evaluation results saved to {results_file}")
    print("="*40)

    # Print final comparison
    for name, stats in all_results.items():
        print(f"\n--- {name} ---")
        print(f"Avg. Passengers Completed: {stats['avg_passengers_completed']:.2f}")
        print(f"Avg. Wait Time:            {stats['avg_wait_time']:.2f}s")
        print(f"Avg. Journey Time:         {stats['avg_journey_time']:.2f}s")
    print("\n" + "="*40)