import time
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from elevator_env import ElevatorEnv
import os
import json
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium import spaces

np.random.seed(42)


# ===== Helper Wrapper for DQN =====
class FlattenMultiDiscreteWrapper(ElevatorEnv):
    """Wraps a MultiDiscrete action space into a single Discrete action for DQN."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.action_space, spaces.MultiDiscrete)
        self.original_action_space = self.action_space
        self.action_space = spaces.Discrete(np.prod(self.original_action_space.nvec))

    def step(self, action):
        # Convert flat action index back to multidiscrete vector
        actions = np.unravel_index(action, self.original_action_space.nvec)
        return super().step(np.array(actions))


# ===== RL Agent Training =====
def train_rl_agent(model_name: str, observation_type: str, reward_type: str,
                   action_type: str, verbose: int = 0, log_evaluation: bool = False):
    """Creates the environment, trains the specified agent, and returns it."""
    if model_name == 'DQN':
        env_class = FlattenMultiDiscreteWrapper
    else:
        env_class = ElevatorEnv

    env = env_class(
        num_floors=10,
        num_elevators=4,
        episode_length=3600,    # 60 minute episodes
        headless=True,
        passenger_generation_rate=1.0,
        observation_type=observation_type,  # 'simple' or 'detailed'
        reward_type=reward_type,            # 'simple' or 'complex'
        action_type=action_type,            # 'discrete' or 'continuous'
        verbose=verbose - 1
    )

    if log_evaluation:
        eval_env = env_class(
            num_floors=10,
            num_elevators=4,
            episode_length=3600,
            headless=True,
            passenger_generation_rate=1.0,
            observation_type=observation_type,
            reward_type=reward_type,
            action_type=action_type,
            verbose=verbose - 1
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

    model_class = model_zoo[model_name]

    # --- Model Specific Configurations ---
    policy = "MlpPolicy"
    model_kwargs = {
        "policy": policy,
        "env": env,
        "verbose": 1 if verbose > 0 else 0,
        "device": "cpu",
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
            "n_steps": 10,
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

    model = model_class(**model_kwargs)

    run_id = f"{model_name}_{observation_type}_{reward_type}_{action_type}"
    save_dir = f"models/{run_id}"
    os.makedirs(save_dir, exist_ok=True)

    if log_evaluation:
        log_dir = f"logs/{run_id}/"
        os.makedirs(log_dir, exist_ok=True)
        eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir,
                                     eval_freq=1024, deterministic=True, render=False)

    if verbose > 0:
        print(f"Starting training for {run_id}...")

    if log_evaluation:
        model.learn(total_timesteps=50000, callback=eval_callback)
    else:
        model.learn(total_timesteps=50000)

    model.save(f"{save_dir}/{run_id}_model.zip")

    if verbose > 0:
        print(f"Training completed and model for {run_id} saved!")

    return model, env


# ===== Evaluation Functions =====
def evaluate_agent(model, env: ElevatorEnv | FlattenMultiDiscreteWrapper, num_episodes=5):
    """Evaluate a trained RL agent."""
    all_stats = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        all_stats.append(info)
        
    avg_completed = np.mean([s['passengers_completed'] for s in all_stats])
    avg_wait = np.mean([s['average_wait_time'] for s in all_stats])
    avg_journey = np.mean([s['average_journey_time'] for s in all_stats])

    return {
        "avg_passengers_completed": avg_completed,
        "avg_wait_time": avg_wait,
        "avg_journey_time": avg_journey
    }


def evaluate_rule_based(observation_type, reward_type, num_episodes=5, verbose: int = 0):
    """Evaluate rule-based elevator logic under given configuration."""
    env = ElevatorEnv(
        num_floors=10,
        num_elevators=4,
        episode_length=3600,
        headless=True,
        passenger_generation_rate=1.0,
        observation_type=observation_type,
        reward_type=reward_type,
        action_type="discrete",
        verbose=verbose - 1
    )

    all_stats = []
    for _ in range(num_episodes):
        env.reset()
        terminated, truncated = False, False
        # The rule-based system runs inside building.step()
        # We provide a dummy action which will be ignored by idle elevators
        dummy_action = np.zeros(env.num_elevators, dtype=int)
        while not terminated and not truncated:
            # The environment's step function calls building.step()
            # which contains the rule-based logic.
            _, _, terminated, truncated, info = env.step(dummy_action)
        all_stats.append(info)

    env.close()

    avg_completed = np.mean([s['passengers_completed'] for s in all_stats])
    avg_wait = np.mean([s['average_wait_time'] for s in all_stats])
    avg_journey = np.mean([s['average_journey_time'] for s in all_stats])

    return {
        "avg_passengers_completed": avg_completed,
        "avg_wait_time": avg_wait,
        "avg_journey_time": avg_journey
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
    VERBOSITY = 1   # 0 for silent, 1 for updates, 2 for detailed
    results_file = "evaluation_results.json"

    # Load existing results
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {results_file}")
    else:
        all_results = {}

    MODEL_COMPATIBILITY = {
        "PPO": ["discrete", "continuous"],
        "A2C": ["discrete", "continuous"],
        "DQN": ["discrete"],
        "SAC": ["continuous"],
        "TD3": ["continuous"],
        "DDPG": ["continuous"],
    }

    observation_types = ["simple", "detailed"]
    reward_types = ["simple", "complex"]

    # === Rule-based combinations ===
    for obs_type in observation_types:
        for reward_type in reward_types:
            if nested_get(all_results, ["rule_based", obs_type, reward_type, "discrete"]) is None:
                print(f"\nEvaluating rule_based_{obs_type}_{reward_type}_discrete")
                start_time = time.time()
                stats = evaluate_rule_based(obs_type, reward_type, num_episodes=5, verbose=VERBOSITY)
                nested_set(all_results, ["rule_based", obs_type, reward_type, "discrete"], stats)
                end_time = time.time()
                print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=4)

    # === RL Model combinations ===
    for model_name, supported_actions in MODEL_COMPATIBILITY.items():
        for obs_type in observation_types:
            for reward_type in reward_types:
                for action_type in supported_actions:
                    keys = [model_name, obs_type, reward_type, action_type]
                    if nested_get(all_results, keys) is not None:
                        print(f"\nSkipping already completed {keys}")
                        continue

                    print("\n" + "=" * 30)
                    print(f"Training {model_name} | Obs:{obs_type} | Reward:{reward_type} | Action:{action_type}")
                    print("=" * 30)
                    start_time = time.time()
                    try:
                        model, env = train_rl_agent(model_name, obs_type, reward_type, action_type, verbose=VERBOSITY)
                        stats = evaluate_agent(model, env, num_episodes=5)
                        end_time = time.time()
                        print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
                        nested_set(all_results, keys, stats)
                        with open(results_file, "w") as f:
                            json.dump(all_results, f, indent=4)
                    except Exception as e:
                        print(f"Error in {keys}: {e}")

    # === Final Summary ===
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n\tAll evaluation results saved!")
