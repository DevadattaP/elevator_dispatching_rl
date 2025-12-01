import numpy as np
from collections import defaultdict
import random
from entities.building import Building
from utils.enums import ElevatorState
import pickle
import json
from pathlib import Path

class TabularElevatorEnv:
    """Simplified discrete environment for tabular RL methods using your existing building logic."""
    
    def __init__(self, 
                 num_floors: int = 10,
                 num_elevators: int = 4,
                 lift_capacity: int = 8,
                 episode_length: int = 1800,
                 passenger_generation_rate: float = 1.0,
                 verbose: int = 0):
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.episode_length = episode_length
        self.passenger_generation_rate = passenger_generation_rate
        self.verbose = verbose
        
        # Use your existing Building class
        self.building = Building(num_floors, num_elevators, 
                               speed_multiplier=10.0,  # Faster for tabular RL
                               capacity=lift_capacity,
                               verbose=(verbose > 1))
        
        # Discrete state space dimensions
        self.state_dims = self._calculate_state_dimensions()
        self.state_space_size = np.prod(self.state_dims)
        
        # Action space: for each elevator, which floor to go to (+1 for stay)
        self.action_space_size = (num_floors + 1) ** num_elevators
        
        self.reset()
    
    def _calculate_state_dimensions(self):
        """Calculate dimensions for discrete state representation."""
        # Elevator positions: 0 to num_floors-1
        # Elevator directions: -1 (down), 0 (idle), 1 (up) -> 3 states
        # Waiting passengers per floor (up/down): 0-3 (0,1,2,3+)
        # Total: (num_floors * 3 * 4 * 4) ^ num_elevators? Let's simplify...
        
        # Simplified state representation:
        # 1. Elevator positions (num_floors possibilities each)
        # 2. Elevator states (idle, moving_up, moving_down)
        # 3. Waiting passengers per floor-direction (0-2: 0,1,2+)
        
        # Let's use a more manageable representation:
        dims = []
        
        # For each elevator: position + state
        for _ in range(self.num_elevators):
            dims.append(self.num_floors)  # Position
            dims.append(3)  # State: idle, moving_up, moving_down
        
        # For each floor: waiting up (0-2), waiting down (0-2)
        for _ in range(self.num_floors):
            dims.append(3)  # Up waiting: 0,1,2+
            dims.append(3)  # Down waiting: 0,1,2+
        
        return dims
    
    def _state_to_index(self, state_tuple):
        """Convert state tuple to unique integer index."""
        state_idx = 0
        multiplier = 1
        
        for i, value in enumerate(state_tuple):
            state_idx += value * multiplier
            multiplier *= self.state_dims[i]
        
        return state_idx
    
    def _get_discrete_state(self):
        """Convert building state to discrete state representation."""
        state_tuple = []
        
        # Elevator information
        for elevator in self.building.elevators:
            # Position (0 to num_floors-1)
            state_tuple.append(min(elevator.current_floor, self.num_floors - 1))
            
            # State: 0=idle, 1=moving_up, 2=moving_down
            if elevator.state == ElevatorState.IDLE:
                state_tuple.append(0)
            elif elevator.state in [ElevatorState.MOVING_UP, ElevatorState.DOOR_OPENING, ElevatorState.DOOR_CLOSING, ElevatorState.DOOR_OPEN]:
                state_tuple.append(1)
            else:  # MOVING_DOWN
                state_tuple.append(2)
        
        # Waiting passengers information
        waiting_up = [0] * self.num_floors
        waiting_down = [0] * self.num_floors
        
        # Count waiting passengers per floor and direction
        for floor, passengers in self.building.active_passengers.items():
            for passenger in passengers:
                if passenger.direction == 'up':  # Up
                    waiting_up[passenger.start_floor] = min(waiting_up[passenger.start_floor] + 1, 2)
                else:  # Down
                    waiting_down[passenger.start_floor] = min(waiting_down[passenger.start_floor] + 1, 2)
        
        # Add waiting counts to state
        for floor in range(self.num_floors):
            state_tuple.append(waiting_up[floor])
            state_tuple.append(waiting_down[floor])
        
        return self._state_to_index(tuple(state_tuple))
    
    def reset(self):
        """Reset environment to initial state."""
        self.building = Building(self.num_floors, self.num_elevators,
                               speed_multiplier=10.0,
                               capacity=self.building.elevators[0].capacity,
                               verbose=(self.verbose > 1))
        
        self.current_step = 0
        self.total_reward = 0
        self.last_completion_count = 0
        
        # Generate initial passengers
        self._generate_initial_passengers()
        
        return self._get_discrete_state()
    
    def _generate_initial_passengers(self):
        """Generate some initial passengers."""
        for _ in range(3):  # Start with 3 passengers
            start_floor = random.randint(0, self.num_floors - 1)
            target_floor = random.choice([f for f in range(self.num_floors) if f != start_floor])
            self.building.add_passenger(start_floor, target_floor)
    
    def _generate_random_passenger(self):
        """Randomly generate a new passenger."""
        if random.random() < self.passenger_generation_rate:
            start_floor = random.randint(0, self.num_floors - 1)
            target_floor = random.choice([f for f in range(self.num_floors) if f != start_floor])
            self.building.add_passenger(start_floor, target_floor)
    
    def step(self, action):
        """Execute one environment step."""
        # Process the action
        self._process_action(action)
        
        # Generate random passengers
        self._generate_random_passenger()
        
        # Advance the building simulation
        self.building.step()
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Get next state
        next_state = self._get_discrete_state()
        
        # Check termination
        done = (self.current_step >= self.episode_length or 
                len(self.building.completed_passengers) >= 50)  # Or complete 50 passengers
        
        info = {
            "passengers_completed": len(self.building.completed_passengers),
            "total_reward": self.total_reward,
            "step": self.current_step
        }
        
        return next_state, reward, done, info
    
    def _process_action(self, action):
        """Process discrete action for all elevators."""
        # Decode action: each elevator gets a target floor (num_floors = stay)
        targets = []
        temp_action = action
        
        for i in range(self.num_elevators):
            target_floor = temp_action % (self.num_floors + 1)
            targets.append(target_floor)
            temp_action = temp_action // (self.num_floors + 1)
        
        # Assign targets to idle elevators
        for i, (elevator, target) in enumerate(zip(self.building.elevators, targets)):
            if (elevator.is_idle() and not elevator.target_floors and 
                target < self.num_floors and target != elevator.current_floor):
                elevator.assign_target(target)
    
    def _calculate_reward(self):
        """Calculate reward for current step."""
        reward = 0.0
        
        # Reward for completed passengers
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        reward += new_completions * 5.0
        
        # Penalty for waiting passengers
        total_waiting = sum(len(passengers) for passengers in self.building.active_passengers.values())
        reward -= total_waiting * 0.1
        
        # Small penalty for movement (encourage efficiency)
        moving_elevators = sum(1 for e in self.building.elevators if e.is_moving())
        reward -= moving_elevators * 0.05
        
        # Bonus for serving long-waiting passengers
        current_time = self.building.time
        for passengers in self.building.active_passengers.values():
            for passenger in passengers:
                if passenger.waiting_time > 30:
                    reward -= 0.5
        
        self.last_completion_count = len(self.building.completed_passengers)
        
        return reward
    
    def get_action_meanings(self):
        """Get meaning of each action dimension for debugging."""
        meanings = []
        for action in range(self.action_space_size):
            targets = []
            temp_action = action
            for i in range(self.num_elevators):
                target_floor = temp_action % (self.num_floors + 1)
                targets.append(f"E{i}:{'Stay' if target_floor == self.num_floors else f'F{target_floor}'}")
                temp_action = temp_action // (self.num_floors + 1)
            meanings.append(" | ".join(targets))
        return meanings


class TabularRL:
    """Implementation of Monte Carlo, Q-learning, and SARSA for elevator control."""
    
    def __init__(self, env, algorithm='q_learning', alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.algorithm = algorithm
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table
        self.Q = defaultdict(lambda: np.zeros(env.action_space_size))
        
        # For tracking
        self.episode_rewards = []
        self.learning_data = []
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.env.action_space_size - 1)
        else:
            return np.argmax(self.Q[state])
    
    def train_episode(self):
        """Train for one episode."""
        if self.algorithm == 'monte_carlo':
            return self._monte_carlo_episode()
        elif self.algorithm == 'sarsa':
            return self._sarsa_episode()
        else:  # Q-learning
            return self._q_learning_episode()
    
    def save(self, filepath):
        """Save the trained model to a file."""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to regular dict for saving
        q_table_dict = {k: v.tolist() for k, v in self.Q.items()}
        
        model_data = {
            'q_table': q_table_dict,
            'algorithm': self.algorithm,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'env_params': {
                'num_floors': self.env.num_floors,
                'num_elevators': self.env.num_elevators,
                'state_space_size': self.env.state_space_size,
                'action_space_size': self.env.action_space_size
            },
            'training_history': {
                'episode_rewards': self.episode_rewards,
                'learning_data': self.learning_data
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, env=None):
        """Load a trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create environment if not provided
        if env is None:
            env_params = model_data['env_params']
            env = TabularElevatorEnv(
                num_floors=env_params['num_floors'],
                num_elevators=env_params['num_elevators'],
                verbose=0
            )
        
        # Create agent instance
        agent = cls(
            env=env,
            algorithm=model_data['algorithm'],
            alpha=model_data['alpha'],
            gamma=model_data['gamma'],
            epsilon=model_data['epsilon']
        )
        
        # Restore Q-table (convert back to defaultdict)
        q_table_dict = model_data['q_table']
        agent.Q = defaultdict(lambda: np.zeros(agent.env.action_space_size))
        for k, v in q_table_dict.items():
            agent.Q[k] = np.array(v)
        
        # Restore training history
        agent.episode_rewards = model_data['training_history']['episode_rewards']
        agent.learning_data = model_data['training_history']['learning_data']
        
        print(f"Model loaded from {filepath}")
        print(f"Algorithm: {agent.algorithm}, Episodes trained: {len(agent.episode_rewards)}")
        
        return agent
    
    def save_training_history(self, filepath):
        """Save training history as JSON for analysis."""
        history_data = {
            'episode_rewards': self.episode_rewards,
            'learning_data': self.learning_data,
            'algorithm': self.algorithm,
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Training history saved to {filepath}")

    def predict(self, state, deterministic=True):
        """Gym-compatible predict method for consistency with stable-baselines3."""
        if isinstance(state, np.ndarray):
            # Convert numpy array to discrete state index
            state = self._continuous_to_discrete_state(state)
        
        action = self.choose_action(state, training=False)
        return action, None  # Return (action, None) to match SB3 interface
    
    def _continuous_to_discrete_state(self, continuous_state):
        """Convert continuous state observation to discrete state index."""
        # This is a simplified conversion - you may need to adjust based on your state representation
        state_tuple = []
        
        # Assuming continuous_state has the same structure as your discrete state
        # but with continuous values. We need to discretize it.
        idx = 0
        
        # Elevator positions (convert continuous to discrete floors)
        for i in range(self.env.num_elevators):
            # Position (first value for each elevator)
            continuous_pos = continuous_state[idx]
            discrete_pos = int(continuous_pos * (self.env.num_floors - 1))
            state_tuple.append(min(discrete_pos, self.env.num_floors - 1))
            idx += 4  # Skip direction, state, passenger_count (4 values per elevator)
        
        # Elevator states (skip for now, use current building state)
        # Use actual building state for more accurate conversion
        building_state = self.env.building.get_state()
        
        # Reconstruct discrete state using actual building data
        return self.env._get_discrete_state()
    
    def _q_learning_episode(self):
        """Q-learning algorithm."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            # Q-learning update
            best_next_action = np.argmax(self.Q[next_state])
            td_target = reward + self.gamma * self.Q[next_state][best_next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        self.learning_data.append({
            'episode': len(self.episode_rewards),
            'total_reward': total_reward,
            'steps': steps,
            'passengers_completed': info['passengers_completed']
        })
        
        return total_reward, steps
    
    def _sarsa_episode(self):
        """SARSA algorithm."""
        state = self.env.reset()
        action = self.choose_action(state)
        total_reward = 0
        steps = 0
        
        while True:
            next_state, reward, done, info = self.env.step(action)
            next_action = self.choose_action(next_state)
            
            # SARSA update
            td_target = reward + self.gamma * self.Q[next_state][next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        self.learning_data.append({
            'episode': len(self.episode_rewards),
            'total_reward': total_reward,
            'steps': steps,
            'passengers_completed': info['passengers_completed']
        })
        
        return total_reward, steps
    
    def _monte_carlo_episode(self):
        """Monte Carlo algorithm."""
        state = self.env.reset()
        episode = []
        total_reward = 0
        steps = 0
        
        # Generate episode
        while True:
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Monte Carlo update (Every Visit)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            self.Q[state][action] += self.alpha * (G - self.Q[state][action])
        
        self.episode_rewards.append(total_reward)
        self.learning_data.append({
            'episode': len(self.episode_rewards),
            'total_reward': total_reward,
            'steps': steps,
            'passengers_completed': info['passengers_completed']
        })
        
        return total_reward, steps
    
    def evaluate(self, num_episodes=10):
        """Evaluate the trained policy."""
        total_rewards = []
        passengers_served = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.choose_action(state, training=False)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            passengers_served.append(info['passengers_completed'])
        
        return (np.mean(total_rewards), np.std(total_rewards),
                np.mean(passengers_served), np.std(passengers_served))
    
    def analyze_policy(self, num_states=5):
        """Analyze the learned policy for some random states."""
        print("\n=== Policy Analysis ===")
        for _ in range(num_states):
            state = random.randint(0, self.env.state_space_size - 1)
            best_action = np.argmax(self.Q[state])
            q_value = np.max(self.Q[state])
            
            print(f"State {state}: Best action = {best_action}, Q-value = {q_value:.3f}")
            
            # Decode action meaning
            targets = []
            temp_action = best_action
            for i in range(self.env.num_elevators):
                target_floor = temp_action % (self.env.num_floors + 1)
                targets.append(f"E{i}->{'Stay' if target_floor == self.env.num_floors else f'F{target_floor}'}")
                temp_action = temp_action // (self.env.num_floors + 1)
            print(f"  Action meaning: {' | '.join(targets)}")


class TabularModelWrapper:
    """Wrapper to make TabularRL models compatible with your existing GUI code."""
    
    def __init__(self, tabular_agent):
        self.tabular_agent = tabular_agent
        self.env = tabular_agent.env
    
    def predict(self, obs, deterministic=True):
        """Match stable-baselines3 predict interface."""
        return self.tabular_agent.predict(obs, deterministic)
    
    @property
    def building(self):
        """Provide access to building for GUI synchronization."""
        return self.env.building
    
    def save(self, path):
        """Save the underlying tabular model."""
        self.tabular_agent.save(path)
    
    @classmethod
    def load(cls, path):
        """Load a tabular model and wrap it."""
        tabular_agent = TabularRL.load(path)
        return cls(tabular_agent)


def train_and_compare_tabular_methods(save_models=True, load_existing=False):
    """Train and compare all three tabular methods with save/load support."""
    env = TabularElevatorEnv(num_floors=10, num_elevators=4, verbose=0)
    
    algorithms = ['q_learning', 'sarsa', 'monte_carlo']
    results = {}
    
    print("Training Tabular RL Methods...")
    print("=" * 50)
    
    for algo in algorithms:
        model_path = f"models/tabular_{algo}_model.pkl"
        history_path = f"models/tabular_{algo}_history.json"
        
        # Try to load existing model
        if load_existing and Path(model_path).exists():
            print(f"\nLoading existing {algo.upper()} model...")
            try:
                agent = TabularRL.load(model_path, env)
                print(f"  Loaded model with {len(agent.episode_rewards)} episodes of training")
            except Exception as e:
                print(f"  Error loading model: {e}. Training new model...")
                agent = TabularRL(env, algorithm=algo, alpha=0.1, gamma=0.95, epsilon=0.1)
        else:
            print(f"\nTraining new {algo.upper()} model...")
            agent = TabularRL(env, algorithm=algo, alpha=0.1, gamma=0.95, epsilon=0.1)
        
        # Training (only if we didn't load or want to continue training)
        if not load_existing or not Path(model_path).exists():
            for episode in range(1000):
                reward, steps = agent.train_episode()
                
                if episode % 100 == 0:
                    completed = agent.learning_data[-1]['passengers_completed']
                    print(f"  Episode {episode}: Reward = {reward:6.1f}, "
                          f"Steps = {steps:3d}, Completed = {completed:2d}")
            
            # Save model and history
            if save_models:
                agent.save(model_path)
                agent.save_training_history(history_path)
        
        # Evaluation
        mean_reward, std_reward, mean_passengers, std_passengers = agent.evaluate(num_episodes=10)
        
        results[algo] = {
            'agent': agent,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_passengers': mean_passengers,
            'std_passengers': std_passengers,
            'model_path': model_path
        }
        
        print(f"  Evaluation: Reward = {mean_reward:.1f} ± {std_reward:.1f}, "
              f"Passengers = {mean_passengers:.1f} ± {std_passengers:.1f}")
        
        # Analyze policy
        agent.analyze_policy(num_states=3)
    
    return results

def train_single_model(algorithm='q_learning', num_episodes=1000, save_model=True):
    """Train a single model with detailed progress tracking."""
    env = TabularElevatorEnv(num_floors=10, num_elevators=4, verbose=0)
    agent = TabularRL(env, algorithm=algorithm, alpha=0.1, gamma=0.95, epsilon=0.1)
    
    print(f"Training {algorithm.upper()} for {num_episodes} episodes...")
    
    # Training progress
    for episode in range(num_episodes):
        reward, steps = agent.train_episode()
        
        if episode % 100 == 0 or episode == num_episodes - 1:
            completed = agent.learning_data[-1]['passengers_completed']
            avg_reward = np.mean(agent.episode_rewards[-100:]) if episode >= 100 else reward
            print(f"Episode {episode:4d}: Reward = {reward:6.1f}, "
                  f"Avg100 = {avg_reward:6.1f}, Completed = {completed:2d}")
    
    # Save model
    if save_model:
        model_path = f"models/tabular_{algorithm}_model.pkl"
        history_path = f"models/tabular_{algorithm}_history.json"
        agent.save(model_path)
        agent.save_training_history(history_path)
    
    return agent

def load_and_evaluate_model(model_path, num_evaluation_episodes=20):
    """Load a saved model and evaluate it."""
    if not Path(model_path).exists():
        print(f"Model file {model_path} not found!")
        return None
    
    print(f"Loading and evaluating model: {model_path}")
    agent = TabularRL.load(model_path)
    
    # Evaluate
    mean_reward, std_reward, mean_passengers, std_passengers = agent.evaluate(
        num_episodes=num_evaluation_episodes
    )
    
    print(f"Evaluation Results ({num_evaluation_episodes} episodes):")
    print(f"  Average Reward: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"  Passengers Completed: {mean_passengers:.1f} ± {std_passengers:.1f}")
    print(f"  Total Training Episodes: {len(agent.episode_rewards)}")
    
    return agent

def compare_saved_models():
    """Compare performance of all saved models."""
    algorithms = ['q_learning', 'sarsa', 'monte_carlo']
    results = {}
    
    print("Comparing Saved Models...")
    print("=" * 50)
    
    for algo in algorithms:
        model_path = f"models/tabular_{algo}_model.pkl"
        
        if Path(model_path).exists():
            agent = load_and_evaluate_model(model_path, num_evaluation_episodes=10)
            if agent:
                results[algo] = {
                    'agent': agent,
                    'model_path': model_path
                }
        else:
            print(f"Model not found: {model_path}")
    
    return results

# Quick testing
if __name__ == "__main__":
    print('Tabular RL is giving memory errors because of the large state space.')
    # results = train_and_compare_tabular_methods()