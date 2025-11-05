import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
from building import Building
from utils.enums import ElevatorState
np.random.seed(42)

class ElevatorEnv(gym.Env):
    def __init__(self, 
                 num_floors: int = 10,
                 num_elevators: int = 4,
                 lift_capacity: int = 8,
                 speed_multiplier: float = 10.0,
                 episode_length: int = 3600,  # 1 hour in simulation seconds
                 headless: bool = True,
                 passenger_generation_rate: float = 1.0,
                 observation_type: str = 'simple', # 'simple' or 'detailed'
                 action_type: str = 'discrete', # 'discrete' or 'continuous'
                 reward_type: str = 'simple', # 'simple', 'complex'
                 verbose: int = 0):
        
        super().__init__()
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.episode_length = episode_length
        self.headless = headless
        self.passenger_generation_rate = passenger_generation_rate
        self.observation_type = observation_type
        self.action_type = action_type
        self.reward_type = reward_type
        self.verbose = verbose
        
        # Initialize building (without GUI)
        self.building = Building(num_floors, num_elevators, speed_multiplier, lift_capacity, verbose=(verbose > 1))
        
        # Define action and observation spaces
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()
        
        # Episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.passenger_sequence = []
        self.last_completion_count = 0
        
        # For GUI mode
        self.gui = None
        if not headless:
            self._init_gui()
    
    def _define_observation_space(self) -> spaces.Space:
        """Define observation space based on the configured type."""
        obs_dim = self._calculate_observation_dimension()
        
        # Use more realistic bounds based on your actual data ranges
        # High value can be episode_length for time-based features
        return spaces.Box(
            low=-1.0, 
            high=float(self.episode_length), 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _calculate_observation_dimension(self) -> int:
        """Calculate exact observation dimension to avoid shape mismatches."""
        obs_dim = 0
        
        # Common states for all observation types
        # Elevator states: position, direction, state, passenger_count
        obs_dim += self.num_elevators * 4
        # Floor waiting queues: waiting_up, waiting_down
        obs_dim += self.num_floors * 2
        # Time of day: sin, cos encoding
        obs_dim += 2
        
        if self.observation_type == 'detailed':
            # Add waiting passenger details: start_floor, target_floor, waiting_time
            # Let's cap this at a reasonable number, e.g., 20 passengers
            self.max_observed_passengers = 20
            obs_dim += self.max_observed_passengers * 3
        
        if self.verbose > 0:
            print(f"Observation type: '{self.observation_type}', Dimension: {obs_dim}")
        return obs_dim

    def _define_action_space(self) -> spaces.Space:
        """Define the action space to directly control each elevator."""
        # Action for each elevator: go to a specific floor (0 to num_floors-1)
        # We can add a special action for 'idle' if needed, e.g., num_floors
        # For now, sending an elevator to its current floor can be interpreted as idle
        if self.action_type == 'discrete':
            return spaces.MultiDiscrete([self.num_floors] * self.num_elevators)
        elif self.action_type == 'continuous':
            # Continuous action space: one float per elevator, from -1 to 1
            # We will scale this to the number of floors
            return spaces.Box(low=-1.0, high=1.0, shape=(self.num_elevators,), dtype=np.float32)
        else:
            raise NotImplementedError(f"Action type '{self.action_type}' not implemented.")

    def _get_state_representation(self) -> np.ndarray:
        """Convert building state to RL observation vector."""
        state = self.building.get_state()
        obs = []
        
        # 1. Elevator information (4 values per elevator)
        for elevator_state in state['elevators']:
            obs.extend([
                elevator_state['position'] / self.num_floors,
                elevator_state['direction'], # -1, 0, 1
                elevator_state['state'] / len(ElevatorState),
                elevator_state['passenger_count'] / self.building.elevators[0].capacity
            ])
        
        # 2. Floor waiting queues (2 values per floor)
        for floor in range(self.num_floors):
            obs.extend([
                state['floors'][floor]['waiting_up'],
                state['floors'][floor]['waiting_down']
            ])
        
        # 3. Time of day (2 values - cyclic encoding)
        current_time = state['time'] % 86400  # Seconds in day
        obs.append(np.sin(2 * np.pi * current_time / 86400))
        obs.append(np.cos(2 * np.pi * current_time / 86400))

        # 4. (Optional) Detailed passenger info
        if self.observation_type == 'detailed':
            waiting_passengers = []
            for floor_passengers in self.building.active_passengers.values():
                waiting_passengers.extend(list(floor_passengers))
            
            # Sort by waiting time to prioritize
            waiting_passengers.sort(key=lambda p: p.waiting_time, reverse=True)
            
            for i in range(self.max_observed_passengers):
                if i < len(waiting_passengers):
                    p = waiting_passengers[i]
                    obs.extend([p.start_floor, p.target_floor, p.waiting_time])
                else:
                    obs.extend([-1, -1, -1]) # Padding

        obs_array = np.array(obs, dtype=np.float32)
        
        # Debug: Check dimension matches
        expected_dim = self._calculate_observation_dimension()
        if len(obs_array) != expected_dim:
            raise ValueError(f"Observation dimension mismatch! Expected {expected_dim}, got {len(obs_array)}")
        
        return obs_array

    def _calculate_reward(self) -> float:
        """Calculate reward based on the configured reward type."""
        if self.reward_type == 'simple':
            return self._calculate_simple_reward()
        elif self.reward_type == 'complex':
            return self._calculate_complex_reward()
        else:
            raise NotImplementedError(f"Reward type '{self.reward_type}' not implemented.")

    def _calculate_simple_reward(self) -> float:
        """A simple reward based on completions and waiting count."""
        reward = 0.0
        
        # Reward for completed passengers
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        reward += new_completions * 10.0
        
        # Penalty for waiting passengers
        total_waiting = sum(len(q) for q in self.building.active_passengers.values())
        reward -= total_waiting * 0.1

        self.last_completion_count = len(self.building.completed_passengers)
        return reward

    def _calculate_complex_reward(self) -> float:
        """A more complex reward considering waiting time, travel time, and energy."""
        reward = 0.0
        state = self.building.get_state()
        
        # 1. Penalty for waiting passengers (most important)
        total_waiting = sum(floor['waiting_up'] + floor['waiting_down'] 
                           for floor in state['floors'].values())
        reward -= total_waiting * 0.5
        
        # 2. Reward for completed passengers
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        reward += new_completions * 2.0
        
        # 3. Reward for efficient service (low travel time)
        if new_completions > 0:
            recent_passengers = self.building.completed_passengers[-new_completions:]
            travel_times = [p.total_time for p in recent_passengers if p.total_time is not None]
            if travel_times:
                avg_travel_time = np.mean(travel_times)
                # Reward decreases as travel time increases
                reward += max(0, 20 - avg_travel_time) * 0.1
        
        # 4. Small penalty for elevator movement (energy cost)
        for elevator in self.building.elevators:
            if elevator.is_moving():
                reward -= 0.05
        
        # 5. Bonus for keeping elevators distributed
        elevator_positions = [elevator.current_floor for elevator in self.building.elevators]
        position_std = np.std(elevator_positions)
        reward += position_std * 0.1  # Reward for spread out elevators
        
        self.last_completion_count = len(self.building.completed_passengers)
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        # Reset building
        self.building = Building(self.num_floors, self.num_elevators, capacity=self.building.elevators[0].capacity, verbose=(self.verbose > 1))
        self.building.set_speed_multiplier(self.building.speed_multiplier)
        
        # Generate passenger sequence for this episode
        self._generate_passenger_sequence()
        
        # Reset episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.last_completion_count = 0
        
        # Add initial passengers
        self._add_scheduled_passengers()
        
        # Get initial state
        observation = self._get_state_representation()
        info = self._get_info()
        
        if self.verbose > 0:
            print("Environment reset.")
        return observation, info

    def _generate_passenger_sequence(self):
        """Generate a fixed passenger sequence for this episode"""
        self.passenger_sequence = []
        current_time = 0
        episode_end = self.episode_length
        
        # Use numpy's random generator for reproducibility with seed
        rng = np.random.default_rng(self.np_random)
        
        while current_time < episode_end:
            # Time to next passenger arrival (Exponential distribution)
            time_increment = rng.exponential(1.0 / (self.passenger_generation_rate / 10.0))
            current_time += time_increment
            if current_time < episode_end:
                start_floor, target_floor = self._generate_passenger_at_time(current_time)
                self.passenger_sequence.append((current_time, start_floor, target_floor))
        
        # Sort by time
        self.passenger_sequence.sort(key=lambda x: x[0])
        self.next_passenger_idx = 0
        
        if self.verbose > 0:
            print(f"Generated {len(self.passenger_sequence)} passengers for the episode.")

    def _generate_passenger_at_time(self, current_time: float):
        """Generate a single random passenger with enhanced distribution"""
        # Get current simulation hour (0-23)
        current_hour = (current_time % 86400) / 3600
        
        # Enhanced distribution based on time of day
        if current_hour < 6:  # Night (12 AM - 6 AM)
            # Very low traffic, mostly random
            if self.np_random.random() < 0.7:  # 30% ground-related
                if self.np_random.random() < 0.5:
                    start_floor, target_floor = 0, np.random.randint(1, self.num_floors - 1)
                else:
                    start_floor, target_floor = np.random.randint(1, self.num_floors - 1), 0
            else:  # 30% inter-floor
                start_floor = np.random.randint(0, self.num_floors - 1)
                target_floor = self.np_random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        elif 6 <= current_hour < 9:  # Morning rush (6 AM - 9 AM)
            # Heavy upward traffic (people coming to work)
            if self.np_random.random() < 0.85:  # 85% ground to floors
                start_floor = 0
                target_floor = np.random.randint(1, self.num_floors - 1)
            else:  # 15% other
                start_floor = np.random.randint(1, self.num_floors - 1)
                target_floor = self.np_random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        elif 9 <= current_hour < 17:  # Daytime (9 AM - 5 PM)
            # Mixed traffic with some inter-floor movement
            rand_val = self.np_random.random()
            if rand_val < 0.8:  # 80% ground-related
                if self.np_random.random() < 0.5:
                    start_floor, target_floor = 0, np.random.randint(1, self.num_floors - 1)
                else:
                    start_floor, target_floor = np.random.randint(1, self.num_floors - 1), 0
            else:  # 20% inter-floor
                start_floor = np.random.randint(1, self.num_floors - 1)
                target_floor = self.np_random.choice([f for f in range(1, self.num_floors) if f != start_floor])
        
        elif 17 <= current_hour < 20:  # Evening rush (5 PM - 8 PM)
            # Heavy downward traffic (people going home)
            if self.np_random.random() < 0.85:  # 85% floors to ground
                start_floor = np.random.randint(1, self.num_floors - 1)
                target_floor = 0
            else:  # 15% other
                start_floor = np.random.randint(0, self.num_floors - 1)
                target_floor = self.np_random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        else:  # Evening (8 PM - 12 AM)
            # Moderate traffic, mostly ground-related
            if self.np_random.random() < 0.7:  # 70% ground-related
                if self.np_random.random() < 0.5:
                    start_floor, target_floor = 0, np.random.randint(1, self.num_floors - 1)
                else:
                    start_floor, target_floor = np.random.randint(1, self.num_floors - 1), 0
            else:  # 30% inter-floor
                start_floor = np.random.randint(1, self.num_floors - 1)
                target_floor = self.np_random.choice([f for f in range(1, self.num_floors) if f != start_floor])
        
        while target_floor == start_floor:
            target_floor = self.np_random.integers(0, self.num_floors)
        
        return (start_floor, target_floor)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Process RL action
        self._process_rl_action(action)
        
        # Add any passengers scheduled for the current time
        self._add_scheduled_passengers()

        # Advance simulation by one step
        self.building.step()
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination
        terminated = self.building.time >= self.episode_length
        truncated = False # For now, we don't use truncation
        
        # Get next observation
        observation = self._get_state_representation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _process_rl_action(self, action: np.ndarray):
        """Assigns the target floor from the action to each idle elevator."""
        if self.action_type == 'continuous':
            # Scale continuous actions from [-1, 1] to [0, num_floors-1]
            action = (action + 1) / 2 * (self.num_floors - 1)
            action = np.round(action).astype(int)

        for i, target_floor in enumerate(action):
            elevator = self.building.elevators[i]
            # Only assign a new target if the elevator is idle
            # This prevents interrupting an elevator that is already servicing requests
            if elevator.is_idle() and not elevator.target_floors:
                # The action is the destination floor
                destination_floor = int(target_floor)
                if destination_floor != elevator.current_floor:
                    elevator.assign_target(destination_floor)
    
    def _add_scheduled_passengers(self):
        """Add passengers scheduled for current time"""
        current_time = self.building.time
        
        while (self.next_passenger_idx < len(self.passenger_sequence) and 
               self.passenger_sequence[self.next_passenger_idx][0] <= current_time):
            _, start_floor, target_floor = self.passenger_sequence[self.next_passenger_idx]
            self.building.add_passenger(start_floor, target_floor)
            self.next_passenger_idx += 1
    
    def _get_info(self) -> Dict:
        """Get additional info for debugging and evaluation."""
        completed_passengers = self.building.completed_passengers
        num_completed = len(completed_passengers)
        
        if num_completed > 0:
            avg_wait_time = sum(p.waiting_time for p in completed_passengers) / num_completed
            avg_journey_time = sum(p.total_time for p in completed_passengers) / num_completed
        else:
            avg_wait_time = 0
            avg_journey_time = 0
            
        return {
            "passengers_completed": num_completed,
            "average_wait_time": avg_wait_time,
            "average_journey_time": avg_journey_time,
            "total_reward": self.total_reward,
            "sim_time": self.building.time
        }
    
    def render(self, mode='human'):
        """Render the environment (not implemented for headless)."""
        if not self.headless and self.gui:
            self.gui.root.update()
    
    def close(self):
        """Clean up resources."""
        if not self.headless and self.gui:
            self.gui.root.destroy()
    
    def _init_gui(self):
        """Initialize the GUI."""
        # This part is complex and depends on running Tkinter in a separate thread
        # For now, we assume headless operation for training.
        print("GUI initialization is not fully supported in this script version for training.")
        self.headless = True