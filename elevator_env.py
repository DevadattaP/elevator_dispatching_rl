import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from building import Building

class ElevatorEnv(gym.Env):
    def __init__(self, 
                 num_floors: int = 10,
                 num_elevators: int = 4,
                 lift_capacity: int = 8,
                 speed_multiplier: float = 10.0,
                 episode_length: int = 3600,  # 1 hour in simulation seconds
                 headless: bool = True,
                 passenger_generation_rate: float = 1.0,
                 verbose: int = 0):
        
        super().__init__()
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.episode_length = episode_length
        self.headless = headless
        self.passenger_generation_rate = passenger_generation_rate
        self.verbose = verbose
        
        # Initialize building (without GUI)
        self.building = Building(num_floors, num_elevators, speed_multiplier, lift_capacity, verbose=(verbose > 0))
        
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
        """Define observation space by calculating exact dimension"""
        obs_dim = self._calculate_observation_dimension()
        
        # Use more realistic bounds based on your actual data ranges
        return spaces.Box(
            low=-1.0, 
            high=100.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _calculate_observation_dimension(self) -> int:
        """Calculate exact observation dimension to avoid shape mismatches"""
        obs_dim = 0
        
        # Elevator states: 4 values per elevator
        obs_dim += self.num_elevators * 4  # position, direction, state, passenger_count
        
        # Floor states: 2 values per floor
        obs_dim += self.num_floors * 2  # waiting_up, waiting_down
        
        # External calls: 2 values per floor per elevator (up, down)
        obs_dim += self.num_floors * self.num_elevators * 2
        
        # Time of day: 2 values (sin, cos encoding)
        obs_dim += 2
        
        if self.verbose > 0:
            print(f"Observation dimension: {obs_dim}")
        return obs_dim

    def _define_action_space(self) -> spaces.Space:
        """Define simplified action space for initial testing"""
        # Start with a simple discrete action space
        # Actions: 0 = use existing rule-based, 1-7 = different dispatching strategies
        return spaces.Discrete(8)
        
        # For more complex actions, we can use:
        # return spaces.MultiDiscrete([self.num_elevators] * self.num_floors * 2)

    def _get_state_representation(self) -> np.ndarray:
        """Convert building state to RL observation vector - FIXED DIMENSION"""
        state = self.building.get_state()
        obs = []
        
        # 1. Elevator information (4 values per elevator)
        for elevator_state in state['elevators']:
            obs.extend([
                elevator_state['position'] / self.num_floors,  # Normalized [0,1]
                (elevator_state['direction'] + 1) / 2,        # [-1,0,1] -> [0,0.5,1]
                elevator_state['state'] / 5.0,                # Normalize state enum [0,1]
                min(elevator_state['passenger_count'] / 8.0, 1.0)  # Cap at 1.0
            ])
        
        # 2. Floor waiting queues (2 values per floor)
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            obs.extend([
                min(floor_state['waiting_up'] / 10.0, 1.0),    # Normalized, capped
                min(floor_state['waiting_down'] / 10.0, 1.0)
            ])
        
        # 3. External calls per elevator (2 values per floor per elevator)
        for floor in range(self.num_floors):
            for elevator_id in range(self.num_elevators):
                calls = state['floors'][floor]['elevator_calls'][elevator_id]
                obs.extend([
                    1.0 if calls['call_up'] else 0.0,
                    1.0 if calls['call_down'] else 0.0
                ])
        
        # 4. Time of day (2 values - cyclic encoding)
        current_time = state['time'] % 86400  # Seconds in day
        obs.append(np.sin(2 * np.pi * current_time / 86400))
        obs.append(np.cos(2 * np.pi * current_time / 86400))
        
        obs_array = np.array(obs, dtype=np.float32)
        
        # Debug: Check dimension matches
        expected_dim = self._calculate_observation_dimension()
        if len(obs_array) != expected_dim:
            if self.verbose > 0:
                print(f"WARNING: Observation dimension mismatch. Expected: {expected_dim}, Got: {len(obs_array)}")
        
        return obs_array

    def _calculate_reward(self) -> float:
        """Calculate reward based on passenger satisfaction"""
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

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        # Reset building
        self.building = Building(self.num_floors, self.num_elevators)
        self.building.set_speed_multiplier(10.0)  # Faster for training
        
        # Generate passenger sequence for this episode
        self._generate_passenger_sequence()
        
        # Reset episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.last_completion_count = 0
        
        # Let building run for a bit to initialize
        for _ in range(10):
            self.building.step()
        
        # Get initial state
        observation = self._get_state_representation()
        info = self._get_info()
        
        if self.verbose > 0:
            print(f"Reset complete. Observation shape: {observation.shape}")
        return observation, info

    def _generate_passenger_sequence(self):
        """Generate a fixed passenger sequence for this episode"""
        self.passenger_sequence = []
        current_time = 0
        episode_end = self.episode_length
        
        while current_time < episode_end:
            # Generate passenger using simplified distribution
            passenger_data = self._generate_passenger_at_time(current_time)
            if passenger_data:
                self.passenger_sequence.append((current_time, passenger_data))
            
            # Time until next passenger
            rate = self.passenger_generation_rate
            time_until_next = np.random.exponential(1.0 / max(rate, 0.1))
            current_time += time_until_next
        
        # Sort by time
        self.passenger_sequence.sort(key=lambda x: x[0])
        self.next_passenger_idx = 0
        
        if self.verbose > 0:
            print(f"Generated {len(self.passenger_sequence)} passengers for episode")

    def _generate_passenger_at_time(self, current_time: float):
        """Generate a single random passenger with enhanced distribution"""
        # Get current simulation hour (0-23)
        current_hour = (current_time % 86400) / 3600
        
        # Enhanced distribution based on time of day
        if current_hour < 6:  # Night (12 AM - 6 AM)
            # Very low traffic, mostly random
            if random.random() < 0.7:  # 30% ground-related
                if random.random() < 0.5:
                    start_floor, target_floor = 0, random.randint(1, self.num_floors - 1)
                else:
                    start_floor, target_floor = random.randint(1, self.num_floors - 1), 0
            else:  # 30% inter-floor
                start_floor = random.randint(0, self.num_floors - 1)
                target_floor = random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        elif 6 <= current_hour < 9:  # Morning rush (6 AM - 9 AM)
            # Heavy upward traffic (people coming to work)
            if random.random() < 0.85:  # 85% ground to floors
                start_floor = 0
                target_floor = random.randint(1, self.num_floors - 1)
            else:  # 15% other
                start_floor = random.randint(1, self.num_floors - 1)
                target_floor = random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        elif 9 <= current_hour < 17:  # Daytime (9 AM - 5 PM)
            # Mixed traffic with some inter-floor movement
            rand_val = random.random()
            if rand_val < 0.8:  # 80% ground-related
                if random.random() < 0.5:
                    start_floor, target_floor = 0, random.randint(1, self.num_floors - 1)
                else:
                    start_floor, target_floor = random.randint(1, self.num_floors - 1), 0
            else:  # 20% inter-floor
                start_floor = random.randint(1, self.num_floors - 1)
                target_floor = random.choice([f for f in range(1, self.num_floors) if f != start_floor])
        
        elif 17 <= current_hour < 20:  # Evening rush (5 PM - 8 PM)
            # Heavy downward traffic (people going home)
            if random.random() < 0.85:  # 85% floors to ground
                start_floor = random.randint(1, self.num_floors - 1)
                target_floor = 0
            else:  # 15% other
                start_floor = random.randint(0, self.num_floors - 1)
                target_floor = random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        else:  # Evening (8 PM - 12 AM)
            # Moderate traffic, mostly ground-related
            if random.random() < 0.7:  # 70% ground-related
                if random.random() < 0.5:
                    start_floor, target_floor = 0, random.randint(1, self.num_floors - 1)
                else:
                    start_floor, target_floor = random.randint(1, self.num_floors - 1), 0
            else:  # 30% inter-floor
                start_floor = random.randint(1, self.num_floors - 1)
                target_floor = random.choice([f for f in range(1, self.num_floors) if f != start_floor])
        
        while target_floor == start_floor:
            target_floor = np.random.randint(0, self.num_floors)
        
        return (start_floor, target_floor)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Process RL action
        self._process_rl_action(action)
        
        # Advance simulation
        simulation_steps = 5  # Fewer steps for stability
        for _ in range(simulation_steps):
            self._add_scheduled_passengers()
            self.building.step()
            self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination
        terminated = self.building.time >= self.episode_length
        truncated = False
        
        # Get next observation
        observation = self._get_state_representation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _process_rl_action(self, action: int):
        """Process RL action - simple implementation for testing"""
        # For now, we'll use action to modify dispatching behavior
        # Action 0: Use existing rule-based system
        # Action 1-7: Can implement different strategies
        
        if action > 0:
            # Example: Modify the ETA calculation weights
            # This is where you'll implement your RL-based dispatching logic
            pass
    
    def _add_scheduled_passengers(self):
        """Add passengers scheduled for current time"""
        current_time = self.building.time
        
        while (self.next_passenger_idx < len(self.passenger_sequence) and 
               self.passenger_sequence[self.next_passenger_idx][0] <= current_time):
            
            spawn_time, (start_floor, target_floor) = self.passenger_sequence[self.next_passenger_idx]
            self.building.add_passenger(start_floor, target_floor)
            self.next_passenger_idx += 1
    
    def _get_info(self) -> Dict:
        """Get additional info for debugging"""
        state = self.building.get_state()
        return {
            'total_waiting': sum(floor['waiting_up'] + floor['waiting_down'] 
                                for floor in state['floors'].values()),
            'completed_passengers': len(self.building.completed_passengers),
            'total_reward': self.total_reward,
            'time': self.building.time
        }
    
    def render(self):
        """Render the environment"""
        if not self.headless and self.gui:
            self.gui.update_display()
        elif self.headless:
            # Print text-based rendering for headless mode
            state = self.building.get_state()
            if self.verbose > 0:
                print(f"Step: {self.current_step}, Time: {state['time']:.1f}s, "
                      f"Waiting: {sum(floor['waiting_up'] + floor['waiting_down'] for floor in state['floors'].values())}, "
                      f"Reward: {self.total_reward:.2f}")
    
    def close(self):
        """Clean up environment"""
        if self.gui:
            self.gui.root.destroy()
    
    def _init_gui(self):
        """Initialize GUI for visualization"""
        # You can modify your existing GUI to work with this environment
        # For now, we'll run headless for training
        pass