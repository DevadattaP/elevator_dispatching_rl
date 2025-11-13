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
                 observation_type: str = 'enhanced', # 'simple', 'detailed', 'enhanced'
                 action_type: str = 'combinatorial', # 'discrete', 'continuous', 'combinatorial', 'assignment'
                 reward_type: str = 'fairness', # 'simple', 'complex', 'fairness', 'squared'
                 use_smdp: bool = True,  # Semi-Markov Decision Process
                 traffic_pattern: str = 'mixed',  # 'up_peak', 'down_peak', 'mixed', 'all_in_one'
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
        self.use_smdp = use_smdp
        self.traffic_pattern = traffic_pattern
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
        self.last_decision_time = 0.0
        
        # For traffic pattern awareness (like Wan et al.)
        self.passenger_arrival_rates = np.zeros(num_floors * 2)  # up/down arrival rates
        self.traffic_context = np.zeros(4)  # up_peak, down_peak, lunch, interfloor indicators
        
        # For reward calculation
        self.accumulated_waiting_penalty = 0.0
        self.last_passenger_count = 0
        
        # For GUI mode
        self.gui = None
        if not headless:
            self._init_gui()
    
    def _define_observation_space(self) -> spaces.Space:
        """Define observation space based on the configured type."""
        obs_dim = self._calculate_observation_dimension()
        
        return spaces.Box(
            low=-1.0, 
            high=float(self.episode_length), 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _calculate_observation_dimension(self) -> int:
        """Calculate exact observation dimension."""
        obs_dim = 0
        
        # Base states for all observation types
        # Elevator states: position, direction, state, passenger_count, destination_count
        obs_dim += self.num_elevators * 5
        
        # Floor waiting queues: waiting_up, waiting_down, elapsed_time_up, elapsed_time_down
        obs_dim += self.num_floors * 4
        
        # Time encoding: sin, cos of time of day
        obs_dim += 2
        
        if self.observation_type in ['detailed', 'enhanced']:
            # Add internal destinations for each elevator (one-hot encoded)
            obs_dim += self.num_elevators * self.num_floors
            
            # Add elevator "busyness" metrics (like ETD scores)
            obs_dim += self.num_elevators * 3  # ETD score, queue_length, load_factor
            
            # Add hall call elapsed times (like Crites & Barto)
            obs_dim += self.num_floors * 2  # up and down call times
            
        if self.observation_type == 'enhanced':
            # Traffic pattern awareness (like Wan et al.)
            obs_dim += 4  # traffic context indicators
            obs_dim += self.num_floors * 2  # passenger arrival rates (up/down)
            
            # Other elevator positions ("footprint" - like Crites & Barto)
            obs_dim += self.num_elevators * self.num_floors
            
            # System-level metrics
            obs_dim += 3  # total_waiting, avg_wait_time, system_load

        if self.verbose > 0:
            print(f"Observation type: '{self.observation_type}', Dimension: {obs_dim}")
        return obs_dim

    def _define_action_space(self) -> spaces.Space:
        """Define the action space based on research paper insights."""
        if self.action_type == 'discrete':
            # Original per-elevator floor assignment
            return spaces.MultiDiscrete([self.num_floors + 1] * self.num_elevators)
        
        elif self.action_type == 'continuous':
            # Continuous action space for floor assignment
            return spaces.Box(low=-1.0, high=1.0, shape=(self.num_elevators,), dtype=np.float32)
        
        elif self.action_type == 'combinatorial':
            # Like Vaartjes et al. - choose which elevators respond to new hall calls
            # Action: binary vector indicating which elevators should respond
            return spaces.MultiBinary(self.num_elevators)
        
        elif self.action_type == 'assignment':
            # Assign new hall call to specific elevator (when hall call occurs)
            return spaces.Discrete(self.num_elevators + 1)  # +1 for "no assignment yet"
        
        else:
            raise NotImplementedError(f"Action type '{self.action_type}' not implemented.")

    def _get_state_representation(self) -> np.ndarray:
        """Convert building state to RL observation vector with enhanced features."""
        state = self.building.get_state()
        obs = []
        
        # 1. Enhanced elevator information (5 values per elevator)
        for i, elevator_state in enumerate(state['elevators']):
            elevator = self.building.elevators[i]
            
            # Handle None direction
            direction = elevator_state['direction'] or 0
            
            # Calculate busyness metrics (like ETD scores)
            destination_count = len(elevator.target_floors)
            load_factor = elevator_state['passenger_count'] / elevator.capacity
            # Simple ETD-like score: estimated time to complete current assignments
            etd_score = destination_count * 10.0  # Simplified estimation
            
            obs.extend([
                elevator_state['position'] / self.num_floors,
                (direction + 1) / 2,
                elevator_state['state'] / len(ElevatorState),
                load_factor,  # Already normalized [0,1]
                min(destination_count / self.num_floors, 1.0)
            ])
        
        # 2. Enhanced floor waiting queues with elapsed times
        max_waiting = 10.0
        max_wait_time = 300.0  # 5 minutes
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            
            # Calculate elapsed times for hall calls
            up_elapsed = self._get_hall_call_elapsed_time(floor, 'up')
            down_elapsed = self._get_hall_call_elapsed_time(floor, 'down')
            
            obs.extend([
                min(floor_state['waiting_up'] / max_waiting, 1.0),
                min(floor_state['waiting_down'] / max_waiting, 1.0),
                min(up_elapsed / max_wait_time, 1.0),
                min(down_elapsed / max_wait_time, 1.0)
            ])
        
        # 3. Time of day encoding
        current_time = state['time'] % 86400
        obs.append(np.sin(2 * np.pi * current_time / 86400))
        obs.append(np.cos(2 * np.pi * current_time / 86400))

        # 4. Detailed features for enhanced observation types
        if self.observation_type in ['detailed', 'enhanced']:
            # Internal destinations (one-hot per elevator)
            for elevator in self.building.elevators:
                internal_requests = [0] * self.num_floors
                for p in elevator.passengers:
                    internal_requests[p.target_floor] = 1
                obs.extend(internal_requests)
            
            # Elevator busyness metrics
            for i, elevator in enumerate(self.building.elevators):
                # More detailed ETD calculation
                queue_length = len(elevator.target_floors)
                passenger_load = len(elevator.passengers) / elevator.capacity
                # Estimated time based on floors to travel
                if elevator.target_floors:
                    next_floor = elevator._get_next_target_floor()
                    travel_estimate = self.building.calculate_eta(elevator.id, next_floor)
                else:
                    travel_estimate = 0.0
                
                obs.extend([
                    min(queue_length / self.num_floors, 1.0),
                    passenger_load,
                    min(travel_estimate / 60.0, 1.0)  # Normalize to 1 minute
                ])
            
            # Hall call elapsed times (like Crites & Barto)
            for floor in range(self.num_floors):
                up_time = self._get_hall_call_elapsed_time(floor, 'up')
                down_time = self._get_hall_call_elapsed_time(floor, 'down')
                obs.extend([
                    min(up_time / max_wait_time, 1.0),
                    min(down_time / max_wait_time, 1.0)
                ])
        
        # 5. Traffic pattern awareness (like Wan et al.)
        if self.observation_type == 'enhanced':
            # Traffic context indicators
            current_hour = (state['time'] % 86400) / 3600
            self._update_traffic_context(current_hour)
            obs.extend(self.traffic_context)
            
            # Passenger arrival rates
            self._update_arrival_rates()
            obs.extend(self.passenger_arrival_rates)
            
            # Other elevator positions ("footprint")
            for i, elevator in enumerate(self.building.elevators):
                position_vector = [0] * self.num_floors
                if 0 <= elevator.current_floor < self.num_floors:
                    position_vector[elevator.current_floor] = 1
                obs.extend(position_vector)
            
            # System-level metrics
            total_waiting = sum(len(q) for q in self.building.active_passengers.values())
            avg_wait = self._calculate_average_waiting_time()
            system_load = total_waiting / (self.num_elevators * elevator.capacity)
            
            obs.extend([
                min(total_waiting / 50.0, 1.0),  # Normalize
                min(avg_wait / 300.0, 1.0),      # Normalize to 5 minutes
                min(system_load, 1.0)
            ])

        obs_array = np.array(obs, dtype=np.float32)
        
        # Safety checks
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        obs_array = np.clip(obs_array, -1.0, 1.0)
        
        expected_dim = self._calculate_observation_dimension()
        if len(obs_array) != expected_dim:
            raise ValueError(f"Observation dimension mismatch! Expected {expected_dim}, got {len(obs_array)}")
        
        return obs_array

    def _get_hall_call_elapsed_time(self, floor: int, direction: str) -> float:
        """Get elapsed time since hall call was made (like Crites & Barto)."""
        # This would need to be implemented in your Building class
        # For now, return a simplified version
        if direction == 'up' and floor in self.building.external_calls.get('up', []):
            return 10.0  # Simplified
        elif direction == 'down' and floor in self.building.external_calls.get('down', []):
            return 10.0  # Simplified
        return 0.0

    def _update_traffic_context(self, current_hour: float):
        """Update traffic pattern context indicators (like Wan et al.)."""
        self.traffic_context = np.zeros(4)
        
        if 7 <= current_hour < 10:  # Morning up-peak
            self.traffic_context[0] = 1.0
        elif 12 <= current_hour < 14:  # Lunch peak
            self.traffic_context[2] = 1.0
        elif 17 <= current_hour < 19:  # Evening down-peak
            self.traffic_context[1] = 1.0
        else:  # Inter-floor
            self.traffic_context[3] = 1.0

    def _update_arrival_rates(self):
        """Update passenger arrival rate estimates."""
        # Simplified implementation - in practice, this would track recent arrivals
        # and use exponential moving average like Wan et al.
        for floor in range(self.num_floors):
            # Placeholder - would need actual arrival tracking
            self.passenger_arrival_rates[floor * 2] = 0.1  # up rate
            self.passenger_arrival_rates[floor * 2 + 1] = 0.1  # down rate

    def _calculate_average_waiting_time(self) -> float:
        """Calculate average waiting time for current waiting passengers."""
        waiting_times = []
        for floor_passengers in self.building.active_passengers.values():
            for passenger in floor_passengers:
                waiting_times.append(passenger.waiting_time)
        return np.mean(waiting_times) if waiting_times else 0.0

    def _calculate_reward(self) -> float:
        """Calculate reward based on research paper insights."""
        if self.reward_type == 'simple':
            return self._calculate_simple_reward()
        elif self.reward_type == 'complex':
            return self._calculate_complex_reward()
        elif self.reward_type == 'fairness':
            return self._calculate_fairness_reward()
        elif self.reward_type == 'squared':
            return self._calculate_squared_reward()
        else:
            raise NotImplementedError(f"Reward type '{self.reward_type}' not implemented.")

    def _calculate_simple_reward(self) -> float:
        """Original simple reward."""
        reward = 0.0
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        reward += new_completions * 1.0
        
        total_waiting = sum(len(q) for q in self.building.active_passengers.values())
        reward -= total_waiting * 0.01

        self.last_completion_count = len(self.building.completed_passengers)
        return np.clip(reward, -10.0, 10.0)

    def _calculate_complex_reward(self) -> float:
        """Enhanced complex reward with multiple components."""
        reward = 0.0
        
        # 1. Passenger completion reward
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        reward += new_completions * 2.0
        
        # 2. Waiting penalty (like all papers)
        total_waiting = sum(len(q) for q in self.building.active_passengers.values())
        reward -= total_waiting * 0.02
        
        # 3. Travel time penalty for new completions
        if new_completions > 0:
            recent_passengers = self.building.completed_passengers[-new_completions:]
            travel_times = [p.total_time for p in recent_passengers if p.total_time is not None]
            if travel_times:
                avg_travel_time = np.mean(travel_times)
                reward -= min(avg_travel_time / 100.0, 1.0)
        
        # 4. Energy penalty (like Vaartjes et al.)
        moving_elevators = sum(1 for e in self.building.elevators if e.is_moving())
        reward -= moving_elevators * 0.01
        
        # 5. Full elevator penalty (like Vaartjes et al.)
        for elevator in self.building.elevators:
            if len(elevator.passengers) >= elevator.capacity:
                reward -= 0.1
        
        self.last_completion_count = len(self.building.completed_passengers)
        return np.clip(reward, -10.0, 10.0)

    def _calculate_fairness_reward(self) -> float:
        """Reward function that encourages fairness (like Crites & Barto)."""
        reward = 0.0
        
        # Completion reward
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        reward += new_completions * 1.5
        
        # Squared waiting penalty to encourage fairness
        waiting_penalty = 0.0
        for floor_passengers in self.building.active_passengers.values():
            for passenger in floor_passengers:
                # Quadratic penalty for long waits (encourages fairness)
                waiting_penalty += (passenger.waiting_time / 100.0) ** 2
        
        reward -= waiting_penalty * 0.01
        
        # Bonus for serving longest-waiting passengers
        longest_waits = []
        for floor_passengers in self.building.active_passengers.values():
            if floor_passengers:
                longest_waits.append(max(p.waiting_time for p in floor_passengers))
        
        if longest_waits:
            max_wait = max(longest_waits)
            if max_wait > 60:  # If someone waiting more than 60 seconds
                reward -= (max_wait - 60) * 0.1
        
        self.last_completion_count = len(self.building.completed_passengers)
        return np.clip(reward, -10.0, 10.0)

    def _calculate_squared_reward(self) -> float:
        """Squared wait time reward like Crites & Barto."""
        reward = 0.0
        
        # Completion reward
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        reward += new_completions * 1.0
        
        # Squared waiting time penalty (main component from Crites & Barto)
        total_squared_wait = 0.0
        current_time = self.building.time
        
        # For waiting passengers
        for floor_passengers in self.building.active_passengers.values():
            for passenger in floor_passengers:
                wait_time = passenger.waiting_time
                total_squared_wait += (wait_time / 100.0) ** 2
        
        # For passengers inside elevators
        for elevator in self.building.elevators:
            for passenger in elevator.passengers:
                travel_time = current_time - passenger.boarding_time
                total_squared_wait += (travel_time / 100.0) ** 2
        
        reward -= total_squared_wait * 0.005
        
        self.last_completion_count = len(self.building.completed_passengers)
        return np.clip(reward, -10.0, 10.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        # Reset building
        self.building = Building(self.num_floors, self.num_elevators, 
                               capacity=self.building.elevators[0].capacity, 
                               verbose=(self.verbose > 1))
        self.building.set_speed_multiplier(self.building.speed_multiplier)
        
        # Generate passenger sequence based on traffic pattern
        self._generate_passenger_sequence()
        
        # Reset episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.last_completion_count = 0
        self.last_decision_time = 0.0
        self.accumulated_waiting_penalty = 0.0
        self.last_passenger_count = 0
        
        # Reset traffic tracking
        self.passenger_arrival_rates = np.zeros(self.num_floors * 2)
        self.traffic_context = np.zeros(4)
        
        # Add initial passengers
        self._add_scheduled_passengers()
        
        # Get initial state
        observation = self._get_state_representation()
        info = self._get_info()
        
        if self.verbose > 0:
            print("Environment reset.")
        return observation, info

    def _generate_passenger_sequence(self):
        """Generate passenger sequence based on traffic pattern."""
        self.passenger_sequence = []
        current_time = 0
        episode_end = self.episode_length
        
        rng = np.random.default_rng(self.np_random)
        
        while current_time < episode_end:
            time_increment = rng.exponential(1.0 / (self.passenger_generation_rate / 10.0))
            current_time += time_increment
            if current_time < episode_end:
                start_floor, target_floor = self._generate_passenger_for_pattern(current_time)
                self.passenger_sequence.append((current_time, start_floor, target_floor))
        
        self.passenger_sequence.sort(key=lambda x: x[0])
        self.next_passenger_idx = 0
        
        if self.verbose > 0:
            print(f"Generated {len(self.passenger_sequence)} passengers for {self.traffic_pattern} pattern.")

    def _generate_passenger_for_pattern(self, current_time: float):
        """Generate passenger based on configured traffic pattern."""
        if self.traffic_pattern == 'up_peak':
            return self._generate_up_peak_passenger(current_time)
        elif self.traffic_pattern == 'down_peak':
            return self._generate_down_peak_passenger(current_time)
        elif self.traffic_pattern == 'mixed':
            return self._generate_mixed_passenger(current_time)
        elif self.traffic_pattern == 'all_in_one':
            return self._generate_all_in_one_passenger(current_time)
        else:
            return self._generate_mixed_passenger(current_time)

    def _generate_up_peak_passenger(self, current_time: float):
        """Generate passenger for up-peak traffic (morning rush)."""
        if self.np_random.random() < 0.85:  # 85% from ground floor up
            start_floor = 0
            target_floor = self.np_random.integers(1, self.num_floors)
        else:  # 15% other traffic
            start_floor = self.np_random.integers(1, self.num_floors)
            target_floor = self.np_random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        return start_floor, target_floor

    def _generate_down_peak_passenger(self, current_time: float):
        """Generate passenger for down-peak traffic (evening rush)."""
        if self.np_random.random() < 0.85:  # 85% from floors down to ground
            start_floor = self.np_random.integers(1, self.num_floors)
            target_floor = 0
        else:  # 15% other traffic
            start_floor = self.np_random.integers(0, self.num_floors)
            target_floor = self.np_random.choice([f for f in range(self.num_floors) if f != start_floor])
        
        return start_floor, target_floor

    def _generate_all_in_one_passenger(self, current_time: float):
        """Generate passenger for mixed traffic throughout day (like Wan et al.)."""
        current_hour = (current_time % 86400) / 3600
        
        if current_hour < 6:  # Night
            return self._generate_mixed_passenger(current_time)
        elif 6 <= current_hour < 9:  # Morning up-peak
            return self._generate_up_peak_passenger(current_time)
        elif 9 <= current_hour < 12:  # Late morning mixed
            return self._generate_mixed_passenger(current_time)
        elif 12 <= current_hour < 14:  # Lunch peak
            # Balanced up/down traffic
            if self.np_random.random() < 0.5:
                start_floor, target_floor = 0, self.np_random.integers(1, self.num_floors)
            else:
                start_floor, target_floor = self.np_random.integers(1, self.num_floors), 0
        elif 14 <= current_hour < 17:  # Afternoon mixed
            return self._generate_mixed_passenger(current_time)
        else:  # Evening down-peak
            return self._generate_down_peak_passenger(current_time)

    def _generate_mixed_passenger(self, current_time: float):
        """Generate passenger for mixed traffic."""
        start_floor = self.np_random.integers(0, self.num_floors)
        target_floor = self.np_random.choice([f for f in range(self.num_floors) if f != start_floor])
        return start_floor, target_floor

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with SMDP support."""
        # Process RL action
        self._process_rl_action(action)
        
        # Add any passengers scheduled for the current time
        self._add_scheduled_passengers()

        # Advance simulation
        self.building.step()
        self.current_step += 1
        
        # Calculate reward with SMDP discounting if enabled
        if self.use_smdp:
            reward = self._calculate_smdp_reward()
        else:
            reward = self._calculate_reward()
            
        self.total_reward += reward
        
        # Check termination
        terminated = self.building.time >= self.episode_length
        truncated = False
        
        # Get next observation
        observation = self._get_state_representation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _calculate_smdp_reward(self) -> float:
        """Calculate reward with SMDP discounting (like Wan et al.)."""
        current_time = self.building.time
        time_since_last = current_time - self.last_decision_time
        
        # Calculate instantaneous reward (like Wan et al.)
        instant_reward = 0.0
        
        # Penalty based on current waiting passengers (r^1 from Wan et al.)
        for floor_passengers in self.building.active_passengers.values():
            for passenger in floor_passengers:
                instant_reward -= passenger.waiting_time / 1000.0  # Small penalty
        
        # Apply exponential discounting for SMDP
        beta = 0.01  # Discount rate (like Wan et al.)
        discounted_reward = instant_reward * (1 - np.exp(-beta * time_since_last)) / beta
        
        # Add completion rewards (undiscounted)
        new_completions = len(self.building.completed_passengers) - self.last_completion_count
        discounted_reward += new_completions * 1.0
        
        self.last_decision_time = current_time
        self.last_completion_count = len(self.building.completed_passengers)
        
        return np.clip(discounted_reward, -10.0, 10.0)

    def _process_rl_action(self, action: np.ndarray):
        """Process action based on action type."""
        # Debug print to see what action we're receiving
        if self.verbose > 1:
            print(f"Processing action: {action}, type: {type(action)}, shape: {getattr(action, 'shape', 'No shape')}")
        
        if self.action_type == 'continuous':
            action = (action + 1) / 2 * self.num_floors
            action = np.round(action).astype(int)

        if self.action_type in ['discrete', 'continuous']:
            # Original per-elevator assignment
            for i, target_floor in enumerate(action):
                elevator = self.building.elevators[i]
                if target_floor == self.num_floors:
                    continue
                if elevator.is_idle() and not elevator.target_floors:
                    destination_floor = int(target_floor)
                    if destination_floor != elevator.current_floor:
                        elevator.assign_target(destination_floor)
                        
        elif self.action_type == 'combinatorial':
            # FIX: Properly handle different action formats
            assigned_elevators = []
            
            # Case 1: Action is a single integer (from DQN discrete space)
            if np.isscalar(action):
                action_int = int(action)
                assigned_elevators = [i for i in range(self.num_elevators) if (action_int >> i) & 1]
            
            # Case 2: Action is a numpy array with single value
            elif isinstance(action, np.ndarray) and action.size == 1:
                action_int = int(action.item())
                assigned_elevators = [i for i in range(self.num_elevators) if (action_int >> i) & 1]
            
            # Case 3: Action is a numpy array with multiple values (binary vector)
            elif isinstance(action, np.ndarray) and action.size > 1:
                # Convert to list and check each element individually
                action_list = action.tolist()
                assigned_elevators = [i for i, assigned in enumerate(action_list) if assigned]
            
            # Case 4: Action is already a list
            elif isinstance(action, (list, tuple)):
                assigned_elevators = [i for i, assigned in enumerate(action) if assigned]
            
            else:
                if self.verbose > 0:
                    print(f"Warning: Unhandled action format: {type(action)}")
                return
            
            if self.verbose > 1:
                print(f"Combinatorial action - Assigned elevators: {assigned_elevators}")
            
            # Assign elevators to serve pending hall calls
            pending_calls = self._get_pending_hall_calls()
            if assigned_elevators and pending_calls:
                for call_floor, call_direction in pending_calls:
                    for elevator_id in assigned_elevators:
                        if elevator_id < len(self.building.elevators):
                            elevator = self.building.elevators[elevator_id]
                            if len(elevator.passengers) < elevator.capacity:
                                elevator.assign_target(call_floor)
                                if self.verbose > 1:
                                    print(f"Assigned elevator {elevator_id} to floor {call_floor}")
                                break  # Assign each call to one elevator
                        
        elif self.action_type == 'assignment':
            # FIX: Handle different action formats for assignment
            if np.isscalar(action):
                elevator_id = int(action)
            elif isinstance(action, np.ndarray):
                elevator_id = int(action.item()) if action.size == 1 else 0
            else:
                elevator_id = 0
                
            if elevator_id < self.num_elevators:
                pending_calls = self._get_pending_hall_calls()
                if pending_calls:
                    call_floor, call_direction = pending_calls[0]
                    elevator = self.building.elevators[elevator_id]
                    if len(elevator.passengers) < elevator.capacity:
                        elevator.assign_target(call_floor)
                        if self.verbose > 1:
                            print(f"Assigned elevator {elevator_id} to floor {call_floor}")

    def _get_pending_hall_calls(self):
        """Get list of pending hall calls that need assignment."""
        pending_calls = []
        
        for floor in range(self.num_floors):
            # Check waiting passengers on this floor
            waiting_passengers = self.building.active_passengers.get(floor, [])
            if waiting_passengers:
                # Group by direction
                up_passengers = [p for p in waiting_passengers if p.direction == 'up']
                down_passengers = [p for p in waiting_passengers if p.direction == 'down']
                
                # Check if elevators are already assigned to this floor
                floor_has_elevator_assigned = False
                for elevator in self.building.elevators:
                    if floor in elevator.target_floors:
                        floor_has_elevator_assigned = True
                        break
                
                if not floor_has_elevator_assigned:
                    if up_passengers and floor < self.num_floors - 1:  # Can go up
                        pending_calls.append((floor, 'up'))
                    if down_passengers and floor > 0:  # Can go down
                        pending_calls.append((floor, 'down'))
        
        return pending_calls

    def _add_scheduled_passengers(self):
        """Add passengers scheduled for current time."""
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
            wait_times = [p.waiting_time for p in completed_passengers]
            journey_times = [p.total_time for p in completed_passengers]
            avg_wait_time = np.mean(wait_times)
            avg_journey_time = np.mean(journey_times)
            max_wait_time = np.max(wait_times)
        else:
            avg_wait_time = avg_journey_time = max_wait_time = 0
            
        # Calculate fairness metric (like Crites & Barto)
        if num_completed > 0:
            squared_wait_times = sum(wt ** 2 for wt in wait_times)
            fairness_metric = squared_wait_times / num_completed
        else:
            fairness_metric = 0
            
        return {
            "passengers_completed": num_completed,
            "average_wait_time": avg_wait_time,
            "average_journey_time": avg_journey_time,
            "max_wait_time": max_wait_time,
            "fairness_metric": fairness_metric,
            "total_reward": self.total_reward,
            "sim_time": self.building.time,
            "traffic_pattern": self.traffic_pattern
        }
    
    def render(self, mode='human'):
        """Render the environment."""
        if not self.headless and self.gui:
            self.gui.root.update()
    
    def close(self):
        """Clean up resources."""
        if not self.headless and self.gui:
            self.gui.root.destroy()
    
    def _init_gui(self):
        """Initialize the GUI."""
        print("GUI initialization is not fully supported in this script version for training.")
        self.headless = True


# ===== Specialized Wrappers =====
class D3QNWrapper(ElevatorEnv):
    """Wrapper optimized for Dueling Double DQN with enhanced observations."""
    def __init__(self, **kwargs):
        kwargs['observation_type'] = 'enhanced'
        kwargs['reward_type'] = 'squared'  # Like Crites & Barto
        super().__init__(**kwargs)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and ensure all attributes are initialized."""
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

class SMDPWrapper(ElevatorEnv):
    """Wrapper for Semi-Markov Decision Process training."""
    def __init__(self, **kwargs):
        kwargs['use_smdp'] = True
        kwargs['observation_type'] = 'enhanced'
        super().__init__(**kwargs)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and ensure all attributes are initialized."""
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

class TrafficAwareWrapper(ElevatorEnv):
    """Wrapper for traffic pattern-aware training (like Wan et al.)."""
    def __init__(self, **kwargs):
        kwargs['traffic_pattern'] = 'all_in_one'
        kwargs['observation_type'] = 'enhanced'
        super().__init__(**kwargs)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and ensure all attributes are initialized."""
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

# ===== Custom Wrappers for Action Space Compatibility =====
class DiscreteCombinatorialWrapper(gym.Wrapper):
    """Convert MultiBinary combinatorial action space to Discrete for DQN."""
    def __init__(self, env):
        super().__init__(env)
        self.n_elevators = env.num_elevators
        # Convert MultiBinary(n_elevators) to Discrete(2^n_elevators)
        self.action_space = spaces.Discrete(2 ** self.n_elevators)
        self.building = env.building
        
    def reset(self, seed=None, options=None):
        """Reset the wrapped environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
        
    def step(self, action):
        # Convert discrete action to binary vector
        # Ensure action is a scalar integer
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        
        binary_action = []
        for i in range(self.n_elevators):
            binary_action.append((action >> i) & 1)
        binary_action = np.array(binary_action, dtype=np.int8)
        
        return self.env.step(binary_action)

class DiscreteAssignmentWrapper(gym.Wrapper):
    """Convert assignment action space to Discrete for DQN."""
    def __init__(self, env):
        super().__init__(env)
        self.n_elevators = env.num_elevators
        # Assignment action space: which elevator to assign (+1 for no assignment)
        self.action_space = spaces.Discrete(self.n_elevators + 1)
        self.building = env.building
        
    def reset(self, seed=None, options=None):
        """Reset the wrapped environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
        
    def step(self, action):
        # Action is already in the right format for assignment
        return self.env.step(action)

class MultiDiscreteWrapper(gym.Wrapper):
    """Convert MultiDiscrete action space to single Discrete for DQN."""
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.action_space = spaces.Discrete(np.prod(self.original_action_space.nvec))
        self.building = env.building
        
    def reset(self, seed=None, options=None):
        """Reset the wrapped environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
        
    def step(self, action):
        # Convert flat action to multi-discrete
        actions = np.unravel_index(action, self.original_action_space.nvec)
        return self.env.step(np.array(actions))

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
