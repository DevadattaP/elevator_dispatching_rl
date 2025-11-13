# elevator_rl_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from building import Building

class ElevatorRLEnv(gym.Env):
    def __init__(self, num_floors=10, num_elevators=4, capacity=8, max_steps=3600, speed_multiplier=10.0):
        super().__init__()
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.capacity = capacity
        self.max_steps = max_steps
        self.current_step = 0
        self.speed_multiplier = speed_multiplier
        # Initialize building
        self.building = Building(num_floors, num_elevators, capacity=capacity, verbose=False, speed_multiplier=self.speed_multiplier)
        self.building.start_passenger_generation()
        
        # Define action and observation space
        # Action: Which elevator to assign to new hall calls (centralized assignment)
        self.action_space = spaces.Discrete(num_elevators)
        
        # State: Combination of elevator states, hall calls, and passenger info
        state_size = (
            num_elevators * 4 +  # elevator: floor, direction, load, state
            num_floors * 2 +     # hall calls: up, down per floor
            num_floors * 2       # waiting passengers: up, down per floor
        )
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(state_size,), dtype=np.float32
        )
        self.total_reward = 0.0
        self.traffic_pattern = "random"  # Placeholder for traffic pattern info
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.building = Building(self.num_floors, self.num_elevators, 
                               capacity=self.capacity, verbose=False, speed_multiplier=self.speed_multiplier)
        self.building.start_passenger_generation()
        self.current_step = 0
        return self._get_state(), self._get_info()
    
    def step(self, action):
        """Execute one time step in the environment"""
        self.current_step += 1
        
        # Process any new hall calls with RL action
        self._process_hall_calls_with_rl(action)
        
        # Advance simulation
        self.building.step()
        
        # Get next state and reward
        next_state = self._get_state()
        reward = self._calculate_reward()
        self.total_reward += reward
        terminated = self.current_step >= self.max_steps
        
        return next_state, reward, terminated, False, self._get_info()
    
    def _get_state(self):
        """Convert building state to RL state vector"""
        state = self.building.get_state()
        feature_vector = []
        
        # Elevator features
        for elevator in state['elevators']:
            # Handle direction (can be None in some states)
            direction = elevator['direction'] or 0  # Convert None to 0
            direction_normalized = (direction + 1) / 2  # -1,0,1 -> 0,0.5,1
            
            # Handle position (ensure it's within bounds)
            position = max(0, min(elevator['position'], self.num_floors - 1))
            position_normalized = position / (self.num_floors - 1) if self.num_floors > 1 else 0
            
            feature_vector.extend([
                position_normalized,
                direction_normalized,
                elevator['passenger_count'] / self.capacity,
                elevator['state'] / 10.0  # Normalize state enum
            ])
        
        # Hall call features
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            
            # Check if there are active up/down calls for ANY elevator on this floor
            has_up_call = any(
                state['floors'][floor]['elevator_calls'][elev_id]['call_up'] 
                for elev_id in range(self.num_elevators)
            )
            has_down_call = any(
                state['floors'][floor]['elevator_calls'][elev_id]['call_down'] 
                for elev_id in range(self.num_elevators)
            )
            
            feature_vector.extend([float(has_up_call), float(has_down_call)])
        
        # Waiting passenger features
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            feature_vector.extend([
                min(floor_state['waiting_up'] / 10.0, 1.0),  # Cap at 1.0
                min(floor_state['waiting_down'] / 10.0, 1.0)
            ])
        
        # Ensure we have the correct state size
        expected_size = (self.num_elevators * 4 + self.num_floors * 2 + self.num_floors * 2)
        if len(feature_vector) != expected_size:
            # Pad or truncate to expected size
            if len(feature_vector) < expected_size:
                feature_vector.extend([0.0] * (expected_size - len(feature_vector)))
            else:
                feature_vector = feature_vector[:expected_size]
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _process_hall_calls_with_rl(self, action):
        """Use RL action to assign new hall calls - FIXED VERSION"""
        state = self.building.get_state()
        
        # Find unassigned hall calls (waiting passengers without active calls)
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            
            # Check for up calls
            if floor_state['waiting_up'] > 0:
                # Check if no elevator is already assigned to this up call
                has_active_up_call = any(
                    state['floors'][floor]['elevator_calls'][elev_id]['call_up']
                    for elev_id in range(self.num_elevators)
                )
                if not has_active_up_call:
                    self.building.call_elevator(floor, action, 'up')
            
            # Check for down calls  
            if floor_state['waiting_down'] > 0:
                # Check if no elevator is already assigned to this down call
                has_active_down_call = any(
                    state['floors'][floor]['elevator_calls'][elev_id]['call_down']
                    for elev_id in range(self.num_elevators)
                )
                if not has_active_down_call:
                    self.building.call_elevator(floor, action, 'down')
    
    def _calculate_reward(self):
        """Calculate reward based on passenger waiting times - IMPROVED VERSION"""
        total_waiting_time = 0
        total_passengers = 0
        
        # Sum waiting times of all active passengers
        for floor in range(self.num_floors):
            for passenger in self.building.active_passengers[floor]:
                total_waiting_time += passenger.waiting_time
                total_passengers += 1
        
        # Also consider passengers in elevators
        for elevator_id in range(self.num_elevators):
            for passenger in self.building.elevator_passengers[elevator_id]:
                if passenger.boarding_time:
                    travel_time = self.building.time - passenger.boarding_time
                    total_waiting_time += travel_time
                    total_passengers += 1
        
        # Multiple reward components
        reward = 0.0
        
        if total_passengers > 0:
            avg_waiting_time = total_waiting_time / total_passengers
            # Negative reward proportional to average waiting time
            reward -= avg_waiting_time * 0.1  # Scale factor
        
        # Bonus for serving passengers
        served_count = len(self.building.completed_passengers)
        reward += served_count * 0.5
        
        # Penalty for long episodes without service
        if total_passengers == 0 and served_count == 0:
            reward -= 0.1
            
        return reward
    
    def _get_info(self) -> dict:
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
        """Optional: Add rendering for training visualization"""
        if mode == 'human':
            # Simple text-based rendering
            state = self.building.get_state()
            print(f"Step: {self.current_step}, Elevators: {[e['floor'] for e in state['elevators']]}, "
                  f"Waiting: {sum(f['total_waiting'] for f in state['floors'].values())}")