from entities.building import Building
import numpy as np
import torch
from agent.elevator_rl_env import ElevatorRLEnv
from agent.elevator_dqn import DQNNetwork

class RLBuilding(Building):
    def __init__(self, num_floors: int = 10, num_elevators: int = 4, 
                 speed_multiplier: float = 10.0, capacity: int = 8, 
                 verbose: bool = False, model_path=None):
        super().__init__(num_floors, num_elevators, speed_multiplier, capacity, verbose)
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.capacity = capacity
        
        # Load trained RL agent
        self.rl_agent = None
        if model_path:
            self.load_rl_agent(model_path)
    
    def load_rl_agent(self, model_path):
        """Load trained RL model"""
        # Create a dummy env to get state dimensions
        dummy_env = ElevatorRLEnv(self.num_floors, self.num_elevators, self.capacity)
        
        # Initialize network architecture (same as during training)
        self.rl_agent = DQNNetwork(
            dummy_env.observation_space.shape[0], 
            dummy_env.action_space.n
        )
        
        # Load trained weights
        self.rl_agent.load_state_dict(torch.load(model_path))
        self.rl_agent.eval()
        print(f"Loaded RL agent from {model_path}")
    
    def step(self):
        """Override step method to use RL for call assignment"""
        # Generate state for RL
        if self.rl_agent:
            state = self._get_rl_state()
            
            # Get RL action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.rl_agent(state_tensor)
                action = q_values.argmax().item()
            
            # Use RL action for new hall calls
            self._process_calls_with_rl(action)
        
        # Continue with normal simulation
        super().step()
    
    def _get_rl_state(self):
        """Convert to RL state format (same as ElevatorRLEnv)"""
        state = self.get_state()
        feature_vector = []
        
        # Same state representation as in ElevatorRLEnv
        for elevator in state['elevators']:
            feature_vector.extend([
                elevator['position'] / self.num_floors,
                (elevator['direction'] + 1) / 2,
                elevator['passenger_count'] / self.capacity,
                elevator['state'] / 10.0
            ])
        
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            feature_vector.extend([
                float(any(state['floors'][f]['elevator_calls'][elev_id]['call_up'] 
                        for f in range(self.num_floors) for elev_id in range(self.num_elevators))),
                float(any(state['floors'][f]['elevator_calls'][elev_id]['call_down'] 
                        for f in range(self.num_floors) for elev_id in range(self.num_elevators)))
            ])
        
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            feature_vector.extend([
                floor_state['waiting_up'] / 10.0,
                floor_state['waiting_down'] / 10.0
            ])
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _process_calls_with_rl(self, action):
        """Use RL action to assign elevators to calls"""
        state = self.get_state()
        
        for floor in range(self.num_floors):
            floor_state = state['floors'][floor]
            
            if (floor_state['waiting_up'] > 0 and 
                not any(state['floors'][floor]['elevator_calls'][elev_id]['call_up'] 
                       for elev_id in range(self.num_elevators))):
                self.call_elevator(floor, action, 'up')
            
            if (floor_state['waiting_down'] > 0 and
                not any(state['floors'][floor]['elevator_calls'][elev_id]['call_down']
                       for elev_id in range(self.num_elevators))):
                self.call_elevator(floor, action, 'down')
