from entities.elevator import Elevator
from entities.passenger import Passenger
from collections import deque
from utils.enums import ElevatorState

class Building:
    def __init__(self, num_floors: int, num_elevators: int = 4):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.elevators = [Elevator(i, num_floors) for i in range(num_elevators)]
        self.time = 0.0
        self.time_step = 1.0/60.0
        
        # EXTERNAL CALL BUTTONS - Each elevator has its own call buttons on each floor
        self.external_calls = {}
        for floor in range(num_floors):
            self.external_calls[floor] = {}
            for elevator_id in range(num_elevators):
                self.external_calls[floor][elevator_id] = {'up': False, 'down': False}
            
            if floor == num_floors - 1:  # Top floor
                for elevator_id in range(num_elevators):
                    self.external_calls[floor][elevator_id]['up'] = None
            if floor == 0:  # Ground floor
                for elevator_id in range(num_elevators):
                    self.external_calls[floor][elevator_id]['down'] = None
        
        # Passenger management - use deque for proper queue behavior
        self.passenger_id_counter = 0
        self.active_passengers = {}  # floor -> deque of waiting passengers
        self.elevator_passengers = {}  # elevator_id -> list of passengers in elevator
        self.completed_passengers = []
        
        # Initialize passenger containers
        for floor in range(num_floors):
            self.active_passengers[floor] = deque()
        for elevator_id in range(num_elevators):
            self.elevator_passengers[elevator_id] = []
    
    def call_elevator(self, floor: int, elevator_id: int, direction: str):
        """External call for a SPECIFIC elevator from a floor button"""
        if not (0 <= floor < self.num_floors and 0 <= elevator_id < self.num_elevators):
            return False
        
        # Set the call button for the specific elevator
        if direction == 'up' and self.external_calls[floor][elevator_id]['up'] is not None:
            self.external_calls[floor][elevator_id]['up'] = True
        elif direction == 'down' and self.external_calls[floor][elevator_id]['down'] is not None:
            self.external_calls[floor][elevator_id]['down'] = True
        else:
            return False
        
        # Assign the target to the specific elevator
        elevator = self.elevators[elevator_id]
        
        # If elevator is already on the same floor and in a state that can accept passengers
        if (elevator.current_floor == floor and 
            elevator.state.value in [ElevatorState.IDLE.value, ElevatorState.DOOR_OPENING.value, 
                                ElevatorState.DOOR_CLOSING.value, ElevatorState.DOOR_OPEN.value]):
            print(f"Elevator {elevator_id} is already on floor {floor}, ensuring door cycle")
            # Ensure doors are open or opening
            if elevator.state.value == ElevatorState.IDLE.value:
                elevator.state = ElevatorState.DOOR_OPENING
                elevator.state_timer = 0
        else:
            elevator.assign_target(floor)
            print(f"Dispatched elevator {elevator_id} to floor {floor} for {direction} call")
        
        return True
    
    def add_passenger(self, start_floor: int, target_floor: int, preferred_elevator_id=None):
        """Add a passenger to the building. 
        NOTE: Preferred elevator is just a suggestion - first available elevator will be used."""
        if start_floor == target_floor:
            return None
        
        direction = 'up' if target_floor > start_floor else 'down'
        
        # Create passenger
        passenger = Passenger(
            passenger_id=self.passenger_id_counter,
            start_floor=start_floor,
            target_floor=target_floor,
            spawn_time=self.time
        )
        self.passenger_id_counter += 1
        
        # Add to waiting queue for the floor (FIFO)
        self.active_passengers[start_floor].append(passenger)
        print(f"Added passenger {passenger.id} from floor {start_floor} to {target_floor} (waiting: {len(self.active_passengers[start_floor])})")
        
        # IGNORE preferred elevator - call ALL elevators to let the first one pick up the passenger
        # This ensures passengers get on the first available elevator
        for elevator_id in range(self.num_elevators):
            self.call_elevator(start_floor, elevator_id, direction)
        
        print(f"Called all elevators to floor {start_floor} for passenger {passenger.id}")
        return passenger
    
    def step(self):
        """Advance simulation by one time step"""
        self.time += self.time_step
        
        # Process external calls for elevators that are idle
        self._process_pending_calls()
        
        # CRITICAL FIX: Process passenger boarding for elevators with open doors
        self._process_passenger_boarding()
        
        # Update all elevators
        for elevator in self.elevators:
            elevator.step(self, self.time_step)
        
        # Update passenger waiting times
        self._update_passenger_times()
    
    def _process_passenger_boarding(self):
        """Process passenger boarding for all elevators with open doors"""
        for elevator_id, elevator in enumerate(self.elevators):
            if elevator.is_door_open():  # Use the helper method
                current_floor = elevator.current_floor
                
                if (current_floor in self.active_passengers and 
                    self.active_passengers[current_floor]):
                    
                    # Board passengers to this elevator
                    direction = 'up' if elevator.direction == 1 else 'down' if elevator.direction == -1 else 'up'
                    boarded_count = self.board_passengers_to_elevator(
                        elevator_id, current_floor, direction, elevator.capacity
                    )
                    
                    if boarded_count > 0:
                        print(f"Boarded {boarded_count} passengers to elevator {elevator_id} on floor {current_floor}")
    
    def _process_pending_calls(self):
        """Process any pending external calls that haven't been served"""
        for floor in range(self.num_floors):
            for elevator_id in range(self.num_elevators):
                floor_calls = self.external_calls[floor][elevator_id]
                elevator = self.elevators[elevator_id]
                
                # Check if elevator is already heading to this floor
                already_coming = floor in elevator.target_floors
                
                # Check up calls
                if floor_calls['up'] and not already_coming:
                    # Only dispatch if elevator is idle or moving in compatible direction
                    if (elevator.state.value == 0 or  # IDLE
                        (elevator.direction == 1 and floor >= elevator.current_floor)):  # Moving up and floor is ahead
                        elevator.assign_target(floor)
                        print(f"Dispatched elevator {elevator_id} to floor {floor} for pending up call")
                
                # Check down calls
                if floor_calls['down'] and not already_coming:
                    if (elevator.state.value == 0 or  # IDLE
                        (elevator.direction == -1 and floor <= elevator.current_floor)):  # Moving down and floor is ahead
                        elevator.assign_target(floor)
                        print(f"Dispatched elevator {elevator_id} to floor {floor} for pending down call")
    
    def _update_passenger_times(self):
        """Update waiting times for all active passengers"""
        # Update waiting passengers
        for floor, passengers in self.active_passengers.items():
            for passenger in passengers:
                passenger.waiting_time = self.time - passenger.spawn_time
        
        # Update passengers in elevators
        for elevator_id, passengers in self.elevator_passengers.items():
            for passenger in passengers:
                if passenger.boarding_time:
                    passenger.waiting_time = self.time - passenger.boarding_time
    
    def clear_call(self, floor: int, elevator_id: int, direction: str):
        """Clear a specific external call"""
        if (0 <= floor < self.num_floors and 
            0 <= elevator_id < self.num_elevators and
            direction in ['up', 'down']):
            
            if direction == 'up' and self.external_calls[floor][elevator_id]['up'] is not None:
                self.external_calls[floor][elevator_id]['up'] = False
            elif direction == 'down' and self.external_calls[floor][elevator_id]['down'] is not None:
                self.external_calls[floor][elevator_id]['down'] = False
    
    def get_available_elevators_for_floor(self, floor: int, direction: str):
        """Get elevators that are available to serve passengers on this floor"""
        available_elevators = []
        
        for elevator_id, elevator in enumerate(self.elevators):
            # Check if elevator is on this floor and in a state that can accept passengers
            if (elevator.current_floor == floor and 
                elevator.can_accept_passengers() and  # Use the helper method
                len(elevator.passengers) < elevator.capacity):
                
                # Check direction compatibility
                if (elevator.direction == 0 or  # IDLE - can take any direction
                    (elevator.direction == 1 and direction == 'up') or  # Moving up and passenger wants up
                    (elevator.direction == -1 and direction == 'down')):  # Moving down and passenger wants down
                    available_elevators.append(elevator_id)
        
        return available_elevators
    
    def board_passengers_to_elevator(self, elevator_id: int, floor: int, direction: str, max_passengers: int):
        """Board passengers from floor to elevator (FIFO order)"""
        elevator = self.elevators[elevator_id]
        available_space = max_passengers - len(elevator.passengers)
        
        if available_space <= 0:
            return 0
        
        # Get compatible passengers in queue order
        compatible_passengers = []
        remaining_passengers = deque()
        
        # Process queue in order
        while self.active_passengers[floor]:
            passenger = self.active_passengers[floor].popleft()
            if passenger.direction == direction:
                compatible_passengers.append(passenger)
            else:
                remaining_passengers.append(passenger)
        
        # Put non-compatible passengers back in queue
        self.active_passengers[floor] = remaining_passengers
        
        # Board passengers up to available space
        boarded_count = 0
        for passenger in compatible_passengers:
            if boarded_count >= available_space:
                # Put excess passengers back in queue
                self.active_passengers[floor].appendleft(passenger)
                continue
                
            # Board the passenger
            passenger.board_elevator(elevator_id, self.time)
            elevator.passengers.append(passenger)
            self.elevator_passengers[elevator_id].append(passenger)
            
            # Press the destination button - THIS IS CRITICAL FOR ELEVATOR MOVEMENT
            elevator.press_internal_button(passenger.target_floor)
            
            boarded_count += 1
            print(f"Passenger {passenger.id} boarded elevator {elevator_id} to floor {passenger.target_floor}")
            
            # CRITICAL: After boarding, ensure elevator has targets to move to
            if not elevator.target_floors:
                elevator.target_floors.add(passenger.target_floor)
        
        # If there are still compatible passengers left, put them back in queue
        for passenger in compatible_passengers[boarded_count:]:
            self.active_passengers[floor].append(passenger)
        
        return boarded_count
    
    def unload_passengers_from_elevator(self, elevator_id: int, floor: int):
        """Unload passengers who have reached their destination"""
        elevator = self.elevators[elevator_id]
        remaining_passengers = []
        unloaded_count = 0
        
        for passenger in elevator.passengers:
            if passenger.target_floor == floor:
                passenger.complete_journey(self.time)
                self.completed_passengers.append(passenger)
                # Remove from elevator passengers list
                if passenger in self.elevator_passengers[elevator_id]:
                    self.elevator_passengers[elevator_id].remove(passenger)
                unloaded_count += 1
                print(f"Passenger {passenger.id} exited elevator {elevator_id} at floor {floor}")
            else:
                remaining_passengers.append(passenger)
        
        elevator.passengers = remaining_passengers
        return unloaded_count
    
    def get_state(self):
        """Get complete state for display"""
        elevator_states = [elevator.get_state() for elevator in self.elevators]
        
        # Floor states with external calls per elevator and passenger counts
        floor_states = {}
        for floor in range(self.num_floors):
            waiting_up = len([p for p in self.active_passengers[floor] if p.direction == 'up'])
            waiting_down = len([p for p in self.active_passengers[floor] if p.direction == 'down'])
            
            # Get call states for each elevator on this floor
            elevator_calls = {}
            for elevator_id in range(self.num_elevators):
                elevator_calls[elevator_id] = {
                    'call_up': self.external_calls[floor][elevator_id]['up'],
                    'call_down': self.external_calls[floor][elevator_id]['down']
                }
            
            floor_states[floor] = {
                'waiting_up': waiting_up,
                'waiting_down': waiting_down,
                'total_waiting': len(self.active_passengers[floor]),
                'elevator_calls': elevator_calls,
                'passengers': list(self.active_passengers[floor])  # Include passenger objects for display
            }
        
        # Elevator passenger details
        elevator_passenger_details = {}
        for elevator_id in range(self.num_elevators):
            elevator_passenger_details[elevator_id] = {
                'passengers': self.elevators[elevator_id].passengers,
                'count': len(self.elevators[elevator_id].passengers)
            }
        
        return {
            'elevators': elevator_states,
            'floors': floor_states,
            'elevator_passengers': elevator_passenger_details,
            'time': self.time,
            'completed_passengers': len(self.completed_passengers)
        }