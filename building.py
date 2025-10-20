from entities.elevator import Elevator
from entities.passenger import Passenger
from collections import deque
from utils.enums import ElevatorState
import math

class Building:
    def __init__(self, num_floors: int, num_elevators: int = 4, speed_multiplier: float = 1.0):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.speed_multiplier = speed_multiplier
        self.elevators = [Elevator(i, num_floors, speed_multiplier) for i in range(num_elevators)]
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
        
        # call assignment tracking
        self.assigned_calls = {}  # (floor, direction) -> elevator_id
    
    def set_speed_multiplier(self, speed_multiplier: float):
        """Update speed multiplier for all elevators"""
        self.speed_multiplier = speed_multiplier
        for elevator in self.elevators:
            elevator.set_speed_multiplier(speed_multiplier)
    
    def calculate_eta(self, elevator_id: int, target_floor: int):
        """Calculate Estimated Time Arrival for elevator to reach target floor"""
        elevator = self.elevators[elevator_id]
        current_pos = elevator.position
        target_pos = target_floor
        
        # If elevator is already on the floor and doors are open/opening
        if (elevator.current_floor == target_floor and 
            elevator.state.value in [ElevatorState.DOOR_OPEN.value, ElevatorState.DOOR_OPENING.value]):
            return 0
        
        # Calculate travel time
        floors_to_travel = abs(target_pos - current_pos)
        
        # Time for acceleration and deceleration
        acceleration_time = 2.0  # seconds to reach max speed
        deceleration_time = 2.0  # seconds to stop
        
        # If distance is short, use simplified calculation
        if floors_to_travel <= elevator.acceleration_distance * 2:
            travel_time = floors_to_travel / (elevator.max_speed * 0.5)
        else:
            # Time for acceleration + constant speed + deceleration
            constant_speed_distance = floors_to_travel - (elevator.acceleration_distance * 2)
            travel_time = acceleration_time + (constant_speed_distance / elevator.max_speed) + deceleration_time
        
        # Add time for door operations if elevator needs to stop
        door_time = elevator.door_operation_time * 2 + elevator.door_open_time
        
        # Add time for existing stops
        existing_stops = len(elevator.target_floors)
        additional_time = existing_stops * door_time
        
        # Penalty for direction changes
        if elevator.direction != 0:
            if (elevator.direction == 1 and target_floor < current_pos) or \
               (elevator.direction == -1 and target_floor > current_pos):
                additional_time += door_time * 2  # Extra penalty for direction change
        
        return travel_time + additional_time
    
    def assign_call_to_best_elevator(self, floor: int, direction: str):
        """Assign floor call to the elevator with minimum ETA"""
        best_elevator_id = None
        min_eta = float('inf')
        
        # Check if this call is already assigned
        call_key = (floor, direction)
        if call_key in self.assigned_calls:
            assigned_elevator = self.assigned_calls[call_key]
            # Verify the assigned elevator is still valid
            if (assigned_elevator in range(self.num_elevators) and 
                floor in self.elevators[assigned_elevator].target_floors):
                return assigned_elevator
        
        # Find elevator with minimum ETA
        for elevator_id in range(self.num_elevators):
            elevator = self.elevators[elevator_id]
            
            # Skip if elevator is at capacity
            if len(elevator.passengers) >= elevator.capacity:
                continue
            
            eta = self.calculate_eta(elevator_id, floor)
            
            # Small bonus for idle elevators
            if elevator.is_idle():
                eta *= 0.8
            
            if eta < min_eta:
                min_eta = eta
                best_elevator_id = elevator_id
        
        if best_elevator_id is not None:
            self.assigned_calls[call_key] = best_elevator_id
            print(f"Assigned floor {floor} {direction} call to elevator {best_elevator_id} (ETA: {min_eta:.1f}s)")
        
        return best_elevator_id
    
    def call_elevator(self, floor: int, elevator_id: int, direction: str):
        """External call for a SPECIFIC elevator from a floor button - CLEANED VERSION"""
        if not (0 <= floor < self.num_floors and 0 <= elevator_id < self.num_elevators):
            return False
        
        # Set the call button for the specific elevator
        if direction == 'up' and self.external_calls[floor][elevator_id]['up'] is not None:
            self.external_calls[floor][elevator_id]['up'] = True
        elif direction == 'down' and self.external_calls[floor][elevator_id]['down'] is not None:
            self.external_calls[floor][elevator_id]['down'] = True
        else:
            return False
        
        elevator = self.elevators[elevator_id]
        
        print(f"External call: Elevator {elevator_id} called to floor {floor} for {direction}")
        
        # If elevator is already on the same floor
        if elevator.current_floor == floor:
            print(f"Elevator {elevator_id} is already on floor {floor}")
            
            # If doors are open or opening, ensure we process boarding
            if elevator.state.value in [ElevatorState.DOOR_OPEN.value, ElevatorState.DOOR_OPENING.value]:
                print(f"Elevator {elevator_id} doors are already open/opening on floor {floor}")
                # Boarding will be handled in the elevator's step method
            else:
                # If idle or moving, trigger door cycle
                if elevator.state.value in [ElevatorState.IDLE.value, ElevatorState.MOVING_UP.value, ElevatorState.MOVING_DOWN.value]:
                    elevator.trigger_door_cycle(self)
            
            # Clear the call immediately since we're already on the floor
            self.clear_call(floor, elevator_id, direction)
        else:
            # Add the floor to elevator's targets
            elevator.assign_target(floor)
            print(f"Dispatched elevator {elevator_id} to floor {floor} for {direction} call")
        
        return True
    
    def add_passenger(self, start_floor: int, target_floor: int, preferred_elevator_id=None):
        """Add a passenger to the building with smart elevator assignment"""
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
        print(f"Added passenger {passenger.id} from floor {start_floor} to {target_floor}")
        
        # SMART ASSIGNMENT: Assign to best elevator only
        assigned_elevator_id = self.assign_call_to_best_elevator(start_floor, direction)
        
        if assigned_elevator_id is not None:
            # Call only the assigned elevator
            self.call_elevator(start_floor, assigned_elevator_id, direction)
            print(f"Assigned elevator {assigned_elevator_id} to pick up passenger {passenger.id}")
        else:
            print(f"No available elevator found for passenger {passenger.id}")
        
        return passenger
    
    def step(self):
        """Advance simulation by one time step"""
        self.time += self.time_step
        
        # Process external calls for elevators that are idle
        self._process_pending_calls()
        
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
            for direction in ['up', 'down']:
                # Check if there are waiting passengers for this direction
                waiting_passengers = [p for p in self.active_passengers[floor] if p.direction == direction]
                if waiting_passengers:
                    # Check if this call is already assigned
                    call_key = (floor, direction)
                    if call_key not in self.assigned_calls:
                        # Assign to best elevator
                        assigned_elevator = self.assign_call_to_best_elevator(floor, direction)
                        if assigned_elevator is not None:
                            self.call_elevator(floor, assigned_elevator, direction)
    
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
        """Clear a specific external call and remove from assigned calls"""
        if (0 <= floor < self.num_floors and 
            0 <= elevator_id < self.num_elevators and
            direction in ['up', 'down']):
            
            # Remove from assigned calls
            call_key = (floor, direction)
            if call_key in self.assigned_calls and self.assigned_calls[call_key] == elevator_id:
                del self.assigned_calls[call_key]
                print(f"Removed assigned call {call_key} for elevator {elevator_id}")
            
            # Always clear the call when elevator arrives at the floor
            # The boarding logic will handle whether passengers actually board
            if direction == 'up' and self.external_calls[floor][elevator_id]['up'] is not None:
                self.external_calls[floor][elevator_id]['up'] = False
                print(f"Cleared up call for elevator {elevator_id} on floor {floor}")
            elif direction == 'down' and self.external_calls[floor][elevator_id]['down'] is not None:
                self.external_calls[floor][elevator_id]['down'] = False
                print(f"Cleared down call for elevator {elevator_id} on floor {floor}")
    
    def _process_passenger_boarding_at_floor(self, floor: int, direction: str):
        """Check if we should clear calls after boarding - called from elevator boarding"""
        # Check if there are still waiting passengers for this direction
        waiting_passengers = [p for p in self.active_passengers[floor] if p.direction == direction]
        
        if not waiting_passengers:
            # No more passengers for this direction, clear all elevator calls
            for elevator_id in range(self.num_elevators):
                if direction == 'up' and self.external_calls[floor][elevator_id]['up'] is not None:
                    self.external_calls[floor][elevator_id]['up'] = False
                elif direction == 'down' and self.external_calls[floor][elevator_id]['down'] is not None:
                    self.external_calls[floor][elevator_id]['down'] = False
            
            # Also remove from assigned calls
            call_key = (floor, direction)
            if call_key in self.assigned_calls:
                del self.assigned_calls[call_key]
            
            print(f"Cleared all {direction} calls on floor {floor} - no more waiting passengers")
    
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
        """Board passengers from floor to elevator (FIFO order) - FIXED DIRECTION LOGIC"""
        elevator = self.elevators[elevator_id]
        available_space = max_passengers - len(elevator.passengers)
        
        if available_space <= 0:
            return 0
        
        # IMPROVED DIRECTION COMPATIBILITY: Determine elevator's intended direction
        def get_elevator_intended_direction(elevator, current_floor):
            """Determine what direction the elevator will go after doors close"""
            if not elevator.target_floors:
                return 'any'  # No targets yet, can go any direction
            
            # Sort targets to see the immediate next direction
            sorted_targets = elevator._sort_target_floors()
            if not sorted_targets:
                return 'any'
            
            next_floor = sorted_targets[0]
            if next_floor > current_floor:
                return 'up'
            elif next_floor < current_floor:
                return 'down'
            else:
                return 'any'
        
        intended_direction = get_elevator_intended_direction(elevator, floor)
        
        def is_direction_compatible(intended_dir, passenger_dir):
            """Check if passenger direction is compatible with elevator's intended direction"""
            # If elevator has no specific direction yet, accept any
            if intended_dir == 'any':
                return True
            
            # If elevator's intended direction matches passenger's direction
            if intended_dir == passenger_dir:
                return True
                
            # If elevator is empty and responding to call, accept both directions
            if len(elevator.passengers) == 0:
                return True
                
            return False
        
        # Get compatible passengers in queue order
        compatible_passengers = []
        incompatible_passengers = deque()
        
        # Process queue in order
        while self.active_passengers[floor]:
            passenger = self.active_passengers[floor].popleft()
            
            if is_direction_compatible(intended_direction, passenger.direction):
                compatible_passengers.append(passenger)
            else:
                incompatible_passengers.append(passenger)
        
        # Put incompatible passengers back in queue
        self.active_passengers[floor] = incompatible_passengers
        
        # Now add compatible passengers to queue (they'll be boarded in order)
        for passenger in compatible_passengers:
            self.active_passengers[floor].append(passenger)
        
        # Board passengers up to available space
        boarded_count = 0
        temp_queue = deque()
        
        while self.active_passengers[floor] and boarded_count < available_space:
            passenger = self.active_passengers[floor].popleft()
            
            # Double-check compatibility
            if is_direction_compatible(intended_direction, passenger.direction):
                # Board the passenger
                passenger.board_elevator(elevator_id, self.time)
                elevator.passengers.append(passenger)
                self.elevator_passengers[elevator_id].append(passenger)
                
                # Press the destination button
                success = elevator.press_internal_button(passenger.target_floor)
                if not success and passenger.target_floor not in elevator.target_floors:
                    elevator.target_floors.add(passenger.target_floor)
                    elevator._sort_target_floors()
                
                boarded_count += 1
                print(f"Passenger {passenger.id} boarded elevator {elevator_id} to floor {passenger.target_floor}")
                
                # Update intended direction after boarding (may have changed)
                intended_direction = get_elevator_intended_direction(elevator, floor)
            else:
                temp_queue.append(passenger)
        
        # Put remaining passengers back in queue
        while self.active_passengers[floor]:
            temp_queue.append(self.active_passengers[floor].popleft())
        
        self.active_passengers[floor] = temp_queue
        
        return boarded_count
    
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