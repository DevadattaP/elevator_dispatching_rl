from entities.passenger import Passenger
from utils.enums import ElevatorState
import random

class Elevator:
    def __init__(self, elevator_id: int, num_floors: int, capacity: int = 8):
        self.id = elevator_id
        self.num_floors = num_floors
        self.capacity = capacity
        
        # Physical properties
        self.current_floor = 0
        self.target_floors = set()
        self.direction = 0
        
        # Movement physics
        self.max_speed = 2.5
        self.acceleration_distance = 0.5
        self.current_speed = 0.0
        
        # Door timing
        self.door_open_time = 3.0
        self.door_operation_time = 2.0
        
        # State management
        self.state = ElevatorState.IDLE
        self.state_timer = 0
        self.passengers = []  # List of Passenger objects currently in elevator
        self.position = 0.0
        
        # INTERNAL BUTTON PANEL
        self.internal_buttons = [False] * num_floors
        
    def is_idle(self):
        return self.state == ElevatorState.IDLE
    
    def is_moving(self):
        return self.state in [ElevatorState.MOVING_UP, ElevatorState.MOVING_DOWN]
    
    def is_door_open(self):
        return self.state == ElevatorState.DOOR_OPEN
    
    def is_door_operating(self):
        return self.state in [ElevatorState.DOOR_OPENING, ElevatorState.DOOR_CLOSING]
    
    def can_accept_passengers(self):
        """Check if elevator can accept passengers (doors open or operating)"""
        return self.state in [ElevatorState.DOOR_OPENING, ElevatorState.DOOR_CLOSING, 
                            ElevatorState.DOOR_OPEN, ElevatorState.IDLE]
    
    def press_internal_button(self, floor: int):
        """Passenger presses a button INSIDE the elevator"""
        if 0 <= floor < self.num_floors and not self.internal_buttons[floor]:
            self.internal_buttons[floor] = True
            self.target_floors.add(floor)
            print(f"Elevator {self.id}: Internal button pressed for floor {floor}")
            return True
        return False
    
    def trigger_door_cycle(self, building):
        """Manually trigger door opening cycle when elevator is already on target floor"""
        if self.is_idle():
            self.state = ElevatorState.DOOR_OPENING
            self.state_timer = 0
            print(f"Elevator {self.id} manually triggered door cycle on floor {self.current_floor}")
    
    def step(self, building, time_step=1.0/60.0):
        self.state_timer += time_step
        
        if self.state == ElevatorState.MOVING_UP:
            self._move_with_physics(building, time_step, direction=1)
        elif self.state == ElevatorState.MOVING_DOWN:
            self._move_with_physics(building, time_step, direction=-1)
        elif self.state == ElevatorState.DOOR_OPENING:
            if self.state_timer >= self.door_operation_time:
                self.state = ElevatorState.DOOR_OPEN
                self.state_timer = 0
                print(f"Elevator {self.id} doors opened at floor {self.current_floor}")
        elif self.state == ElevatorState.DOOR_OPEN:
            if self.state_timer >= self.door_open_time:
                self.state = ElevatorState.DOOR_CLOSING
                self.state_timer = 0
                print(f"Elevator {self.id} doors closing at floor {self.current_floor}")
        elif self.state == ElevatorState.DOOR_CLOSING:
            if self.state_timer >= self.door_operation_time:
                self._choose_next_action(building)
        
        # CRITICAL FIX: If idle but have targets, start moving
        if self.is_idle() and self.target_floors:
            print(f"Elevator {self.id} is IDLE but has targets {self.target_floors}, starting movement")
            self._choose_next_action(building)
    
    def _move_with_physics(self, building, time_step, direction):
        target_floor = self._get_next_target_floor()
        if target_floor is None:
            self.state = ElevatorState.IDLE
            self.current_speed = 0.0
            return
        
        distance_to_target = abs(target_floor - self.position)
        
        # Speed control with acceleration/deceleration
        if distance_to_target <= self.acceleration_distance:
            # Decelerating
            self.current_speed = max(0.1, (distance_to_target / self.acceleration_distance) * self.max_speed)
        elif distance_to_target >= 2 * self.acceleration_distance:
            # Full speed
            self.current_speed = min(self.current_speed + 0.5 * time_step, self.max_speed)
        else:
            # Accelerating
            self.current_speed = min(self.current_speed + 0.3 * time_step, self.max_speed)
        
        # Move elevator
        movement = self.current_speed * time_step * direction
        self.position += movement
        
        # Check if we've reached the target floor
        if direction == 1 and self.position >= target_floor - 0.01:
            self.current_floor = target_floor
            self.position = float(self.current_floor)
            self._stop_at_floor(building)
        elif direction == -1 and self.position <= target_floor + 0.01:
            self.current_floor = target_floor
            self.position = float(self.current_floor)
            self._stop_at_floor(building)
        
        # Update current floor display
        self.current_floor = int(round(self.position))
    
    def _get_next_target_floor(self):
        if not self.target_floors:
            return None
        
        if self.direction == 1:
            above_floors = [f for f in self.target_floors if f > self.position]
            return min(above_floors) if above_floors else max(self.target_floors)
        elif self.direction == -1:
            below_floors = [f for f in self.target_floors if f < self.position]
            return max(below_floors) if below_floors else min(self.target_floors)
        else:
            return min(self.target_floors, key=lambda f: abs(f - self.position))
    
    def _stop_at_floor(self, building):
        self.state = ElevatorState.DOOR_OPENING
        self.state_timer = 0
        self.current_speed = 0.0
        
        # Clear this floor from targets and internal button
        self.target_floors.discard(self.current_floor)
        self.internal_buttons[self.current_floor] = False
        
        print(f"Elevator {self.id} stopping at floor {self.current_floor}")
        
        # FIRST: Unload passengers who have reached their destination
        unloaded_count = building.unload_passengers_from_elevator(self.id, self.current_floor)
        if unloaded_count > 0:
            print(f"Unloaded {unloaded_count} passengers from elevator {self.id}")
        
        # Clear external calls for this elevator on this floor
        if self.current_floor in building.external_calls:
            building.clear_call(self.current_floor, self.id, 'up')
            building.clear_call(self.current_floor, self.id, 'down')
    
    def _choose_next_action(self, building):
        if self.target_floors:
            next_floor = self._get_next_target_floor()
            if next_floor > self.current_floor:
                self.state = ElevatorState.MOVING_UP
                self.direction = 1
                print(f"Elevator {self.id} starting to move UP to floor {next_floor}")
            elif next_floor < self.current_floor:
                self.state = ElevatorState.MOVING_DOWN
                self.direction = -1
                print(f"Elevator {self.id} starting to move DOWN to floor {next_floor}")
            else:
                # If target is current floor, trigger door cycle
                print(f"Elevator {self.id} target {next_floor} is current floor, triggering door cycle")
                self.trigger_door_cycle(building)
        else:
            self.state = ElevatorState.IDLE
            self.direction = 0
            print(f"Elevator {self.id} is now IDLE (no targets)")
    
    def assign_target(self, floor: int):
        """Called when elevator is assigned to respond to external call"""
        if floor not in self.target_floors:
            self.target_floors.add(floor)
            print(f"Elevator {self.id} assigned to floor {floor}")
            
            # CRITICAL: If elevator is idle, immediately start moving
            if self.is_idle():
                print(f"Elevator {self.id} was IDLE, starting movement to {floor}")
                self._choose_next_action(None)  # Pass None since we don't need building here
    
    def get_state(self):
        return {
            'floor': self.current_floor,
            'position': self.position,
            'state': self.state.value,  # Use .value for the numeric value
            'direction': self.direction,
            'speed': self.current_speed,
            'passenger_count': len(self.passengers),
            'target_floors': list(self.target_floors),
            'internal_buttons': self.internal_buttons.copy(),
            'passengers': self.passengers  # Include passenger objects for display
        }