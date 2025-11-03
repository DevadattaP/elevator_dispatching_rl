from entities.passenger import Passenger
from utils.enums import ElevatorState
import random

class Elevator:
    def __init__(self, elevator_id: int, num_floors: int, speed_multiplier: float = 1.0, capacity: int = 8):
        self.id = elevator_id
        self.num_floors = num_floors
        self.capacity = capacity
        self.speed_multiplier = speed_multiplier
        
        # Physical properties - make base values that get multiplied by speed_multiplier
        self.base_max_speed = 2.5  # Base speed without multiplier (floors per second)
        self.base_acceleration_distance = 1.0  # Distance to accelerate/decelerate (in floors)
        self.base_door_open_time = 3.0
        self.base_door_operation_time = 2.0
        
        # Current floor and movement
        self.current_floor = 0
        self.target_floors = set()
        self.direction = 0
        
        # Movement physics - these will be calculated based on speed_multiplier
        self.max_speed = self.base_max_speed * speed_multiplier
        self.acceleration_distance = self.base_acceleration_distance
        self.current_speed = 0.0
        
        # Door timing - calculated based on speed_multiplier
        self.door_open_time = self.base_door_open_time / speed_multiplier
        self.door_operation_time = self.base_door_operation_time / speed_multiplier
        
        # State management
        self.state = ElevatorState.IDLE
        self.state_timer = 0
        self.passengers: list[Passenger] = []
        self.position = 0.0
        
        # INTERNAL BUTTON PANEL
        self.internal_buttons = [False] * num_floors
    
    def set_speed_multiplier(self, speed_multiplier: float):
        """Update speed multiplier and recalculate all timing parameters"""
        self.speed_multiplier = max(0.1, speed_multiplier)  # Minimum 0.1x speed
        
        # Update movement parameters
        self.max_speed = self.base_max_speed * speed_multiplier
        self.acceleration_distance = self.base_acceleration_distance
        
        # Update door timing (inverse relationship - faster speed = shorter door times)
        self.door_open_time = self.base_door_open_time / speed_multiplier
        self.door_operation_time = self.base_door_operation_time / speed_multiplier
        
        print(f"Elevator {self.id} speed set to {speed_multiplier}x: "
              f"max_speed={self.max_speed:.1f}, "
              f"door_open_time={self.door_open_time:.1f}s, "
              f"door_operation_time={self.door_operation_time:.1f}s")
    
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
        # Only allow internal button presses when there are passengers in the elevator
        if len(self.passengers) == 0:
            print(f"Elevator {self.id}: Internal button for floor {floor} ignored - elevator is empty")
            return False
        
        if 0 <= floor < self.num_floors and not self.internal_buttons[floor]:
            self.internal_buttons[floor] = True
            if floor not in self.target_floors:
                self.target_floors.add(floor)
                # Sort targets after adding new internal call
                self._sort_target_floors()
                print(f"Elevator {self.id}: Internal button pressed for floor {floor}, targets: {self.target_floors}")
            return True
        return False
    
    def trigger_door_cycle(self, building):
        """Manually trigger door opening cycle when elevator is already on target floor"""
        if self.is_idle():
            self.state = ElevatorState.DOOR_OPENING
            self.state_timer = 0
            print(f"Elevator {self.id} manually triggered door cycle on floor {self.current_floor}")
    
    def step(self, building, time_step=1.0/60.0):
        self.state_timer += time_step*self.speed_multiplier
        
        if self.state == ElevatorState.MOVING_UP:
            self._move_with_physics(building, time_step, direction=1)
        elif self.state == ElevatorState.MOVING_DOWN:
            self._move_with_physics(building, time_step, direction=-1)
        elif self.state == ElevatorState.DOOR_OPENING:
            if self.state_timer >= self.door_operation_time:
                self.state = ElevatorState.DOOR_OPEN
                self.state_timer = 0
                # Call the door opened handler
                self._on_doors_opened(building)
        elif self.state == ElevatorState.DOOR_OPEN:
            # Periodic boarding checks
            check_interval = 0.3
            current_check = int(self.state_timer / check_interval)
            previous_check = int((self.state_timer - time_step) / check_interval)
            
            if current_check > previous_check:
                self._process_passenger_boarding_at_stop(building)
            
            if self.state_timer >= self.door_open_time:
                self.state = ElevatorState.DOOR_CLOSING
                self.state_timer = 0
                print(f"Elevator {self.id} doors closing at floor {self.current_floor}")
                # Final boarding check before closing
                self._process_passenger_boarding_at_stop(building)
        elif self.state == ElevatorState.DOOR_CLOSING:
            if self.state_timer >= self.door_operation_time:
                self._choose_next_action(building)
    
    def _on_doors_opened(self, building):
        """Called when doors are fully open - handle passenger exchange"""
        print(f"Elevator {self.id} doors fully opened at floor {self.current_floor}")
        
        # FIRST: Unload passengers who have reached their destination
        unloaded_count = self._unload_passengers_at_stop(building)
        
        # THEN: Board new passengers
        self._process_passenger_boarding_at_stop(building)
        
        # Update direction after passenger exchange (may have new targets from boarding)
        self._update_direction_based_on_targets()
    
    def _move_with_physics(self, building, time_step, direction):
        target_floor = self._get_next_target_floor()
        if target_floor is None:
            self.state = ElevatorState.IDLE
            self.current_speed = 0.0
            return
        
        # Calculate distance to target (in floors)
        distance_to_target = abs(target_floor - self.position)
        
        # NEW PHYSICS: Calculate speed based on acceleration/deceleration profile
        if distance_to_target <= self.acceleration_distance:
            # DECELERATION PHASE: Last acceleration_distance floors
            # Linearly decrease speed from max_speed to 0
            deceleration_progress = distance_to_target / self.acceleration_distance
            target_speed = self.max_speed * deceleration_progress
            
            # Smooth deceleration - approach target speed gradually
            if self.current_speed > target_speed:
                self.current_speed = max(target_speed, self.current_speed - self.max_speed * 2 * time_step)
            else:
                self.current_speed = min(target_speed, self.current_speed + self.max_speed * 2 * time_step)
                
        elif distance_to_target >= 2 * self.acceleration_distance:
            # CRUISING PHASE: Middle section at constant max speed
            # We're beyond acceleration distance from start and end
            target_speed = self.max_speed
            
            # Smooth acceleration to cruising speed
            if self.current_speed < target_speed:
                self.current_speed = min(target_speed, self.current_speed + self.max_speed * 2 * time_step)
            else:
                self.current_speed = max(target_speed, self.current_speed - self.max_speed * 0.5 * time_step)
                
        else:
            # ACCELERATION PHASE: First acceleration_distance floors
            # Linearly increase speed from 0 to max_speed
            acceleration_progress = 1.0 - (distance_to_target - self.acceleration_distance) / self.acceleration_distance
            target_speed = self.max_speed * acceleration_progress
            
            # Smooth acceleration - approach target speed gradually
            if self.current_speed < target_speed:
                self.current_speed = min(target_speed, self.current_speed + self.max_speed * 2 * time_step)
            else:
                self.current_speed = max(target_speed, self.current_speed - self.max_speed * 0.5 * time_step)
        
        # Ensure speed doesn't go below minimum or above maximum
        self.current_speed = max(0.1, min(self.max_speed, self.current_speed))
        
        # Move elevator
        movement = self.current_speed * time_step * direction
        self.position += movement
        
        # Debug physics (optional)
        if random.random() < 0.01:  # Print occasionally to avoid spam
            print(f"Elevator {self.id}: pos={self.position:.2f}, target={target_floor}, "
                  f"dist={distance_to_target:.2f}, speed={self.current_speed:.2f}")
        
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
        """Called when elevator physically reaches the target floor"""
        # Reset movement
        self.state = ElevatorState.DOOR_OPENING
        self.state_timer = 0
        self.current_speed = 0.0
        
        print(f"Elevator {self.id} reached floor {self.current_floor}, starting door opening sequence")
        
        # Clear this floor from targets and internal button
        self.target_floors.discard(self.current_floor)
        self.internal_buttons[self.current_floor] = False
        
        # Clear external calls for this elevator on this floor
        if self.current_floor in building.external_calls:
            building.clear_call(self.current_floor, self.id, 'up')
            building.clear_call(self.current_floor, self.id, 'down')
        
        # Update direction for boarding compatibility
        self._update_direction_based_on_targets()
        
        # IMPORTANT: Passengers will unload when doors open in the step method
        # This happens in DOOR_OPENING -> DOOR_OPEN transition
    
    def _unload_passengers_at_stop(self, building):
        """Unload passengers who have reached their destination - called when doors open"""
        current_floor = self.current_floor
        unloaded_count = 0
        
        # Create a copy of passengers list to avoid modification during iteration
        passengers_to_check = self.passengers.copy()
        
        for passenger in passengers_to_check:
            if passenger.target_floor == current_floor:
                # MARK PASSENGER AS COMPLETED FIRST to calculate total_time correctly
                passenger.complete_journey(building.time)
                
                # UPDATE METRICS DIRECTLY IN BUILDING
                if hasattr(building, 'elevator_metrics') and self.id < len(building.elevator_metrics):
                    metrics = building.elevator_metrics[self.id]
                    metrics['passengers_served'] += 1
                    metrics['total_waiting_time'] += passenger.waiting_time
                    if passenger.total_time:  # Only add if journey is completed
                        metrics['total_travel_time'] += passenger.total_time
                
                # Remove passenger from elevator
                if passenger in self.passengers:
                    self.passengers.remove(passenger)
                if passenger in building.elevator_passengers[self.id]:
                    building.elevator_passengers[self.id].remove(passenger)
                
                # Complete passenger journey
                building.completed_passengers.append(passenger)
                
                unloaded_count += 1
                print(f"Passenger {passenger.id} exited elevator {self.id} at floor {current_floor}")
        
        if unloaded_count > 0:
            print(f"Unloaded {unloaded_count} passengers from elevator {self.id}")
            
            # Clear all internal buttons when elevator becomes empty
            if len(self.passengers) == 0:
                print(f"Elevator {self.id} is now empty, clearing all internal buttons")
                self.internal_buttons = [False] * self.num_floors
                # Keep external targets but clear internal ones
                external_targets = set()
                for floor_num in self.target_floors:
                    # Check if this target came from an external call
                    if (floor_num in building.external_calls and 
                        any(building.external_calls[floor_num][self.id][d] 
                            for d in ['up', 'down'] if building.external_calls[floor_num][self.id][d] is not None)):
                        external_targets.add(floor_num)
                self.target_floors = external_targets
                if self.target_floors:
                    self._sort_target_floors()
                    print(f"Elevator {self.id} keeping external targets: {self.target_floors}")
                else:
                    print(f"Elevator {self.id} has no remaining targets")
        
        return unloaded_count
    
    def _update_direction_based_on_targets(self):
        """Update elevator direction based on target floors"""
        if not self.target_floors:
            self.direction = 0  # IDLE
            return
        
        # Find the next target floor
        next_floor = self._get_next_target_floor()
        if next_floor is None:
            self.direction = 0
            return
        
        if next_floor > self.current_floor:
            self.direction = 1  # UP
        elif next_floor < self.current_floor:
            self.direction = -1  # DOWN
        else:
            self.direction = 0  # IDLE
    
    def _process_passenger_boarding_at_stop(self, building):
        """Process passenger boarding when elevator stops at a floor - FIXED VERSION"""
        current_floor = self.current_floor
        
        if (current_floor in building.active_passengers and 
            building.active_passengers[current_floor]):
            
            # FIX: Use the building's boarding logic which now considers intended direction
            boarded_count = building.board_passengers_to_elevator(
                self.id, current_floor, 'any', self.capacity  # 'any' lets building determine compatibility
            )
            
            if boarded_count > 0:
                print(f"Boarded {boarded_count} passengers to elevator {self.id} on floor {current_floor}")
                
                # After boarding, re-sort targets since we may have new internal calls
                self._sort_target_floors()
                # Update direction based on new targets
                self._update_direction_based_on_targets()
    
    def _choose_next_action(self, building):
        """Choose next action after door closing"""
        # Re-check if we have targets after door closing
        if not self.target_floors:
            self.state = ElevatorState.IDLE
            self.direction = 0
            print(f"Elevator {self.id} is now IDLE (no targets)")
            return
        
        # Get the next target floor
        next_floor = self._get_next_target_floor()
        if next_floor is None:
            self.state = ElevatorState.IDLE
            self.direction = 0
            print(f"Elevator {self.id} is now IDLE (no valid targets)")
            return
        
        print(f"Elevator {self.id} choosing action: current={self.current_floor}, next={next_floor}, all_targets={self.target_floors}")
        
        if next_floor > self.current_floor:
            self.state = ElevatorState.MOVING_UP
            self.direction = 1
            print(f"Elevator {self.id} starting to move UP to floor {next_floor}")
        elif next_floor < self.current_floor:
            self.state = ElevatorState.MOVING_DOWN
            self.direction = -1
            print(f"Elevator {self.id} starting to move DOWN to floor {next_floor}")
        else:
            # If target is current floor, this shouldn't happen but handle it
            print(f"Elevator {self.id} target {next_floor} is current floor, triggering door cycle")
            self.trigger_door_cycle(building)   
    
    def _sort_target_floors(self):
        """Sort target floors using SCAN algorithm based on current direction and position"""
        if not self.target_floors:
            return
        
        # Convert to list and sort based on direction
        targets_list = list(self.target_floors)
        
        if self.direction == 1:  # Moving UP
            # Sort: floors above current position in ascending order, then floors below in descending order
            above = [f for f in targets_list if f >= self.position]
            below = [f for f in targets_list if f < self.position]
            above.sort()
            below.sort(reverse=True)
            sorted_targets = above + below
        elif self.direction == -1:  # Moving DOWN
            # Sort: floors below current position in descending order, then floors above in ascending order
            below = [f for f in targets_list if f <= self.position]
            above = [f for f in targets_list if f > self.position]
            below.sort(reverse=True)
            above.sort()
            sorted_targets = below + above
        else:  # IDLE - find closest floor
            sorted_targets = sorted(targets_list, key=lambda f: abs(f - self.position))
        
        # Update target_floors (maintain as set for O(1) lookups, but use sorted list for movement)
        self.target_floors = set(sorted_targets)
        return sorted_targets
    
    def _get_next_target_floor(self):
        """Get the next target floor based on sorted queue"""
        if not self.target_floors:
            return None
        
        # Sort targets based on current strategy
        sorted_targets = self._sort_target_floors()
        
        if not sorted_targets:
            return None
        
        # For moving elevators, use the first target in the sorted list
        if self.direction == 1:  # UP
            # Find next floor above current position
            above_floors = [f for f in sorted_targets if f >= self.position]
            return above_floors[0] if above_floors else sorted_targets[-1]
        elif self.direction == -1:  # DOWN
            # Find next floor below current position
            below_floors = [f for f in sorted_targets if f <= self.position]
            return below_floors[0] if below_floors else sorted_targets[0]
        else:  # IDLE
            return sorted_targets[0]
    
    def assign_target(self, floor: int):
        """Called when elevator is assigned to respond to external call"""
        if floor not in self.target_floors:
            self.target_floors.add(floor)
            print(f"Elevator {self.id} assigned to floor {floor}, current targets: {self.target_floors}")
            
            # Sort the targets after adding new one
            self._sort_target_floors()
            
            # If elevator is idle, determine direction and start moving
            if self.is_idle():
                print(f"Elevator {self.id} was IDLE, starting movement to {floor}")
                self._choose_next_action(None)
    
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