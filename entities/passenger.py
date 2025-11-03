import time

class Passenger:
    def __init__(self, passenger_id: int, start_floor: int, target_floor: int, spawn_time: float):
        self.id = passenger_id
        self.start_floor = start_floor
        self.target_floor = target_floor
        self.spawn_time = spawn_time
        self.completion_time = None
        self.waiting_time = 0
        self.boarding_time = None
        self.elevator_id = None  # Which elevator the passenger is in
        
        # Determine direction
        if target_floor > start_floor:
            self.direction = 'up'
        else:
            self.direction = 'down'
    
    @property
    def total_time(self):
        if self.completion_time is not None:
            return self.completion_time - self.spawn_time
        return None
    
    @property
    def is_completed(self):
        return self.completion_time is not None
    
    @property
    def is_waiting(self):
        return self.boarding_time is None and not self.is_completed
    
    @property
    def is_in_elevator(self):
        return self.boarding_time is not None and not self.is_completed
    
    def board_elevator(self, elevator_id: int, current_time: float):
        """Mark passenger as boarded on an elevator"""
        self.boarding_time = current_time
        self.elevator_id = elevator_id
        self.waiting_time = current_time - self.spawn_time
    
    def complete_journey(self, current_time: float):
        """Mark passenger journey as completed"""
        self.completion_time = current_time
    
    def __str__(self):
        status = "waiting" if self.is_waiting else "in_elevator" if self.is_in_elevator else "completed"
        return f"Passenger {self.id}: {self.start_floor}->{self.target_floor} ({status})"