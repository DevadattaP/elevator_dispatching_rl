from enum import Enum

class ElevatorState(Enum):
    IDLE = 0
    MOVING_UP = 1
    MOVING_DOWN = 2
    DOOR_OPENING = 3
    DOOR_CLOSING = 4
    DOOR_OPEN = 5