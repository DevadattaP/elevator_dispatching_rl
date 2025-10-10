# Elevator Dispatch System with Reinforcement Learning

This project implements an elevator dispatch system using reinforcement learning techniques. The system simulates a building with multiple elevators and floors, and uses the PPO (Proximal Policy Optimization) algorithm from Stable Baselines3 to optimize elevator dispatching.

## Project Structure

- `main.py`: Entry point of the application, handles training, evaluation, and GUI simulation.
- `elevator_env.py`: Defines the OpenAI Gym environment for the elevator system.
- `building.py`: Defines the `Building` class, which represents the environment.
- `elevator.py`: Defines the `Elevator` class.
- `gui.py`: Implements a graphical user interface for the simulation using tkinter.
- `test_elevator_system.py`: Contains unit tests for the core components.

## Features

- Simulated building with configurable number of floors and elevators
- OpenAI Gym environment for standardized RL training
- PPO algorithm from Stable Baselines3 for elevator dispatching
- GUI for visual representation of the elevator system
- Options for training, evaluating, and visualizing the RL agent's performance

### Enhanced simulation environment
- Realistic Passenger Patterns:
  - Morning rush hour (8-10 AM) with upward traffic
  - Evening rush hour (5-7 PM) with downward traffic
  - Configurable base probability and peak multipliers
- Detailed Elevator Physics:
  - Movement speed and capacity constraints
  - Door operation timing
  - Directional indicators (up/down/idle)

### Advanced Reinforcement Learning
- Custom Observation Space:
  - Elevator positions, directions, and loads
  - Waiting passenger counts and wait times
  - Time step information for temporal awareness
- Optimized Reward Function:
  - Delivery bonuses (+5 per passenger)
  - Wait time penalties (capped at 20 steps)
  - Movement efficiency penalties
  - Properly scaled and clipped rewards

### Interactive Visualization
- Real-time GUI:
  - Building layout with elevator positions
  - Passenger destination visualization
  - Color-coded elevator status
- Comprehensive Statistics:
  - Reward history tracking
  - Wait time distribution
  - Elevator utilization metrics
  - Passenger flow analysis

## Setup and Running

1. Ensure you have Python 3.7+ installed.
2. Install the required libraries:
   ```
   pip install gymnasium stable-baselines3 numpy matplotlib
   ```
3. Run the simulation:

   - To train the RL agent:
     ```
     python main.py --train --floors 5 --elevators 1 --timesteps 200000
     ```

   - To evaluate the trained agent:
     ```
     python main.py --evaluate <model path> --episodes 20 --floors 5 --elevators 1
     ```

   - To run the GUI simulation:
     ```
     python main.py --gui --floors 5 --elevators 1
     ```

   - You can also customize the simulation parameters:
     ```
     python main.py --train --floors 10 --elevators 4 --timesteps 500000
     ```

4. To run the unit tests:
   ```
   python -m unittest test_elevator_system.py
   ```

## How it Works

1. The `ElevatorEnv` class defines an OpenAI Gym environment that simulates the elevator system.
2. The PPO algorithm from Stable Baselines3 is used to train an RL agent to optimize elevator dispatching.
3. During training, the agent learns to make decisions on which elevator to move and where, based on the current state of the building.
4. The trained model can be evaluated to assess its performance in managing the elevator system.
5. The GUI provides a visual representation of the elevator system's behavior, allowing users to step through the simulation and observe the agent's decisions.

## Current State and Limitations

The current implementation provides a production-ready elevator dispatch system with several advanced features, but still has some limitations:

1. **Passenger Generation**:
   - While we've implemented time-based peak patterns (morning/evening rushes), the underlying probability distributions could benefit from real-world traffic data
   - Currently lacks special events (e.g., lunchtime spikes, conference schedules)

2. **Reward Function**:
   - The multi-component reward (delivery bonus, wait penalty, movement penalty) works well but could be fine-tuned
   - Energy efficiency metrics are not currently incorporated
   - No differentiation between short/long distance trips

3. **Algorithm Selection**:
   - Currently optimized for PPO
   - Other algorithms like SAC might better handle the continuous action space
   - No comparative benchmarking between algorithms

4. **Performance Scaling**:
   - The Gym environment becomes resource-intensive with >20 floors/elevators
   - Visualization slows down with many simultaneous passengers

## Future Improvements Roadmap

### Immediate Priorities (v1.1)
1. **Enhanced Passenger Modeling**:
   - [ ] Add lunchtime traffic patterns (12-1 PM)
   - [ ] Implement special event generators
   - [ ] Add passenger groups traveling together
   - [ ] Enhanced state representation with temporal/spatial features
   - [ ] Predictive passenger flow modeling

2. **Reward Function Extensions**:
   - [ ] Incorporate energy consumption metrics
   - [ ] Add differential rewards for express vs local elevators
   - [ ] Implement priority scheduling (emergency/disabled)
   - [ ] Multi-objective reward balancing (efficiency, comfort, energy, fairness)
   - [ ] Adaptive reward weights based on operational conditions

3. **Algorithm Expansion**:
   - [ ] Add SAC implementation comparison
   - [ ] Develop hybrid rule-based/RL approach
   - [ ] Create benchmark suite for algorithm comparison
   - [ ] Multi-algorithm framework (PPO, A2C, DDPG, SAC, TD3)
   - [ ] Hyperparameter optimization with Optuna

### Mid-Term Goals (v1.5)
4. **GUI Enhancements**:
   - [ ] Real-time 3D visualization option
   - [ ] Interactive scenario builder
   - [ ] Playback controls for simulation review
   - [ ] Enhanced real-time statistics and analytics
   - [ ] Comparative algorithm performance dashboard

5. **Performance Optimization**:
   - [ ] Vectorized environment for faster training
   - [ ] Cython acceleration for core simulation
   - [ ] Distributed training support
   - [ ] Realistic physics engine with kinematic movement
   - [ ] Large-scale simulation support (50+ floors)

### ROS2 Integration (v2.0)
6. **Distributed Architecture**:
   - [ ] ROS2 node structure with custom message definitions
   - [ ] Distributed components (building manager, elevator controllers, RL agent)
   - [ ] Real-time communication between simulation components
   - [ ] Hardware-in-the-loop testing capability
   - [ ] ROS2-based visualization and monitoring

### Long-Term Vision (v3.0)
7. **Advanced Features**:
   - [ ] Predictive maintenance integration
   - [ ] Dynamic floor importance weighting
   - [ ] Multi-building coordination
   - [ ] Transfer learning between building configurations
   - [ ] Continuous learning from operational data

8. **Deployment Ready**:
   - [ ] Docker containerization
   - [ ] REST API interface
   - [ ] Cloud training pipeline
   - [ ] ROS2 deployment for real-world testing
   - [ ] Real-world data integration and adaptation

### Research & Development Focus
- **State Representation Optimization**: Finding most efficient observation space
- **Algorithm Performance**: Comparative analysis in elevator control domain  
- **Real-world Transfer**: Bridging simulation-to-reality gap through ROS2
- **Scalability Studies**: Performance with ultra-tall buildings (50+ floors)

## How to Contribute

We welcome contributions through:
- Pull requests (with accompanying tests)
- Issue reports (bug/feature)
- Real-world traffic pattern datasets
- Algorithm benchmarking results

See our [Contribution Guidelines](CONTRIBUTING.md) for details.

> **Note**: This roadmap reflects our current development priorities. Specific features may change based on community feedback and research advancements.