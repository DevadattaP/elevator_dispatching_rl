import tkinter as tk
from tkinter import ttk
import threading
import time
import os
from elevator_env import ElevatorEnv, D3QNWrapper, SMDPWrapper, TrafficAwareWrapper, DiscreteCombinatorialWrapper, DiscreteAssignmentWrapper, MultiDiscreteWrapper
from elevator_dqn import ElevatorDQN, ElevatorDDQN, ElevatorTDQN
from elevator_rl_env import ElevatorRLEnv
import numpy as np
import torch
from building import Building
from utils.enums import ElevatorState
from graphs import GraphWindow
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG


class ElevatorGUI:
    def __init__(self, root, num_floors=4, num_elevators=2, capacity=8, verbose: bool = False):
        self.root = root
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.capacity = capacity
        self.verbose = verbose
        self.building = Building(num_floors, num_elevators, capacity=capacity, verbose=verbose)
        
        self.is_running = False
        self.simulation_thread = None
        self.elevator_buttons = {}  # Internal buttons
        self.external_call_buttons = {}  # External call buttons
        self.generation_enabled = False
        
        self.panels_visible = True  # Track panel visibility state
        
        # Graph window reference
        self.graph_window = None
        
        self.model = None
        self.env = None
        self.agent_type = tk.StringVar(value="rule_based")  # track mode
        
        self.setup_gui()
        self.update_display()
        
        # Auto-open graph window on startup
        self.open_graph_window()
        
    def setup_gui(self):
        self.root.title("Elevator Simulation - Individual Elevator Call Buttons")
        self.root.geometry("1600x1000")
        
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill="x")
        
        # Agent selector
        ttk.Label(control_frame, text="Agent:").pack(side="left", padx=5)
        # agent_choices = ["rule_based", "PPO", "A2C", "DQN", "SAC", "TD3", "DDPG"]
        agent_choices = ["rule_based", "dqn", 'ddqn', 'tdqn']
        self.agent_menu = ttk.Combobox(control_frame, textvariable=self.agent_type, values=agent_choices, width=10, state="readonly")
        self.agent_menu.pack(side="left", padx=5)

        # Load model button
        ttk.Button(control_frame, text="Load Model", command=self.load_agent_model).pack(side="left", padx=5)

        # Simulation controls
        ttk.Button(control_frame, text="Start", command=self.start_simulation).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Pause", command=self.pause_simulation).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).pack(side="left", padx=5)
        
        # Speed control
        ttk.Label(control_frame, text="Sim Speed:").pack(side="left", padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=10.0, variable=self.speed_var, 
                               orient="horizontal", length=150, command=self.on_speed_change)
        speed_scale.pack(side="left", padx=5)
        
        # Speed value display
        self.speed_label = ttk.Label(control_frame, text="1.0x")
        self.speed_label.pack(side="left", padx=5)
        
        # Add generation controls to passenger frame
        ttk.Button(control_frame, text="Start Auto Generation", 
                  command=self.start_passenger_generation, width=15).pack(side="left", padx=2)
        ttk.Button(control_frame, text="Stop Auto Generation", 
                  command=self.stop_passenger_generation, width=15).pack(side="left", padx=2)
        
        # Generation rate control
        ttk.Label(control_frame, text="Rate:").pack(side="left", padx=2)
        self.generation_rate_var = tk.DoubleVar(value=1.0)
        ttk.Scale(control_frame, from_=1.0, to=10.0, variable=self.generation_rate_var,
                 orient="horizontal", length=80, command=self.on_generation_rate_change).pack(side="left", padx=2)
        self.generation_speed_label = ttk.Label(control_frame, text="1.0x")
        self.generation_speed_label.pack(side="left", padx=2)
        
        # Passanger generation status label
        self.generation_status = ttk.Label(control_frame, text="Passanger Generation: OFF", foreground="red")
        self.generation_status.pack(side="left", padx=10)
        
        # Add panel visibility controls to control panel
        ttk.Button(control_frame, text="Hide Panels", command=self.toggle_panels_visibility).pack(side="right", padx=5)
        
        # Graph window button
        ttk.Button(control_frame, text="Show Graphs", 
                  command=self.open_graph_window).pack(side="right", padx=5)
        
        # Stats
        self.stats_label = ttk.Label(control_frame, text="Time: 00:00:00 | Waiting: 0")
        self.stats_label.pack(side="right", padx=10)
        
        # Main display frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Middle: Elevator visualization
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(side="left", fill="both", expand=True)
        
        self.canvas = tk.Canvas(middle_frame, bg="white", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack(fill="both", expand=True)
        
        # Left: External call buttons for ALL elevators on each floor
        self.left_frame = ttk.Frame(main_frame)
        self.left_frame.pack(side="left", fill="y", padx=10)
        
        ttk.Label(self.left_frame, text="External Call Buttons", font=("Arial", 12, "bold")).pack(pady=5)
        self.setup_external_controls(self.left_frame)
        
        # Right: Individual elevator internal panels
        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side="right", fill="y", padx=10)
        
        ttk.Label(self.right_frame, text="Elevator Internal Panels", font=("Arial", 12, "bold")).pack(pady=5)
        self.setup_elevator_panels(self.right_frame)

    def load_agent_model(self):
        """Load pre-trained RL models with exact training configurations."""
        model_name = self.agent_type.get()
        if model_name == "rule_based":
            self.model = None
            # Create environment for rule-based system
            self.env = ElevatorEnv(
                num_floors=self.num_floors,
                num_elevators=self.num_elevators,
                observation_type="simple",
                reward_type="simple", 
                action_type="discrete",
                traffic_pattern="mixed",
                use_smdp=False,
                verbose=0
            )
            self.building = self.env.building
            return
        elif model_name in ['dqn', 'ddqn', 'tdqn']:
            dummy_env = ElevatorRLEnv(self.num_floors, self.num_elevators, self.capacity)
            # dummy_env = ElevatorEnv(
            #     num_floors=self.num_floors,
            #     num_elevators=self.num_elevators,
            #     lift_capacity=self.capacity,
            #     observation_type='enhanced',  # Use your enhanced observation
            #     action_type='assignment',     # Or 'combinatorial' based on your preference
            #     reward_type='fairness',       # Use fairness reward like Crites & Barto
            #     use_smdp=True,               # Use SMDP for better performance
            #     traffic_pattern='all_in_one', # Mixed traffic patterns
            #     verbose=0
            # )
        
            if model_name == 'dqn':
                rl_agent = ElevatorDQN(env=dummy_env,resume=True)
            elif model_name == 'ddqn':
                rl_agent = ElevatorDDQN(env=dummy_env,resume=True)
            elif model_name == 'tdqn':
                rl_agent = ElevatorTDQN(env=dummy_env,resume=True)
            self.env = dummy_env
            self.building = self.env.building
            self.model = rl_agent
            return
        
        # Model path mapping with exact training configurations
        model_configs = {
            "DQN": {
                "path": "./models/DQN_d3qn_enhanced_squared_combinatorial_all_in_one/DQN_d3qn_enhanced_squared_combinatorial_all_in_one_model.zip",
                "config": {
                    "env_wrapper": "d3qn",
                    "observation_type": "enhanced",  # Auto-set by d3qn wrapper
                    "reward_type": "squared",        # Auto-set by d3qn wrapper  
                    "action_type": "combinatorial",
                    "traffic_pattern": "all_in_one",
                    "use_smdp": False
                }
            },
            "A2C": {
                "path": "./models/A2C_default_detailed_fairness_discrete_mixed/A2C_default_detailed_fairness_discrete_mixed_model.zip", 
                "config": {
                    "env_wrapper": "default",
                    "observation_type": "detailed",
                    "reward_type": "fairness",
                    "action_type": "discrete", 
                    "traffic_pattern": "mixed",
                    "use_smdp": False
                }
            },
            "SAC": {
                "path": "./models/SAC_default_enhanced_complex_continuous_mixed/SAC_default_enhanced_complex_continuous_mixed_model.zip",
                "config": {
                    "env_wrapper": "default", 
                    "observation_type": "enhanced",
                    "reward_type": "complex",
                    "action_type": "continuous",
                    "traffic_pattern": "mixed",
                    "use_smdp": False
                }
            },
            "PPO": {
                "path": "./models/PPO_smdp_enhanced_complex_discrete_mixed_smdp/PPO_smdp_enhanced_complex_discrete_mixed_smdp_model.zip",
                "config": {
                    "env_wrapper": "smdp",
                    "observation_type": "enhanced",  # Auto-set by smdp wrapper
                    "reward_type": "complex",
                    "action_type": "discrete",
                    "traffic_pattern": "mixed", 
                    "use_smdp": True  # Auto-set by smdp wrapper
                }
            }
        }
        
        if model_name not in model_configs:
            print(f"No configuration found for {model_name}")
            # Fallback to default environment
            self.env = ElevatorEnv(
                num_floors=self.num_floors,
                num_elevators=self.num_elevators,
                observation_type="simple",
                reward_type="simple",
                action_type="discrete",
                verbose=0
            )
            self.building = self.env.building
            return
        
        config = model_configs[model_name]
        model_path = config["path"]
        env_config = config["config"]
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            # Fallback to default environment
            self.env = ElevatorEnv(
                num_floors=self.num_floors,
                num_elevators=self.num_elevators, 
                observation_type="simple",
                reward_type="simple",
                action_type="discrete",
                verbose=0
            )
            self.building = self.env.building
            return
        
        print(f"Loading {model_name} from: {model_path}")
                
        # Choose environment wrapper
        wrapper_classes = {
            "default": ElevatorEnv,
            "d3qn": D3QNWrapper, 
            "smdp": SMDPWrapper,
            "traffic_aware": TrafficAwareWrapper
        }
        
        env_class = wrapper_classes[env_config['env_wrapper']]
        
        # Environment parameters - match training exactly
        env_kwargs = {
            'num_floors': self.num_floors,
            'num_elevators': self.num_elevators,
            'episode_length': 3600,
            'headless': True,  # GUI handles rendering
            'passenger_generation_rate': 1.0,
            'observation_type': env_config['observation_type'],
            'reward_type': env_config['reward_type'],
            'action_type': env_config['action_type'], 
            'traffic_pattern': env_config['traffic_pattern'],
            'use_smdp': env_config['use_smdp'],
            'verbose': 0
        }
        
        # Apply wrapper-specific overrides (same as training)
        if env_config['env_wrapper'] == "d3qn":
            # D3QN wrapper auto-sets observation_type and reward_type
            env_kwargs.pop('observation_type', None)
            env_kwargs.pop('reward_type', None)
        elif env_config['env_wrapper'] == "smdp":
            # SMDP wrapper auto-sets use_smdp and observation_type
            env_kwargs['use_smdp'] = True
            env_kwargs['observation_type'] = 'enhanced'
        elif env_config['env_wrapper'] == "traffic_aware":
            # Traffic-aware wrapper auto-sets traffic_pattern and observation_type  
            env_kwargs['traffic_pattern'] = 'all_in_one'
            env_kwargs['observation_type'] = 'enhanced'
        
        # Create base environment first (for building access)
        base_env = env_class(**env_kwargs)
        self.base_env = base_env
        
        # Apply action space wrappers for DQN (same as training)
        if model_name == "DQN":
            if env_config['action_type'] == "combinatorial":
                self.env = DiscreteCombinatorialWrapper(base_env)
            elif env_config['action_type'] == "assignment":
                self.env = DiscreteAssignmentWrapper(base_env) 
            elif env_config['action_type'] == "discrete":
                self.env = MultiDiscreteWrapper(base_env)
            else:
                self.env = base_env
        else:
            self.env = base_env
        
        # Store building reference for GUI access
        self.building = base_env.building
        
        # Model mapping
        MODEL_MAP = {
            "PPO": PPO,
            "A2C": A2C, 
            "DQN": DQN,
            "SAC": SAC,
            "TD3": TD3,
            "DDPG": DDPG
        }
        
        # Load model with appropriate settings
        ModelClass = MODEL_MAP[model_name]
        
        try:
            # Load with environment for proper setup
            self.model = ModelClass.load(model_path, env=self.env)
            
            print(f"Successfully loaded {model_name}")
            print(f"   Configuration:")
            print(f"   - Wrapper: {env_config['env_wrapper']}")
            print(f"   - Observation: {env_config['observation_type']}")
            print(f"   - Reward: {env_config['reward_type']}") 
            print(f"   - Action: {env_config['action_type']}")
            print(f"   - Traffic: {env_config['traffic_pattern']}")
            print(f"   - SMDP: {env_config['use_smdp']}")
            
            if model_name == "DQN":
                # Set action converter for proper DQN action processing
                if env_config['action_type'] == "combinatorial":
                    self.action_converter = "combinatorial"
                    print(f"DQN configured with combinatorial action converter")
                elif env_config['action_type'] == "assignment":
                    self.action_converter = "assignment"
                    print(f"DQN configured with assignment action converter")
                elif env_config['action_type'] == "discrete":
                    self.action_converter = "discrete" 
                    print(f"DQN configured with discrete action converter")
                else:
                    self.action_converter = None
                    print(f"DQN with no action converter (action_type: {env_config['action_type']})")
            else:
                self.action_converter = None
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Trying to load without environment...")
            try:
                self.model = ModelClass.load(model_path)
                print("Model loaded without environment (may have prediction issues)")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                self.model = None
                
                # Still set up environment for rule-based fallback
                self.env = ElevatorEnv(
                    num_floors=self.num_floors,
                    num_elevators=self.num_elevators,
                    observation_type="simple", 
                    reward_type="simple",
                    action_type="discrete",
                    verbose=0
                )
                self.building = self.env.building

    def _process_dqn_action(self, action, converter_type):
        """Convert DQN discrete action back to original action space."""
        if converter_type == "combinatorial":
            # Convert discrete action (0-15) to binary vector for 4 elevators
            binary_action = []
            for i in range(self.num_elevators):
                binary_action.append((action >> i) & 1)
            return np.array(binary_action, dtype=np.int8)
            
        elif converter_type == "assignment":
            # Assignment action is already in correct format (0-4 for 4 elevators + no assignment)
            return action
            
        elif converter_type == "discrete":
            # Convert flat discrete action back to multi-discrete
            actions = np.unravel_index(action, [self.num_floors + 1] * self.num_elevators)
            return np.array(actions)
            
        else:
            return action
      
    def open_graph_window(self):
        """Open or focus the graph window"""
        if self.graph_window is None or not self.graph_window.root.winfo_exists():
            self.graph_window = GraphWindow(self.building, self.num_elevators)
            # When graph window closes, clear the reference
            self.graph_window.root.protocol("WM_DELETE_WINDOW", self.on_graph_window_close)
        else:
            # Bring existing window to front
            self.graph_window.root.lift()
            self.graph_window.root.focus_force()
    
    def on_graph_window_close(self):
        """Handle graph window closing"""
        if self.graph_window:
            self.graph_window.close()
            self.graph_window = None
            
    def toggle_panels_visibility(self):
        """Toggle visibility of all elevator panels"""
        self.panels_visible = not self.panels_visible
        
        if self.panels_visible:
            # Show panels
            self.left_frame.pack(side="left", fill="y", padx=10)
            self.right_frame.pack(side="right", fill="y", padx=10)
            # Update button text
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button) and child.cget("text") == "Show Panels":
                            child.config(text="Hide Panels")
        else:
            # Hide panels
            self.left_frame.pack_forget()
            self.right_frame.pack_forget()
            # Update button text
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button) and child.cget("text") == "Hide Panels":
                            child.config(text="Show Panels")
        
        # Update the display to use full width
        self.update_display()

    def setup_external_controls(self, parent):
        """Setup external call buttons for EACH elevator on each floor"""
        # External call buttons frame
        call_frame = ttk.LabelFrame(parent, text="Call Specific Elevators", padding="10")
        call_frame.pack(fill="x", pady=5)
        
        # Create a grid: floors vs elevators
        for floor in range(self.num_floors-1, -1, -1):
            floor_frame = ttk.Frame(call_frame)
            floor_frame.pack(fill="x", pady=3)
            
            # Floor label
            ttk.Label(floor_frame, text=f"Floor {floor}", width=8, 
                     font=("Arial", 10, "bold")).pack(side="left")
            
            # Create call buttons for each elevator on this floor
            for elevator_id in range(self.num_elevators):
                elevator_frame = ttk.Frame(floor_frame)
                elevator_frame.pack(side="left", padx=5)
                
                ttk.Label(elevator_frame, text=f"E{elevator_id}", 
                         font=("Arial", 8)).pack()
                
                # Store button references
                if floor not in self.external_call_buttons:
                    self.external_call_buttons[floor] = {}
                
                # Up button (not available on top floor)
                if floor < self.num_floors - 1:
                    up_btn = ttk.Button(elevator_frame, text="↑", width=3,
                                      command=lambda f=floor, e=elevator_id: self.call_elevator(f, e, 'up'))
                    up_btn.pack(side="left", padx=1)
                    self.external_call_buttons[floor][(elevator_id, 'up')] = up_btn
                else:
                    ttk.Label(elevator_frame, text="   ", width=3).pack(side="left", padx=1)
                
                # Down button (not available on ground floor)
                if floor > 0:
                    down_btn = ttk.Button(elevator_frame, text="↓", width=3,
                                        command=lambda f=floor, e=elevator_id: self.call_elevator(f, e, 'down'))
                    down_btn.pack(side="left", padx=1)
                    self.external_call_buttons[floor][(elevator_id, 'down')] = down_btn
                else:
                    ttk.Label(elevator_frame, text="   ", width=3).pack(side="left", padx=1)
        
        # Passenger addition frame
        passenger_frame = ttk.LabelFrame(parent, text="Add Passenger", padding="10")
        passenger_frame.pack(fill="x", pady=5)
        
        ttk.Label(passenger_frame, text="From:").grid(row=0, column=0, padx=2)
        self.start_floor_var = tk.IntVar(value=0)
        ttk.Spinbox(passenger_frame, from_=0, to=self.num_floors-1, 
                   textvariable=self.start_floor_var, width=5).grid(row=0, column=1, padx=2)
        
        ttk.Label(passenger_frame, text="To:").grid(row=0, column=2, padx=2)
        self.target_floor_var = tk.IntVar(value=self.num_floors-1)
        ttk.Spinbox(passenger_frame, from_=0, to=self.num_floors-1,
                   textvariable=self.target_floor_var, width=5).grid(row=0, column=3, padx=2)
        
        # Preferred elevator selection
        ttk.Label(passenger_frame, text="Pref Elevator:").grid(row=0, column=4, padx=2)
        self.preferred_elevator_var = tk.StringVar(value="Any")
        elevator_choices = ["Any"] + [f"E{i}" for i in range(self.num_elevators)]
        ttk.Combobox(passenger_frame, textvariable=self.preferred_elevator_var,
                    values=elevator_choices, width=6, state="readonly").grid(row=0, column=5, padx=2)
        
        ttk.Button(passenger_frame, text="Add Passenger",
                  command=self.add_passenger).grid(row=0, column=6, padx=5)
    
    def setup_elevator_panels(self, parent):
        """Setup individual internal control panels for each elevator"""
        for elevator_id in range(self.num_elevators):
            elevator_frame = ttk.LabelFrame(parent, text=f"Elevator {elevator_id} Internal Panel", padding="10")
            elevator_frame.pack(fill="x", pady=5)
            
            # Create a grid of buttons for this elevator
            self.elevator_buttons[elevator_id] = {}
            
            # Calculate grid layout
            cols = 3
            rows = (self.num_floors + cols - 1) // cols
            
            for floor in range(self.num_floors-1, -1, -1):
                row = (self.num_floors - 1 - floor) // cols
                col = (self.num_floors - 1 - floor) % cols
                
                btn = ttk.Button(elevator_frame, text=str(floor), width=3,
                               command=lambda e=elevator_id, f=floor: self.press_elevator_button(e, f))
                btn.grid(row=row, column=col, padx=2, pady=2)
                self.elevator_buttons[elevator_id][floor] = btn
       
    def on_speed_change(self, event=None):
        """Handle speed slider changes"""
        speed_value = self.speed_var.get()
        self.speed_label.config(text=f"{speed_value:.1f}x")
        self.building.set_speed_multiplier(speed_value)

    def on_generation_rate_change(self, event=None):
        """Handle generation rate changes"""
        rate_value = self.generation_rate_var.get()
        # Update the label next to the scale
        self.generation_speed_label.config(text=f"{rate_value:.1f}x")
        
        # Update building generation probability
        if hasattr(self.building, 'base_generation_probability'):
            # Scale probability linearly with rate
            self.building.generation_probability = self.building.base_generation_probability * rate_value
            if self.verbose:
                print(f"Generation rate set to {rate_value:.1f}x - probability: {self.building.generation_probability:.4f}")
    
    def start_passenger_generation(self):
        """Start automatic passenger generation"""
        if not self.generation_enabled:
            self.generation_enabled = True
            self.building.start_passenger_generation()
            self.generation_status.config(text="Passanger Generation: ON", foreground="green")
            if self.verbose:
                print("Started automatic passenger generation")
    
    def stop_passenger_generation(self):
        """Stop automatic passenger generation"""
        if self.generation_enabled:
            self.generation_enabled = False
            self.building.stop_passenger_generation()
            self.generation_status.config(text="Passanger Generation: OFF", foreground="red")
            if self.verbose:
                print("Stopped automatic passenger generation")

    def call_elevator(self, floor: int, elevator_id: int, direction: str):
        """Handle external call button press for SPECIFIC elevator"""
        success = self.building.call_elevator(floor, elevator_id, direction)
        if success:
            if self.verbose:
                print(f"External call: Elevator {elevator_id} called to floor {floor} going {direction}")
    
    def press_elevator_button(self, elevator_id: int, floor: int):
        """Handle internal elevator button press"""
        elevator = self.building.elevators[elevator_id]
        success = elevator.press_internal_button(floor)
        
        if success:
            # Visual feedback - change button color
            btn = self.elevator_buttons[elevator_id][floor]
            btn.configure(style="Pressed.TButton")
    
    def add_passenger(self):
        """Add a passenger to the simulation"""
        start_floor = self.start_floor_var.get()
        target_floor = self.target_floor_var.get()
        preferred_elevator = self.preferred_elevator_var.get()
        
        if start_floor == target_floor:
            return
        
        # Convert preferred elevator selection
        if preferred_elevator == "Any":
            preferred_elevator_id = None
        else:
            preferred_elevator_id = int(preferred_elevator[1:])  # Extract number from "E0", "E1", etc.
        
        passenger = self.building.add_passenger(start_floor, target_floor, preferred_elevator_id)
        if passenger:
            if self.verbose:
                print(f"Added passenger from floor {start_floor} to {target_floor}" + 
                  (f" (prefers {preferred_elevator})" if preferred_elevator != "Any" else ""))
    
    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.simulation_thread.start()
    
    def pause_simulation(self):
        self.is_running = False
    
    def reset_simulation(self):
        self.stop_passenger_generation()
        self.pause_simulation()
        self.building = Building(self.num_floors, self.num_elevators, self.speed_var.get(), self.capacity, self.verbose)
        self.generation_enabled = False
        self.generation_status.config(text="Passanger Generation: OFF", foreground="red")
        # Reset button colors
        for elevator_id in self.elevator_buttons:
            for floor, btn in self.elevator_buttons[elevator_id].items():
                btn.configure(style="TButton")
        self.update_display()
    
    def run_simulation(self):
        if self.model is not None:
            print(f"Starting RL simulation with {self.agent_type.get()} agent")
            # Make sure environment is properly reset
            obs, info = self.env.reset()
            terminated, truncated = False, False
            
        step_count = 0
        while self.is_running:
            if self.model is None:
                # Rule-based control
                self.building.step()
            else:
                # RL-based control using Gym interface
                if self.agent_type.get() in ['dqn', 'ddqn', 'tdqn']:
                    action = self.model.predict(obs)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                
                # if step_count % 100 == 0:  # Print every 100 steps for debugging
                #     print(f"Step {step_count}: Action={action}, Agent={self.agent_type.get()}")
                
                # Handle DQN action conversion
                if hasattr(self, 'action_converter') and self.action_converter:
                    processed_action = self._process_dqn_action(action, self.action_converter)
                    # if step_count % 100 == 0:
                    #     print(f"  Converted action: {processed_action}")
                else:
                    processed_action = action
                    
                obs, reward, terminated, truncated, info = self.env.step(processed_action)
                
                if terminated or truncated:
                    print("Episode terminated, resetting environment")
                    obs, info = self.env.reset()
                    step_count = 0
                    
                self.building = self.env.building
                step_count += 1
                
            self.root.after(0, self.update_display)
            time.sleep(0.016)
           
    def draw_passengers(self):
        """Draw passenger indicators on floors and in elevators"""
        canvas = self.canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        margin = 60
        floor_height = (height - 2 * margin) / self.num_floors
        elevator_width = 100
        spacing = 20
        
        state = self.building.get_state()
        
        # Draw waiting passengers on floors
        for floor, floor_state in state['floors'].items():
            y_center = height - margin - (floor + 0.5) * floor_height
            
            # Draw passenger indicators for waiting passengers
            waiting_passengers = floor_state['passengers']
            if waiting_passengers:
                # Group by direction for display
                up_passengers = [p for p in waiting_passengers if p.direction == 'up']
                down_passengers = [p for p in waiting_passengers if p.direction == 'down']
                
                # Draw up passengers on right side
                if up_passengers:
                    x = width - margin - 40
                    for i, passenger in enumerate(up_passengers[:5]):  # Show max 5
                        passenger_y = y_center - 15 - i * 8
                        canvas.create_oval(x - 4, passenger_y - 4, x + 4, passenger_y + 4, 
                                        fill="blue", outline="darkblue")
                        # Show passenger ID for small numbers
                        if len(up_passengers) <= 3:
                            canvas.create_text(x, passenger_y, text=str(passenger.id), 
                                            font=("Arial", 6), fill="white")
                    
                    # Show count if more than 5
                    if len(up_passengers) > 5:
                        canvas.create_text(x, y_center - 60, text=f"+{len(up_passengers)-5}", 
                                        font=("Arial", 8), fill="blue")
                
                # Draw down passengers on left side
                if down_passengers:
                    x = width - margin - 80
                    for i, passenger in enumerate(down_passengers[:5]):  # Show max 5
                        passenger_y = y_center + 15 + i * 8
                        canvas.create_oval(x - 4, passenger_y - 4, x + 4, passenger_y + 4, 
                                        fill="red", outline="darkred")
                        # Show passenger ID for small numbers
                        if len(down_passengers) <= 3:
                            canvas.create_text(x, passenger_y, text=str(passenger.id), 
                                            font=("Arial", 6), fill="white")
                    
                    # Show count if more than 5
                    if len(down_passengers) > 5:
                        canvas.create_text(x, y_center + 60, text=f"+{len(down_passengers)-5}", 
                                        font=("Arial", 8), fill="red")
        
        # Draw passengers inside elevators
        for i, elevator_state in enumerate(state['elevators']):
            x1 = margin + i * (elevator_width + spacing)
            x2 = x1 + elevator_width
            y_center = height - margin - (elevator_state['position'] + 0.5) * floor_height
            y1 = y_center - floor_height * 0.4
            y2 = y_center + floor_height * 0.4
            
            # Draw passengers inside elevator
            passengers_in_elevator = elevator_state['passengers']
            if passengers_in_elevator:
                # Calculate positions for passenger dots
                max_display = 8
                displayed_passengers = passengers_in_elevator[:max_display]
                
                for j, passenger in enumerate(displayed_passengers):
                    # Calculate position in a grid inside elevator
                    cols = 2
                    row = j // cols
                    col = j % cols
                    
                    dot_x = x1 + 15 + col * 20
                    dot_y = y1 + 15 + row * 15
                    
                    # Color code by destination floor
                    colors = ["green", "blue", "purple", "orange", "brown", "pink", "cyan", "magenta"]
                    color_idx = passenger.target_floor % len(colors)
                    
                    canvas.create_oval(dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3,
                                    fill=colors[color_idx], outline="black")
                    
                    # Show passenger ID for small numbers
                    if len(displayed_passengers) <= 4:
                        canvas.create_text(dot_x, dot_y, text=str(passenger.id), 
                                        font=("Arial", 5), fill="white")
                
                # Show count if more than max_display
                if len(passengers_in_elevator) > max_display:
                    canvas.create_text((x1 + x2) // 2, y2 - 10, 
                                    text=f"+{len(passengers_in_elevator) - max_display}",
                                    font=("Arial", 7), fill="darkred")

    def update_display(self):
        self.canvas.delete("all")
        agent_name = self.agent_type.get()
        self.root.title(f"Elevator Simulation - {agent_name.upper()} Agent")
        self.draw_building()
        self.draw_elevators()
        self.draw_passengers()
        self.update_elevator_buttons()
        self.update_external_call_buttons()
        self.update_stats()
    
    def draw_building(self):
        canvas = self.canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        margin = 60
        floor_height = (height - 2 * margin) / self.num_floors
        elevator_width = 100
        spacing = 20
        
        # Draw floors
        for floor in range(self.num_floors):
            y = height - margin - (floor + 0.5) * floor_height
            
            # Floor line
            canvas.create_line(margin, y, width - margin, y, fill="gray", width=1)
            
            # Floor label
            canvas.create_text(margin - 20, y, text=f"F{floor}", 
                             anchor="e", font=("Arial", 10, "bold"))
    
    def draw_elevators(self):
        canvas = self.canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        margin = 60
        floor_height = (height - 2 * margin) / self.num_floors
        elevator_width = 100
        spacing = 20
        
        state = self.building.get_state()
        
        for i, elevator_state in enumerate(state['elevators']):
            # Calculate position
            x1 = margin + i * (elevator_width + spacing)
            x2 = x1 + elevator_width
            
            # Use continuous position for smooth movement
            position = elevator_state['position']
            y_center = height - margin - (position + 0.5) * floor_height
            y1 = y_center - floor_height * 0.4
            y2 = y_center + floor_height * 0.4
            
            # Elevator color based on state
            color = self.get_elevator_color(elevator_state['state'])
            
            # Draw elevator cabin
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", width=2)
            
            # Elevator ID and capacity
            info = f"E{i}\n{len(self.building.elevators[i].passengers)}/{self.building.elevators[i].capacity}"
            canvas.create_text((x1 + x2) // 2, y_center, text=info, 
                             font=("Arial", 9, "bold"), justify="center")
            
            # Direction indicator
            if elevator_state['direction'] == 1:
                canvas.create_text(x2 + 8, y_center - 15, text="▲", font=("Arial", 12), fill="green")
            elif elevator_state['direction'] == -1:
                canvas.create_text(x2 + 8, y_center + 15, text="▼", font=("Arial", 12), fill="blue")
            
            # Speed indicator
            speed_text = f"{elevator_state['speed']:.1f}f/s"
            canvas.create_text((x1 + x2) // 2, y1 - 12, text=speed_text, 
                             font=("Arial", 8), fill="darkblue")
            
            # Draw external call indicators for this elevator
            self.draw_elevator_call_indicators(canvas, i, x1, x2, y1, y2)
    
    def draw_elevator_call_indicators(self, canvas, elevator_id, x1, x2, y1, y2):
        """Draw external call indicators for a specific elevator"""
        state = self.building.get_state()
        call_indicator_size = 8
        
        for floor, floor_state in state['floors'].items():
            calls = floor_state['elevator_calls'][elevator_id]
            
            if calls['call_up']:
                # Draw up call indicator near the elevator
                indicator_x = x1 - 15
                indicator_y = y1 + (y2 - y1) * 0.3
                canvas.create_oval(indicator_x - call_indicator_size, indicator_y - call_indicator_size,
                                 indicator_x + call_indicator_size, indicator_y + call_indicator_size,
                                 fill="red", outline="darkred")
                canvas.create_text(indicator_x, indicator_y, text="↑", font=("Arial", 8), fill="white")
            
            if calls['call_down']:
                # Draw down call indicator near the elevator
                indicator_x = x1 - 15
                indicator_y = y1 + (y2 - y1) * 0.7
                canvas.create_oval(indicator_x - call_indicator_size, indicator_y - call_indicator_size,
                                 indicator_x + call_indicator_size, indicator_y + call_indicator_size,
                                 fill="red", outline="darkred")
                canvas.create_text(indicator_x, indicator_y, text="↓", font=("Arial", 8), fill="white")
    
    def update_elevator_buttons(self):
        """Update the appearance and state of elevator internal buttons"""
        state = self.building.get_state()
        
        for elevator_id, elevator_state in enumerate(state['elevators']):
            internal_buttons = elevator_state['internal_buttons']
            passenger_count = elevator_state['passenger_count']
            
            for floor, pressed in enumerate(internal_buttons):
                btn = self.elevator_buttons[elevator_id][floor]
                
                # Disable button if elevator is empty
                if passenger_count == 0:
                    btn.configure(state="disabled", style="TButton")
                    if pressed:
                        # If button was pressed but elevator is empty, reset it
                        self.elevator_buttons[elevator_id][floor].configure(style="TButton")
                else:
                    btn.configure(state="normal")
                    if pressed:
                        btn.configure(style="Pressed.TButton")
                    else:
                        btn.configure(style="TButton")
    
    def update_external_call_buttons(self):
        """Update the appearance of external call buttons based on current state"""
        state = self.building.get_state()
        
        for floor, floor_state in state['floors'].items():
            for elevator_id in range(self.num_elevators):
                calls = floor_state['elevator_calls'][elevator_id]
                
                # Update up button
                up_key = (elevator_id, 'up')
                if floor in self.external_call_buttons and up_key in self.external_call_buttons[floor]:
                    btn = self.external_call_buttons[floor][up_key]
                    if calls['call_up']:
                        btn.configure(style="Pressed.TButton")
                    else:
                        btn.configure(style="TButton")
                
                # Update down button  
                down_key = (elevator_id, 'down')
                if floor in self.external_call_buttons and down_key in self.external_call_buttons[floor]:
                    btn = self.external_call_buttons[floor][down_key]
                    if calls['call_down']:
                        btn.configure(style="Pressed.TButton")
                    else:
                        btn.configure(style="TButton")
    
    def get_elevator_color(self, state_value):
        colors = {
            ElevatorState.IDLE.value: "lightgray",
            ElevatorState.MOVING_UP.value: "lightgreen", 
            ElevatorState.MOVING_DOWN.value: "lightblue",
            ElevatorState.DOOR_OPENING.value: "orange",
            ElevatorState.DOOR_CLOSING.value: "orange",
            ElevatorState.DOOR_OPEN.value: "red"
        }
        return colors.get(state_value, "white")
    
    def update_stats(self):
        state = self.building.get_state()
        total_waiting = sum(floor['waiting_up'] + floor['waiting_down'] 
                           for floor in state['floors'].values())
        
        total_seconds = int(state['time'])
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        stats_text = f"Time: {hours:02d}:{minutes:02d}:{seconds:02d} | Waiting Passengers: {total_waiting}"
        self.stats_label.config(text=stats_text)

def main():
    root = tk.Tk()
    
    # Create styles for pressed buttons
    style = ttk.Style()
    style.configure("Pressed.TButton", background="red", foreground="white")
    
    # Configuration
    NUM_FLOORS = 10  # Including ground floor (0)
    NUM_ELEVATORS = 4
    CAPACITY = 8
    
    app = ElevatorGUI(root, NUM_FLOORS, NUM_ELEVATORS, CAPACITY, verbose=False)
    root.mainloop()

if __name__ == "__main__":
    main()