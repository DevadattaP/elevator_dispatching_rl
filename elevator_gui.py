import tkinter as tk
from tkinter import ttk
import threading
import time
from building import Building
from utils.enums import ElevatorState
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque


class GraphWindow:
    def __init__(self, building: Building, num_elevators):
        self.building = building
        self.num_elevators = num_elevators
        self.is_running = False
        self.update_thread = None
        
        # Create the graph window
        self.root = tk.Toplevel()
        self.root.title("Elevator Performance Metrics")
        self.root.geometry("1200x800")
        
        # Data storage for graphs
        self.history_length = 100
        self.time_data = deque(maxlen=self.history_length)
        
        # Passenger metrics
        self.passengers_served = [deque(maxlen=self.history_length) for _ in range(num_elevators)]
        self.avg_waiting_time = [deque(maxlen=self.history_length) for _ in range(num_elevators)]
        self.avg_travel_time = [deque(maxlen=self.history_length) for _ in range(num_elevators)]
        self.idle_percentage = [deque(maxlen=self.history_length) for _ in range(num_elevators)]
        
        self.setup_graphs()
        self.start_updates()
    
    def setup_graphs(self):
        """Setup the matplotlib figures and axes - UPDATED"""
        # Create a figure with 4 subplots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Create canvas for tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize empty plots for each elevator
        self.lines_served = []
        self.lines_waiting = []
        self.lines_travel = []
        self.lines_idle = []
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for i in range(self.num_elevators):
            color = colors[i % len(colors)]
            # Passengers served per minute (5-min average)
            line1, = self.ax1.plot([], [], label=f'E{i}', color=color, linewidth=2)
            self.lines_served.append(line1)
            
            # Average waiting time
            line2, = self.ax2.plot([], [], label=f'E{i}', color=color, linewidth=2)
            self.lines_waiting.append(line2)
            
            # Average travel time
            line3, = self.ax3.plot([], [], label=f'E{i}', color=color, linewidth=2)
            self.lines_travel.append(line3)
            
            # Average idle time in minutes (5-min average)
            line4, = self.ax4.plot([], [], label=f'E{i}', color=color, linewidth=2)
            self.lines_idle.append(line4)
        
        # Setup axes with new labels
        self.setup_axes()

    def setup_axes(self):
        """Setup the axes labels and formatting - UPDATED FOR 5-MIN AVG"""
        # Passengers served per minute (5-minute moving average)
        self.ax1.set_title('Passenger Service Rate (5-min Moving Average)')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Passengers/Minute')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Average waiting time
        self.ax2.set_title('Average Waiting Time')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Time (s)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Average travel time
        self.ax3.set_title('Average Travel Time')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Time (s)')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # Average idle time in minutes (5-minute average) - CHANGED FROM PERCENTAGE
        self.ax4.set_title('Average Idle Time (5-min Moving Average)')
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Idle Time (Seconds)')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)

    def setup_controls(self):
        """Add control buttons to the graph window"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(control_frame, text="Close", 
                  command=self.close).pack(side="right", padx=5)
        
        ttk.Button(control_frame, text="Pause Updates", 
                  command=self.toggle_updates).pack(side="right", padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Updating...", foreground="green")
        self.status_label.pack(side="left", padx=5)
    
    def start_updates(self):
        """Start the graph update thread"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
    
    def toggle_updates(self):
        """Toggle graph updates on/off"""
        self.is_running = not self.is_running
        if self.is_running:
            self.status_label.config(text="Updating...", foreground="green")
        else:
            self.status_label.config(text="Paused", foreground="red")
    
    def update_loop(self):
        """Main update loop running in separate thread"""
        while True:
            if self.is_running:
                try:
                    self.update_graphs()
                except Exception as e:
                    print(f"Graph update error: {e}")
            time.sleep(0.5)  # Update every 500ms
    
    def update_graphs(self):
        """Update the graphs with current data - UPDATED FOR NEW METRICS"""
        current_time = getattr(self.building, 'time', 0)
        
        # Add current time to time data
        self.time_data.append(current_time)
        
        # Get actual metrics from building
        for elevator_id in range(self.num_elevators):
            metrics = self.building.get_elevator_metrics_for_display(elevator_id)
            
            # Use the new 5-minute average metrics
            passengers_per_minute_5min = metrics['passengers_per_minute_5min']
            avg_waiting = metrics['avg_waiting_time']
            avg_travel = metrics['avg_travel_time']
            avg_idle_time_5min = metrics['avg_idle_time_5min']  # Now in minutes
            
            self.passengers_served[elevator_id].append(passengers_per_minute_5min)
            self.avg_waiting_time[elevator_id].append(avg_waiting)
            self.avg_travel_time[elevator_id].append(avg_travel)
            self.idle_percentage[elevator_id].append(avg_idle_time_5min)  # Now stores idle time in minutes
        
        # Update each plot
        for i in range(self.num_elevators):
            if len(self.time_data) > 0:
                # Passengers served per minute (5-min average)
                if len(self.passengers_served[i]) > 0:
                    self.lines_served[i].set_data(self.time_data, self.passengers_served[i])
                
                # Average waiting time
                if len(self.avg_waiting_time[i]) > 0:
                    self.lines_waiting[i].set_data(self.time_data, self.avg_waiting_time[i])
                
                # Average travel time
                if len(self.avg_travel_time[i]) > 0:
                    self.lines_travel[i].set_data(self.time_data, self.avg_travel_time[i])
                
                # Average idle time in minutes (5-min average)
                if len(self.idle_percentage[i]) > 0:
                    self.lines_idle[i].set_data(self.time_data, self.idle_percentage[i])
        
        # Adjust axes limits
        self.adjust_axes_limits()
        
        # Redraw the canvas in the main thread
        self.root.after(0, self.canvas.draw)

    def adjust_axes_limits(self):
        """Adjust the axes limits based on current data - UPDATED"""
        if len(self.time_data) > 0:
            time_min, time_max = min(self.time_data), max(self.time_data)
            time_range = max(1, time_max - time_min)
            
            # Passengers served per minute
            all_served = [max(data) if data else 0 for data in self.passengers_served]
            max_served = max(all_served) if all_served else 5  # Lower default since it's rate
            self.ax1.set_xlim(time_min - time_range * 0.1, time_max + time_range * 0.1)
            self.ax1.set_ylim(0, max_served * 1.1)
            
            # Waiting time
            all_waiting = [max(data) if data else 0 for data in self.avg_waiting_time]
            max_waiting = max(all_waiting) if all_waiting else 10
            self.ax2.set_xlim(time_min - time_range * 0.1, time_max + time_range * 0.1)
            self.ax2.set_ylim(0, max_waiting * 1.1)
            
            # Travel time
            all_travel = [max(data) if data else 0 for data in self.avg_travel_time]
            max_travel = max(all_travel) if all_travel else 10
            self.ax3.set_xlim(time_min - time_range * 0.1, time_max + time_range * 0.1)
            self.ax3.set_ylim(0, max_travel * 1.1)
            
            # Idle time in minutes - dynamic scaling
            all_idle = [max(data) if data else 0 for data in self.idle_percentage]
            min_idle = min([min(data) if data else 0 for data in self.idle_percentage])
            max_idle = max(all_idle) if all_idle else 5  # Default to 5 minutes
            
            self.ax4.set_xlim(time_min - time_range * 0.1, time_max + time_range * 0.1)
            self.ax4.set_ylim(0, max_idle * 1.1) 

    def close(self):
        """Close the graph window"""
        self.is_running = False
        self.root.destroy()


class ElevatorGUI:
    def __init__(self, root, num_floors=4, num_elevators=2):
        self.root = root
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.building = Building(num_floors, num_elevators)
        
        self.is_running = False
        self.simulation_thread = None
        self.elevator_buttons = {}  # Internal buttons
        self.external_call_buttons = {}  # External call buttons
        self.generation_enabled = False
        
        self.panels_visible = True  # Track panel visibility state
        
        # Graph window reference
        self.graph_window = None
        
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
            print(f"Generation rate set to {rate_value:.1f}x - probability: {self.building.generation_probability:.4f}")
    
    def start_passenger_generation(self):
        """Start automatic passenger generation"""
        if not self.generation_enabled:
            self.generation_enabled = True
            self.building.start_passenger_generation()
            self.generation_status.config(text="Passanger Generation: ON", foreground="green")
            print("Started automatic passenger generation")
    
    def stop_passenger_generation(self):
        """Stop automatic passenger generation"""
        if self.generation_enabled:
            self.generation_enabled = False
            self.building.stop_passenger_generation()
            self.generation_status.config(text="Passanger Generation: OFF", foreground="red")
            print("Stopped automatic passenger generation")

    def call_elevator(self, floor: int, elevator_id: int, direction: str):
        """Handle external call button press for SPECIFIC elevator"""
        success = self.building.call_elevator(floor, elevator_id, direction)
        if success:
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
        self.building = Building(self.num_floors, self.num_elevators, self.speed_var.get())
        self.generation_enabled = False
        self.generation_status.config(text="Passanger Generation: OFF", foreground="red")
        # Reset button colors
        for elevator_id in self.elevator_buttons:
            for floor, btn in self.elevator_buttons[elevator_id].items():
                btn.configure(style="TButton")
        self.update_display()
    
    def run_simulation(self):
        while self.is_running:
            self.building.step()
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
    
    app = ElevatorGUI(root, NUM_FLOORS, NUM_ELEVATORS)
    root.mainloop()

if __name__ == "__main__":
    main()