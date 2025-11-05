import tkinter as tk
from tkinter import ttk
import threading
import time
from building import Building
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
