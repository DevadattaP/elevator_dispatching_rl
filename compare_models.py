import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os

class ElevatorModelComparator:
    def __init__(self, results_file: str = "enhanced_evaluation_results.json"):
        self.results_file = results_file
        self.results_data = None
        self.df = None
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def load_results(self):
        """Load results from JSON file."""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file {self.results_file} not found")
            
        with open(self.results_file, 'r') as f:
            self.results_data = json.load(f)
        print(f"Loaded results from {self.results_file}")
        
    def flatten_results(self):
        """Flatten nested JSON structure into a pandas DataFrame."""
        flattened_data = []
        
        def extract_configurations(data, path=None):
            if path is None:
                path = []
                
            if isinstance(data, dict):
                if 'avg_wait_time' in data:  # This is a result entry
                    config_dict = {}
                    # Reconstruct configuration from path
                    for i, key in enumerate(path):
                        if i == 0:
                            config_dict['agent_type'] = key
                        elif i == 1:
                            config_dict['wrapper'] = key
                        elif i == 2:
                            config_dict['observation_type'] = key
                        elif i == 3:
                            config_dict['reward_type'] = key
                        elif i == 4:
                            config_dict['action_type'] = key
                        elif i == 5:
                            config_dict['traffic_pattern'] = key
                        elif i == 6:
                            config_dict['smdp'] = key
                    
                    # Add all metrics
                    config_dict.update(data)
                    flattened_data.append(config_dict)
                else:
                    for key, value in data.items():
                        extract_configurations(value, path + [key])
        
        extract_configurations(self.results_data)
        self.df = pd.DataFrame(flattened_data)
        
        # Create a unique identifier for each configuration
        self.df['config_id'] = self.df.apply(
            lambda x: f"{x['agent_type']}_{x['wrapper']}_{x['observation_type']}_{x['reward_type']}_{x['action_type']}_{x['traffic_pattern']}", 
            axis=1
        )
        
        print(f"Flattened {len(self.df)} configurations")
        return self.df
    
    def get_best_configurations(self, metric: str = 'avg_wait_time', top_k: int = 10, ascending: bool = True):
        """Get top K configurations based on specified metric."""
        if self.df is None:
            self.flatten_results()
            
        sorted_df = self.df.sort_values(by=metric, ascending=ascending).head(top_k)
        return sorted_df
    
    def plot_comparison_bar_chart(self, metric: str, group_by: str = 'agent_type', 
                                 title: str = None, figsize: tuple = (12, 8)):
        """Create bar chart comparing models based on specified metric."""
        if self.df is None:
            self.flatten_results()
            
        if title is None:
            title = f'Comparison of {metric.replace("_", " ").title()}'
            
        plt.figure(figsize=figsize)
        
        # Group data
        grouped = self.df.groupby(group_by)[metric].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped.sort_values('mean')
        
        # Create bar plot
        bars = plt.bar(range(len(grouped)), grouped['mean'], 
                      yerr=grouped['std'], capsize=5, alpha=0.7,
                      color=self.colors[:len(grouped)])
        
        plt.xlabel(group_by.replace('_', ' ').title())
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.xticks(range(len(grouped)), grouped[group_by], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(grouped['mean'], grouped['std'])):
            plt.text(i, mean + std + max(grouped['mean']) * 0.01, 
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        return grouped
    
    def plot_metric_comparison_grid(self, metrics: List[str] = None, 
                                   group_by: str = 'agent_type',
                                   figsize: tuple = (18, 12)):
        """Create a grid of bar charts for multiple metrics."""
        if metrics is None:
            metrics = ['avg_wait_time', 'avg_journey_time', 'avg_passengers_completed', 
                      'max_wait_time', 'fairness_metric', 'avg_episode_reward']
        
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            grouped = self.df.groupby(group_by)[metric].agg(['mean', 'std']).reset_index()
            grouped = grouped.sort_values('mean')
            
            bars = ax.bar(range(len(grouped)), grouped['mean'], 
                         yerr=grouped['std'], capsize=5, alpha=0.7,
                         color=self.colors[:len(grouped)])
            
            ax.set_xlabel(group_by.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped[group_by], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for j, (mean, std) in enumerate(zip(grouped['mean'], grouped['std'])):
                ax.text(j, mean + std + max(grouped['mean']) * 0.02, 
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_traffic_pattern_comparison(self, metric: str = 'avg_wait_time',
                                       agent_types: List[str] = None,
                                       figsize: tuple = (14, 8)):
        """Compare performance across different traffic patterns."""
        if self.df is None:
            self.flatten_results()
            
        if agent_types is None:
            agent_types = self.df['agent_type'].unique()
            
        plt.figure(figsize=figsize)
        
        traffic_patterns = self.df['traffic_pattern'].unique()
        x = np.arange(len(traffic_patterns))
        width = 0.8 / len(agent_types)
        
        for i, agent_type in enumerate(agent_types):
            agent_data = []
            for pattern in traffic_patterns:
                mask = (self.df['agent_type'] == agent_type) & (self.df['traffic_pattern'] == pattern)
                if mask.any():
                    agent_data.append(self.df[mask][metric].mean())
                else:
                    agent_data.append(0)
            
            plt.bar(x + i * width - width * (len(agent_types) - 1) / 2, 
                   agent_data, width, label=agent_type, alpha=0.7)
        
        plt.xlabel('Traffic Pattern')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} by Traffic Pattern')
        plt.xticks(x, traffic_patterns, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_reward_type_comparison(self, metric: str = 'avg_wait_time',
                                   agent_types: List[str] = None,
                                   figsize: tuple = (14, 8)):
        """Compare performance across different reward types."""
        if self.df is None:
            self.flatten_results()
            
        if agent_types is None:
            agent_types = self.df['agent_type'].unique()
            
        plt.figure(figsize=figsize)
        
        reward_types = self.df['reward_type'].unique()
        x = np.arange(len(reward_types))
        width = 0.8 / len(agent_types)
        
        for i, agent_type in enumerate(agent_types):
            agent_data = []
            for reward_type in reward_types:
                mask = (self.df['agent_type'] == agent_type) & (self.df['reward_type'] == reward_type)
                if mask.any():
                    agent_data.append(self.df[mask][metric].mean())
                else:
                    agent_data.append(0)
            
            plt.bar(x + i * width - width * (len(agent_types) - 1) / 2, 
                   agent_data, width, label=agent_type, alpha=0.7)
        
        plt.xlabel('Reward Type')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} by Reward Type')
        plt.xticks(x, reward_types, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_best_configurations_radar(self, top_k: int = 8, metrics: List[str] = None):
        """Create radar chart for top configurations."""
        if metrics is None:
            metrics = ['avg_wait_time', 'avg_journey_time', 'avg_passengers_completed', 
                      'max_wait_time', 'fairness_metric']
        
        best_configs = self.get_best_configurations('avg_wait_time', top_k, ascending=True)
        
        # Normalize metrics for radar chart (lower is better for wait times, higher for completions)
        normalized_data = []
        config_names = []
        
        for _, config in best_configs.iterrows():
            normalized_row = []
            for metric in metrics:
                if metric in ['avg_passengers_completed']:
                    # Higher is better
                    max_val = self.df[metric].max()
                    min_val = self.df[metric].min()
                    normalized_val = (config[metric] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                else:
                    # Lower is better
                    max_val = self.df[metric].max()
                    min_val = self.df[metric].min()
                    normalized_val = 1 - (config[metric] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                normalized_row.append(normalized_val)
            
            normalized_data.append(normalized_row)
            config_names.append(config['config_id'][:30] + '...')  # Truncate long names
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        normalized_data = [row + row[:1] for row in normalized_data]  # Complete the circle for each config
        
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        for i, data in enumerate(normalized_data):
            ax.plot(angles, data, 'o-', linewidth=2, label=config_names[i], 
                   color=self.colors[i % len(self.colors)])
            ax.fill(angles, data, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(f'Top {top_k} Configurations - Radar Chart\n(Higher values are better)', size=14)
        plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def create_comprehensive_report(self):
        """Generate a comprehensive comparison report."""
        if self.df is None:
            self.flatten_results()
            
        print("=" * 60)
        print("COMPREHENSIVE ELEVATOR MODEL COMPARISON REPORT")
        print("=" * 60)
        
        # Overall statistics
        print(f"\nTotal configurations evaluated: {len(self.df)}")
        print(f"Agent types: {', '.join(self.df['agent_type'].unique())}")
        print(f"Traffic patterns: {', '.join(self.df['traffic_pattern'].unique())}")
        
        # Best configurations
        print("\n" + "=" * 40)
        print("TOP 5 CONFIGURATIONS (by Average Wait Time)")
        print("=" * 40)
        best_configs = self.get_best_configurations('avg_wait_time', 5, True)
        for i, (_, config) in enumerate(best_configs.iterrows()):
            print(f"{i+1}. {config['config_id']}")
            print(f"   Wait Time: {config['avg_wait_time']:.2f}s | "
                  f"Journey Time: {config['avg_journey_time']:.2f}s | "
                  f"Completed: {config['avg_passengers_completed']:.1f}")
        
        # Agent type comparison
        print("\n" + "=" * 40)
        print("AGENT TYPE PERFORMANCE COMPARISON")
        print("=" * 40)
        agent_stats = self.df.groupby('agent_type').agg({
            'avg_wait_time': ['mean', 'std', 'min'],
            'avg_journey_time': ['mean', 'std'],
            'avg_passengers_completed': ['mean', 'std']
        }).round(2)
        print(agent_stats)
        
        # Rule-based vs RL comparison
        print("\n" + "=" * 40)
        print("RULE-BASED vs RL COMPARISON")
        print("=" * 40)
        rule_based_avg = self.df[self.df['agent_type'] == 'rule_based']['avg_wait_time'].mean()
        rl_avg = self.df[self.df['agent_type'] != 'rule_based']['avg_wait_time'].mean()
        improvement = ((rule_based_avg - rl_avg) / rule_based_avg) * 100
        print(f"Rule-based average wait time: {rule_based_avg:.2f}s")
        print(f"RL average wait time: {rl_avg:.2f}s")
        print(f"Improvement: {improvement:.1f}%")
    
    def run_full_comparison(self):
        """Run complete comparison analysis."""
        print("Starting comprehensive model comparison...")
        
        self.load_results()
        self.flatten_results()
        
        # Generate comprehensive report
        self.create_comprehensive_report()
        
        # Create all comparison plots
        print("\nGenerating comparison plots...")
        
        # 1. Main metric comparisons
        self.plot_metric_comparison_grid()
        
        # # 2. Agent type comparisons
        # self.plot_comparison_bar_chart('avg_wait_time', 'agent_type', 
        #                               'Average Wait Time by Agent Type')
        # self.plot_comparison_bar_chart('avg_passengers_completed', 'agent_type',
        #                               'Average Passengers Completed by Agent Type')
        
        # # 3. Traffic pattern analysis
        # self.plot_traffic_pattern_comparison()
        
        # # 4. Reward type analysis
        # self.plot_reward_type_comparison()
        
        # # 5. Radar chart for top configurations
        self.plot_best_configurations_radar()
        
        print("\nComparison complete! All plots generated.")


# ===== Quick Comparison Function =====
def quick_comparison(results_file: str = "enhanced_evaluation_results.json"):
    """Quick comparison for immediate insights."""
    comparator = ElevatorModelComparator(results_file)
    comparator.load_results()
    comparator.flatten_results()
    
    # Quick summary
    print("QUICK COMPARISON SUMMARY")
    print("=" * 40)
    
    # Best overall
    best = comparator.get_best_configurations('avg_wait_time', 1, True).iloc[0]
    print(f"Best configuration: {best['config_id']}")
    print(f"Wait Time: {best['avg_wait_time']:.2f}s | "
          f"Journey Time: {best['avg_journey_time']:.2f}s")
    
    # Rule-based baseline
    rule_based = comparator.df[comparator.df['agent_type'] == 'rule_based']
    if len(rule_based) > 0:
        rb_avg = rule_based['avg_wait_time'].mean()
        improvement = ((rb_avg - best['avg_wait_time']) / rb_avg) * 100
        print(f"vs Rule-based ({rb_avg:.2f}s): {improvement:.1f}% improvement")
    
    # Top 3 agent types
    agent_perf = comparator.df.groupby('agent_type')['avg_wait_time'].mean().sort_values()
    print(f"\nTop 3 Agent Types:")
    for i, (agent, perf) in enumerate(agent_perf.head(3).items()):
        print(f"  {i+1}. {agent}: {perf:.2f}s")
    
    # Generate key plots
    comparator.plot_comparison_bar_chart('avg_wait_time', 'agent_type')
    comparator.plot_traffic_pattern_comparison()


# ===== Main Execution =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare elevator model evaluation results')
    parser.add_argument('--results-file', default='evaluation_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick comparison only')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Run full comprehensive analysis')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_comparison(args.results_file)
    elif args.full:
        comparator = ElevatorModelComparator(args.results_file)
        comparator.run_full_comparison()
    else:
        # Default: run full analysis
        comparator = ElevatorModelComparator(args.results_file)
        comparator.run_full_comparison()