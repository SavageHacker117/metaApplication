"""
Visualization Manager for RL-LLM System

This module provides comprehensive visualization capabilities for RL training,
including real-time monitoring, performance plots, and interactive dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import time

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VisualizationManager:
    """
    Main visualization manager for RL training monitoring and analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualization manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', './visualizations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.training_data = defaultdict(list)
        self.episode_data = []
        self.real_time_data = deque(maxlen=1000)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Plot configurations
        self.plot_config = config.get('plots', {})
        self.figure_size = self.plot_config.get('figure_size', (12, 8))
        self.dpi = self.plot_config.get('dpi', 100)
        
        logger.info(f"Initialized VisualizationManager with output dir: {self.output_dir}")
    
    def add_training_data(self, step: int, metrics: Dict[str, float]):
        """
        Add training data point.
        
        Args:
            step: Training step
            metrics: Dictionary of metric name -> value
        """
        self.training_data['step'].append(step)
        self.training_data['timestamp'].append(datetime.now())
        
        for metric_name, value in metrics.items():
            self.training_data[metric_name].append(value)
        
        # Add to real-time data
        data_point = {'step': step, 'timestamp': datetime.now(), **metrics}
        self.real_time_data.append(data_point)
    
    def add_episode_data(self, episode: int, episode_info: Dict[str, Any]):
        """
        Add episode data.
        
        Args:
            episode: Episode number
            episode_info: Episode information dictionary
        """
        episode_data = {
            'episode': episode,
            'timestamp': datetime.now(),
            **episode_info
        }
        self.episode_data.append(episode_data)
    
    def plot_training_curves(self, metrics: Optional[List[str]] = None, 
                           save: bool = True, show: bool = False) -> plt.Figure:
        """
        Plot training curves for specified metrics.
        
        Args:
            metrics: List of metrics to plot (None for all)
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.training_data['step']:
            logger.warning("No training data available for plotting")
            return None
        
        if metrics is None:
            # Auto-detect numeric metrics
            metrics = [key for key in self.training_data.keys() 
                      if key not in ['step', 'timestamp'] and self.training_data[key]]
        
        # Create subplots
        n_metrics = len(metrics)
        if n_metrics == 0:
            logger.warning("No valid metrics found for plotting")
            return None
        
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figure_size, dpi=self.dpi)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        steps = self.training_data['step']
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = self.training_data[metric]
            
            if not values:
                continue
            
            # Plot the metric
            ax.plot(steps[:len(values)], values, linewidth=2, alpha=0.8)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            
            # Add moving average if data is noisy
            if len(values) > 20:
                window_size = min(50, len(values) // 10)
                if window_size > 1:
                    moving_avg = pd.Series(values).rolling(window=window_size, center=True).mean()
                    ax.plot(steps[:len(values)], moving_avg, 
                           color='red', linewidth=2, alpha=0.7, 
                           label=f'Moving Avg ({window_size})')
                    ax.legend()
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved training curves to {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_episode_analysis(self, save: bool = True, show: bool = False) -> plt.Figure:
        """
        Plot episode-based analysis.
        
        Args:
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.episode_data:
            logger.warning("No episode data available for plotting")
            return None
        
        df = pd.DataFrame(self.episode_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # Episode rewards
        if 'total_reward' in df.columns:
            axes[0, 0].plot(df['episode'], df['total_reward'], 'b-', linewidth=2, alpha=0.7)
            axes[0, 0].set_title('Episode Rewards', fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add moving average
            if len(df) > 10:
                window_size = min(20, len(df) // 5)
                moving_avg = df['total_reward'].rolling(window=window_size, center=True).mean()
                axes[0, 0].plot(df['episode'], moving_avg, 'r-', linewidth=2, alpha=0.8, 
                               label=f'Moving Avg ({window_size})')
                axes[0, 0].legend()
        
        # Episode lengths
        if 'episode_length' in df.columns:
            axes[0, 1].plot(df['episode'], df['episode_length'], 'g-', linewidth=2, alpha=0.7)
            axes[0, 1].set_title('Episode Lengths', fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Reward distribution
        if 'total_reward' in df.columns:
            axes[1, 0].hist(df['total_reward'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('Reward Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Total Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate (if available)
        if 'success' in df.columns:
            success_rate = df.groupby(df.index // 10)['success'].mean()  # Group by 10s
            axes[1, 1].plot(success_rate.index * 10, success_rate.values, 'o-', linewidth=2)
            axes[1, 1].set_title('Success Rate (per 10 episodes)', fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Plot episode duration if available
            if 'duration' in df.columns:
                axes[1, 1].plot(df['episode'], df['duration'], 'm-', linewidth=2, alpha=0.7)
                axes[1, 1].set_title('Episode Duration', fontweight='bold')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Duration (s)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'episode_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved episode analysis to {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_interactive_dashboard(self, save: bool = True) -> go.Figure:
        """
        Create interactive Plotly dashboard.
        
        Args:
            save: Whether to save the dashboard as HTML
            
        Returns:
            Plotly figure
        """
        if not self.training_data['step'] and not self.episode_data:
            logger.warning("No data available for dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Episode Rewards', 'Learning Rate', 'Performance Metrics'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training data plots
        if self.training_data['step']:
            steps = self.training_data['step']
            
            # Loss plot
            if 'loss' in self.training_data:
                fig.add_trace(
                    go.Scatter(x=steps, y=self.training_data['loss'], 
                             name='Loss', line=dict(color='red', width=2)),
                    row=1, col=1
                )
            
            # Learning rate plot
            if 'learning_rate' in self.training_data:
                fig.add_trace(
                    go.Scatter(x=steps, y=self.training_data['learning_rate'], 
                             name='Learning Rate', line=dict(color='blue', width=2)),
                    row=2, col=1
                )
        
        # Episode data plots
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            
            # Episode rewards
            if 'total_reward' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['episode'], y=df['total_reward'], 
                             name='Episode Reward', line=dict(color='green', width=2)),
                    row=1, col=2
                )
                
                # Add moving average
                if len(df) > 10:
                    window_size = min(20, len(df) // 5)
                    moving_avg = df['total_reward'].rolling(window=window_size, center=True).mean()
                    fig.add_trace(
                        go.Scatter(x=df['episode'], y=moving_avg, 
                                 name=f'Moving Avg ({window_size})', 
                                 line=dict(color='orange', width=2, dash='dash')),
                        row=1, col=2
                    )
            
            # Performance metrics
            if 'episode_length' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['episode'], y=df['episode_length'], 
                             name='Episode Length', line=dict(color='purple', width=2)),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="RL Training Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Training Step", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_xaxes(title_text="Training Step", row=2, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=2)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Reward", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        fig.update_yaxes(title_text="Steps", row=2, col=2)
        
        if save:
            filename = self.output_dir / f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(filename)
            logger.info(f"Saved interactive dashboard to {filename}")
        
        return fig
    
    def plot_heatmap(self, data: np.ndarray, title: str = "Heatmap", 
                    labels: Optional[List[str]] = None, save: bool = True, 
                    show: bool = False) -> plt.Figure:
        """
        Create a heatmap visualization.
        
        Args:
            data: 2D numpy array
            title: Plot title
            labels: Optional labels for axes
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create heatmap
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if labels:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'heatmap_{title.lower().replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved heatmap to {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def start_real_time_monitoring(self, update_interval: float = 1.0):
        """
        Start real-time monitoring thread.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Real-time monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(update_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started real-time monitoring")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped real-time monitoring")
    
    def _monitoring_loop(self, update_interval: float):
        """Real-time monitoring loop."""
        while self.monitoring_active:
            try:
                # Update real-time plots here
                # This is a placeholder - implement actual real-time plotting
                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break
    
    def save_data(self, filename: Optional[str] = None):
        """
        Save all visualization data to file.
        
        Args:
            filename: Optional filename (auto-generated if None)
        """
        if filename is None:
            filename = f'visualization_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        filepath = self.output_dir / filename
        
        # Prepare data for JSON serialization
        data_to_save = {
            'training_data': dict(self.training_data),
            'episode_data': self.episode_data,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        for key, values in data_to_save['training_data'].items():
            if key == 'timestamp':
                data_to_save['training_data'][key] = [
                    ts.isoformat() if isinstance(ts, datetime) else ts 
                    for ts in values
                ]
        
        for episode in data_to_save['episode_data']:
            if 'timestamp' in episode and isinstance(episode['timestamp'], datetime):
                episode['timestamp'] = episode['timestamp'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        logger.info(f"Saved visualization data to {filepath}")
    
    def load_data(self, filepath: Union[str, Path]):
        """
        Load visualization data from file.
        
        Args:
            filepath: Path to data file
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore training data
        self.training_data = defaultdict(list, data.get('training_data', {}))
        
        # Convert timestamp strings back to datetime objects
        if 'timestamp' in self.training_data:
            self.training_data['timestamp'] = [
                datetime.fromisoformat(ts) for ts in self.training_data['timestamp']
            ]
        
        # Restore episode data
        self.episode_data = data.get('episode_data', [])
        for episode in self.episode_data:
            if 'timestamp' in episode:
                episode['timestamp'] = datetime.fromisoformat(episode['timestamp'])
        
        logger.info(f"Loaded visualization data from {filepath}")
    
    def generate_report(self, save: bool = True) -> str:
        """
        Generate a comprehensive visualization report.
        
        Args:
            save: Whether to save the report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "# RL Training Visualization Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
        ]
        
        # Training data summary
        if self.training_data['step']:
            total_steps = max(self.training_data['step'])
            report_lines.extend([
                f"- Total training steps: {total_steps:,}",
                f"- Data points collected: {len(self.training_data['step']):,}",
            ])
        
        # Episode data summary
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            report_lines.extend([
                f"- Total episodes: {len(self.episode_data):,}",
            ])
            
            if 'total_reward' in df.columns:
                report_lines.extend([
                    f"- Average reward: {df['total_reward'].mean():.2f}",
                    f"- Best reward: {df['total_reward'].max():.2f}",
                    f"- Worst reward: {df['total_reward'].min():.2f}",
                ])
        
        report_content = "\n".join(report_lines)
        
        if save:
            filename = self.output_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
            with open(filename, 'w') as f:
                f.write(report_content)
            logger.info(f"Saved visualization report to {filename}")
        
        return report_content


def create_visualization_manager(config: Dict[str, Any]) -> VisualizationManager:
    """
    Factory function to create visualization manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VisualizationManager instance
    """
    return VisualizationManager(config)

