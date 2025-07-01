"""
Advanced Analysis and Visualization System for RL-LLM

This module provides comprehensive analysis and visualization capabilities including
interactive dashboards, statistical analysis, performance profiling, and real-time
monitoring with advanced plotting and data exploration tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
from collections import defaultdict, deque
import threading
import time
from abc import ABC, abstractmethod
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import networkx as nx
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization."""
    output_dir: Path = Path('./analysis_output')
    dashboard_port: int = 8050
    dashboard_host: str = "127.0.0.1"
    auto_refresh_interval: int = 30  # seconds
    plot_style: str = "plotly_white"
    color_palette: str = "viridis"
    figure_width: int = 1200
    figure_height: int = 600
    statistical_significance: float = 0.05
    smoothing_window: int = 10
    enable_real_time: bool = True
    cache_size: int = 1000


class DataAnalyzer:
    """Advanced data analysis engine."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_timestamps = {}
        
        logger.info("Initialized DataAnalyzer")
    
    def analyze_training_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive training performance analysis."""
        analysis = {}
        
        # Basic statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        analysis['basic_stats'] = data[numeric_columns].describe()
        
        # Learning curves analysis
        if 'episode' in data.columns and 'reward' in data.columns:
            analysis['learning_curve'] = self._analyze_learning_curve(data)
        
        # Convergence analysis
        analysis['convergence'] = self._analyze_convergence(data)
        
        # Performance trends
        analysis['trends'] = self._analyze_trends(data)
        
        # Statistical tests
        analysis['statistical_tests'] = self._perform_statistical_tests(data)
        
        # Anomaly detection
        analysis['anomalies'] = self._detect_anomalies(data)
        
        return analysis
    
    def _analyze_learning_curve(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze learning curve characteristics."""
        rewards = data['reward'].values
        episodes = data['episode'].values
        
        # Smooth the curve
        if len(rewards) > self.config.smoothing_window:
            smoothed_rewards = savgol_filter(rewards, self.config.smoothing_window, 3)
        else:
            smoothed_rewards = rewards
        
        # Calculate metrics
        initial_performance = np.mean(rewards[:min(100, len(rewards))])
        final_performance = np.mean(rewards[-min(100, len(rewards)):])
        improvement = final_performance - initial_performance
        
        # Find convergence point
        convergence_point = self._find_convergence_point(smoothed_rewards)
        
        # Calculate learning rate
        learning_rate = self._calculate_learning_rate(episodes, smoothed_rewards)
        
        return {
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'improvement': improvement,
            'convergence_episode': convergence_point,
            'learning_rate': learning_rate,
            'smoothed_curve': smoothed_rewards.tolist()
        }
    
    def _analyze_convergence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze convergence properties."""
        convergence_metrics = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            if column in ['episode', 'step']:
                continue
            
            values = data[column].dropna().values
            if len(values) < 10:
                continue
            
            # Calculate convergence metrics
            window_size = min(50, len(values) // 4)
            if window_size < 5:
                continue
            
            # Moving variance
            moving_var = pd.Series(values).rolling(window_size).var()
            
            # Convergence point (where variance stabilizes)
            var_threshold = np.std(moving_var.dropna()) * 0.1
            convergence_idx = None
            
            for i in range(len(moving_var) - window_size):
                if i < window_size:
                    continue
                recent_var = moving_var.iloc[i:i+window_size]
                if recent_var.std() < var_threshold:
                    convergence_idx = i
                    break
            
            convergence_metrics[column] = {
                'converged': convergence_idx is not None,
                'convergence_point': convergence_idx,
                'final_variance': moving_var.iloc[-1] if len(moving_var) > 0 else None,
                'stability_score': 1.0 / (1.0 + moving_var.iloc[-window_size:].mean()) if len(moving_var) >= window_size else 0
            }
        
        return convergence_metrics
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends."""
        trends = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            if column in ['episode', 'step']:
                continue
            
            values = data[column].dropna().values
            if len(values) < 10:
                continue
            
            # Linear trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Trend classification
            if p_value < self.config.statistical_significance:
                if slope > 0:
                    trend_type = "increasing"
                else:
                    trend_type = "decreasing"
            else:
                trend_type = "stable"
            
            # Change points detection
            change_points = self._detect_change_points(values)
            
            trends[column] = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_type': trend_type,
                'change_points': change_points,
                'recent_trend': self._calculate_recent_trend(values)
            }
        
        return trends
    
    def _perform_statistical_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}
        
        # Normality tests
        for column in data.select_dtypes(include=[np.number]).columns:
            values = data[column].dropna().values
            if len(values) < 8:
                continue
            
            # Shapiro-Wilk test for normality
            try:
                stat, p_value = stats.shapiro(values[:5000])  # Limit for performance
                tests[f"{column}_normality"] = {
                    'test': 'shapiro_wilk',
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > self.config.statistical_significance
                }
            except Exception as e:
                logger.debug(f"Normality test failed for {column}: {e}")
        
        # Stationarity tests
        for column in data.select_dtypes(include=[np.number]).columns:
            if column in ['episode', 'step']:
                continue
            
            values = data[column].dropna().values
            if len(values) < 20:
                continue
            
            try:
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(values)
                tests[f"{column}_stationarity"] = {
                    'test': 'augmented_dickey_fuller',
                    'statistic': result[0],
                    'p_value': result[1],
                    'is_stationary': result[1] < self.config.statistical_significance
                }
            except Exception as e:
                logger.debug(f"Stationarity test failed for {column}: {e}")
        
        return tests
    
    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the data."""
        anomalies = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            values = data[column].dropna().values
            if len(values) < 10:
                continue
            
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(values))
            z_anomalies = np.where(z_scores > 3)[0]
            
            # IQR based anomaly detection
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = np.where((values < lower_bound) | (values > upper_bound))[0]
            
            anomalies[column] = {
                'z_score_anomalies': z_anomalies.tolist(),
                'iqr_anomalies': iqr_anomalies.tolist(),
                'anomaly_rate': len(set(z_anomalies.tolist() + iqr_anomalies.tolist())) / len(values)
            }
        
        return anomalies
    
    def _find_convergence_point(self, values: np.ndarray, window_size: int = 50) -> Optional[int]:
        """Find the convergence point in a time series."""
        if len(values) < window_size * 2:
            return None
        
        # Calculate moving average and variance
        moving_avg = pd.Series(values).rolling(window_size).mean()
        moving_var = pd.Series(values).rolling(window_size).var()
        
        # Find point where variance becomes stable
        var_threshold = np.std(moving_var.dropna()) * 0.2
        
        for i in range(window_size, len(values) - window_size):
            recent_vars = moving_var.iloc[i:i+window_size]
            if recent_vars.std() < var_threshold:
                return i
        
        return None
    
    def _calculate_learning_rate(self, episodes: np.ndarray, rewards: np.ndarray) -> float:
        """Calculate the learning rate from episode-reward data."""
        if len(episodes) < 10:
            return 0.0
        
        # Fit exponential curve: reward = a * (1 - exp(-b * episode)) + c
        try:
            from scipy.optimize import curve_fit
            
            def exp_func(x, a, b, c):
                return a * (1 - np.exp(-b * x)) + c
            
            popt, _ = curve_fit(exp_func, episodes, rewards, maxfev=1000)
            return popt[1]  # Learning rate parameter
        except Exception:
            # Fallback to linear approximation
            slope, _, _, _, _ = stats.linregress(episodes, rewards)
            return max(0, slope)
    
    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        """Detect change points in time series."""
        if len(values) < 20:
            return []
        
        change_points = []
        window_size = max(10, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            before = values[i-window_size:i]
            after = values[i:i+window_size]
            
            # T-test for difference in means
            try:
                _, p_value = stats.ttest_ind(before, after)
                if p_value < self.config.statistical_significance:
                    change_points.append(i)
            except Exception:
                continue
        
        return change_points
    
    def _calculate_recent_trend(self, values: np.ndarray, window_ratio: float = 0.2) -> Dict[str, float]:
        """Calculate trend for recent portion of data."""
        window_size = max(10, int(len(values) * window_ratio))
        recent_values = values[-window_size:]
        
        if len(recent_values) < 3:
            return {'slope': 0, 'r_squared': 0}
        
        x = np.arange(len(recent_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value
        }
    
    def compare_experiments(self, experiment_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare multiple experiments statistically."""
        comparison = {}
        
        # Get common metrics
        common_metrics = set.intersection(*[set(df.columns) for df in experiment_data.values()])
        common_metrics = [col for col in common_metrics if col not in ['episode', 'step', 'timestamp']]
        
        for metric in common_metrics:
            metric_comparison = {}
            
            # Extract final performance for each experiment
            final_performances = {}
            for exp_name, df in experiment_data.items():
                if metric in df.columns:
                    final_values = df[metric].dropna().tail(100)  # Last 100 values
                    if len(final_values) > 0:
                        final_performances[exp_name] = final_values.mean()
            
            if len(final_performances) < 2:
                continue
            
            # Statistical comparison
            exp_names = list(final_performances.keys())
            exp_values = [experiment_data[name][metric].dropna().tail(100).values 
                         for name in exp_names]
            
            # ANOVA test
            try:
                f_stat, p_value = stats.f_oneway(*exp_values)
                metric_comparison['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.statistical_significance
                }
            except Exception as e:
                logger.debug(f"ANOVA failed for {metric}: {e}")
            
            # Pairwise t-tests
            pairwise_tests = {}
            for i in range(len(exp_names)):
                for j in range(i+1, len(exp_names)):
                    name1, name2 = exp_names[i], exp_names[j]
                    try:
                        t_stat, p_val = stats.ttest_ind(exp_values[i], exp_values[j])
                        pairwise_tests[f"{name1}_vs_{name2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < self.config.statistical_significance
                        }
                    except Exception as e:
                        logger.debug(f"T-test failed for {name1} vs {name2}: {e}")
            
            metric_comparison['pairwise_tests'] = pairwise_tests
            metric_comparison['final_performances'] = final_performances
            
            comparison[metric] = metric_comparison
        
        return comparison


class AdvancedVisualizer:
    """Advanced visualization engine with interactive plots."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.config.color_palette)
        
        logger.info("Initialized AdvancedVisualizer")
    
    def create_training_dashboard(self, data: pd.DataFrame) -> go.Figure:
        """Create comprehensive training dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Learning Curve', 'Loss Evolution', 'Performance Distribution', 
                          'Training Metrics', 'Convergence Analysis', 'Recent Performance'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Learning curve
        if 'episode' in data.columns and 'reward' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['episode'], y=data['reward'], 
                          mode='lines', name='Reward', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add smoothed curve
            if len(data) > self.config.smoothing_window:
                smoothed = savgol_filter(data['reward'], self.config.smoothing_window, 3)
                fig.add_trace(
                    go.Scatter(x=data['episode'], y=smoothed, 
                              mode='lines', name='Smoothed Reward', 
                              line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
        
        # Loss evolution
        loss_columns = [col for col in data.columns if 'loss' in col.lower()]
        for loss_col in loss_columns[:3]:  # Limit to 3 loss types
            fig.add_trace(
                go.Scatter(x=data.index, y=data[loss_col], 
                          mode='lines', name=loss_col),
                row=1, col=2
            )
        
        # Performance distribution
        if 'reward' in data.columns:
            fig.add_trace(
                go.Histogram(x=data['reward'], name='Reward Distribution', 
                           nbinsx=30, opacity=0.7),
                row=2, col=1
            )
        
        # Training metrics heatmap
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.columns,
                          colorscale='RdBu', 
                          name='Correlation'),
                row=2, col=2
            )
        
        # Convergence analysis
        if 'episode' in data.columns and 'reward' in data.columns:
            window_size = min(50, len(data) // 10)
            if window_size > 5:
                rolling_std = data['reward'].rolling(window_size).std()
                fig.add_trace(
                    go.Scatter(x=data['episode'], y=rolling_std, 
                              mode='lines', name='Rolling Std',
                              line=dict(color='green')),
                    row=3, col=1
                )
        
        # Recent performance
        recent_data = data.tail(min(1000, len(data)))
        if 'reward' in recent_data.columns:
            fig.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['reward'], 
                          mode='lines+markers', name='Recent Rewards',
                          line=dict(color='purple')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=self.config.figure_height * 1.5,
            width=self.config.figure_width,
            title_text="Training Dashboard",
            showlegend=True,
            template=self.config.plot_style
        )
        
        return fig
    
    def create_comparison_plot(self, experiment_data: Dict[str, pd.DataFrame], 
                             metric: str = 'reward') -> go.Figure:
        """Create experiment comparison plot."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (exp_name, df) in enumerate(experiment_data.items()):
            if metric not in df.columns:
                continue
            
            color = colors[i % len(colors)]
            
            # Main line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[metric],
                    mode='lines',
                    name=exp_name,
                    line=dict(color=color, width=2),
                    opacity=0.7
                )
            )
            
            # Smoothed line
            if len(df) > self.config.smoothing_window:
                smoothed = savgol_filter(df[metric], self.config.smoothing_window, 3)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=smoothed,
                        mode='lines',
                        name=f'{exp_name} (smoothed)',
                        line=dict(color=color, width=3, dash='dash'),
                        showlegend=False
                    )
                )
            
            # Confidence interval
            window_size = min(100, len(df) // 10)
            if window_size > 5:
                rolling_mean = df[metric].rolling(window_size).mean()
                rolling_std = df[metric].rolling(window_size).std()
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rolling_mean + rolling_std,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rolling_mean - rolling_std,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
        
        fig.update_layout(
            title=f'Experiment Comparison: {metric}',
            xaxis_title='Episode/Step',
            yaxis_title=metric,
            height=self.config.figure_height,
            width=self.config.figure_width,
            template=self.config.plot_style,
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Create detailed performance analysis visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Over Time', 'Performance Distribution',
                          'Moving Statistics', 'Anomaly Detection'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if 'reward' not in data.columns:
            return fig
        
        rewards = data['reward'].dropna()
        
        # Performance over time
        fig.add_trace(
            go.Scatter(x=data.index, y=rewards, mode='lines', 
                      name='Performance', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add trend line
        x_numeric = np.arange(len(rewards))
        z = np.polyfit(x_numeric, rewards, 1)
        trend_line = np.poly1d(z)(x_numeric)
        
        fig.add_trace(
            go.Scatter(x=data.index, y=trend_line, mode='lines',
                      name='Trend', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Performance distribution
        fig.add_trace(
            go.Histogram(x=rewards, nbinsx=50, name='Distribution'),
            row=1, col=2
        )
        
        # Add normal distribution overlay
        mu, sigma = stats.norm.fit(rewards)
        x_norm = np.linspace(rewards.min(), rewards.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(rewards) * (rewards.max() - rewards.min()) / 50
        
        fig.add_trace(
            go.Scatter(x=x_norm, y=y_norm, mode='lines',
                      name='Normal Fit', line=dict(color='red')),
            row=1, col=2
        )
        
        # Moving statistics
        window_size = min(100, len(rewards) // 10)
        if window_size > 5:
            rolling_mean = rewards.rolling(window_size).mean()
            rolling_std = rewards.rolling(window_size).std()
            
            fig.add_trace(
                go.Scatter(x=data.index, y=rolling_mean, mode='lines',
                          name='Rolling Mean', line=dict(color='green')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=data.index, y=rolling_std, mode='lines',
                          name='Rolling Std', line=dict(color='orange')),
                row=2, col=1
            )
        
        # Anomaly detection
        z_scores = np.abs(stats.zscore(rewards))
        anomalies = rewards[z_scores > 3]
        anomaly_indices = data.index[z_scores > 3]
        
        fig.add_trace(
            go.Scatter(x=data.index, y=rewards, mode='lines',
                      name='Performance', line=dict(color='blue'), opacity=0.5),
            row=2, col=2
        )
        
        if len(anomalies) > 0:
            fig.add_trace(
                go.Scatter(x=anomaly_indices, y=anomalies, mode='markers',
                          name='Anomalies', marker=dict(color='red', size=8)),
                row=2, col=2
            )
        
        fig.update_layout(
            height=self.config.figure_height * 1.2,
            width=self.config.figure_width,
            title_text="Performance Analysis",
            template=self.config.plot_style
        )
        
        return fig
    
    def create_3d_analysis(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> go.Figure:
        """Create 3D analysis visualization."""
        fig = go.Figure()
        
        # 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=data[x_col],
                y=data[y_col],
                z=data[z_col],
                mode='markers',
                marker=dict(
                    size=3,
                    color=data.index,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                ),
                text=[f"Episode: {i}" for i in data.index],
                hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z}}<br>%{{text}}"
            )
        )
        
        fig.update_layout(
            title=f'3D Analysis: {x_col} vs {y_col} vs {z_col}',
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            height=self.config.figure_height,
            width=self.config.figure_width,
            template=self.config.plot_style
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str, formats: List[str] = ['html', 'png']):
        """Save figure in multiple formats."""
        filepath = self.config.output_dir / filename
        
        for fmt in formats:
            if fmt == 'html':
                fig.write_html(f"{filepath}.html")
            elif fmt == 'png':
                fig.write_image(f"{filepath}.png", width=self.config.figure_width, 
                               height=self.config.figure_height)
            elif fmt == 'pdf':
                fig.write_image(f"{filepath}.pdf", width=self.config.figure_width, 
                               height=self.config.figure_height)
        
        logger.info(f"Saved figure: {filename}")


class InteractiveDashboard:
    """Interactive dashboard using Dash."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_sources = {}
        self.setup_layout()
        self.setup_callbacks()
        
        logger.info("Initialized InteractiveDashboard")
    
    def add_data_source(self, name: str, data: pd.DataFrame):
        """Add data source to dashboard."""
        self.data_sources[name] = data
        logger.info(f"Added data source: {name}")
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("RL-LLM Analysis Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Data Source"),
                            dcc.Dropdown(
                                id='data-source-dropdown',
                                options=[],
                                value=None,
                                placeholder="Select data source"
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Metric"),
                            dcc.Dropdown(
                                id='metric-dropdown',
                                options=[],
                                value=None,
                                placeholder="Select metric"
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Analysis Type"),
                            dcc.Dropdown(
                                id='analysis-type-dropdown',
                                options=[
                                    {'label': 'Time Series', 'value': 'timeseries'},
                                    {'label': 'Distribution', 'value': 'distribution'},
                                    {'label': 'Correlation', 'value': 'correlation'},
                                    {'label': 'Anomaly Detection', 'value': 'anomaly'}
                                ],
                                value='timeseries'
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Auto Refresh"),
                            dbc.Switch(
                                id="auto-refresh-switch",
                                label="Enable",
                                value=self.config.enable_real_time
                            )
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='main-plot', style={'height': '600px'})
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='secondary-plot-1', style={'height': '400px'})
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id='secondary-plot-2', style={'height': '400px'})
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Statistics Summary"),
                            html.Div(id='stats-summary')
                        ])
                    ])
                ], width=12)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.config.auto_refresh_interval * 1000,
                n_intervals=0,
                disabled=not self.config.enable_real_time
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            Output('data-source-dropdown', 'options'),
            Input('interval-component', 'n_intervals')
        )
        def update_data_sources(n):
            return [{'label': name, 'value': name} for name in self.data_sources.keys()]
        
        @self.app.callback(
            Output('metric-dropdown', 'options'),
            Input('data-source-dropdown', 'value')
        )
        def update_metrics(data_source):
            if not data_source or data_source not in self.data_sources:
                return []
            
            df = self.data_sources[data_source]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return [{'label': col, 'value': col} for col in numeric_cols]
        
        @self.app.callback(
            [Output('main-plot', 'figure'),
             Output('secondary-plot-1', 'figure'),
             Output('secondary-plot-2', 'figure'),
             Output('stats-summary', 'children')],
            [Input('data-source-dropdown', 'value'),
             Input('metric-dropdown', 'value'),
             Input('analysis-type-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_plots(data_source, metric, analysis_type, n):
            if not data_source or not metric or data_source not in self.data_sources:
                empty_fig = go.Figure()
                return empty_fig, empty_fig, empty_fig, "No data selected"
            
            df = self.data_sources[data_source]
            
            # Main plot
            if analysis_type == 'timeseries':
                main_fig = self._create_timeseries_plot(df, metric)
            elif analysis_type == 'distribution':
                main_fig = self._create_distribution_plot(df, metric)
            elif analysis_type == 'correlation':
                main_fig = self._create_correlation_plot(df)
            elif analysis_type == 'anomaly':
                main_fig = self._create_anomaly_plot(df, metric)
            else:
                main_fig = go.Figure()
            
            # Secondary plots
            sec_fig_1 = self._create_rolling_stats_plot(df, metric)
            sec_fig_2 = self._create_histogram_plot(df, metric)
            
            # Statistics summary
            stats_summary = self._create_stats_summary(df, metric)
            
            return main_fig, sec_fig_1, sec_fig_2, stats_summary
    
    def _create_timeseries_plot(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """Create time series plot."""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines',
                name=metric,
                line=dict(color='blue')
            )
        )
        
        # Add smoothed line
        if len(df) > self.config.smoothing_window:
            smoothed = savgol_filter(df[metric], self.config.smoothing_window, 3)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=smoothed,
                    mode='lines',
                    name='Smoothed',
                    line=dict(color='red', dash='dash')
                )
            )
        
        fig.update_layout(
            title=f'{metric} Over Time',
            xaxis_title='Episode/Step',
            yaxis_title=metric,
            template=self.config.plot_style
        )
        
        return fig
    
    def _create_distribution_plot(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """Create distribution plot."""
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=df[metric],
                nbinsx=50,
                name='Distribution',
                opacity=0.7
            )
        )
        
        fig.update_layout(
            title=f'{metric} Distribution',
            xaxis_title=metric,
            yaxis_title='Frequency',
            template=self.config.plot_style
        )
        
        return fig
    
    def _create_correlation_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            )
        )
        
        fig.update_layout(
            title='Correlation Matrix',
            template=self.config.plot_style
        )
        
        return fig
    
    def _create_anomaly_plot(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """Create anomaly detection plot."""
        fig = go.Figure()
        
        values = df[metric].dropna()
        z_scores = np.abs(stats.zscore(values))
        
        # Normal points
        normal_mask = z_scores <= 3
        fig.add_trace(
            go.Scatter(
                x=df.index[normal_mask],
                y=values[normal_mask],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4)
            )
        )
        
        # Anomalies
        anomaly_mask = z_scores > 3
        if anomaly_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.index[anomaly_mask],
                    y=values[anomaly_mask],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=8)
                )
            )
        
        fig.update_layout(
            title=f'{metric} Anomaly Detection',
            xaxis_title='Episode/Step',
            yaxis_title=metric,
            template=self.config.plot_style
        )
        
        return fig
    
    def _create_rolling_stats_plot(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """Create rolling statistics plot."""
        fig = go.Figure()
        
        window_size = min(50, len(df) // 10)
        if window_size > 5:
            rolling_mean = df[metric].rolling(window_size).mean()
            rolling_std = df[metric].rolling(window_size).std()
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rolling_mean,
                    mode='lines',
                    name='Rolling Mean',
                    line=dict(color='green')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rolling_std,
                    mode='lines',
                    name='Rolling Std',
                    line=dict(color='orange')
                )
            )
        
        fig.update_layout(
            title=f'{metric} Rolling Statistics',
            xaxis_title='Episode/Step',
            yaxis_title='Value',
            template=self.config.plot_style
        )
        
        return fig
    
    def _create_histogram_plot(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """Create histogram with statistics."""
        fig = go.Figure()
        
        values = df[metric].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=30,
                name='Histogram',
                opacity=0.7
            )
        )
        
        # Add mean line
        mean_val = values.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}"
        )
        
        fig.update_layout(
            title=f'{metric} Histogram',
            xaxis_title=metric,
            yaxis_title='Frequency',
            template=self.config.plot_style
        )
        
        return fig
    
    def _create_stats_summary(self, df: pd.DataFrame, metric: str) -> html.Div:
        """Create statistics summary."""
        values = df[metric].dropna()
        
        stats_data = {
            'Count': len(values),
            'Mean': values.mean(),
            'Std': values.std(),
            'Min': values.min(),
            'Max': values.max(),
            'Median': values.median(),
            'Skewness': stats.skew(values),
            'Kurtosis': stats.kurtosis(values)
        }
        
        stats_cards = []
        for stat_name, stat_value in stats_data.items():
            stats_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(stat_name, className="card-title"),
                            html.H4(f"{stat_value:.4f}" if isinstance(stat_value, float) else str(stat_value))
                        ])
                    ])
                ], width=3)
            )
        
        return dbc.Row(stats_cards)
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        self.app.run_server(
            host=self.config.dashboard_host,
            port=self.config.dashboard_port,
            debug=debug
        )


def create_analysis_system(config: Dict[str, Any]) -> Tuple[DataAnalyzer, AdvancedVisualizer, InteractiveDashboard]:
    """
    Factory function to create complete analysis system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (DataAnalyzer, AdvancedVisualizer, InteractiveDashboard)
    """
    analysis_config = AnalysisConfig(**config)
    
    analyzer = DataAnalyzer(analysis_config)
    visualizer = AdvancedVisualizer(analysis_config)
    dashboard = InteractiveDashboard(analysis_config)
    
    return analyzer, visualizer, dashboard

