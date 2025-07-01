"""
Data Collector for RL-LLM Training Metrics

This module provides comprehensive data collection capabilities for RL training,
including metrics tracking, episode data storage, and analysis tools.
"""

import time
import numpy as np
import pandas as pd
import pickle
import json
import h5py
import sqlite3
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import threading
from queue import Queue, Empty
import psutil
import torch
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
    """Data structure for episode information."""
    episode_id: str
    episode_number: int
    agent_name: str
    environment_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    observations: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    timestamp: datetime
    episode: int
    step: int
    metrics: Dict[str, float]
    agent_name: str
    environment_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataBuffer:
    """Circular buffer for efficient data storage."""
    
    def __init__(self, capacity: int):
        """
        Initialize data buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.RLock()
    
    def add(self, item: Any):
        """Add item to buffer."""
        with self.lock:
            self.buffer.append(item)
    
    def get_all(self) -> List[Any]:
        """Get all items from buffer."""
        with self.lock:
            return list(self.buffer)
    
    def get_recent(self, n: int) -> List[Any]:
        """Get n most recent items."""
        with self.lock:
            return list(self.buffer)[-n:] if n <= len(self.buffer) else list(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get buffer size."""
        with self.lock:
            return len(self.buffer)


class DatabaseManager:
    """Database manager for persistent data storage."""
    
    def __init__(self, db_path: Path):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized database at {db_path}")
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Episodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    episode_number INTEGER,
                    agent_name TEXT,
                    environment_name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    total_reward REAL,
                    episode_length INTEGER,
                    success BOOLEAN,
                    metrics TEXT,
                    metadata TEXT
                )
            ''')
            
            # Training metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    episode INTEGER,
                    step INTEGER,
                    agent_name TEXT,
                    environment_name TEXT,
                    metrics TEXT,
                    metadata TEXT
                )
            ''')
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_mb REAL,
                    gpu_memory_used_mb REAL,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
    
    def save_episode(self, episode_data: EpisodeData):
        """Save episode data to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO episodes 
                (episode_id, episode_number, agent_name, environment_name, 
                 start_time, end_time, total_reward, episode_length, success, 
                 metrics, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                episode_data.episode_id,
                episode_data.episode_number,
                episode_data.agent_name,
                episode_data.environment_name,
                episode_data.start_time.isoformat(),
                episode_data.end_time.isoformat() if episode_data.end_time else None,
                episode_data.total_reward,
                episode_data.episode_length,
                episode_data.success,
                json.dumps(episode_data.metrics),
                json.dumps(episode_data.metadata)
            ))
            
            conn.commit()
    
    def save_training_metrics(self, metrics: TrainingMetrics):
        """Save training metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_metrics 
                (timestamp, episode, step, agent_name, environment_name, metrics, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.episode,
                metrics.step,
                metrics.agent_name,
                metrics.environment_name,
                json.dumps(metrics.metrics),
                json.dumps(metrics.metadata)
            ))
            
            conn.commit()
    
    def get_episodes(self, agent_name: Optional[str] = None, 
                    environment_name: Optional[str] = None,
                    limit: Optional[int] = None) -> pd.DataFrame:
        """Get episodes from database."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM episodes"
            params = []
            
            conditions = []
            if agent_name:
                conditions.append("agent_name = ?")
                params.append(agent_name)
            
            if environment_name:
                conditions.append("environment_name = ?")
                params.append(environment_name)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY episode_number DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Parse JSON columns
            if not df.empty:
                df['metrics'] = df['metrics'].apply(lambda x: json.loads(x) if x else {})
                df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
            
            return df
    
    def get_training_metrics(self, agent_name: Optional[str] = None,
                           environment_name: Optional[str] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Get training metrics from database."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM training_metrics"
            params = []
            
            conditions = []
            if agent_name:
                conditions.append("agent_name = ?")
                params.append(agent_name)
            
            if environment_name:
                conditions.append("environment_name = ?")
                params.append(environment_name)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Parse JSON columns
            if not df.empty:
                df['metrics'] = df['metrics'].apply(lambda x: json.loads(x) if x else {})
                df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
            
            return df


class DataCollector:
    """Main data collector for RL training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data collector.
        
        Args:
            config: Data collector configuration
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', './data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage configuration
        self.buffer_size = config.get('buffer_size', 10000)
        self.save_frequency = config.get('save_frequency', 100)
        self.enable_database = config.get('enable_database', True)
        self.enable_hdf5 = config.get('enable_hdf5', False)
        self.collect_system_metrics = config.get('collect_system_metrics', True)
        
        # Data buffers
        self.episode_buffer = DataBuffer(self.buffer_size)
        self.metrics_buffer = DataBuffer(self.buffer_size * 10)  # More metrics than episodes
        
        # Database
        if self.enable_database:
            self.db_manager = DatabaseManager(self.output_dir / 'training_data.db')
        else:
            self.db_manager = None
        
        # HDF5 storage
        if self.enable_hdf5:
            self.hdf5_path = self.output_dir / 'training_data.h5'
        else:
            self.hdf5_path = None
        
        # Current episode tracking
        self.current_episode: Optional[EpisodeData] = None
        self.episode_counter = 0
        self.step_counter = 0
        
        # System monitoring
        if self.collect_system_metrics:
            self.system_monitor = SystemMonitor(self)
            self.system_monitor.start()
        else:
            self.system_monitor = None
        
        # Callbacks
        self.episode_callbacks: List[Callable[[EpisodeData], None]] = []
        self.metrics_callbacks: List[Callable[[TrainingMetrics], None]] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized DataCollector")
    
    def start_episode(self, agent_name: str, environment_name: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start new episode data collection.
        
        Args:
            agent_name: Name of the agent
            environment_name: Name of the environment
            metadata: Optional episode metadata
            
        Returns:
            Episode ID
        """
        with self.lock:
            # Finish previous episode if exists
            if self.current_episode:
                self.end_episode()
            
            # Create new episode
            episode_id = str(uuid.uuid4())
            self.current_episode = EpisodeData(
                episode_id=episode_id,
                episode_number=self.episode_counter,
                agent_name=agent_name,
                environment_name=environment_name,
                start_time=datetime.now(),
                metadata=metadata or {}
            )
            
            self.episode_counter += 1
            
            logger.debug(f"Started episode {self.episode_counter}: {episode_id}")
            return episode_id
    
    def record_step(self, observation: Any, action: Any, reward: float, 
                   done: bool, info: Dict[str, Any]):
        """
        Record a single step in the current episode.
        
        Args:
            observation: Environment observation
            action: Agent action
            reward: Step reward
            done: Episode termination flag
            info: Additional step information
        """
        if not self.current_episode:
            logger.warning("No active episode for step recording")
            return
        
        with self.lock:
            # Store step data (optionally compress observations)
            if self.config.get('store_observations', True):
                self.current_episode.observations.append(observation)
            
            self.current_episode.actions.append(action)
            self.current_episode.rewards.append(reward)
            self.current_episode.dones.append(done)
            self.current_episode.infos.append(info)
            
            # Update episode metrics
            self.current_episode.total_reward += reward
            self.current_episode.episode_length += 1
            
            self.step_counter += 1
    
    def end_episode(self, success: Optional[bool] = None, 
                   metrics: Optional[Dict[str, Any]] = None):
        """
        End current episode data collection.
        
        Args:
            success: Episode success flag
            metrics: Additional episode metrics
        """
        if not self.current_episode:
            logger.warning("No active episode to end")
            return
        
        with self.lock:
            # Finalize episode data
            self.current_episode.end_time = datetime.now()
            
            if success is not None:
                self.current_episode.success = success
            elif self.current_episode.infos:
                # Try to infer success from last info
                last_info = self.current_episode.infos[-1]
                self.current_episode.success = last_info.get('success', False)
            
            if metrics:
                self.current_episode.metrics.update(metrics)
            
            # Add to buffer
            self.episode_buffer.add(self.current_episode)
            
            # Save to persistent storage
            if self.db_manager:
                self.db_manager.save_episode(self.current_episode)
            
            # Execute callbacks
            for callback in self.episode_callbacks:
                try:
                    callback(self.current_episode)
                except Exception as e:
                    logger.error(f"Episode callback failed: {e}")
            
            # Periodic save
            if self.episode_counter % self.save_frequency == 0:
                self.save_data()
            
            logger.debug(f"Ended episode {self.current_episode.episode_number}")
            self.current_episode = None
    
    def record_metrics(self, metrics: Dict[str, float], 
                      agent_name: str, environment_name: str,
                      episode: Optional[int] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Record training metrics.
        
        Args:
            metrics: Dictionary of metric values
            agent_name: Name of the agent
            environment_name: Name of the environment
            episode: Episode number (optional)
            metadata: Additional metadata
        """
        training_metrics = TrainingMetrics(
            timestamp=datetime.now(),
            episode=episode or self.episode_counter,
            step=self.step_counter,
            metrics=metrics,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.metrics_buffer.add(training_metrics)
        
        # Save to database
        if self.db_manager:
            self.db_manager.save_training_metrics(training_metrics)
        
        # Execute callbacks
        for callback in self.metrics_callbacks:
            try:
                callback(training_metrics)
            except Exception as e:
                logger.error(f"Metrics callback failed: {e}")
    
    def add_episode_callback(self, callback: Callable[[EpisodeData], None]):
        """Add episode completion callback."""
        self.episode_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[TrainingMetrics], None]):
        """Add metrics recording callback."""
        self.metrics_callbacks.append(callback)
    
    def get_recent_episodes(self, n: int = 100) -> List[EpisodeData]:
        """Get recent episodes."""
        return self.episode_buffer.get_recent(n)
    
    def get_recent_metrics(self, n: int = 1000) -> List[TrainingMetrics]:
        """Get recent metrics."""
        return self.metrics_buffer.get_recent(n)
    
    def get_episode_statistics(self, n: int = 100) -> Dict[str, float]:
        """Get episode statistics."""
        recent_episodes = self.get_recent_episodes(n)
        
        if not recent_episodes:
            return {}
        
        rewards = [ep.total_reward for ep in recent_episodes]
        lengths = [ep.episode_length for ep in recent_episodes]
        success_rate = np.mean([ep.success for ep in recent_episodes])
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': success_rate,
            'num_episodes': len(recent_episodes)
        }
    
    def save_data(self):
        """Save buffered data to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save episodes
        episodes = self.episode_buffer.get_all()
        if episodes:
            episodes_file = self.output_dir / f'episodes_{timestamp}.pkl'
            with open(episodes_file, 'wb') as f:
                pickle.dump(episodes, f)
            
            logger.info(f"Saved {len(episodes)} episodes to {episodes_file}")
        
        # Save metrics
        metrics = self.metrics_buffer.get_all()
        if metrics:
            metrics_file = self.output_dir / f'metrics_{timestamp}.pkl'
            with open(metrics_file, 'wb') as f:
                pickle.dump(metrics, f)
            
            logger.info(f"Saved {len(metrics)} metrics to {metrics_file}")
        
        # Save to HDF5 if enabled
        if self.enable_hdf5:
            self._save_to_hdf5(episodes, metrics)
    
    def _save_to_hdf5(self, episodes: List[EpisodeData], metrics: List[TrainingMetrics]):
        """Save data to HDF5 format."""
        try:
            with h5py.File(self.hdf5_path, 'a') as f:
                # Save episodes
                if episodes:
                    episode_group = f.require_group('episodes')
                    
                    for i, episode in enumerate(episodes):
                        ep_group = episode_group.require_group(f'episode_{episode.episode_number}')
                        
                        # Save episode metadata
                        ep_group.attrs['episode_id'] = episode.episode_id
                        ep_group.attrs['total_reward'] = episode.total_reward
                        ep_group.attrs['episode_length'] = episode.episode_length
                        ep_group.attrs['success'] = episode.success
                        
                        # Save trajectories
                        if episode.actions:
                            ep_group.create_dataset('actions', data=np.array(episode.actions))
                        if episode.rewards:
                            ep_group.create_dataset('rewards', data=np.array(episode.rewards))
                
                # Save metrics
                if metrics:
                    metrics_group = f.require_group('metrics')
                    
                    # Aggregate metrics by name
                    metric_data = defaultdict(list)
                    timestamps = []
                    
                    for metric in metrics:
                        timestamps.append(metric.timestamp.timestamp())
                        for name, value in metric.metrics.items():
                            metric_data[name].append(value)
                    
                    # Save aggregated data
                    metrics_group.create_dataset('timestamps', data=np.array(timestamps))
                    for name, values in metric_data.items():
                        metrics_group.create_dataset(name, data=np.array(values))
            
            logger.info(f"Saved data to HDF5: {self.hdf5_path}")
            
        except Exception as e:
            logger.error(f"Failed to save HDF5 data: {e}")
    
    def export_to_csv(self, output_path: Optional[Path] = None):
        """Export data to CSV files."""
        if output_path is None:
            output_path = self.output_dir / 'csv_export'
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export episodes
        if self.db_manager:
            episodes_df = self.db_manager.get_episodes()
            if not episodes_df.empty:
                episodes_df.to_csv(output_path / 'episodes.csv', index=False)
                logger.info(f"Exported episodes to {output_path / 'episodes.csv'}")
            
            # Export metrics
            metrics_df = self.db_manager.get_training_metrics()
            if not metrics_df.empty:
                metrics_df.to_csv(output_path / 'training_metrics.csv', index=False)
                logger.info(f"Exported metrics to {output_path / 'training_metrics.csv'}")
    
    def cleanup(self):
        """Cleanup data collector resources."""
        logger.info("Cleaning up DataCollector")
        
        # End current episode
        if self.current_episode:
            self.end_episode()
        
        # Save remaining data
        self.save_data()
        
        # Stop system monitor
        if self.system_monitor:
            self.system_monitor.stop()
        
        # Clear buffers
        self.episode_buffer.clear()
        self.metrics_buffer.clear()


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, data_collector: DataCollector):
        """
        Initialize system monitor.
        
        Args:
            data_collector: Parent data collector
        """
        self.data_collector = data_collector
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 30.0  # seconds
    
    def start(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started system monitoring")
    
    def stop(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped system monitoring")
    
    def _monitor_loop(self):
        """System monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                process = psutil.Process()
                
                metrics = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': process.memory_info().rss / 1024 / 1024,
                    'disk_usage_percent': psutil.disk_usage('/').percent
                }
                
                # GPU metrics (if available)
                if torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        metrics['gpu_memory_used_mb'] = gpu_memory
                    except Exception:
                        pass
                
                # Record metrics
                self.data_collector.record_metrics(
                    metrics=metrics,
                    agent_name='system',
                    environment_name='system',
                    metadata={'type': 'system_metrics'}
                )
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(self.monitor_interval)


def create_data_collector(config: Dict[str, Any]) -> DataCollector:
    """
    Factory function to create data collector.
    
    Args:
        config: Data collector configuration
        
    Returns:
        DataCollector instance
    """
    return DataCollector(config)

