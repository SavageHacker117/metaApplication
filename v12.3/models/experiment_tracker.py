"""
Experiment Tracker for RL-LLM System

This module provides comprehensive experiment tracking capabilities including
experiment management, hyperparameter optimization, result comparison, and
reproducibility features.
"""

import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import uuid
import hashlib
import sqlite3
from collections import defaultdict
import threading
import shutil
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_id: str
    name: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    status: str = "running"  # 'running', 'completed', 'failed', 'paused'
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_experiment_id: Optional[str] = None
    git_commit: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Experiment result data."""
    experiment_id: str
    timestamp: datetime
    episode: int
    step: int
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentArtifact:
    """Experiment artifact information."""
    artifact_id: str
    experiment_id: str
    name: str
    artifact_type: str  # 'model', 'plot', 'data', 'log', 'config'
    file_path: Path
    created_at: datetime
    size_mb: float
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """Main experiment tracking system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config
        self.experiments_dir = Path(config.get('experiments_dir', './experiments'))
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for experiment metadata
        self.db_path = self.experiments_dir / 'experiments.db'
        self._init_database()
        
        # Current experiment tracking
        self.current_experiment: Optional[ExperimentConfig] = None
        self.current_experiment_dir: Optional[Path] = None
        
        # Result buffers
        self.result_buffer: List[ExperimentResult] = []
        self.buffer_size = config.get('buffer_size', 1000)
        self.save_frequency = config.get('save_frequency', 100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Auto-save timer
        self.auto_save_interval = config.get('auto_save_interval', 300)  # 5 minutes
        self.last_save_time = datetime.now()
        
        logger.info("Initialized ExperimentTracker")
    
    def _init_database(self):
        """Initialize experiment database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT DEFAULT 'running',
                    tags TEXT,
                    hyperparameters TEXT,
                    environment_config TEXT,
                    agent_config TEXT,
                    training_config TEXT,
                    metadata TEXT,
                    parent_experiment_id TEXT,
                    git_commit TEXT,
                    python_version TEXT,
                    dependencies TEXT
                )
            ''')
            
            # Experiment results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    timestamp TEXT,
                    episode INTEGER,
                    step INTEGER,
                    metrics TEXT,
                    metadata TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            # Experiment artifacts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT,
                    artifact_type TEXT,
                    file_path TEXT,
                    created_at TEXT,
                    size_mb REAL,
                    checksum TEXT,
                    metadata TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            conn.commit()
    
    def create_experiment(self, name: str, description: str = "",
                         hyperparameters: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None,
                         parent_experiment_id: Optional[str] = None) -> str:
        """
        Create new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            hyperparameters: Hyperparameter configuration
            tags: Experiment tags
            parent_experiment_id: Parent experiment ID for experiment chains
            
        Returns:
            Experiment ID
        """
        with self.lock:
            experiment_id = str(uuid.uuid4())
            
            # Create experiment directory
            experiment_dir = self.experiments_dir / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (experiment_dir / 'artifacts').mkdir(exist_ok=True)
            (experiment_dir / 'checkpoints').mkdir(exist_ok=True)
            (experiment_dir / 'logs').mkdir(exist_ok=True)
            (experiment_dir / 'plots').mkdir(exist_ok=True)
            
            # Get system information
            git_commit = self._get_git_commit()
            python_version = self._get_python_version()
            dependencies = self._get_dependencies()
            
            # Create experiment config
            experiment_config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                author=os.getenv('USER', 'unknown'),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=tags or [],
                hyperparameters=hyperparameters or {},
                parent_experiment_id=parent_experiment_id,
                git_commit=git_commit,
                python_version=python_version,
                dependencies=dependencies
            )
            
            # Save to database
            self._save_experiment_config(experiment_config)
            
            # Save config file
            config_file = experiment_dir / 'experiment_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(asdict(experiment_config), f, default_flow_style=False)
            
            self.current_experiment = experiment_config
            self.current_experiment_dir = experiment_dir
            
            logger.info(f"Created experiment: {name} ({experiment_id})")
            return experiment_id
    
    def set_current_experiment(self, experiment_id: str):
        """Set current active experiment."""
        with self.lock:
            experiment_config = self.get_experiment(experiment_id)
            if not experiment_config:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            self.current_experiment = experiment_config
            self.current_experiment_dir = self.experiments_dir / experiment_id
            
            logger.info(f"Set current experiment to {experiment_id}")
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters for current experiment."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        with self.lock:
            self.current_experiment.hyperparameters.update(hyperparameters)
            self.current_experiment.updated_at = datetime.now()
            
            # Update database
            self._save_experiment_config(self.current_experiment)
            
            logger.debug(f"Logged hyperparameters: {list(hyperparameters.keys())}")
    
    def log_config(self, config_type: str, config: Dict[str, Any]):
        """
        Log configuration for current experiment.
        
        Args:
            config_type: Type of config ('environment', 'agent', 'training')
            config: Configuration dictionary
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        with self.lock:
            if config_type == 'environment':
                self.current_experiment.environment_config.update(config)
            elif config_type == 'agent':
                self.current_experiment.agent_config.update(config)
            elif config_type == 'training':
                self.current_experiment.training_config.update(config)
            else:
                self.current_experiment.metadata[f"{config_type}_config"] = config
            
            self.current_experiment.updated_at = datetime.now()
            
            # Update database
            self._save_experiment_config(self.current_experiment)
            
            logger.debug(f"Logged {config_type} config")
    
    def log_metrics(self, metrics: Dict[str, float], episode: int = 0, 
                   step: int = 0, metadata: Optional[Dict[str, Any]] = None):
        """
        Log metrics for current experiment.
        
        Args:
            metrics: Dictionary of metric values
            episode: Episode number
            step: Step number
            metadata: Additional metadata
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        result = ExperimentResult(
            experiment_id=self.current_experiment.experiment_id,
            timestamp=datetime.now(),
            episode=episode,
            step=step,
            metrics=metrics,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.result_buffer.append(result)
        
        # Auto-save if buffer is full or time interval passed
        if (len(self.result_buffer) >= self.save_frequency or
            datetime.now() - self.last_save_time > timedelta(seconds=self.auto_save_interval)):
            self._save_results_buffer()
        
        logger.debug(f"Logged metrics for episode {episode}, step {step}")
    
    def log_artifact(self, name: str, file_path: Union[str, Path], 
                    artifact_type: str = "data",
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Log artifact for current experiment.
        
        Args:
            name: Artifact name
            file_path: Path to artifact file
            artifact_type: Type of artifact
            metadata: Additional metadata
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        
        # Copy artifact to experiment directory
        artifacts_dir = self.current_experiment_dir / 'artifacts'
        artifact_file = artifacts_dir / file_path.name
        shutil.copy2(file_path, artifact_file)
        
        # Create artifact record
        artifact = ExperimentArtifact(
            artifact_id=str(uuid.uuid4()),
            experiment_id=self.current_experiment.experiment_id,
            name=name,
            artifact_type=artifact_type,
            file_path=artifact_file,
            created_at=datetime.now(),
            size_mb=artifact_file.stat().st_size / (1024 * 1024),
            checksum=self._calculate_checksum(artifact_file),
            metadata=metadata or {}
        )
        
        # Save to database
        self._save_artifact(artifact)
        
        logger.info(f"Logged artifact: {name} ({artifact_type})")
    
    def log_model(self, model: Any, name: str, episode: int = 0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Log model checkpoint for current experiment.
        
        Args:
            model: Model object
            name: Model name
            episode: Episode number
            metadata: Additional metadata
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        checkpoints_dir = self.current_experiment_dir / 'checkpoints'
        model_file = checkpoints_dir / f"{name}_episode_{episode}.pth"
        
        # Save model
        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), model_file)
        else:
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Log as artifact
        self.log_artifact(
            name=f"{name}_episode_{episode}",
            file_path=model_file,
            artifact_type="model",
            metadata=metadata
        )
        
        logger.info(f"Logged model checkpoint: {name} at episode {episode}")
    
    def finish_experiment(self, status: str = "completed", 
                         final_metrics: Optional[Dict[str, float]] = None):
        """
        Finish current experiment.
        
        Args:
            status: Final experiment status
            final_metrics: Final performance metrics
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        with self.lock:
            # Save any remaining results
            self._save_results_buffer()
            
            # Update experiment status
            self.current_experiment.status = status
            self.current_experiment.updated_at = datetime.now()
            
            if final_metrics:
                self.current_experiment.metadata['final_metrics'] = final_metrics
            
            # Save to database
            self._save_experiment_config(self.current_experiment)
            
            # Generate experiment summary
            self._generate_experiment_summary()
            
            logger.info(f"Finished experiment {self.current_experiment.experiment_id} with status: {status}")
            
            self.current_experiment = None
            self.current_experiment_dir = None
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM experiments WHERE experiment_id = ?', (experiment_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to ExperimentConfig
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            # Parse JSON fields
            json_fields = ['tags', 'hyperparameters', 'environment_config', 
                          'agent_config', 'training_config', 'metadata', 'dependencies']
            for field in json_fields:
                if data[field]:
                    data[field] = json.loads(data[field])
                else:
                    data[field] = {} if field != 'tags' else []
            
            # Parse datetime fields
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
            return ExperimentConfig(**data)
    
    def list_experiments(self, status: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        limit: Optional[int] = None) -> List[ExperimentConfig]:
        """List experiments with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM experiments'
            params = []
            conditions = []
            
            if status:
                conditions.append('status = ?')
                params.append(status)
            
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY created_at DESC'
            
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            experiments = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                data = dict(zip(columns, row))
                
                # Parse JSON fields
                json_fields = ['tags', 'hyperparameters', 'environment_config', 
                              'agent_config', 'training_config', 'metadata', 'dependencies']
                for field in json_fields:
                    if data[field]:
                        data[field] = json.loads(data[field])
                    else:
                        data[field] = {} if field != 'tags' else []
                
                # Parse datetime fields
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                
                experiment = ExperimentConfig(**data)
                
                # Filter by tags if specified
                if tags and not any(tag in experiment.tags for tag in tags):
                    continue
                
                experiments.append(experiment)
            
            return experiments
    
    def get_experiment_results(self, experiment_id: str,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Get experiment results as DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT timestamp, episode, step, metrics, metadata
                FROM experiment_results
                WHERE experiment_id = ?
                ORDER BY timestamp
            '''
            params = [experiment_id]
            
            if limit:
                query += f' LIMIT {limit}'
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                # Parse JSON columns
                df['metrics'] = df['metrics'].apply(json.loads)
                df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
                
                # Expand metrics into separate columns
                metrics_df = pd.json_normalize(df['metrics'])
                df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
            
            return df
    
    def compare_experiments(self, experiment_ids: List[str],
                          metric: str = "mean_reward") -> Dict[str, Any]:
        """Compare experiments by metric."""
        comparison = {}
        
        for exp_id in experiment_ids:
            results_df = self.get_experiment_results(exp_id)
            
            if not results_df.empty and metric in results_df.columns:
                values = results_df[metric].dropna()
                
                comparison[exp_id] = {
                    'final_value': values.iloc[-1] if len(values) > 0 else None,
                    'max_value': values.max(),
                    'mean_value': values.mean(),
                    'std_value': values.std(),
                    'num_points': len(values)
                }
            else:
                comparison[exp_id] = {
                    'final_value': None,
                    'max_value': None,
                    'mean_value': None,
                    'std_value': None,
                    'num_points': 0
                }
        
        return comparison
    
    def plot_experiment_comparison(self, experiment_ids: List[str],
                                 metric: str = "mean_reward",
                                 save_path: Optional[Path] = None):
        """Plot experiment comparison."""
        plt.figure(figsize=(12, 8))
        
        for exp_id in experiment_ids:
            results_df = self.get_experiment_results(exp_id)
            
            if not results_df.empty and metric in results_df.columns:
                experiment = self.get_experiment(exp_id)
                label = experiment.name if experiment else exp_id[:8]
                
                plt.plot(results_df['episode'], results_df[metric], 
                        label=label, alpha=0.8)
        
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.title(f'Experiment Comparison: {metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _save_experiment_config(self, config: ExperimentConfig):
        """Save experiment configuration to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO experiments
                (experiment_id, name, description, author, created_at, updated_at,
                 status, tags, hyperparameters, environment_config, agent_config,
                 training_config, metadata, parent_experiment_id, git_commit,
                 python_version, dependencies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.experiment_id, config.name, config.description, config.author,
                config.created_at.isoformat(), config.updated_at.isoformat(),
                config.status, json.dumps(config.tags), json.dumps(config.hyperparameters),
                json.dumps(config.environment_config), json.dumps(config.agent_config),
                json.dumps(config.training_config), json.dumps(config.metadata),
                config.parent_experiment_id, config.git_commit, config.python_version,
                json.dumps(config.dependencies)
            ))
            
            conn.commit()
    
    def _save_results_buffer(self):
        """Save results buffer to database."""
        if not self.result_buffer:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in self.result_buffer:
                cursor.execute('''
                    INSERT INTO experiment_results
                    (experiment_id, timestamp, episode, step, metrics, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.experiment_id, result.timestamp.isoformat(),
                    result.episode, result.step, json.dumps(result.metrics),
                    json.dumps(result.metadata)
                ))
            
            conn.commit()
        
        logger.debug(f"Saved {len(self.result_buffer)} results to database")
        self.result_buffer.clear()
        self.last_save_time = datetime.now()
    
    def _save_artifact(self, artifact: ExperimentArtifact):
        """Save artifact to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO experiment_artifacts
                (artifact_id, experiment_id, name, artifact_type, file_path,
                 created_at, size_mb, checksum, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                artifact.artifact_id, artifact.experiment_id, artifact.name,
                artifact.artifact_type, str(artifact.file_path),
                artifact.created_at.isoformat(), artifact.size_mb,
                artifact.checksum, json.dumps(artifact.metadata)
            ))
            
            conn.commit()
    
    def _generate_experiment_summary(self):
        """Generate experiment summary report."""
        if not self.current_experiment:
            return
        
        summary_file = self.current_experiment_dir / 'experiment_summary.md'
        
        # Get experiment results
        results_df = self.get_experiment_results(self.current_experiment.experiment_id)
        
        with open(summary_file, 'w') as f:
            f.write(f"# Experiment Summary: {self.current_experiment.name}\n\n")
            f.write(f"**Experiment ID:** {self.current_experiment.experiment_id}\n")
            f.write(f"**Author:** {self.current_experiment.author}\n")
            f.write(f"**Created:** {self.current_experiment.created_at}\n")
            f.write(f"**Status:** {self.current_experiment.status}\n")
            f.write(f"**Description:** {self.current_experiment.description}\n\n")
            
            if self.current_experiment.tags:
                f.write(f"**Tags:** {', '.join(self.current_experiment.tags)}\n\n")
            
            # Hyperparameters
            if self.current_experiment.hyperparameters:
                f.write("## Hyperparameters\n\n")
                for key, value in self.current_experiment.hyperparameters.items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            # Results summary
            if not results_df.empty:
                f.write("## Results Summary\n\n")
                numeric_columns = results_df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    if col not in ['episode', 'step']:
                        final_value = results_df[col].iloc[-1] if len(results_df) > 0 else None
                        max_value = results_df[col].max()
                        mean_value = results_df[col].mean()
                        
                        f.write(f"- **{col}:**\n")
                        f.write(f"  - Final: {final_value:.4f}\n")
                        f.write(f"  - Maximum: {max_value:.4f}\n")
                        f.write(f"  - Mean: {mean_value:.4f}\n")
        
        logger.info(f"Generated experiment summary: {summary_file}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get package dependencies."""
        try:
            import pkg_resources
            dependencies = {}
            for dist in pkg_resources.working_set:
                dependencies[dist.project_name] = dist.version
            return dependencies
        except Exception:
            return {}
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


def create_experiment_tracker(config: Dict[str, Any]) -> ExperimentTracker:
    """
    Factory function to create experiment tracker.
    
    Args:
        config: Tracker configuration
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(config)

