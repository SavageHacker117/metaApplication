"""
Model Registry for RL-LLM System

This module provides comprehensive model management capabilities including
model versioning, metadata tracking, performance comparison, and deployment
preparation.
"""

import os
import json
import pickle
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Type
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import shutil
import sqlite3
from abc import ABC, abstractmethod
import threading
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata information."""
    model_id: str
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    model_type: str  # 'agent', 'environment', 'policy', etc.
    architecture: str
    framework: str  # 'pytorch', 'tensorflow', etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    file_size_mb: float = 0.0
    checksum: str = ""
    parent_model_id: Optional[str] = None
    status: str = "active"  # 'active', 'archived', 'deprecated'


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_id: str
    version_number: str
    created_at: datetime
    file_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_latest: bool = False


class ModelStorage(ABC):
    """Abstract base class for model storage backends."""
    
    @abstractmethod
    def save_model(self, model: Any, model_id: str, version: str) -> Path:
        """Save model to storage."""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str, version: str = "latest") -> Any:
        """Load model from storage."""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete model from storage."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List all model IDs."""
        pass


class LocalFileStorage(ModelStorage):
    """Local file system storage backend."""
    
    def __init__(self, storage_dir: Path):
        """
        Initialize local file storage.
        
        Args:
            storage_dir: Directory for model storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized LocalFileStorage at {storage_dir}")
    
    def save_model(self, model: Any, model_id: str, version: str) -> Path:
        """Save model to local file system."""
        model_dir = self.storage_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file format based on model type
        if hasattr(model, 'state_dict') and callable(getattr(model, 'state_dict')):
            # PyTorch model
            file_path = model_dir / f"{version}.pth"
            torch.save(model.state_dict(), file_path)
        elif hasattr(model, 'save'):
            # Model with save method
            file_path = model_dir / f"{version}.pkl"
            model.save(str(file_path))
        else:
            # Generic pickle save
            file_path = model_dir / f"{version}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Saved model {model_id} version {version} to {file_path}")
        return file_path
    
    def load_model(self, model_id: str, version: str = "latest") -> Any:
        """Load model from local file system."""
        model_dir = self.storage_dir / model_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        
        if version == "latest":
            # Find latest version
            version_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pkl"))
            if not version_files:
                raise FileNotFoundError(f"No versions found for model {model_id}")
            
            # Sort by modification time
            latest_file = max(version_files, key=lambda p: p.stat().st_mtime)
            file_path = latest_file
        else:
            # Specific version
            file_path = model_dir / f"{version}.pth"
            if not file_path.exists():
                file_path = model_dir / f"{version}.pkl"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Model {model_id} version {version} not found")
        
        # Load based on file extension
        if file_path.suffix == '.pth':
            model = torch.load(file_path, map_location='cpu')
        else:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
        
        logger.info(f"Loaded model {model_id} version {version} from {file_path}")
        return model
    
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete model from local file system."""
        model_dir = self.storage_dir / model_id
        
        if not model_dir.exists():
            return False
        
        if version is None:
            # Delete entire model directory
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model {model_id}")
        else:
            # Delete specific version
            for ext in ['.pth', '.pkl']:
                file_path = model_dir / f"{version}{ext}"
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted model {model_id} version {version}")
                    return True
            return False
        
        return True
    
    def list_models(self) -> List[str]:
        """List all model IDs."""
        if not self.storage_dir.exists():
            return []
        
        return [d.name for d in self.storage_dir.iterdir() if d.is_dir()]


class ModelRegistry:
    """Central model registry for managing ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model registry.
        
        Args:
            config: Registry configuration
        """
        self.config = config
        self.registry_dir = Path(config.get('registry_dir', './model_registry'))
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backend
        storage_type = config.get('storage_type', 'local')
        if storage_type == 'local':
            storage_dir = self.registry_dir / 'models'
            self.storage = LocalFileStorage(storage_dir)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        # Initialize metadata database
        self.db_path = self.registry_dir / 'registry.db'
        self._init_database()
        
        # Model cache
        self.model_cache: Dict[str, Any] = {}
        self.cache_size = config.get('cache_size', 10)
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized ModelRegistry")
    
    def _init_database(self):
        """Initialize registry database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    model_type TEXT,
                    architecture TEXT,
                    framework TEXT,
                    parameters TEXT,
                    performance_metrics TEXT,
                    training_config TEXT,
                    tags TEXT,
                    file_size_mb REAL,
                    checksum TEXT,
                    parent_model_id TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Model versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    version_number TEXT,
                    created_at TEXT,
                    file_path TEXT,
                    metadata TEXT,
                    performance_metrics TEXT,
                    is_latest BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    version_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    environment TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
    
    def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """
        Register a new model.
        
        Args:
            model: Model object to register
            metadata: Model metadata
            
        Returns:
            Model ID
        """
        with self.lock:
            # Generate model ID if not provided
            if not metadata.model_id:
                metadata.model_id = self._generate_model_id(metadata.name)
            
            # Save model to storage
            file_path = self.storage.save_model(model, metadata.model_id, metadata.version)
            
            # Calculate file size and checksum
            metadata.file_size_mb = file_path.stat().st_size / (1024 * 1024)
            metadata.checksum = self._calculate_checksum(file_path)
            
            # Save metadata to database
            self._save_model_metadata(metadata)
            
            # Create version entry
            version = ModelVersion(
                version_id=f"{metadata.model_id}_{metadata.version}",
                model_id=metadata.model_id,
                version_number=metadata.version,
                created_at=datetime.now(),
                file_path=file_path,
                is_latest=True
            )
            self._save_model_version(version)
            
            # Update cache
            cache_key = f"{metadata.model_id}_{metadata.version}"
            if len(self.model_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.model_cache))
                del self.model_cache[oldest_key]
            
            self.model_cache[cache_key] = model
            
            logger.info(f"Registered model {metadata.model_id} version {metadata.version}")
            return metadata.model_id
    
    def get_model(self, model_id: str, version: str = "latest") -> Any:
        """
        Get model by ID and version.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Model object
        """
        with self.lock:
            # Check cache first
            cache_key = f"{model_id}_{version}"
            if cache_key in self.model_cache:
                logger.debug(f"Retrieved model {model_id} from cache")
                return self.model_cache[cache_key]
            
            # Load from storage
            model = self.storage.load_model(model_id, version)
            
            # Update cache
            if len(self.model_cache) >= self.cache_size:
                oldest_key = next(iter(self.model_cache))
                del self.model_cache[oldest_key]
            
            self.model_cache[cache_key] = model
            
            return model
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM models WHERE model_id = ?', (model_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to ModelMetadata
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            # Parse JSON fields
            for field in ['parameters', 'performance_metrics', 'training_config', 'tags']:
                if data[field]:
                    data[field] = json.loads(data[field])
                else:
                    data[field] = {} if field != 'tags' else []
            
            # Parse datetime fields
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
            return ModelMetadata(**data)
    
    def list_models(self, model_type: Optional[str] = None, 
                   status: str = "active") -> List[ModelMetadata]:
        """List registered models."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM models WHERE status = ?'
            params = [status]
            
            if model_type:
                query += ' AND model_type = ?'
                params.append(model_type)
            
            query += ' ORDER BY updated_at DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            models = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                data = dict(zip(columns, row))
                
                # Parse JSON fields
                for field in ['parameters', 'performance_metrics', 'training_config', 'tags']:
                    if data[field]:
                        data[field] = json.loads(data[field])
                    else:
                        data[field] = {} if field != 'tags' else []
                
                # Parse datetime fields
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                
                models.append(ModelMetadata(**data))
            
            return models
    
    def update_model_performance(self, model_id: str, version: str,
                               metrics: Dict[str, float], 
                               environment: str = "",
                               metadata: Optional[Dict[str, Any]] = None):
        """Update model performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            version_id = f"{model_id}_{version}"
            timestamp = datetime.now().isoformat()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute('''
                    INSERT INTO model_performance 
                    (model_id, version_id, metric_name, metric_value, timestamp, environment, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id, version_id, metric_name, metric_value,
                    timestamp, environment, json.dumps(metadata or {})
                ))
            
            conn.commit()
        
        logger.info(f"Updated performance metrics for model {model_id} version {version}")
    
    def compare_models(self, model_ids: List[str], 
                      metric: str = "mean_reward") -> Dict[str, Any]:
        """Compare models by performance metric."""
        comparison = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for model_id in model_ids:
                cursor.execute('''
                    SELECT metric_value, timestamp FROM model_performance
                    WHERE model_id = ? AND metric_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (model_id, metric))
                
                values = [row[0] for row in cursor.fetchall()]
                
                if values:
                    comparison[model_id] = {
                        'latest_value': values[0],
                        'mean_value': np.mean(values),
                        'std_value': np.std(values),
                        'num_evaluations': len(values)
                    }
                else:
                    comparison[model_id] = {
                        'latest_value': None,
                        'mean_value': None,
                        'std_value': None,
                        'num_evaluations': 0
                    }
        
        return comparison
    
    def create_model_version(self, model: Any, model_id: str, 
                           version: str, metadata: Optional[Dict[str, Any]] = None):
        """Create new version of existing model."""
        with self.lock:
            # Check if model exists
            if not self.get_model_metadata(model_id):
                raise ValueError(f"Model {model_id} not found")
            
            # Save new version
            file_path = self.storage.save_model(model, model_id, version)
            
            # Create version entry
            version_obj = ModelVersion(
                version_id=f"{model_id}_{version}",
                model_id=model_id,
                version_number=version,
                created_at=datetime.now(),
                file_path=file_path,
                metadata=metadata or {},
                is_latest=True
            )
            
            # Update previous latest version
            self._update_latest_version(model_id, version_obj.version_id)
            
            # Save new version
            self._save_model_version(version_obj)
            
            logger.info(f"Created version {version} for model {model_id}")
    
    def delete_model(self, model_id: str, version: Optional[str] = None):
        """Delete model or specific version."""
        with self.lock:
            # Delete from storage
            self.storage.delete_model(model_id, version)
            
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if version is None:
                    # Delete entire model
                    cursor.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
                    cursor.execute('DELETE FROM model_versions WHERE model_id = ?', (model_id,))
                    cursor.execute('DELETE FROM model_performance WHERE model_id = ?', (model_id,))
                else:
                    # Delete specific version
                    version_id = f"{model_id}_{version}"
                    cursor.execute('DELETE FROM model_versions WHERE version_id = ?', (version_id,))
                    cursor.execute('DELETE FROM model_performance WHERE version_id = ?', (version_id,))
                
                conn.commit()
            
            # Clear from cache
            cache_keys_to_remove = [k for k in self.model_cache.keys() if k.startswith(f"{model_id}_")]
            for key in cache_keys_to_remove:
                del self.model_cache[key]
            
            logger.info(f"Deleted model {model_id}" + (f" version {version}" if version else ""))
    
    def export_model(self, model_id: str, version: str = "latest",
                    export_path: Path = None, format: str = "onnx") -> Path:
        """Export model to different format."""
        model = self.get_model(model_id, version)
        
        if export_path is None:
            export_path = self.registry_dir / 'exports' / f"{model_id}_{version}.{format}"
        
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "onnx" and hasattr(model, 'to_onnx'):
            model.to_onnx(str(export_path))
        elif format == "torchscript" and hasattr(model, 'to_torchscript'):
            torch.jit.save(torch.jit.script(model), str(export_path))
        else:
            # Default pickle export
            with open(export_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Exported model {model_id} to {export_path}")
        return export_path
    
    def _generate_model_id(self, name: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{name}_{timestamp}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_model_metadata(self, metadata: ModelMetadata):
        """Save model metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO models 
                (model_id, name, version, description, author, created_at, updated_at,
                 model_type, architecture, framework, parameters, performance_metrics,
                 training_config, tags, file_size_mb, checksum, parent_model_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_id, metadata.name, metadata.version, metadata.description,
                metadata.author, metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                metadata.model_type, metadata.architecture, metadata.framework,
                json.dumps(metadata.parameters), json.dumps(metadata.performance_metrics),
                json.dumps(metadata.training_config), json.dumps(metadata.tags),
                metadata.file_size_mb, metadata.checksum, metadata.parent_model_id,
                metadata.status
            ))
            
            conn.commit()
    
    def _save_model_version(self, version: ModelVersion):
        """Save model version to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_versions
                (version_id, model_id, version_number, created_at, file_path,
                 metadata, performance_metrics, is_latest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version.version_id, version.model_id, version.version_number,
                version.created_at.isoformat(), str(version.file_path),
                json.dumps(version.metadata), json.dumps(version.performance_metrics),
                version.is_latest
            ))
            
            conn.commit()
    
    def _update_latest_version(self, model_id: str, new_latest_version_id: str):
        """Update latest version flag."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear previous latest
            cursor.execute('''
                UPDATE model_versions SET is_latest = FALSE 
                WHERE model_id = ?
            ''', (model_id,))
            
            # Set new latest
            cursor.execute('''
                UPDATE model_versions SET is_latest = TRUE 
                WHERE version_id = ?
            ''', (new_latest_version_id,))
            
            conn.commit()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total models
            cursor.execute('SELECT COUNT(*) FROM models WHERE status = "active"')
            total_models = cursor.fetchone()[0]
            
            # Models by type
            cursor.execute('''
                SELECT model_type, COUNT(*) FROM models 
                WHERE status = "active" GROUP BY model_type
            ''')
            models_by_type = dict(cursor.fetchall())
            
            # Total storage size
            cursor.execute('SELECT SUM(file_size_mb) FROM models WHERE status = "active"')
            total_size_mb = cursor.fetchone()[0] or 0
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) FROM models 
                WHERE status = "active" AND updated_at > datetime('now', '-7 days')
            ''')
            recent_models = cursor.fetchone()[0]
            
            return {
                'total_models': total_models,
                'models_by_type': models_by_type,
                'total_storage_mb': total_size_mb,
                'recent_models_7days': recent_models,
                'cache_size': len(self.model_cache)
            }


def create_model_registry(config: Dict[str, Any]) -> ModelRegistry:
    """
    Factory function to create model registry.
    
    Args:
        config: Registry configuration
        
    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(config)

