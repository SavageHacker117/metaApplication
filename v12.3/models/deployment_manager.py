"""
Deployment Manager for RL-LLM System

This module provides comprehensive deployment capabilities for trained RL models
including containerization, API serving, monitoring, and scaling management.
"""

import os
import json
import yaml
import docker
import subprocess
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
import time
import requests
from abc import ABC, abstractmethod
import tempfile
import shutil
import sqlite3
from collections import defaultdict
import psutil

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    deployment_id: str
    name: str
    model_id: str
    model_version: str
    deployment_type: str  # 'api', 'batch', 'streaming'
    environment: str  # 'development', 'staging', 'production'
    created_at: datetime
    updated_at: datetime
    status: str = "pending"  # 'pending', 'deploying', 'running', 'stopped', 'failed'
    endpoint_url: Optional[str] = None
    container_id: Optional[str] = None
    port: Optional[int] = None
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """Deployment metrics."""
    deployment_id: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    request_count: int
    response_time_ms: float
    error_rate: float
    throughput: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeploymentBackend(ABC):
    """Abstract base class for deployment backends."""
    
    @abstractmethod
    def deploy(self, config: DeploymentConfig, model_path: Path) -> str:
        """Deploy model and return deployment ID."""
        pass
    
    @abstractmethod
    def stop(self, deployment_id: str) -> bool:
        """Stop deployment."""
        pass
    
    @abstractmethod
    def get_status(self, deployment_id: str) -> str:
        """Get deployment status."""
        pass
    
    @abstractmethod
    def get_logs(self, deployment_id: str, lines: int = 100) -> List[str]:
        """Get deployment logs."""
        pass
    
    @abstractmethod
    def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment."""
        pass


class DockerBackend(DeploymentBackend):
    """Docker-based deployment backend."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Docker backend.
        
        Args:
            config: Backend configuration
        """
        self.config = config
        self.client = docker.from_env()
        self.base_image = config.get('base_image', 'python:3.9-slim')
        self.network_name = config.get('network_name', 'rl-llm-network')
        
        # Ensure network exists
        self._ensure_network()
        
        logger.info("Initialized DockerBackend")
    
    def _ensure_network(self):
        """Ensure Docker network exists."""
        try:
            self.client.networks.get(self.network_name)
        except docker.errors.NotFound:
            self.client.networks.create(self.network_name)
            logger.info(f"Created Docker network: {self.network_name}")
    
    def deploy(self, config: DeploymentConfig, model_path: Path) -> str:
        """Deploy model using Docker."""
        try:
            # Create deployment directory
            deploy_dir = Path(tempfile.mkdtemp(prefix=f"deploy_{config.deployment_id}_"))
            
            # Copy model files
            model_deploy_path = deploy_dir / 'model'
            shutil.copytree(model_path.parent, model_deploy_path)
            
            # Create Dockerfile
            dockerfile_content = self._generate_dockerfile(config)
            dockerfile_path = deploy_dir / 'Dockerfile'
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Create requirements.txt
            requirements_path = deploy_dir / 'requirements.txt'
            with open(requirements_path, 'w') as f:
                f.write(self._generate_requirements())
            
            # Create serving script
            serving_script = self._generate_serving_script(config)
            script_path = deploy_dir / 'serve.py'
            with open(script_path, 'w') as f:
                f.write(serving_script)
            
            # Build Docker image
            image_tag = f"rl-llm-model:{config.deployment_id}"
            logger.info(f"Building Docker image: {image_tag}")
            
            image, build_logs = self.client.images.build(
                path=str(deploy_dir),
                tag=image_tag,
                rm=True
            )
            
            # Run container
            port = config.port or self._find_available_port()
            
            container = self.client.containers.run(
                image_tag,
                detach=True,
                ports={f'{port}/tcp': port},
                network=self.network_name,
                environment=config.environment_vars,
                name=f"rl-llm-{config.deployment_id}",
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Update config
            config.container_id = container.id
            config.port = port
            config.endpoint_url = f"http://localhost:{port}"
            config.status = "running"
            
            # Cleanup temporary directory
            shutil.rmtree(deploy_dir)
            
            logger.info(f"Deployed model {config.model_id} as container {container.id}")
            return container.id
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            config.status = "failed"
            raise
    
    def stop(self, deployment_id: str) -> bool:
        """Stop Docker deployment."""
        try:
            container = self.client.containers.get(f"rl-llm-{deployment_id}")
            container.stop()
            container.remove()
            
            logger.info(f"Stopped deployment {deployment_id}")
            return True
            
        except docker.errors.NotFound:
            logger.warning(f"Container for deployment {deployment_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to stop deployment {deployment_id}: {e}")
            return False
    
    def get_status(self, deployment_id: str) -> str:
        """Get Docker deployment status."""
        try:
            container = self.client.containers.get(f"rl-llm-{deployment_id}")
            return container.status
        except docker.errors.NotFound:
            return "not_found"
        except Exception as e:
            logger.error(f"Failed to get status for {deployment_id}: {e}")
            return "error"
    
    def get_logs(self, deployment_id: str, lines: int = 100) -> List[str]:
        """Get Docker deployment logs."""
        try:
            container = self.client.containers.get(f"rl-llm-{deployment_id}")
            logs = container.logs(tail=lines).decode('utf-8')
            return logs.split('\n')
        except Exception as e:
            logger.error(f"Failed to get logs for {deployment_id}: {e}")
            return []
    
    def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale Docker deployment (restart with new configuration)."""
        # For Docker, we need to stop and restart with multiple containers
        # This is a simplified implementation
        logger.warning("Docker scaling not fully implemented - would need orchestration")
        return False
    
    def _generate_dockerfile(self, config: DeploymentConfig) -> str:
        """Generate Dockerfile for deployment."""
        return f"""
FROM {self.base_image}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and serving code
COPY model/ ./model/
COPY serve.py .

# Expose port
EXPOSE {config.port or 8000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{config.port or 8000}/health || exit 1

# Run serving script
CMD ["python", "serve.py"]
"""
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt for deployment."""
        return """
flask==2.3.3
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
requests>=2.25.0
gunicorn>=20.1.0
"""
    
    def _generate_serving_script(self, config: DeploymentConfig) -> str:
        """Generate serving script for deployment."""
        return f"""
import os
import json
import pickle
import torch
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model
MODEL_PATH = Path('./model')
model = None

def load_model():
    global model
    try:
        # Try to load PyTorch model
        model_files = list(MODEL_PATH.glob('*.pth'))
        if model_files:
            model = torch.load(model_files[0], map_location='cpu')
            logger.info(f"Loaded PyTorch model from {{model_files[0]}}")
        else:
            # Try to load pickle model
            pkl_files = list(MODEL_PATH.glob('*.pkl'))
            if pkl_files:
                with open(pkl_files[0], 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded pickle model from {{pkl_files[0]}}")
            else:
                raise FileNotFoundError("No model files found")
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy', 'model_loaded': model is not None}})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if not data or 'input' not in data:
            return jsonify({{'error': 'Missing input data'}}), 400
        
        # Convert input to appropriate format
        input_data = np.array(data['input'])
        
        # Make prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
        elif hasattr(model, 'act'):
            prediction = model.act(input_data)
        elif callable(model):
            prediction = model(input_data)
        else:
            return jsonify({{'error': 'Model does not support prediction'}}), 500
        
        # Convert prediction to serializable format
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        
        return jsonify({{'prediction': prediction}})
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        return jsonify({{'error': str(e)}}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({{
        'model_id': '{config.model_id}',
        'model_version': '{config.model_version}',
        'deployment_id': '{config.deployment_id}',
        'deployment_type': '{config.deployment_type}'
    }})

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', {config.port or 8000}))
    app.run(host='0.0.0.0', port=port, debug=False)
"""
    
    def _find_available_port(self, start_port: int = 8000) -> int:
        """Find available port starting from start_port."""
        import socket
        
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        
        raise RuntimeError("No available ports found")


class DeploymentManager:
    """Main deployment manager for RL models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize deployment manager.
        
        Args:
            config: Manager configuration
        """
        self.config = config
        self.deployments_dir = Path(config.get('deployments_dir', './deployments'))
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.deployments_dir / 'deployments.db'
        self._init_database()
        
        # Initialize backend
        backend_type = config.get('backend_type', 'docker')
        if backend_type == 'docker':
            self.backend = DockerBackend(config.get('docker_config', {}))
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        # Active deployments
        self.deployments: Dict[str, DeploymentConfig] = {}
        self._load_active_deployments()
        
        # Monitoring
        self.monitoring_enabled = config.get('monitoring_enabled', True)
        self.monitoring_interval = config.get('monitoring_interval', 60)  # seconds
        self.monitoring_thread = None
        
        if self.monitoring_enabled:
            self._start_monitoring()
        
        logger.info("Initialized DeploymentManager")
    
    def _init_database(self):
        """Initialize deployment database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Deployments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    deployment_type TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT DEFAULT 'pending',
                    endpoint_url TEXT,
                    container_id TEXT,
                    port INTEGER,
                    replicas INTEGER DEFAULT 1,
                    resources TEXT,
                    environment_vars TEXT,
                    health_check_url TEXT,
                    metadata TEXT
                )
            ''')
            
            # Deployment metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deployment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT,
                    timestamp TEXT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    request_count INTEGER,
                    response_time_ms REAL,
                    error_rate REAL,
                    throughput REAL,
                    metadata TEXT,
                    FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
                )
            ''')
            
            conn.commit()
    
    def deploy_model(self, model_path: Path, config: Dict[str, Any]) -> str:
        """
        Deploy model to production.
        
        Args:
            model_path: Path to model files
            config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        # Create deployment config
        deployment_id = config.get('deployment_id', f"deploy_{int(time.time())}")
        
        deployment_config = DeploymentConfig(
            deployment_id=deployment_id,
            name=config['name'],
            model_id=config['model_id'],
            model_version=config.get('model_version', 'latest'),
            deployment_type=config.get('deployment_type', 'api'),
            environment=config.get('environment', 'development'),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            port=config.get('port'),
            replicas=config.get('replicas', 1),
            resources=config.get('resources', {}),
            environment_vars=config.get('environment_vars', {}),
            metadata=config.get('metadata', {})
        )
        
        try:
            # Deploy using backend
            deployment_config.status = "deploying"
            self._save_deployment_config(deployment_config)
            
            container_id = self.backend.deploy(deployment_config, model_path)
            
            # Wait for deployment to be ready
            self._wait_for_deployment(deployment_config)
            
            # Store deployment
            self.deployments[deployment_id] = deployment_config
            self._save_deployment_config(deployment_config)
            
            logger.info(f"Successfully deployed model {config['model_id']} as {deployment_id}")
            return deployment_id
            
        except Exception as e:
            deployment_config.status = "failed"
            self._save_deployment_config(deployment_config)
            logger.error(f"Deployment failed: {e}")
            raise
    
    def stop_deployment(self, deployment_id: str) -> bool:
        """Stop deployment."""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        try:
            success = self.backend.stop(deployment_id)
            
            if success:
                self.deployments[deployment_id].status = "stopped"
                self.deployments[deployment_id].updated_at = datetime.now()
                self._save_deployment_config(self.deployments[deployment_id])
                
                logger.info(f"Stopped deployment {deployment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to stop deployment {deployment_id}: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[str]:
        """Get deployment status."""
        if deployment_id not in self.deployments:
            return None
        
        # Get live status from backend
        live_status = self.backend.get_status(deployment_id)
        
        # Update stored status if different
        if live_status != self.deployments[deployment_id].status:
            self.deployments[deployment_id].status = live_status
            self.deployments[deployment_id].updated_at = datetime.now()
            self._save_deployment_config(self.deployments[deployment_id])
        
        return live_status
    
    def list_deployments(self, environment: Optional[str] = None) -> List[DeploymentConfig]:
        """List all deployments."""
        deployments = list(self.deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        
        return deployments
    
    def get_deployment_logs(self, deployment_id: str, lines: int = 100) -> List[str]:
        """Get deployment logs."""
        if deployment_id not in self.deployments:
            return []
        
        return self.backend.get_logs(deployment_id, lines)
    
    def scale_deployment(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment."""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        try:
            success = self.backend.scale(deployment_id, replicas)
            
            if success:
                self.deployments[deployment_id].replicas = replicas
                self.deployments[deployment_id].updated_at = datetime.now()
                self._save_deployment_config(self.deployments[deployment_id])
                
                logger.info(f"Scaled deployment {deployment_id} to {replicas} replicas")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_id}: {e}")
            return False
    
    def test_deployment(self, deployment_id: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test deployment with sample data."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.deployments[deployment_id]
        
        if not deployment.endpoint_url:
            raise ValueError(f"No endpoint URL for deployment {deployment_id}")
        
        try:
            # Test health endpoint
            health_response = requests.get(f"{deployment.endpoint_url}/health", timeout=10)
            health_status = health_response.status_code == 200
            
            # Test prediction endpoint
            predict_response = requests.post(
                f"{deployment.endpoint_url}/predict",
                json=test_data,
                timeout=30
            )
            
            prediction_success = predict_response.status_code == 200
            prediction_result = predict_response.json() if prediction_success else None
            
            return {
                'deployment_id': deployment_id,
                'health_status': health_status,
                'prediction_success': prediction_success,
                'prediction_result': prediction_result,
                'response_time_ms': predict_response.elapsed.total_seconds() * 1000,
                'status_code': predict_response.status_code
            }
            
        except Exception as e:
            logger.error(f"Deployment test failed: {e}")
            return {
                'deployment_id': deployment_id,
                'health_status': False,
                'prediction_success': False,
                'error': str(e)
            }
    
    def _wait_for_deployment(self, config: DeploymentConfig, timeout: int = 300):
        """Wait for deployment to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if config.endpoint_url:
                    response = requests.get(f"{config.endpoint_url}/health", timeout=5)
                    if response.status_code == 200:
                        config.status = "running"
                        return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(5)
        
        raise TimeoutError(f"Deployment {config.deployment_id} did not become ready within {timeout} seconds")
    
    def _load_active_deployments(self):
        """Load active deployments from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM deployments 
                WHERE status IN ('running', 'deploying')
            ''')
            
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                
                # Parse JSON fields
                for field in ['resources', 'environment_vars', 'metadata']:
                    if data[field]:
                        data[field] = json.loads(data[field])
                    else:
                        data[field] = {}
                
                # Parse datetime fields
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                
                deployment = DeploymentConfig(**data)
                self.deployments[deployment.deployment_id] = deployment
        
        logger.info(f"Loaded {len(self.deployments)} active deployments")
    
    def _save_deployment_config(self, config: DeploymentConfig):
        """Save deployment configuration to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO deployments
                (deployment_id, name, model_id, model_version, deployment_type,
                 environment, created_at, updated_at, status, endpoint_url,
                 container_id, port, replicas, resources, environment_vars,
                 health_check_url, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.deployment_id, config.name, config.model_id, config.model_version,
                config.deployment_type, config.environment, config.created_at.isoformat(),
                config.updated_at.isoformat(), config.status, config.endpoint_url,
                config.container_id, config.port, config.replicas,
                json.dumps(config.resources), json.dumps(config.environment_vars),
                config.health_check_url, json.dumps(config.metadata)
            ))
            
            conn.commit()
    
    def _start_monitoring(self):
        """Start deployment monitoring."""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started deployment monitoring")
    
    def _monitoring_loop(self):
        """Monitoring loop for deployments."""
        while True:
            try:
                for deployment_id, deployment in self.deployments.items():
                    if deployment.status == "running":
                        metrics = self._collect_deployment_metrics(deployment)
                        if metrics:
                            self._save_deployment_metrics(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_deployment_metrics(self, deployment: DeploymentConfig) -> Optional[DeploymentMetrics]:
        """Collect metrics for deployment."""
        try:
            # Get container stats if available
            cpu_usage = 0.0
            memory_usage = 0.0
            
            if deployment.container_id:
                try:
                    import docker
                    client = docker.from_env()
                    container = client.containers.get(deployment.container_id)
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                    cpu_usage = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
                    
                    # Calculate memory usage
                    memory_usage = (stats['memory_stats']['usage'] / stats['memory_stats']['limit']) * 100.0
                    
                except Exception as e:
                    logger.debug(f"Failed to get container stats: {e}")
            
            # Test endpoint response time
            response_time_ms = 0.0
            if deployment.endpoint_url:
                try:
                    start_time = time.time()
                    response = requests.get(f"{deployment.endpoint_url}/health", timeout=5)
                    response_time_ms = (time.time() - start_time) * 1000
                except Exception:
                    response_time_ms = -1  # Indicate failure
            
            return DeploymentMetrics(
                deployment_id=deployment.deployment_id,
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_count=0,  # Would need request tracking
                response_time_ms=response_time_ms,
                error_rate=0.0,  # Would need error tracking
                throughput=0.0   # Would need throughput tracking
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {deployment.deployment_id}: {e}")
            return None
    
    def _save_deployment_metrics(self, metrics: DeploymentMetrics):
        """Save deployment metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO deployment_metrics
                (deployment_id, timestamp, cpu_usage, memory_usage,
                 request_count, response_time_ms, error_rate, throughput, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.deployment_id, metrics.timestamp.isoformat(),
                metrics.cpu_usage, metrics.memory_usage, metrics.request_count,
                metrics.response_time_ms, metrics.error_rate, metrics.throughput,
                json.dumps(metrics.metadata)
            ))
            
            conn.commit()
    
    def get_deployment_metrics(self, deployment_id: str, 
                             hours: int = 24) -> List[DeploymentMetrics]:
        """Get deployment metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            since_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT * FROM deployment_metrics
                WHERE deployment_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (deployment_id, since_time.isoformat()))
            
            columns = [desc[0] for desc in cursor.description]
            metrics = []
            
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                data['metadata'] = json.loads(data['metadata']) if data['metadata'] else {}
                
                # Remove database ID
                data.pop('id', None)
                
                metrics.append(DeploymentMetrics(**data))
            
            return metrics


def create_deployment_manager(config: Dict[str, Any]) -> DeploymentManager:
    """
    Factory function to create deployment manager.
    
    Args:
        config: Manager configuration
        
    Returns:
        DeploymentManager instance
    """
    return DeploymentManager(config)

