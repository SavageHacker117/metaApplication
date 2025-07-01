"""
Advanced Model Serving and Inference System for RL-LLM

This module provides comprehensive model serving capabilities including model versioning,
A/B testing, auto-scaling, caching, batch inference, real-time serving, and performance
optimization for RL models and other ML models.
"""

import asyncio
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple, Union, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import uuid
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.jit import ScriptModule
import onnx
import onnxruntime as ort
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer
import redis
import aiohttp
from aiohttp import web
import aiofiles
import asyncpg
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import psutil
import docker
from kubernetes import client, config as k8s_config
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import grpc
from grpc import aio as aio_grpc
import tritonclient.http as httpclient
import mlflow
import wandb
from abc import ABC, abstractmethod
import hashlib
import base64
import gzip
import lz4.frame
import msgpack

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    TORCHSCRIPT = "torchscript"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class InferenceMode(Enum):
    """Inference execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAMING = "streaming"


class ModelStatus(Enum):
    """Model deployment status."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"
    UNLOADING = "unloading"


@dataclass
class ModelMetadata:
    """Model metadata and configuration."""
    model_id: str
    name: str
    version: str
    format: ModelFormat
    framework: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    model_path: Path
    config_path: Optional[Path] = None
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    postprocessing_config: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""


@dataclass
class InferenceRequest:
    """Inference request structure."""
    request_id: str
    model_id: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    mode: InferenceMode = InferenceMode.SYNC
    priority: int = 0  # Higher values = higher priority
    timeout: Optional[float] = None
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceResponse:
    """Inference response structure."""
    request_id: str
    model_id: str
    outputs: Dict[str, Any]
    status: str = "success"  # success, error, timeout
    error_message: str = ""
    inference_time: float = 0.0
    queue_time: float = 0.0
    total_time: float = 0.0
    model_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelInstance:
    """Model instance for serving."""
    instance_id: str
    model_metadata: ModelMetadata
    model: Any  # The actual model object
    status: ModelStatus = ModelStatus.LOADING
    load_time: Optional[datetime] = None
    last_used: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    average_inference_time: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    device: str = "cpu"


class ModelLoader(ABC):
    """Abstract model loader interface."""
    
    @abstractmethod
    async def load_model(self, metadata: ModelMetadata) -> Any:
        """Load model from metadata."""
        pass
    
    @abstractmethod
    async def unload_model(self, model: Any):
        """Unload model and free resources."""
        pass
    
    @abstractmethod
    async def predict(self, model: Any, inputs: Dict[str, Any], 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on model."""
        pass


class PyTorchModelLoader(ModelLoader):
    """PyTorch model loader."""
    
    async def load_model(self, metadata: ModelMetadata) -> torch.nn.Module:
        """Load PyTorch model."""
        try:
            if metadata.format == ModelFormat.TORCHSCRIPT:
                model = torch.jit.load(metadata.model_path)
            else:
                model = torch.load(metadata.model_path, map_location='cpu')
            
            # Move to appropriate device
            device = metadata.resource_requirements.get('device', 'cpu')
            if device == 'gpu' and torch.cuda.is_available():
                model = model.cuda()
                device = f"cuda:{torch.cuda.current_device()}"
            
            model.eval()
            logger.info(f"Loaded PyTorch model {metadata.model_id} on {device}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load PyTorch model {metadata.model_id}: {e}")
            raise
    
    async def unload_model(self, model: torch.nn.Module):
        """Unload PyTorch model."""
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def predict(self, model: torch.nn.Module, inputs: Dict[str, Any], 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run PyTorch inference."""
        try:
            # Convert inputs to tensors
            tensor_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, (list, np.ndarray)):
                    tensor_inputs[key] = torch.tensor(value)
                elif isinstance(value, torch.Tensor):
                    tensor_inputs[key] = value
                else:
                    tensor_inputs[key] = torch.tensor([value])
            
            # Move to model device
            device = next(model.parameters()).device
            tensor_inputs = {k: v.to(device) for k, v in tensor_inputs.items()}
            
            # Run inference
            with torch.no_grad():
                if len(tensor_inputs) == 1:
                    outputs = model(list(tensor_inputs.values())[0])
                else:
                    outputs = model(**tensor_inputs)
            
            # Convert outputs to serializable format
            if isinstance(outputs, torch.Tensor):
                result = {"output": outputs.cpu().numpy().tolist()}
            elif isinstance(outputs, (tuple, list)):
                result = {f"output_{i}": out.cpu().numpy().tolist() 
                         for i, out in enumerate(outputs)}
            elif isinstance(outputs, dict):
                result = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                         for k, v in outputs.items()}
            else:
                result = {"output": str(outputs)}
            
            return result
        
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            raise


class ONNXModelLoader(ModelLoader):
    """ONNX model loader."""
    
    async def load_model(self, metadata: ModelMetadata) -> ort.InferenceSession:
        """Load ONNX model."""
        try:
            # Configure providers
            providers = ['CPUExecutionProvider']
            if metadata.resource_requirements.get('device') == 'gpu':
                if ort.get_device() == 'GPU':
                    providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(str(metadata.model_path), providers=providers)
            logger.info(f"Loaded ONNX model {metadata.model_id}")
            return session
        
        except Exception as e:
            logger.error(f"Failed to load ONNX model {metadata.model_id}: {e}")
            raise
    
    async def unload_model(self, model: ort.InferenceSession):
        """Unload ONNX model."""
        del model
    
    async def predict(self, model: ort.InferenceSession, inputs: Dict[str, Any], 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run ONNX inference."""
        try:
            # Prepare inputs
            onnx_inputs = {}
            for input_meta in model.get_inputs():
                input_name = input_meta.name
                if input_name in inputs:
                    value = inputs[input_name]
                    if isinstance(value, (list, tuple)):
                        onnx_inputs[input_name] = np.array(value, dtype=np.float32)
                    elif isinstance(value, np.ndarray):
                        onnx_inputs[input_name] = value.astype(np.float32)
                    else:
                        onnx_inputs[input_name] = np.array([value], dtype=np.float32)
            
            # Run inference
            outputs = model.run(None, onnx_inputs)
            
            # Format outputs
            output_names = [output.name for output in model.get_outputs()]
            result = {}
            for i, output in enumerate(outputs):
                output_name = output_names[i] if i < len(output_names) else f"output_{i}"
                result[output_name] = output.tolist()
            
            return result
        
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise


class HuggingFaceModelLoader(ModelLoader):
    """Hugging Face model loader."""
    
    async def load_model(self, metadata: ModelMetadata) -> Tuple[Any, Any]:
        """Load Hugging Face model and tokenizer."""
        try:
            model_path = str(metadata.model_path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            model = AutoModel.from_pretrained(model_path)
            
            # Move to device
            device = metadata.resource_requirements.get('device', 'cpu')
            if device == 'gpu' and torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            logger.info(f"Loaded Hugging Face model {metadata.model_id}")
            return (model, tokenizer)
        
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {metadata.model_id}: {e}")
            raise
    
    async def unload_model(self, model_tuple: Tuple[Any, Any]):
        """Unload Hugging Face model."""
        model, tokenizer = model_tuple
        if hasattr(model, 'cpu'):
            model.cpu()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def predict(self, model_tuple: Tuple[Any, Any], inputs: Dict[str, Any], 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Hugging Face inference."""
        try:
            model, tokenizer = model_tuple
            
            # Get text input
            text = inputs.get('text', inputs.get('input_text', ''))
            if not text:
                raise ValueError("No text input provided")
            
            # Tokenize
            tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            
            # Move to model device
            device = next(model.parameters()).device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**tokens)
            
            # Extract embeddings or logits
            if hasattr(outputs, 'last_hidden_state'):
                result = {"embeddings": outputs.last_hidden_state.cpu().numpy().tolist()}
            elif hasattr(outputs, 'logits'):
                result = {"logits": outputs.logits.cpu().numpy().tolist()}
            else:
                result = {"output": str(outputs)}
            
            return result
        
        except Exception as e:
            logger.error(f"Hugging Face inference failed: {e}")
            raise


class ModelRegistry:
    """Model registry for managing model metadata and versions."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.models = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize model registry database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    format TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    config_path TEXT,
                    input_schema TEXT NOT NULL,
                    output_schema TEXT NOT NULL,
                    preprocessing_config TEXT,
                    postprocessing_config TEXT,
                    resource_requirements TEXT,
                    performance_targets TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_name_version 
                ON models (name, version)
            ''')
            
            conn.commit()
    
    def register_model(self, metadata: ModelMetadata):
        """Register model in registry."""
        self.models[metadata.model_id] = metadata
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO models 
                (model_id, name, version, format, framework, model_path, config_path,
                 input_schema, output_schema, preprocessing_config, postprocessing_config,
                 resource_requirements, performance_targets, tags, metadata, 
                 created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_id, metadata.name, metadata.version, metadata.format.value,
                metadata.framework, str(metadata.model_path), 
                str(metadata.config_path) if metadata.config_path else None,
                json.dumps(metadata.input_schema), json.dumps(metadata.output_schema),
                json.dumps(metadata.preprocessing_config), 
                json.dumps(metadata.postprocessing_config),
                json.dumps(metadata.resource_requirements),
                json.dumps(metadata.performance_targets),
                json.dumps(metadata.tags), json.dumps(metadata.metadata),
                metadata.created_at.isoformat(), metadata.created_by
            ))
            
            conn.commit()
        
        logger.info(f"Registered model: {metadata.name} v{metadata.version}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    def list_models(self, name: Optional[str] = None, 
                   active_only: bool = True) -> List[ModelMetadata]:
        """List registered models."""
        models = list(self.models.values())
        
        if name:
            models = [m for m in models if m.name == name]
        
        return models
    
    def get_latest_version(self, name: str) -> Optional[ModelMetadata]:
        """Get latest version of a model."""
        models = [m for m in self.models.values() if m.name == name]
        if not models:
            return None
        
        # Sort by version (assuming semantic versioning)
        models.sort(key=lambda m: tuple(map(int, m.version.split('.'))), reverse=True)
        return models[0]


class InferenceCache:
    """Caching system for inference results."""
    
    def __init__(self, backend: str = "redis", redis_url: str = "redis://localhost:6379/1"):
        self.backend = backend
        if backend == "redis":
            self.redis_client = redis.from_url(redis_url)
        else:
            self.memory_cache = {}
            self.cache_timestamps = {}
    
    def _generate_cache_key(self, model_id: str, inputs: Dict[str, Any], 
                          parameters: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        cache_data = {
            'model_id': model_id,
            'inputs': inputs,
            'parameters': parameters
        }
        
        # Create deterministic hash
        serialized = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    async def get(self, model_id: str, inputs: Dict[str, Any], 
                 parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached inference result."""
        cache_key = self._generate_cache_key(model_id, inputs, parameters)
        
        try:
            if self.backend == "redis":
                cached_data = self.redis_client.get(f"inference:{cache_key}")
                if cached_data:
                    # Decompress and deserialize
                    decompressed = lz4.frame.decompress(cached_data)
                    return msgpack.unpackb(decompressed, raw=False)
            else:
                if cache_key in self.memory_cache:
                    timestamp, data = self.memory_cache[cache_key]
                    if time.time() - timestamp < 3600:  # 1 hour TTL
                        return data
                    else:
                        del self.memory_cache[cache_key]
        
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    async def set(self, model_id: str, inputs: Dict[str, Any], 
                 parameters: Dict[str, Any], result: Dict[str, Any], ttl: int = 3600):
        """Set cached inference result."""
        cache_key = self._generate_cache_key(model_id, inputs, parameters)
        
        try:
            if self.backend == "redis":
                # Serialize and compress
                serialized = msgpack.packb(result)
                compressed = lz4.frame.compress(serialized)
                self.redis_client.setex(f"inference:{cache_key}", ttl, compressed)
            else:
                self.memory_cache[cache_key] = (time.time(), result)
        
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")


class ModelServer:
    """Main model serving engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ModelRegistry(Path(config.get('registry_db_path', './models.db')))
        self.cache = InferenceCache(
            backend=config.get('cache_backend', 'redis'),
            redis_url=config.get('redis_url', 'redis://localhost:6379/1')
        )
        
        # Model loaders
        self.loaders = {
            ModelFormat.PYTORCH: PyTorchModelLoader(),
            ModelFormat.TORCHSCRIPT: PyTorchModelLoader(),
            ModelFormat.ONNX: ONNXModelLoader(),
            ModelFormat.HUGGINGFACE: HuggingFaceModelLoader(),
        }
        
        # Model instances
        self.instances = {}
        self.instance_lock = threading.RLock()
        
        # Request queue and processing
        self.request_queue = asyncio.Queue(maxsize=config.get('max_queue_size', 1000))
        self.batch_queue = defaultdict(list)
        self.processing_tasks = []
        
        # Metrics
        self.setup_metrics()
        
        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        self.process_pool = ProcessPoolExecutor(max_workers=config.get('max_processes', 2))
        
        logger.info("Initialized ModelServer")
    
    def setup_metrics(self):
        """Setup Prometheus metrics."""
        self.request_counter = Counter('inference_requests_total', 
                                     'Total inference requests', ['model_id', 'status'])
        self.request_duration = Histogram('inference_duration_seconds',
                                        'Inference request duration', ['model_id'])
        self.queue_size_gauge = Gauge('inference_queue_size', 'Current queue size')
        self.active_models_gauge = Gauge('active_models', 'Number of active models')
        self.memory_usage_gauge = Gauge('model_memory_usage_bytes', 
                                      'Model memory usage', ['model_id'])
    
    async def load_model(self, model_id: str) -> bool:
        """Load model for serving."""
        metadata = self.registry.get_model(model_id)
        if not metadata:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        try:
            with self.instance_lock:
                if model_id in self.instances:
                    logger.info(f"Model {model_id} already loaded")
                    return True
            
            # Get appropriate loader
            loader = self.loaders.get(metadata.format)
            if not loader:
                logger.error(f"No loader available for format {metadata.format}")
                return False
            
            # Load model
            start_time = time.time()
            model = await loader.load_model(metadata)
            load_time = time.time() - start_time
            
            # Create instance
            instance = ModelInstance(
                instance_id=str(uuid.uuid4()),
                model_metadata=metadata,
                model=model,
                status=ModelStatus.READY,
                load_time=datetime.now(),
                device=metadata.resource_requirements.get('device', 'cpu')
            )
            
            with self.instance_lock:
                self.instances[model_id] = instance
            
            # Update metrics
            self.active_models_gauge.inc()
            
            logger.info(f"Loaded model {model_id} in {load_time:.2f}s")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload model from serving."""
        try:
            with self.instance_lock:
                instance = self.instances.get(model_id)
                if not instance:
                    return False
                
                instance.status = ModelStatus.UNLOADING
            
            # Get loader and unload
            loader = self.loaders.get(instance.model_metadata.format)
            if loader:
                await loader.unload_model(instance.model)
            
            with self.instance_lock:
                del self.instances[model_id]
            
            # Update metrics
            self.active_models_gauge.dec()
            
            logger.info(f"Unloaded model {model_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on model."""
        start_time = time.time()
        queue_time = 0.0
        
        try:
            # Check cache first
            if self.config.get('enable_cache', True):
                cached_result = await self.cache.get(
                    request.model_id, request.inputs, request.parameters
                )
                if cached_result:
                    self.request_counter.labels(
                        model_id=request.model_id, status='cache_hit'
                    ).inc()
                    
                    return InferenceResponse(
                        request_id=request.request_id,
                        model_id=request.model_id,
                        outputs=cached_result,
                        status="success",
                        inference_time=0.0,
                        queue_time=0.0,
                        total_time=time.time() - start_time,
                        metadata={"cache_hit": True}
                    )
            
            # Get model instance
            with self.instance_lock:
                instance = self.instances.get(request.model_id)
                if not instance or instance.status != ModelStatus.READY:
                    raise ValueError(f"Model {request.model_id} not available")
            
            # Run inference
            inference_start = time.time()
            loader = self.loaders.get(instance.model_metadata.format)
            
            outputs = await loader.predict(
                instance.model, request.inputs, request.parameters
            )
            
            inference_time = time.time() - inference_start
            
            # Update instance metrics
            instance.request_count += 1
            instance.last_used = datetime.now()
            instance.average_inference_time = (
                (instance.average_inference_time * (instance.request_count - 1) + inference_time) /
                instance.request_count
            )
            
            # Cache result
            if self.config.get('enable_cache', True):
                await self.cache.set(
                    request.model_id, request.inputs, request.parameters, outputs
                )
            
            # Update metrics
            self.request_counter.labels(
                model_id=request.model_id, status='success'
            ).inc()
            self.request_duration.labels(model_id=request.model_id).observe(inference_time)
            
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                outputs=outputs,
                status="success",
                inference_time=inference_time,
                queue_time=queue_time,
                total_time=time.time() - start_time,
                model_version=instance.model_metadata.version
            )
        
        except Exception as e:
            logger.error(f"Inference failed for request {request.request_id}: {e}")
            
            # Update error metrics
            self.request_counter.labels(
                model_id=request.model_id, status='error'
            ).inc()
            
            if request.model_id in self.instances:
                self.instances[request.model_id].error_count += 1
            
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                outputs={},
                status="error",
                error_message=str(e),
                inference_time=0.0,
                queue_time=queue_time,
                total_time=time.time() - start_time
            )
    
    async def batch_predict(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Run batch inference."""
        if not requests:
            return []
        
        # Group by model
        model_requests = defaultdict(list)
        for request in requests:
            model_requests[request.model_id].append(request)
        
        # Process each model's requests
        all_responses = []
        for model_id, model_reqs in model_requests.items():
            try:
                # Get model instance
                with self.instance_lock:
                    instance = self.instances.get(model_id)
                    if not instance or instance.status != ModelStatus.READY:
                        # Return error responses
                        for req in model_reqs:
                            all_responses.append(InferenceResponse(
                                request_id=req.request_id,
                                model_id=model_id,
                                outputs={},
                                status="error",
                                error_message=f"Model {model_id} not available"
                            ))
                        continue
                
                # Batch process
                batch_inputs = [req.inputs for req in model_reqs]
                batch_parameters = [req.parameters for req in model_reqs]
                
                # Run batch inference (simplified - would need model-specific batching)
                responses = []
                for i, req in enumerate(model_reqs):
                    response = await self.predict(req)
                    responses.append(response)
                
                all_responses.extend(responses)
            
            except Exception as e:
                logger.error(f"Batch inference failed for model {model_id}: {e}")
                for req in model_reqs:
                    all_responses.append(InferenceResponse(
                        request_id=req.request_id,
                        model_id=model_id,
                        outputs={},
                        status="error",
                        error_message=str(e)
                    ))
        
        return all_responses
    
    async def start_processing(self):
        """Start background request processing."""
        # Start request processors
        num_processors = self.config.get('num_processors', 4)
        for i in range(num_processors):
            task = asyncio.create_task(self._process_requests())
            self.processing_tasks.append(task)
        
        # Start batch processor
        if self.config.get('enable_batching', True):
            batch_task = asyncio.create_task(self._process_batches())
            self.processing_tasks.append(batch_task)
        
        logger.info(f"Started {len(self.processing_tasks)} processing tasks")
    
    async def _process_requests(self):
        """Process individual requests from queue."""
        while True:
            try:
                request = await self.request_queue.get()
                
                if request.mode == InferenceMode.BATCH:
                    # Add to batch queue
                    self.batch_queue[request.model_id].append(request)
                else:
                    # Process immediately
                    response = await self.predict(request)
                    
                    # Handle callback if specified
                    if request.callback_url:
                        await self._send_callback(request.callback_url, response)
                
                self.request_queue.task_done()
            
            except Exception as e:
                logger.error(f"Request processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batches(self):
        """Process batched requests."""
        batch_size = self.config.get('batch_size', 8)
        batch_timeout = self.config.get('batch_timeout', 0.1)
        
        while True:
            try:
                await asyncio.sleep(batch_timeout)
                
                for model_id, requests in list(self.batch_queue.items()):
                    if len(requests) >= batch_size or (
                        requests and 
                        time.time() - requests[0].created_at.timestamp() > batch_timeout
                    ):
                        # Process batch
                        batch_requests = requests[:batch_size]
                        self.batch_queue[model_id] = requests[batch_size:]
                        
                        responses = await self.batch_predict(batch_requests)
                        
                        # Handle callbacks
                        for request, response in zip(batch_requests, responses):
                            if request.callback_url:
                                await self._send_callback(request.callback_url, response)
            
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(1)
    
    async def _send_callback(self, callback_url: str, response: InferenceResponse):
        """Send callback with inference response."""
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    callback_url,
                    json={
                        'request_id': response.request_id,
                        'model_id': response.model_id,
                        'outputs': response.outputs,
                        'status': response.status,
                        'error_message': response.error_message,
                        'inference_time': response.inference_time,
                        'total_time': response.total_time
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                )
        except Exception as e:
            logger.error(f"Callback failed for {callback_url}: {e}")
    
    async def submit_request(self, request: InferenceRequest) -> Union[InferenceResponse, str]:
        """Submit inference request."""
        if request.mode == InferenceMode.SYNC:
            return await self.predict(request)
        else:
            # Add to queue for async processing
            await self.request_queue.put(request)
            self.queue_size_gauge.set(self.request_queue.qsize())
            return request.request_id
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get model status and metrics."""
        with self.instance_lock:
            instance = self.instances.get(model_id)
            if not instance:
                return {"status": "not_loaded"}
            
            return {
                "status": instance.status.value,
                "load_time": instance.load_time.isoformat() if instance.load_time else None,
                "last_used": instance.last_used.isoformat() if instance.last_used else None,
                "request_count": instance.request_count,
                "error_count": instance.error_count,
                "average_inference_time": instance.average_inference_time,
                "memory_usage": instance.memory_usage,
                "gpu_usage": instance.gpu_usage,
                "device": instance.device,
                "version": instance.model_metadata.version
            }
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status and metrics."""
        return {
            "active_models": len(self.instances),
            "queue_size": self.request_queue.qsize(),
            "processing_tasks": len(self.processing_tasks),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.cpu_percent(),
            "models": {
                model_id: self.get_model_status(model_id)
                for model_id in self.instances.keys()
            }
        }
    
    async def cleanup(self):
        """Cleanup server resources."""
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Unload all models
        model_ids = list(self.instances.keys())
        for model_id in model_ids:
            await self.unload_model(model_id)
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("ModelServer cleanup completed")


class ModelServerAPI:
    """REST API for model server."""
    
    def __init__(self, model_server: ModelServer, host: str = "0.0.0.0", port: int = 8000):
        self.model_server = model_server
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        self.app.router.add_post('/models/{model_id}/load', self.load_model)
        self.app.router.add_delete('/models/{model_id}', self.unload_model)
        self.app.router.add_post('/models/{model_id}/predict', self.predict)
        self.app.router.add_post('/models/{model_id}/batch_predict', self.batch_predict)
        self.app.router.add_get('/models/{model_id}/status', self.model_status)
        self.app.router.add_get('/models', self.list_models)
        self.app.router.add_get('/status', self.server_status)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.metrics)
    
    async def load_model(self, request: web.Request) -> web.Response:
        """Load model endpoint."""
        model_id = request.match_info['model_id']
        success = await self.model_server.load_model(model_id)
        
        if success:
            return web.json_response({"status": "success", "message": f"Model {model_id} loaded"})
        else:
            return web.json_response(
                {"status": "error", "message": f"Failed to load model {model_id}"},
                status=400
            )
    
    async def unload_model(self, request: web.Request) -> web.Response:
        """Unload model endpoint."""
        model_id = request.match_info['model_id']
        success = await self.model_server.unload_model(model_id)
        
        if success:
            return web.json_response({"status": "success", "message": f"Model {model_id} unloaded"})
        else:
            return web.json_response(
                {"status": "error", "message": f"Failed to unload model {model_id}"},
                status=400
            )
    
    async def predict(self, request: web.Request) -> web.Response:
        """Prediction endpoint."""
        model_id = request.match_info['model_id']
        
        try:
            data = await request.json()
            
            inference_request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                model_id=model_id,
                inputs=data.get('inputs', {}),
                parameters=data.get('parameters', {}),
                mode=InferenceMode.SYNC
            )
            
            response = await self.model_server.submit_request(inference_request)
            
            return web.json_response({
                "request_id": response.request_id,
                "outputs": response.outputs,
                "status": response.status,
                "error_message": response.error_message,
                "inference_time": response.inference_time,
                "total_time": response.total_time,
                "model_version": response.model_version
            })
        
        except Exception as e:
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=400
            )
    
    async def batch_predict(self, request: web.Request) -> web.Response:
        """Batch prediction endpoint."""
        model_id = request.match_info['model_id']
        
        try:
            data = await request.json()
            requests_data = data.get('requests', [])
            
            inference_requests = []
            for req_data in requests_data:
                inference_requests.append(InferenceRequest(
                    request_id=str(uuid.uuid4()),
                    model_id=model_id,
                    inputs=req_data.get('inputs', {}),
                    parameters=req_data.get('parameters', {}),
                    mode=InferenceMode.BATCH
                ))
            
            responses = await self.model_server.batch_predict(inference_requests)
            
            return web.json_response({
                "responses": [
                    {
                        "request_id": resp.request_id,
                        "outputs": resp.outputs,
                        "status": resp.status,
                        "error_message": resp.error_message,
                        "inference_time": resp.inference_time,
                        "total_time": resp.total_time
                    }
                    for resp in responses
                ]
            })
        
        except Exception as e:
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=400
            )
    
    async def model_status(self, request: web.Request) -> web.Response:
        """Model status endpoint."""
        model_id = request.match_info['model_id']
        status = self.model_server.get_model_status(model_id)
        return web.json_response(status)
    
    async def list_models(self, request: web.Request) -> web.Response:
        """List models endpoint."""
        models = self.model_server.registry.list_models()
        return web.json_response({
            "models": [
                {
                    "model_id": model.model_id,
                    "name": model.name,
                    "version": model.version,
                    "format": model.format.value,
                    "framework": model.framework,
                    "status": self.model_server.get_model_status(model.model_id).get("status", "not_loaded")
                }
                for model in models
            ]
        })
    
    async def server_status(self, request: web.Request) -> web.Response:
        """Server status endpoint."""
        status = self.model_server.get_server_status()
        return web.json_response(status)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    async def metrics(self, request: web.Request) -> web.Response:
        """Prometheus metrics endpoint."""
        return web.Response(
            text=prometheus_client.generate_latest().decode('utf-8'),
            content_type='text/plain'
        )
    
    async def start_server(self):
        """Start the API server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Model server API started on {self.host}:{self.port}")


def create_model_server(config: Dict[str, Any]) -> ModelServer:
    """
    Factory function to create model server.
    
    Args:
        config: Server configuration
        
    Returns:
        ModelServer instance
    """
    return ModelServer(config)


# Example usage
async def main():
    """Example usage of model server."""
    config = {
        'registry_db_path': './models.db',
        'cache_backend': 'redis',
        'redis_url': 'redis://localhost:6379/1',
        'enable_cache': True,
        'enable_batching': True,
        'batch_size': 8,
        'batch_timeout': 0.1,
        'max_queue_size': 1000,
        'num_processors': 4,
        'max_workers': 10
    }
    
    # Create server
    server = create_model_server(config)
    
    # Register example model
    metadata = ModelMetadata(
        model_id="example_pytorch_model",
        name="example_model",
        version="1.0.0",
        format=ModelFormat.PYTORCH,
        framework="pytorch",
        input_schema={"input": {"type": "tensor", "shape": [1, 784]}},
        output_schema={"output": {"type": "tensor", "shape": [1, 10]}},
        model_path=Path("./models/example_model.pth"),
        resource_requirements={"device": "cpu"}
    )
    
    server.registry.register_model(metadata)
    
    # Start processing
    await server.start_processing()
    
    # Load model
    await server.load_model("example_pytorch_model")
    
    # Create API
    api = ModelServerAPI(server, host="0.0.0.0", port=8000)
    await api.start_server()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down model server")
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

