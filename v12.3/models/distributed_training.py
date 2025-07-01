"""
Distributed Training System for RL-LLM

This module provides comprehensive distributed training capabilities including
multi-GPU training, multi-node coordination, parameter synchronization, and
communication protocols for scalable RL training.
"""

import os
import time
import socket
import threading
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import json
import queue
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import zmq
import redis
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # 'nccl', 'gloo', 'mpi'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    communication_backend: str = "pytorch"  # 'pytorch', 'zmq', 'redis'
    gradient_compression: bool = False
    async_communication: bool = True
    parameter_server: bool = False
    federation_enabled: bool = False
    checkpoint_frequency: int = 1000
    log_frequency: int = 100


@dataclass
class WorkerInfo:
    """Information about a distributed worker."""
    worker_id: str
    rank: int
    local_rank: int
    hostname: str
    gpu_count: int
    status: str = "idle"  # 'idle', 'training', 'syncing', 'failed'
    last_heartbeat: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class CommunicationBackend(ABC):
    """Abstract base class for communication backends."""
    
    @abstractmethod
    def initialize(self, config: DistributedConfig):
        """Initialize communication backend."""
        pass
    
    @abstractmethod
    def send(self, data: Any, destination: int):
        """Send data to destination rank."""
        pass
    
    @abstractmethod
    def receive(self, source: int) -> Any:
        """Receive data from source rank."""
        pass
    
    @abstractmethod
    def broadcast(self, data: Any, root: int = 0):
        """Broadcast data from root to all ranks."""
        pass
    
    @abstractmethod
    def all_reduce(self, data: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce operation on tensor."""
        pass
    
    @abstractmethod
    def barrier(self):
        """Synchronization barrier."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup communication backend."""
        pass


class PyTorchBackend(CommunicationBackend):
    """PyTorch distributed communication backend."""
    
    def __init__(self):
        self.initialized = False
    
    def initialize(self, config: DistributedConfig):
        """Initialize PyTorch distributed."""
        os.environ['MASTER_ADDR'] = config.master_addr
        os.environ['MASTER_PORT'] = config.master_port
        
        init_process_group(
            backend=config.backend,
            rank=config.rank,
            world_size=config.world_size
        )
        
        self.initialized = True
        logger.info(f"Initialized PyTorch distributed backend (rank {config.rank}/{config.world_size})")
    
    def send(self, data: Any, destination: int):
        """Send data using PyTorch distributed."""
        if isinstance(data, torch.Tensor):
            dist.send(data, dst=destination)
        else:
            # Serialize non-tensor data
            tensor_data = torch.tensor(pickle.dumps(data), dtype=torch.uint8)
            size_tensor = torch.tensor([len(tensor_data)], dtype=torch.long)
            dist.send(size_tensor, dst=destination)
            dist.send(tensor_data, dst=destination)
    
    def receive(self, source: int) -> Any:
        """Receive data using PyTorch distributed."""
        # First receive size
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(size_tensor, src=source)
        
        # Then receive data
        data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8)
        dist.recv(data_tensor, src=source)
        
        return pickle.loads(data_tensor.numpy().tobytes())
    
    def broadcast(self, data: Any, root: int = 0):
        """Broadcast data using PyTorch distributed."""
        if isinstance(data, torch.Tensor):
            dist.broadcast(data, src=root)
            return data
        else:
            if dist.get_rank() == root:
                serialized = pickle.dumps(data)
                size_tensor = torch.tensor([len(serialized)], dtype=torch.long)
                data_tensor = torch.tensor(list(serialized), dtype=torch.uint8)
            else:
                size_tensor = torch.tensor([0], dtype=torch.long)
                data_tensor = None
            
            dist.broadcast(size_tensor, src=root)
            
            if dist.get_rank() != root:
                data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8)
            
            dist.broadcast(data_tensor, src=root)
            
            if dist.get_rank() != root:
                return pickle.loads(data_tensor.numpy().tobytes())
            else:
                return data
    
    def all_reduce(self, data: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce operation using PyTorch distributed."""
        if op == "sum":
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            data /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(data, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(data, op=dist.ReduceOp.MIN)
        
        return data
    
    def barrier(self):
        """Synchronization barrier."""
        dist.barrier()
    
    def cleanup(self):
        """Cleanup PyTorch distributed."""
        if self.initialized:
            destroy_process_group()
            self.initialized = False


class ZMQBackend(CommunicationBackend):
    """ZeroMQ communication backend."""
    
    def __init__(self):
        self.context = None
        self.sockets = {}
        self.config = None
    
    def initialize(self, config: DistributedConfig):
        """Initialize ZMQ backend."""
        self.config = config
        self.context = zmq.Context()
        
        # Create sockets for each rank
        for rank in range(config.world_size):
            if rank != config.rank:
                socket = self.context.socket(zmq.DEALER)
                socket.connect(f"tcp://{config.master_addr}:{int(config.master_port) + rank}")
                self.sockets[rank] = socket
        
        # Create listening socket
        self.listen_socket = self.context.socket(zmq.ROUTER)
        self.listen_socket.bind(f"tcp://*:{int(config.master_port) + config.rank}")
        
        logger.info(f"Initialized ZMQ backend (rank {config.rank})")
    
    def send(self, data: Any, destination: int):
        """Send data using ZMQ."""
        serialized = pickle.dumps(data)
        self.sockets[destination].send(serialized)
    
    def receive(self, source: int) -> Any:
        """Receive data using ZMQ."""
        identity, message = self.listen_socket.recv_multipart()
        return pickle.loads(message)
    
    def broadcast(self, data: Any, root: int = 0):
        """Broadcast data using ZMQ."""
        if self.config.rank == root:
            for rank in range(self.config.world_size):
                if rank != root:
                    self.send(data, rank)
            return data
        else:
            return self.receive(root)
    
    def all_reduce(self, data: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce operation using ZMQ."""
        # Simple implementation - gather to rank 0, reduce, then broadcast
        if self.config.rank == 0:
            tensors = [data]
            for rank in range(1, self.config.world_size):
                received = self.receive(rank)
                tensors.append(received)
            
            # Reduce
            if op == "sum":
                result = sum(tensors)
            elif op == "mean":
                result = sum(tensors) / len(tensors)
            elif op == "max":
                result = torch.max(torch.stack(tensors), dim=0)[0]
            elif op == "min":
                result = torch.min(torch.stack(tensors), dim=0)[0]
            
            # Broadcast result
            for rank in range(1, self.config.world_size):
                self.send(result, rank)
            
            return result
        else:
            self.send(data, 0)
            return self.receive(0)
    
    def barrier(self):
        """Synchronization barrier using ZMQ."""
        # Simple barrier implementation
        if self.config.rank == 0:
            for rank in range(1, self.config.world_size):
                self.receive(rank)  # Wait for all ranks
            for rank in range(1, self.config.world_size):
                self.send("barrier_release", rank)
        else:
            self.send("barrier_ready", 0)
            self.receive(0)  # Wait for release
    
    def cleanup(self):
        """Cleanup ZMQ backend."""
        for socket in self.sockets.values():
            socket.close()
        if hasattr(self, 'listen_socket'):
            self.listen_socket.close()
        if self.context:
            self.context.term()


class ParameterServer:
    """Parameter server for distributed training."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.parameters = {}
        self.gradients = defaultdict(list)
        self.version = 0
        self.lock = threading.Lock()
        
        # Redis connection for parameter storage
        if config.communication_backend == "redis":
            self.redis_client = redis.Redis(host=config.master_addr, port=6379, db=0)
        else:
            self.redis_client = None
        
        logger.info("Initialized Parameter Server")
    
    def push_gradients(self, worker_id: str, gradients: Dict[str, torch.Tensor]):
        """Push gradients from worker."""
        with self.lock:
            for name, grad in gradients.items():
                self.gradients[name].append((worker_id, grad))
    
    def pull_parameters(self, worker_id: str) -> Dict[str, torch.Tensor]:
        """Pull latest parameters for worker."""
        with self.lock:
            return self.parameters.copy()
    
    def aggregate_gradients(self, aggregation_method: str = "mean"):
        """Aggregate gradients from all workers."""
        with self.lock:
            aggregated = {}
            
            for param_name, grad_list in self.gradients.items():
                if not grad_list:
                    continue
                
                grads = torch.stack([grad for _, grad in grad_list])
                
                if aggregation_method == "mean":
                    aggregated[param_name] = torch.mean(grads, dim=0)
                elif aggregation_method == "sum":
                    aggregated[param_name] = torch.sum(grads, dim=0)
                elif aggregation_method == "median":
                    aggregated[param_name] = torch.median(grads, dim=0)[0]
                
            # Clear gradients
            self.gradients.clear()
            
            return aggregated
    
    def update_parameters(self, updates: Dict[str, torch.Tensor], learning_rate: float = 0.001):
        """Update parameters with aggregated gradients."""
        with self.lock:
            for name, update in updates.items():
                if name in self.parameters:
                    self.parameters[name] -= learning_rate * update
                else:
                    self.parameters[name] = -learning_rate * update
            
            self.version += 1
            
            # Store in Redis if available
            if self.redis_client:
                serialized = pickle.dumps(self.parameters)
                self.redis_client.set(f"parameters_v{self.version}", serialized)


class DistributedTrainer:
    """Main distributed training coordinator."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.workers = {}
        self.communication_backend = self._create_communication_backend()
        
        if config.parameter_server:
            self.parameter_server = ParameterServer(config)
        else:
            self.parameter_server = None
        
        # Training statistics
        self.training_stats = defaultdict(list)
        self.sync_times = deque(maxlen=100)
        self.communication_overhead = deque(maxlen=100)
        
        logger.info("Initialized DistributedTrainer")
    
    def _create_communication_backend(self) -> CommunicationBackend:
        """Create communication backend."""
        if self.config.communication_backend == "pytorch":
            return PyTorchBackend()
        elif self.config.communication_backend == "zmq":
            return ZMQBackend()
        else:
            raise ValueError(f"Unknown communication backend: {self.config.communication_backend}")
    
    def initialize(self):
        """Initialize distributed training."""
        self.communication_backend.initialize(self.config)
        
        # Register this worker
        worker_info = WorkerInfo(
            worker_id=f"worker_{self.config.rank}",
            rank=self.config.rank,
            local_rank=self.config.local_rank,
            hostname=socket.gethostname(),
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        
        self.workers[self.config.rank] = worker_info
        
        logger.info(f"Initialized distributed training (rank {self.config.rank}/{self.config.world_size})")
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if self.config.backend == "nccl" and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.local_rank}")
            model = model.to(device)
            model = DDP(model, device_ids=[self.config.local_rank])
        else:
            model = DDP(model)
        
        logger.info("Wrapped model for distributed training")
        return model
    
    def synchronize_parameters(self, model: torch.nn.Module) -> float:
        """Synchronize model parameters across workers."""
        start_time = time.time()
        
        if self.parameter_server:
            # Parameter server approach
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            # Push gradients to parameter server
            self.parameter_server.push_gradients(f"worker_{self.config.rank}", gradients)
            
            # Pull updated parameters
            updated_params = self.parameter_server.pull_parameters(f"worker_{self.config.rank}")
            
            # Update model parameters
            for name, param in model.named_parameters():
                if name in updated_params:
                    param.data.copy_(updated_params[name])
        
        else:
            # All-reduce approach
            for param in model.parameters():
                if param.grad is not None:
                    self.communication_backend.all_reduce(param.grad, op="mean")
        
        sync_time = time.time() - start_time
        self.sync_times.append(sync_time)
        
        return sync_time
    
    def broadcast_model(self, model: torch.nn.Module, root: int = 0):
        """Broadcast model parameters from root to all workers."""
        for param in model.parameters():
            self.communication_backend.broadcast(param.data, root=root)
        
        logger.info(f"Broadcasted model parameters from rank {root}")
    
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, List[float]]:
        """Gather metrics from all workers."""
        all_metrics = defaultdict(list)
        
        if self.config.rank == 0:
            # Collect from all workers
            all_metrics[self.config.rank] = metrics
            
            for rank in range(1, self.config.world_size):
                worker_metrics = self.communication_backend.receive(rank)
                all_metrics[rank] = worker_metrics
        else:
            # Send to rank 0
            self.communication_backend.send(metrics, 0)
        
        return dict(all_metrics)
    
    def checkpoint_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, checkpoint_dir: Path):
        """Save distributed checkpoint."""
        if self.config.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': self.config,
                'training_stats': dict(self.training_stats)
            }
            
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       checkpoint_path: Path) -> int:
        """Load distributed checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.config.local_rank}")
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Broadcast to ensure all workers have same parameters
        self.broadcast_model(model, root=0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def monitor_workers(self) -> Dict[str, WorkerInfo]:
        """Monitor worker status and performance."""
        if self.config.rank == 0:
            # Collect status from all workers
            worker_status = {}
            
            for rank in range(self.config.world_size):
                if rank == 0:
                    worker_status[rank] = self.workers[rank]
                else:
                    try:
                        status = self.communication_backend.receive(rank)
                        worker_status[rank] = status
                    except Exception as e:
                        logger.warning(f"Failed to get status from rank {rank}: {e}")
            
            return worker_status
        else:
            # Send status to rank 0
            self.communication_backend.send(self.workers[self.config.rank], 0)
            return {}
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get distributed training statistics."""
        stats = {
            'sync_times': {
                'mean': np.mean(self.sync_times) if self.sync_times else 0,
                'std': np.std(self.sync_times) if self.sync_times else 0,
                'max': np.max(self.sync_times) if self.sync_times else 0
            },
            'communication_overhead': {
                'mean': np.mean(self.communication_overhead) if self.communication_overhead else 0,
                'std': np.std(self.communication_overhead) if self.communication_overhead else 0
            },
            'world_size': self.config.world_size,
            'backend': self.config.backend,
            'parameter_server': self.config.parameter_server
        }
        
        return stats
    
    def cleanup(self):
        """Cleanup distributed training."""
        self.communication_backend.cleanup()
        logger.info("Cleaned up distributed training")


def setup_distributed_training(rank: int, world_size: int, master_addr: str = "localhost",
                             master_port: str = "12355", backend: str = "nccl") -> DistributedTrainer:
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend
        
    Returns:
        DistributedTrainer instance
    """
    config = DistributedConfig(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=rank % torch.cuda.device_count() if torch.cuda.is_available() else 0,
        master_addr=master_addr,
        master_port=master_port
    )
    
    trainer = DistributedTrainer(config)
    trainer.initialize()
    
    return trainer


def run_distributed_training(train_fn: Callable, world_size: int, **kwargs):
    """
    Run distributed training with multiprocessing.
    
    Args:
        train_fn: Training function to run on each process
        world_size: Number of processes
        **kwargs: Additional arguments for training function
    """
    torch_mp.spawn(
        train_fn,
        args=(world_size, kwargs),
        nprocs=world_size,
        join=True
    )


class FederatedLearning:
    """Federated learning coordinator."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.clients = {}
        self.global_model = None
        self.round_number = 0
        
        logger.info("Initialized Federated Learning coordinator")
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]):
        """Register federated learning client."""
        self.clients[client_id] = {
            'info': client_info,
            'last_update': datetime.now(),
            'performance': {}
        }
        
        logger.info(f"Registered client: {client_id}")
    
    def aggregate_models(self, client_models: Dict[str, torch.nn.Module],
                        aggregation_method: str = "fedavg") -> torch.nn.Module:
        """Aggregate models from federated clients."""
        if aggregation_method == "fedavg":
            # FedAvg: weighted average based on data size
            total_samples = sum(info['data_size'] for info in self.clients.values())
            
            aggregated_state = {}
            for name, param in list(client_models.values())[0].state_dict().items():
                weighted_params = []
                
                for client_id, model in client_models.items():
                    weight = self.clients[client_id]['info']['data_size'] / total_samples
                    weighted_params.append(weight * model.state_dict()[name])
                
                aggregated_state[name] = sum(weighted_params)
            
            # Update global model
            self.global_model.load_state_dict(aggregated_state)
        
        self.round_number += 1
        logger.info(f"Aggregated models for round {self.round_number}")
        
        return self.global_model
    
    def select_clients(self, fraction: float = 1.0) -> List[str]:
        """Select clients for federated round."""
        num_clients = max(1, int(len(self.clients) * fraction))
        selected = np.random.choice(list(self.clients.keys()), num_clients, replace=False)
        
        logger.info(f"Selected {len(selected)} clients for round {self.round_number + 1}")
        return selected.tolist()


def create_distributed_trainer(config: Dict[str, Any]) -> DistributedTrainer:
    """
    Factory function to create distributed trainer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DistributedTrainer instance
    """
    dist_config = DistributedConfig(**config)
    return DistributedTrainer(dist_config)

