"""
Parallel Execution Framework for RL-LLM Tree
Multi-environment orchestration and asynchronous learning coordination

This module implements sophisticated parallel processing capabilities for
distributed RL training and multi-environment execution.
"""

import asyncio
import threading
import multiprocessing as mp
import queue
import time
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from collections import defaultdict, deque
import json
import pickle

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for parallel processing."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"


class TaskPriority(Enum):
    """Priority levels for task execution."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResourceRequirement:
    """Specification of resource requirements for a task."""
    resource_type: ResourceType
    amount: float
    is_exclusive: bool = False
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None


@dataclass
class ExecutionTask:
    """Represents a task for parallel execution."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    worker_id: Optional[str] = None
    completed_at: float = field(default_factory=time.time)


@dataclass
class WorkerStatus:
    """Status information for a worker."""
    worker_id: str
    is_active: bool
    current_task: Optional[str]
    completed_tasks: int
    failed_tasks: int
    total_execution_time: float
    resource_utilization: Dict[ResourceType, float]
    last_heartbeat: float


class ResourceManager:
    """Manages computational resources across workers."""
    
    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {
            ResourceType.CPU: float(mp.cpu_count()),
            ResourceType.GPU: 0.0,  # Will be detected dynamically
            ResourceType.MEMORY: 8.0,  # GB, simplified
            ResourceType.NETWORK: 100.0,  # Mbps, simplified
            ResourceType.STORAGE: 1000.0  # GB, simplified
        }
        
        self.allocated_resources: Dict[str, Dict[ResourceType, float]] = {}
        self.resource_lock = threading.Lock()
        
        # Detect GPU resources
        self._detect_gpu_resources()
    
    def _detect_gpu_resources(self):
        """Detect available GPU resources."""
        try:
            import torch
            if torch.cuda.is_available():
                self.available_resources[ResourceType.GPU] = float(torch.cuda.device_count())
                logger.info(f"Detected {torch.cuda.device_count()} GPU(s)")
        except ImportError:
            logger.info("PyTorch not available, no GPU detection")
    
    def can_allocate(self, task_id: str, requirements: List[ResourceRequirement]) -> bool:
        """Check if resources can be allocated for a task."""
        with self.resource_lock:
            for req in requirements:
                available = self.available_resources.get(req.resource_type, 0.0)
                allocated = sum(
                    worker_resources.get(req.resource_type, 0.0)
                    for worker_resources in self.allocated_resources.values()
                )
                
                remaining = available - allocated
                
                if remaining < req.amount:
                    return False
        
        return True
    
    def allocate_resources(self, task_id: str, requirements: List[ResourceRequirement]) -> bool:
        """Allocate resources for a task."""
        if not self.can_allocate(task_id, requirements):
            return False
        
        with self.resource_lock:
            self.allocated_resources[task_id] = {}
            for req in requirements:
                self.allocated_resources[task_id][req.resource_type] = req.amount
        
        return True
    
    def release_resources(self, task_id: str):
        """Release resources allocated to a task."""
        with self.resource_lock:
            if task_id in self.allocated_resources:
                del self.allocated_resources[task_id]
    
    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization."""
        with self.resource_lock:
            utilization = {}
            
            for resource_type in ResourceType:
                available = self.available_resources.get(resource_type, 0.0)
                allocated = sum(
                    worker_resources.get(resource_type, 0.0)
                    for worker_resources in self.allocated_resources.values()
                )
                
                if available > 0:
                    utilization[resource_type] = allocated / available
                else:
                    utilization[resource_type] = 0.0
            
            return utilization


class Worker(ABC):
    """Abstract base class for workers."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.is_active = False
        self.current_task: Optional[ExecutionTask] = None
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        self.last_heartbeat = time.time()
    
    @abstractmethod
    def execute_task(self, task: ExecutionTask) -> ExecutionResult:
        """Execute a task and return the result."""
        pass
    
    def get_status(self) -> WorkerStatus:
        """Get current worker status."""
        return WorkerStatus(
            worker_id=self.worker_id,
            is_active=self.is_active,
            current_task=self.current_task.task_id if self.current_task else None,
            completed_tasks=self.completed_tasks,
            failed_tasks=self.failed_tasks,
            total_execution_time=self.total_execution_time,
            resource_utilization={},  # To be implemented by subclasses
            last_heartbeat=self.last_heartbeat
        )
    
    def heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()


class ThreadWorker(Worker):
    """Worker that executes tasks in threads."""
    
    def __init__(self, worker_id: str):
        super().__init__(worker_id)
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def execute_task(self, task: ExecutionTask) -> ExecutionResult:
        """Execute task in thread."""
        self.current_task = task
        self.is_active = True
        start_time = time.time()
        
        try:
            # Submit task to thread executor
            future = self.executor.submit(task.function, *task.args, **task.kwargs)
            
            # Wait for completion with timeout
            if task.timeout:
                result = future.result(timeout=task.timeout)
            else:
                result = future.result()
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.completed_tasks += 1
            
            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=self.worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.failed_tasks += 1
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                worker_id=self.worker_id
            )
        
        finally:
            self.current_task = None
            self.is_active = False
            self.heartbeat()


class ProcessWorker(Worker):
    """Worker that executes tasks in separate processes."""
    
    def __init__(self, worker_id: str):
        super().__init__(worker_id)
        self.executor = ProcessPoolExecutor(max_workers=1)
    
    def execute_task(self, task: ExecutionTask) -> ExecutionResult:
        """Execute task in separate process."""
        self.current_task = task
        self.is_active = True
        start_time = time.time()
        
        try:
            # Serialize task for process execution
            serialized_task = self._serialize_task(task)
            
            # Submit task to process executor
            future = self.executor.submit(self._execute_serialized_task, serialized_task)
            
            # Wait for completion with timeout
            if task.timeout:
                result = future.result(timeout=task.timeout)
            else:
                result = future.result()
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.completed_tasks += 1
            
            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=self.worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.failed_tasks += 1
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                worker_id=self.worker_id
            )
        
        finally:
            self.current_task = None
            self.is_active = False
            self.heartbeat()
    
    def _serialize_task(self, task: ExecutionTask) -> bytes:
        """Serialize task for process execution."""
        # Create a simplified task representation for serialization
        task_data = {
            'task_id': task.task_id,
            'function': task.function,
            'args': task.args,
            'kwargs': task.kwargs,
            'metadata': task.metadata
        }
        return pickle.dumps(task_data)
    
    @staticmethod
    def _execute_serialized_task(serialized_task: bytes) -> Any:
        """Execute a serialized task in a separate process."""
        task_data = pickle.loads(serialized_task)
        function = task_data['function']
        args = task_data['args']
        kwargs = task_data['kwargs']
        
        return function(*args, **kwargs)


class TaskScheduler:
    """Schedules tasks for execution based on priority and dependencies."""
    
    def __init__(self):
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.pending_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.reverse_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.scheduler_lock = threading.Lock()
    
    def add_task(self, task: ExecutionTask):
        """Add a task to the scheduler."""
        with self.scheduler_lock:
            self.pending_tasks[task.task_id] = task
            
            # Build dependency graph
            self.dependency_graph[task.task_id] = task.dependencies.copy()
            for dep in task.dependencies:
                self.reverse_dependencies[dep].append(task.task_id)
            
            # Check if task can be scheduled immediately
            if self._can_schedule(task.task_id):
                self._schedule_task(task)
    
    def _can_schedule(self, task_id: str) -> bool:
        """Check if a task can be scheduled (all dependencies completed)."""
        dependencies = self.dependency_graph.get(task_id, [])
        return all(dep in self.completed_tasks for dep in dependencies)
    
    def _schedule_task(self, task: ExecutionTask):
        """Schedule a task for execution."""
        # Priority queue uses negative priority for max-heap behavior
        priority = -task.priority.value
        self.task_queue.put((priority, task.created_at, task))
    
    def get_next_task(self) -> Optional[ExecutionTask]:
        """Get the next task to execute."""
        try:
            _, _, task = self.task_queue.get_nowait()
            return task
        except queue.Empty:
            return None
    
    def complete_task(self, result: ExecutionResult):
        """Mark a task as completed and schedule dependent tasks."""
        with self.scheduler_lock:
            self.completed_tasks[result.task_id] = result
            
            if result.task_id in self.pending_tasks:
                del self.pending_tasks[result.task_id]
            
            # Schedule dependent tasks
            dependent_tasks = self.reverse_dependencies.get(result.task_id, [])
            for dep_task_id in dependent_tasks:
                if dep_task_id in self.pending_tasks and self._can_schedule(dep_task_id):
                    task = self.pending_tasks[dep_task_id]
                    self._schedule_task(task)
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        with self.scheduler_lock:
            return {
                "queued_tasks": self.task_queue.qsize(),
                "pending_tasks": len(self.pending_tasks),
                "completed_tasks": len(self.completed_tasks),
                "dependency_chains": len(self.dependency_graph)
            }


class ParallelExecutor:
    """Main parallel execution coordinator."""
    
    def __init__(self, max_thread_workers: int = 4, max_process_workers: int = 2):
        self.resource_manager = ResourceManager()
        self.scheduler = TaskScheduler()
        
        # Worker pools
        self.thread_workers: List[ThreadWorker] = []
        self.process_workers: List[ProcessWorker] = []
        
        # Initialize workers
        for i in range(max_thread_workers):
            worker = ThreadWorker(f"thread_worker_{i}")
            self.thread_workers.append(worker)
        
        for i in range(max_process_workers):
            worker = ProcessWorker(f"process_worker_{i}")
            self.process_workers.append(worker)
        
        self.all_workers = self.thread_workers + self.process_workers
        
        # Execution control
        self.is_running = False
        self.execution_thread: Optional[threading.Thread] = None
        self.result_callbacks: List[Callable[[ExecutionResult], None]] = []
        
        # Statistics
        self.total_tasks_executed = 0
        self.total_execution_time = 0.0
        self.task_history: deque = deque(maxlen=1000)
    
    def add_result_callback(self, callback: Callable[[ExecutionResult], None]):
        """Add a callback to be called when tasks complete."""
        self.result_callbacks.append(callback)
    
    def submit_task(self, task: ExecutionTask) -> str:
        """Submit a task for execution."""
        # Check resource requirements
        if not self.resource_manager.can_allocate(task.task_id, task.resource_requirements):
            raise RuntimeError(f"Insufficient resources for task {task.task_id}")
        
        # Add to scheduler
        self.scheduler.add_task(task)
        
        logger.info(f"Submitted task {task.task_id} for execution")
        return task.task_id
    
    def submit_function(self, func: Callable, *args, task_id: Optional[str] = None, 
                       priority: TaskPriority = TaskPriority.MEDIUM, 
                       timeout: Optional[float] = None, **kwargs) -> str:
        """Submit a function for execution."""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        task = ExecutionTask(
            task_id=task_id,
            function=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        return self.submit_task(task)
    
    def start(self):
        """Start the parallel execution system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        
        logger.info("Parallel executor started")
    
    def stop(self):
        """Stop the parallel execution system."""
        self.is_running = False
        
        if self.execution_thread:
            self.execution_thread.join(timeout=5.0)
        
        # Shutdown worker executors
        for worker in self.all_workers:
            if hasattr(worker, 'executor'):
                worker.executor.shutdown(wait=True)
        
        logger.info("Parallel executor stopped")
    
    def _execution_loop(self):
        """Main execution loop."""
        while self.is_running:
            try:
                # Get next task
                task = self.scheduler.get_next_task()
                if task is None:
                    time.sleep(0.1)  # No tasks available, wait briefly
                    continue
                
                # Find available worker
                worker = self._find_available_worker(task)
                if worker is None:
                    # No workers available, put task back and wait
                    self.scheduler.add_task(task)
                    time.sleep(0.1)
                    continue
                
                # Allocate resources
                if not self.resource_manager.allocate_resources(task.task_id, task.resource_requirements):
                    # Resources not available, put task back
                    self.scheduler.add_task(task)
                    time.sleep(0.1)
                    continue
                
                # Execute task asynchronously
                self._execute_task_async(worker, task)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                time.sleep(1.0)
    
    def _find_available_worker(self, task: ExecutionTask) -> Optional[Worker]:
        """Find an available worker for the task."""
        # Simple strategy: prefer thread workers for lightweight tasks
        available_workers = [w for w in self.all_workers if not w.is_active]
        
        if not available_workers:
            return None
        
        # Prefer thread workers for most tasks
        thread_workers = [w for w in available_workers if isinstance(w, ThreadWorker)]
        if thread_workers:
            return thread_workers[0]
        
        # Use process workers if no thread workers available
        return available_workers[0]
    
    def _execute_task_async(self, worker: Worker, task: ExecutionTask):
        """Execute a task asynchronously."""
        def execute_and_handle():
            try:
                result = worker.execute_task(task)
                self._handle_task_completion(task, result)
            except Exception as e:
                error_result = ExecutionResult(
                    task_id=task.task_id,
                    success=False,
                    error=e,
                    worker_id=worker.worker_id
                )
                self._handle_task_completion(task, error_result)
        
        # Execute in separate thread to avoid blocking
        execution_thread = threading.Thread(target=execute_and_handle, daemon=True)
        execution_thread.start()
    
    def _handle_task_completion(self, task: ExecutionTask, result: ExecutionResult):
        """Handle task completion."""
        # Release resources
        self.resource_manager.release_resources(task.task_id)
        
        # Update statistics
        self.total_tasks_executed += 1
        self.total_execution_time += result.execution_time
        self.task_history.append({
            "task_id": task.task_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "worker_id": result.worker_id,
            "completed_at": result.completed_at
        })
        
        # Notify scheduler
        self.scheduler.complete_task(result)
        
        # Call result callbacks
        for callback in self.result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
        
        logger.info(f"Task {task.task_id} completed: success={result.success}")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks to complete."""
        start_time = time.time()
        
        while True:
            status = self.get_status()
            if status["active_tasks"] == 0 and status["queued_tasks"] == 0:
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the execution system."""
        active_workers = sum(1 for w in self.all_workers if w.is_active)
        
        return {
            "is_running": self.is_running,
            "total_workers": len(self.all_workers),
            "active_workers": active_workers,
            "thread_workers": len(self.thread_workers),
            "process_workers": len(self.process_workers),
            "total_tasks_executed": self.total_tasks_executed,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": (
                self.total_execution_time / self.total_tasks_executed 
                if self.total_tasks_executed > 0 else 0.0
            ),
            "resource_utilization": self.resource_manager.get_resource_utilization(),
            "scheduler_status": self.scheduler.get_status(),
            "active_tasks": active_workers,
            "queued_tasks": self.scheduler.task_queue.qsize(),
            "worker_status": [w.get_status() for w in self.all_workers]
        }


# Example usage and testing
if __name__ == "__main__":
    # Example functions to execute
    def simple_computation(x: int, y: int) -> int:
        """Simple computation function."""
        time.sleep(0.1)  # Simulate work
        return x + y
    
    def complex_computation(n: int) -> int:
        """More complex computation function."""
        time.sleep(0.5)  # Simulate work
        result = sum(i * i for i in range(n))
        return result
    
    def failing_function() -> None:
        """Function that always fails."""
        raise ValueError("This function always fails")
    
    # Create parallel executor
    executor = ParallelExecutor(max_thread_workers=3, max_process_workers=2)
    
    # Add result callback
    def result_callback(result: ExecutionResult):
        print(f"Task {result.task_id} completed: success={result.success}")
        if result.success:
            print(f"Result: {result.result}")
        else:
            print(f"Error: {result.error}")
    
    executor.add_result_callback(result_callback)
    
    # Start executor
    executor.start()
    
    try:
        # Submit various tasks
        task_ids = []
        
        # Simple tasks
        for i in range(5):
            task_id = executor.submit_function(
                simple_computation, 
                i, i + 1, 
                task_id=f"simple_{i}",
                priority=TaskPriority.MEDIUM
            )
            task_ids.append(task_id)
        
        # Complex tasks
        for i in range(3):
            task_id = executor.submit_function(
                complex_computation, 
                100 + i * 50,
                task_id=f"complex_{i}",
                priority=TaskPriority.HIGH
            )
            task_ids.append(task_id)
        
        # Failing task
        task_id = executor.submit_function(
            failing_function,
            task_id="failing_task",
            priority=TaskPriority.LOW
        )
        task_ids.append(task_id)
        
        print(f"Submitted {len(task_ids)} tasks")
        
        # Wait for completion
        print("Waiting for tasks to complete...")
        completed = executor.wait_for_completion(timeout=10.0)
        
        if completed:
            print("All tasks completed!")
        else:
            print("Timeout waiting for tasks to complete")
        
        # Get final status
        status = executor.get_status()
        print(f"Final status: {json.dumps(status, indent=2, default=str)}")
        
    finally:
        # Stop executor
        executor.stop()

