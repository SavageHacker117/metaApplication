"""
Workflow Orchestration and Pipeline Management System for RL-LLM

This module provides comprehensive workflow orchestration capabilities including
task scheduling, dependency management, pipeline execution, resource allocation,
monitoring, and failure recovery for complex RL training workflows.
"""

import asyncio
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import pickle
import sqlite3
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import psutil
import docker
import kubernetes
from kubernetes import client, config
import redis
import celery
from celery import Celery
import schedule
import cron_descriptor
from croniter import croniter
import jinja2
import requests

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    CREATED = "created"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ResourceType(Enum):
    """Resource type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    resource_type: ResourceType
    amount: float
    unit: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDefinition:
    """Task definition with execution parameters."""
    task_id: str
    name: str
    task_type: str  # 'python', 'shell', 'docker', 'kubernetes', 'function'
    command: Optional[str] = None
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_count: int = 0
    retry_delay: int = 60
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    docker_image: Optional[str] = None
    kubernetes_spec: Optional[Dict[str, Any]] = None


@dataclass
class TaskExecution:
    """Task execution instance."""
    execution_id: str
    task_id: str
    pipeline_id: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    error_message: str = ""
    retry_count: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class PipelineDefinition:
    """Pipeline definition with tasks and configuration."""
    pipeline_id: str
    name: str
    description: str = ""
    tasks: List[TaskDefinition] = field(default_factory=list)
    schedule: Optional[str] = None  # Cron expression
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    max_parallel_tasks: int = 10
    failure_policy: str = "fail_fast"  # 'fail_fast', 'continue', 'retry_failed'
    notification_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_by: str = ""
    created_at: Optional[datetime] = None


@dataclass
class PipelineExecution:
    """Pipeline execution instance."""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.CREATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    triggered_by: str = ""
    trigger_type: str = "manual"  # 'manual', 'scheduled', 'webhook', 'event'
    logs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class TaskExecutor(ABC):
    """Abstract task executor interface."""
    
    @abstractmethod
    async def execute(self, task_def: TaskDefinition, execution: TaskExecution) -> TaskExecution:
        """Execute task and return updated execution."""
        pass
    
    @abstractmethod
    async def cancel(self, execution_id: str) -> bool:
        """Cancel running task."""
        pass
    
    @abstractmethod
    async def get_status(self, execution_id: str) -> TaskStatus:
        """Get task execution status."""
        pass


class PythonTaskExecutor(TaskExecutor):
    """Python function task executor."""
    
    async def execute(self, task_def: TaskDefinition, execution: TaskExecution) -> TaskExecution:
        """Execute Python function task."""
        execution.status = TaskStatus.RUNNING
        execution.start_time = datetime.now()
        
        try:
            if task_def.function:
                # Execute function with parameters
                if asyncio.iscoroutinefunction(task_def.function):
                    result = await task_def.function(**task_def.parameters)
                else:
                    result = task_def.function(**task_def.parameters)
                
                execution.result = result
                execution.status = TaskStatus.SUCCESS
                execution.exit_code = 0
            else:
                raise ValueError("No function specified for Python task")
        
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            execution.exit_code = 1
            logger.error(f"Python task {task_def.task_id} failed: {e}")
        
        finally:
            execution.end_time = datetime.now()
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
        
        return execution
    
    async def cancel(self, execution_id: str) -> bool:
        """Cancel Python task (limited cancellation support)."""
        # Python function cancellation is limited
        return False
    
    async def get_status(self, execution_id: str) -> TaskStatus:
        """Get Python task status."""
        # Status is managed by the execution object
        return TaskStatus.UNKNOWN


class ShellTaskExecutor(TaskExecutor):
    """Shell command task executor."""
    
    def __init__(self):
        self.running_processes = {}
    
    async def execute(self, task_def: TaskDefinition, execution: TaskExecution) -> TaskExecution:
        """Execute shell command task."""
        execution.status = TaskStatus.RUNNING
        execution.start_time = datetime.now()
        
        try:
            # Prepare environment
            env = dict(os.environ)
            env.update(task_def.environment)
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                task_def.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=task_def.working_directory
            )
            
            self.running_processes[execution.execution_id] = process
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task_def.timeout
                )
                
                execution.stdout = stdout.decode('utf-8') if stdout else ""
                execution.stderr = stderr.decode('utf-8') if stderr else ""
                execution.exit_code = process.returncode
                
                if process.returncode == 0:
                    execution.status = TaskStatus.SUCCESS
                else:
                    execution.status = TaskStatus.FAILED
                    execution.error_message = execution.stderr
            
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                execution.status = TaskStatus.FAILED
                execution.error_message = "Task timed out"
                execution.exit_code = -1
        
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            execution.exit_code = -1
            logger.error(f"Shell task {task_def.task_id} failed: {e}")
        
        finally:
            execution.end_time = datetime.now()
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Clean up
            if execution.execution_id in self.running_processes:
                del self.running_processes[execution.execution_id]
        
        return execution
    
    async def cancel(self, execution_id: str) -> bool:
        """Cancel shell task."""
        if execution_id in self.running_processes:
            process = self.running_processes[execution_id]
            try:
                process.kill()
                await process.wait()
                return True
            except Exception as e:
                logger.error(f"Failed to cancel task {execution_id}: {e}")
                return False
        return False
    
    async def get_status(self, execution_id: str) -> TaskStatus:
        """Get shell task status."""
        if execution_id in self.running_processes:
            process = self.running_processes[execution_id]
            if process.returncode is None:
                return TaskStatus.RUNNING
            elif process.returncode == 0:
                return TaskStatus.SUCCESS
            else:
                return TaskStatus.FAILED
        return TaskStatus.UNKNOWN


class DockerTaskExecutor(TaskExecutor):
    """Docker container task executor."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.running_containers = {}
    
    async def execute(self, task_def: TaskDefinition, execution: TaskExecution) -> TaskExecution:
        """Execute Docker container task."""
        execution.status = TaskStatus.RUNNING
        execution.start_time = datetime.now()
        
        try:
            # Prepare container configuration
            container_config = {
                'image': task_def.docker_image,
                'command': task_def.command,
                'environment': task_def.environment,
                'detach': True,
                'remove': True
            }
            
            # Add resource limits
            for req in task_def.resource_requirements:
                if req.resource_type == ResourceType.CPU:
                    container_config['cpu_quota'] = int(req.amount * 100000)
                elif req.resource_type == ResourceType.MEMORY:
                    container_config['mem_limit'] = f"{int(req.amount)}{req.unit}"
            
            # Run container
            container = self.docker_client.containers.run(**container_config)
            self.running_containers[execution.execution_id] = container
            
            # Wait for completion
            result = container.wait(timeout=task_def.timeout)
            
            # Get logs
            execution.stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            execution.stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
            execution.exit_code = result['StatusCode']
            
            if result['StatusCode'] == 0:
                execution.status = TaskStatus.SUCCESS
            else:
                execution.status = TaskStatus.FAILED
                execution.error_message = execution.stderr
        
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            execution.exit_code = -1
            logger.error(f"Docker task {task_def.task_id} failed: {e}")
        
        finally:
            execution.end_time = datetime.now()
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Clean up
            if execution.execution_id in self.running_containers:
                del self.running_containers[execution.execution_id]
        
        return execution
    
    async def cancel(self, execution_id: str) -> bool:
        """Cancel Docker task."""
        if execution_id in self.running_containers:
            container = self.running_containers[execution_id]
            try:
                container.kill()
                return True
            except Exception as e:
                logger.error(f"Failed to cancel Docker task {execution_id}: {e}")
                return False
        return False
    
    async def get_status(self, execution_id: str) -> TaskStatus:
        """Get Docker task status."""
        if execution_id in self.running_containers:
            container = self.running_containers[execution_id]
            container.reload()
            
            if container.status == 'running':
                return TaskStatus.RUNNING
            elif container.status == 'exited':
                if container.attrs['State']['ExitCode'] == 0:
                    return TaskStatus.SUCCESS
                else:
                    return TaskStatus.FAILED
        
        return TaskStatus.UNKNOWN


class ResourceManager:
    """Resource allocation and management."""
    
    def __init__(self):
        self.available_resources = {
            ResourceType.CPU: psutil.cpu_count(),
            ResourceType.MEMORY: psutil.virtual_memory().total / (1024**3),  # GB
            ResourceType.GPU: self._get_gpu_count(),
            ResourceType.STORAGE: psutil.disk_usage('/').free / (1024**3),  # GB
        }
        
        self.allocated_resources = defaultdict(float)
        self.resource_locks = defaultdict(threading.Lock)
        
        logger.info(f"Initialized ResourceManager with resources: {self.available_resources}")
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0
    
    async def allocate_resources(self, requirements: List[ResourceRequirement]) -> bool:
        """Allocate resources for task execution."""
        # Check if resources are available
        for req in requirements:
            with self.resource_locks[req.resource_type]:
                available = self.available_resources.get(req.resource_type, 0)
                allocated = self.allocated_resources[req.resource_type]
                
                if available - allocated < req.amount:
                    return False
        
        # Allocate resources
        for req in requirements:
            with self.resource_locks[req.resource_type]:
                self.allocated_resources[req.resource_type] += req.amount
        
        return True
    
    async def release_resources(self, requirements: List[ResourceRequirement]):
        """Release allocated resources."""
        for req in requirements:
            with self.resource_locks[req.resource_type]:
                self.allocated_resources[req.resource_type] = max(
                    0, self.allocated_resources[req.resource_type] - req.amount
                )
    
    def get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Get current resource usage."""
        usage = {}
        
        for resource_type in ResourceType:
            available = self.available_resources.get(resource_type, 0)
            allocated = self.allocated_resources[resource_type]
            
            usage[resource_type.value] = {
                'available': available,
                'allocated': allocated,
                'utilization': (allocated / available * 100) if available > 0 else 0
            }
        
        return usage


class DependencyResolver:
    """Task dependency resolution and execution ordering."""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
    
    def add_task(self, task: TaskDefinition):
        """Add task to dependency graph."""
        self.dependency_graph.add_node(task.task_id, task=task)
        
        # Add dependency edges
        for dep_id in task.dependencies:
            self.dependency_graph.add_edge(dep_id, task.task_id)
    
    def get_execution_order(self) -> List[List[str]]:
        """Get task execution order respecting dependencies."""
        try:
            # Topological sort to get execution levels
            levels = []
            graph_copy = self.dependency_graph.copy()
            
            while graph_copy.nodes():
                # Find nodes with no incoming edges
                ready_nodes = [node for node in graph_copy.nodes() 
                              if graph_copy.in_degree(node) == 0]
                
                if not ready_nodes:
                    # Circular dependency detected
                    raise ValueError("Circular dependency detected in task graph")
                
                levels.append(ready_nodes)
                graph_copy.remove_nodes_from(ready_nodes)
            
            return levels
        
        except Exception as e:
            logger.error(f"Failed to resolve dependencies: {e}")
            raise
    
    def validate_dependencies(self) -> List[str]:
        """Validate task dependencies and return issues."""
        issues = []
        
        # Check for circular dependencies
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            cycles = list(nx.simple_cycles(self.dependency_graph))
            issues.append(f"Circular dependencies detected: {cycles}")
        
        # Check for missing dependencies
        for node in self.dependency_graph.nodes():
            task = self.dependency_graph.nodes[node]['task']
            for dep_id in task.dependencies:
                if dep_id not in self.dependency_graph.nodes():
                    issues.append(f"Task {node} depends on non-existent task {dep_id}")
        
        return issues
    
    def visualize_dependencies(self, output_path: Optional[Path] = None):
        """Visualize task dependency graph."""
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.dependency_graph, k=2, iterations=50)
        
        # Draw graph
        nx.draw(self.dependency_graph, pos, 
                with_labels=True, 
                node_color='lightblue',
                node_size=3000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20)
        
        plt.title("Task Dependency Graph")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class WorkflowOrchestrator:
    """Main workflow orchestration engine."""
    
    def __init__(self, db_path: Path, redis_url: str = "redis://localhost:6379/0"):
        self.db_path = db_path
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize components
        self.resource_manager = ResourceManager()
        self.dependency_resolver = DependencyResolver()
        
        # Task executors
        self.executors = {
            'python': PythonTaskExecutor(),
            'shell': ShellTaskExecutor(),
            'docker': DockerTaskExecutor(),
        }
        
        # Pipeline and execution storage
        self.pipelines = {}
        self.executions = {}
        
        # Scheduler
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Initialize database
        self._init_database()
        
        logger.info("Initialized WorkflowOrchestrator")
    
    def _init_database(self):
        """Initialize SQLite database for persistence."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Pipelines table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipelines (
                    pipeline_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    execution_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (pipeline_id) REFERENCES pipelines (pipeline_id)
                )
            ''')
            
            # Task executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_executions (
                    execution_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    pipeline_execution_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    execution_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (pipeline_execution_id) REFERENCES executions (execution_id)
                )
            ''')
            
            conn.commit()
    
    def register_pipeline(self, pipeline: PipelineDefinition):
        """Register pipeline definition."""
        pipeline.created_at = datetime.now()
        self.pipelines[pipeline.pipeline_id] = pipeline
        
        # Build dependency graph
        self.dependency_resolver = DependencyResolver()
        for task in pipeline.tasks:
            self.dependency_resolver.add_task(task)
        
        # Validate dependencies
        issues = self.dependency_resolver.validate_dependencies()
        if issues:
            raise ValueError(f"Pipeline validation failed: {issues}")
        
        # Save to database
        self._save_pipeline(pipeline)
        
        logger.info(f"Registered pipeline: {pipeline.name}")
    
    def _save_pipeline(self, pipeline: PipelineDefinition):
        """Save pipeline to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            pipeline_data = {
                'name': pipeline.name,
                'description': pipeline.description,
                'tasks': [self._task_to_dict(task) for task in pipeline.tasks],
                'schedule': pipeline.schedule,
                'parameters': pipeline.parameters,
                'timeout': pipeline.timeout,
                'max_parallel_tasks': pipeline.max_parallel_tasks,
                'failure_policy': pipeline.failure_policy,
                'notification_config': pipeline.notification_config,
                'tags': pipeline.tags,
                'metadata': pipeline.metadata,
                'version': pipeline.version,
                'created_by': pipeline.created_by
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO pipelines 
                (pipeline_id, name, definition, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                pipeline.pipeline_id,
                pipeline.name,
                json.dumps(pipeline_data),
                pipeline.created_at.isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
    
    def _task_to_dict(self, task: TaskDefinition) -> Dict[str, Any]:
        """Convert task definition to dictionary."""
        return {
            'task_id': task.task_id,
            'name': task.name,
            'task_type': task.task_type,
            'command': task.command,
            'parameters': task.parameters,
            'dependencies': task.dependencies,
            'resource_requirements': [
                {
                    'resource_type': req.resource_type.value,
                    'amount': req.amount,
                    'unit': req.unit,
                    'constraints': req.constraints
                }
                for req in task.resource_requirements
            ],
            'timeout': task.timeout,
            'retry_count': task.retry_count,
            'retry_delay': task.retry_delay,
            'tags': task.tags,
            'metadata': task.metadata,
            'environment': task.environment,
            'working_directory': task.working_directory,
            'docker_image': task.docker_image,
            'kubernetes_spec': task.kubernetes_spec
        }
    
    async def execute_pipeline(self, pipeline_id: str, parameters: Dict[str, Any] = None,
                             triggered_by: str = "manual") -> str:
        """Execute pipeline and return execution ID."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.pipelines[pipeline_id]
        execution_id = str(uuid.uuid4())
        
        # Create pipeline execution
        execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            status=PipelineStatus.SCHEDULED,
            parameters=parameters or {},
            triggered_by=triggered_by
        )
        
        self.executions[execution_id] = execution
        
        # Start execution in background
        asyncio.create_task(self._execute_pipeline_async(execution))
        
        logger.info(f"Started pipeline execution: {execution_id}")
        return execution_id
    
    async def _execute_pipeline_async(self, execution: PipelineExecution):
        """Execute pipeline asynchronously."""
        pipeline = self.pipelines[execution.pipeline_id]
        
        try:
            execution.status = PipelineStatus.RUNNING
            execution.start_time = datetime.now()
            
            # Get execution order
            execution_levels = self.dependency_resolver.get_execution_order()
            
            # Execute tasks level by level
            for level in execution_levels:
                # Execute tasks in parallel within each level
                tasks_in_level = [
                    task for task in pipeline.tasks 
                    if task.task_id in level
                ]
                
                # Limit parallelism
                semaphore = asyncio.Semaphore(pipeline.max_parallel_tasks)
                
                async def execute_task_with_semaphore(task_def):
                    async with semaphore:
                        return await self._execute_task(task_def, execution)
                
                # Execute tasks concurrently
                task_results = await asyncio.gather(
                    *[execute_task_with_semaphore(task) for task in tasks_in_level],
                    return_exceptions=True
                )
                
                # Check for failures
                failed_tasks = []
                for i, result in enumerate(task_results):
                    if isinstance(result, Exception):
                        failed_tasks.append(tasks_in_level[i].task_id)
                    elif result.status == TaskStatus.FAILED:
                        failed_tasks.append(result.task_id)
                
                # Handle failures based on policy
                if failed_tasks:
                    if pipeline.failure_policy == "fail_fast":
                        execution.status = PipelineStatus.FAILED
                        break
                    elif pipeline.failure_policy == "continue":
                        continue
                    elif pipeline.failure_policy == "retry_failed":
                        # Implement retry logic here
                        pass
            
            # Set final status
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.SUCCESS
        
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.logs.append(f"Pipeline execution failed: {str(e)}")
            logger.error(f"Pipeline execution {execution.execution_id} failed: {e}")
        
        finally:
            execution.end_time = datetime.now()
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Save execution to database
            self._save_execution(execution)
            
            # Send notifications
            await self._send_notifications(execution)
    
    async def _execute_task(self, task_def: TaskDefinition, 
                          pipeline_execution: PipelineExecution) -> TaskExecution:
        """Execute individual task."""
        task_execution_id = str(uuid.uuid4())
        
        task_execution = TaskExecution(
            execution_id=task_execution_id,
            task_id=task_def.task_id,
            pipeline_id=pipeline_execution.pipeline_id
        )
        
        pipeline_execution.task_executions[task_def.task_id] = task_execution
        
        try:
            # Allocate resources
            if task_def.resource_requirements:
                allocated = await self.resource_manager.allocate_resources(
                    task_def.resource_requirements
                )
                if not allocated:
                    task_execution.status = TaskStatus.FAILED
                    task_execution.error_message = "Insufficient resources"
                    return task_execution
            
            # Get executor
            executor = self.executors.get(task_def.task_type)
            if not executor:
                raise ValueError(f"No executor found for task type: {task_def.task_type}")
            
            # Execute task with retries
            for attempt in range(task_def.retry_count + 1):
                try:
                    task_execution = await executor.execute(task_def, task_execution)
                    
                    if task_execution.status == TaskStatus.SUCCESS:
                        break
                    elif attempt < task_def.retry_count:
                        task_execution.status = TaskStatus.RETRYING
                        task_execution.retry_count = attempt + 1
                        await asyncio.sleep(task_def.retry_delay)
                
                except Exception as e:
                    if attempt == task_def.retry_count:
                        task_execution.status = TaskStatus.FAILED
                        task_execution.error_message = str(e)
                        break
                    else:
                        await asyncio.sleep(task_def.retry_delay)
            
            # Call success/failure callbacks
            if task_execution.status == TaskStatus.SUCCESS and task_def.on_success:
                try:
                    if asyncio.iscoroutinefunction(task_def.on_success):
                        await task_def.on_success(task_execution)
                    else:
                        task_def.on_success(task_execution)
                except Exception as e:
                    logger.warning(f"Success callback failed for task {task_def.task_id}: {e}")
            
            elif task_execution.status == TaskStatus.FAILED and task_def.on_failure:
                try:
                    if asyncio.iscoroutinefunction(task_def.on_failure):
                        await task_def.on_failure(task_execution)
                    else:
                        task_def.on_failure(task_execution)
                except Exception as e:
                    logger.warning(f"Failure callback failed for task {task_def.task_id}: {e}")
        
        finally:
            # Release resources
            if task_def.resource_requirements:
                await self.resource_manager.release_resources(task_def.resource_requirements)
        
        return task_execution
    
    def _save_execution(self, execution: PipelineExecution):
        """Save execution to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            execution_data = {
                'status': execution.status.value,
                'start_time': execution.start_time.isoformat() if execution.start_time else None,
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'duration': execution.duration,
                'parameters': execution.parameters,
                'triggered_by': execution.triggered_by,
                'trigger_type': execution.trigger_type,
                'logs': execution.logs,
                'artifacts': execution.artifacts,
                'metrics': execution.metrics,
                'task_executions': {
                    task_id: {
                        'execution_id': te.execution_id,
                        'status': te.status.value,
                        'start_time': te.start_time.isoformat() if te.start_time else None,
                        'end_time': te.end_time.isoformat() if te.end_time else None,
                        'duration': te.duration,
                        'exit_code': te.exit_code,
                        'stdout': te.stdout,
                        'stderr': te.stderr,
                        'error_message': te.error_message,
                        'retry_count': te.retry_count,
                        'resource_usage': te.resource_usage,
                        'logs': te.logs,
                        'artifacts': te.artifacts
                    }
                    for task_id, te in execution.task_executions.items()
                }
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO executions 
                (execution_id, pipeline_id, status, start_time, end_time, execution_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id,
                execution.pipeline_id,
                execution.status.value,
                execution.start_time.isoformat() if execution.start_time else None,
                execution.end_time.isoformat() if execution.end_time else None,
                json.dumps(execution_data),
                datetime.now().isoformat()
            ))
            
            conn.commit()
    
    async def _send_notifications(self, execution: PipelineExecution):
        """Send execution notifications."""
        pipeline = self.pipelines[execution.pipeline_id]
        notification_config = pipeline.notification_config
        
        if not notification_config:
            return
        
        # Prepare notification data
        notification_data = {
            'pipeline_name': pipeline.name,
            'execution_id': execution.execution_id,
            'status': execution.status.value,
            'duration': execution.duration,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'failed_tasks': [
                task_id for task_id, te in execution.task_executions.items()
                if te.status == TaskStatus.FAILED
            ]
        }
        
        # Send notifications based on configuration
        for notification_type, config in notification_config.items():
            try:
                if notification_type == 'email':
                    await self._send_email_notification(notification_data, config)
                elif notification_type == 'webhook':
                    await self._send_webhook_notification(notification_data, config)
                elif notification_type == 'slack':
                    await self._send_slack_notification(notification_data, config)
            except Exception as e:
                logger.error(f"Failed to send {notification_type} notification: {e}")
    
    async def _send_email_notification(self, data: Dict[str, Any], config: Dict[str, Any]):
        """Send email notification."""
        # Implementation would depend on email service configuration
        logger.info(f"Email notification sent for execution {data['execution_id']}")
    
    async def _send_webhook_notification(self, data: Dict[str, Any], config: Dict[str, Any]):
        """Send webhook notification."""
        webhook_url = config.get('url')
        if webhook_url:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=data) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for execution {data['execution_id']}")
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
    
    async def _send_slack_notification(self, data: Dict[str, Any], config: Dict[str, Any]):
        """Send Slack notification."""
        # Implementation would use Slack API
        logger.info(f"Slack notification sent for execution {data['execution_id']}")
    
    def get_execution_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get pipeline execution status."""
        return self.executions.get(execution_id)
    
    def list_pipelines(self) -> List[PipelineDefinition]:
        """List all registered pipelines."""
        return list(self.pipelines.values())
    
    def list_executions(self, pipeline_id: Optional[str] = None) -> List[PipelineExecution]:
        """List pipeline executions."""
        executions = list(self.executions.values())
        
        if pipeline_id:
            executions = [e for e in executions if e.pipeline_id == pipeline_id]
        
        return executions
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return self.resource_manager.get_resource_usage()
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel pipeline execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False
        
        execution.status = PipelineStatus.CANCELLED
        
        # Cancel running tasks
        cancelled_tasks = []
        for task_id, task_execution in execution.task_executions.items():
            if task_execution.status == TaskStatus.RUNNING:
                pipeline = self.pipelines[execution.pipeline_id]
                task_def = next(t for t in pipeline.tasks if t.task_id == task_id)
                executor = self.executors.get(task_def.task_type)
                
                if executor:
                    success = await executor.cancel(task_execution.execution_id)
                    if success:
                        task_execution.status = TaskStatus.CANCELLED
                        cancelled_tasks.append(task_id)
        
        logger.info(f"Cancelled execution {execution_id}, cancelled tasks: {cancelled_tasks}")
        return True


def create_workflow_orchestrator(config: Dict[str, Any]) -> WorkflowOrchestrator:
    """
    Factory function to create workflow orchestrator.
    
    Args:
        config: Orchestrator configuration
        
    Returns:
        WorkflowOrchestrator instance
    """
    db_path = Path(config.get('db_path', './workflow.db'))
    redis_url = config.get('redis_url', 'redis://localhost:6379/0')
    
    return WorkflowOrchestrator(db_path, redis_url)


# Example usage and pipeline templates
def create_rl_training_pipeline() -> PipelineDefinition:
    """Create example RL training pipeline."""
    
    # Define tasks
    tasks = [
        TaskDefinition(
            task_id="data_preparation",
            name="Prepare Training Data",
            task_type="python",
            function=lambda **kwargs: print("Preparing data..."),
            parameters={"dataset_path": "/data/training"},
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 2),
                ResourceRequirement(ResourceType.MEMORY, 4, "GB")
            ],
            timeout=3600
        ),
        
        TaskDefinition(
            task_id="model_training",
            name="Train RL Model",
            task_type="python",
            function=lambda **kwargs: print("Training model..."),
            dependencies=["data_preparation"],
            parameters={"epochs": 100, "learning_rate": 0.001},
            resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 1),
                ResourceRequirement(ResourceType.MEMORY, 8, "GB")
            ],
            timeout=7200,
            retry_count=2
        ),
        
        TaskDefinition(
            task_id="model_evaluation",
            name="Evaluate Model",
            task_type="python",
            function=lambda **kwargs: print("Evaluating model..."),
            dependencies=["model_training"],
            parameters={"test_episodes": 100},
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 1),
                ResourceRequirement(ResourceType.MEMORY, 2, "GB")
            ],
            timeout=1800
        ),
        
        TaskDefinition(
            task_id="model_deployment",
            name="Deploy Model",
            task_type="docker",
            docker_image="rl-model-server:latest",
            command="python deploy.py",
            dependencies=["model_evaluation"],
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 1),
                ResourceRequirement(ResourceType.MEMORY, 2, "GB")
            ],
            timeout=600
        )
    ]
    
    # Create pipeline
    pipeline = PipelineDefinition(
        pipeline_id="rl_training_pipeline",
        name="RL Model Training Pipeline",
        description="Complete RL model training, evaluation, and deployment pipeline",
        tasks=tasks,
        max_parallel_tasks=2,
        failure_policy="fail_fast",
        notification_config={
            "webhook": {
                "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
            }
        },
        tags=["rl", "training", "ml"],
        version="1.0.0",
        created_by="rl-system"
    )
    
    return pipeline

