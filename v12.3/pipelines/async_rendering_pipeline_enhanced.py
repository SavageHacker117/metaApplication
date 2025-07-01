"""
Enhanced Async Rendering Pipeline v3

Improvements based on feedback:
- Profile queue wait times and scale workers dynamically
- Minimize setup overhead for rendering
- Add "mock renderer" or low-res proxy for early RL episodes
- Optimized batch processing
- Advanced performance monitoring
- Intelligent resource management
"""

import asyncio
import threading
import queue
import time
import logging
import statistics
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import json
import numpy as np
import psutil
from collections import deque, defaultdict
import pickle
import hashlib
import weakref

# GPU monitoring
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

@dataclass
class EnhancedRenderConfig:
    """Enhanced configuration for async rendering pipeline."""
    # Rendering settings
    width: int = 1024
    height: int = 768
    samples: int = 4
    headless: bool = True
    gpu_accelerated: bool = True
    
    # Dynamic worker scaling
    min_workers: int = 2
    max_workers: int = mp.cpu_count() * 2
    initial_workers: int = 4
    scale_up_threshold: float = 2.0  # seconds average queue wait
    scale_down_threshold: float = 0.5  # seconds average queue wait
    worker_scale_interval: float = 10.0  # seconds
    
    # Performance optimization
    enable_batch_processing: bool = True
    max_batch_size: int = 16
    batch_timeout: float = 0.1  # seconds
    enable_render_cache: bool = True
    cache_size: int = 1000
    
    # Low-res proxy settings
    enable_proxy_renderer: bool = True
    proxy_scale_factor: float = 0.25  # 25% of original resolution
    proxy_quality_threshold: int = 100  # episodes before switching to full quality
    
    # Queue management
    max_queue_size: int = 1000
    priority_levels: int = 3
    queue_timeout: float = 30.0
    
    # Memory management
    max_memory_mb: int = 8192
    memory_cleanup_interval: int = 50
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    
    # Monitoring
    enable_profiling: bool = True
    profile_window_size: int = 100
    log_performance_interval: float = 60.0

@dataclass
class EnhancedRenderTask:
    """Enhanced rendering task with priority and metadata."""
    task_id: str
    scene_data: Dict[str, Any]
    render_config: Dict[str, Any]
    priority: int = 1  # 0=highest, 2=lowest
    use_proxy: bool = False
    output_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)

@dataclass
class RenderResult:
    """Enhanced rendering result with detailed metrics."""
    task_id: str
    success: bool
    output_path: Optional[str] = None
    render_time: float = 0.0
    queue_wait_time: float = 0.0
    setup_time: float = 0.0
    actual_render_time: float = 0.0
    memory_used: float = 0.0
    was_cached: bool = False
    was_proxy: bool = False
    worker_id: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceProfiler:
    """Profiles rendering pipeline performance."""
    
    def __init__(self, config: EnhancedRenderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.queue_wait_times = deque(maxlen=config.profile_window_size)
        self.render_times = deque(maxlen=config.profile_window_size)
        self.setup_times = deque(maxlen=config.profile_window_size)
        self.memory_usage = deque(maxlen=config.profile_window_size)
        
        # Worker performance tracking
        self.worker_performance = defaultdict(lambda: {
            'render_times': deque(maxlen=50),
            'success_count': 0,
            'error_count': 0,
            'last_active': time.time()
        })
        
        # System metrics
        self.cpu_usage = deque(maxlen=config.profile_window_size)
        self.gpu_usage = deque(maxlen=config.profile_window_size)
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Monitor CPU
                cpu_percent = psutil.cpu_percent()
                self.cpu_usage.append(cpu_percent)
                
                # Monitor GPU if available
                if GPU_MONITORING_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_load = gpus[0].load * 100
                            self.gpu_usage.append(gpu_load)
                    except:
                        pass
                
                # Monitor memory
                memory_info = psutil.virtual_memory()
                self.memory_usage.append(memory_info.percent)
                
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.warning(f"Monitoring error: {e}")
    
    def record_task_metrics(self, result: RenderResult):
        """Record metrics from completed task."""
        self.queue_wait_times.append(result.queue_wait_time)
        self.render_times.append(result.render_time)
        self.setup_times.append(result.setup_time)
        
        # Update worker performance
        worker_stats = self.worker_performance[result.worker_id]
        worker_stats['render_times'].append(result.actual_render_time)
        worker_stats['last_active'] = time.time()
        
        if result.success:
            worker_stats['success_count'] += 1
        else:
            worker_stats['error_count'] += 1
    
    def get_average_queue_wait(self) -> float:
        """Get average queue wait time."""
        return statistics.mean(self.queue_wait_times) if self.queue_wait_times else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'queue_wait': {
                'average': statistics.mean(self.queue_wait_times) if self.queue_wait_times else 0,
                'median': statistics.median(self.queue_wait_times) if self.queue_wait_times else 0,
                'max': max(self.queue_wait_times) if self.queue_wait_times else 0
            },
            'render_time': {
                'average': statistics.mean(self.render_times) if self.render_times else 0,
                'median': statistics.median(self.render_times) if self.render_times else 0,
                'max': max(self.render_times) if self.render_times else 0
            },
            'setup_time': {
                'average': statistics.mean(self.setup_times) if self.setup_times else 0,
                'median': statistics.median(self.setup_times) if self.setup_times else 0
            },
            'system': {
                'cpu_usage': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'memory_usage': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'gpu_usage': statistics.mean(self.gpu_usage) if self.gpu_usage else 0
            },
            'workers': {
                worker_id: {
                    'avg_render_time': statistics.mean(stats['render_times']) if stats['render_times'] else 0,
                    'success_rate': stats['success_count'] / max(stats['success_count'] + stats['error_count'], 1),
                    'total_tasks': stats['success_count'] + stats['error_count']
                }
                for worker_id, stats in self.worker_performance.items()
            }
        }
        
        return stats
    
    def should_scale_up(self) -> bool:
        """Determine if workers should be scaled up."""
        avg_wait = self.get_average_queue_wait()
        return avg_wait > self.config.scale_up_threshold
    
    def should_scale_down(self) -> bool:
        """Determine if workers should be scaled down."""
        avg_wait = self.get_average_queue_wait()
        return avg_wait < self.config.scale_down_threshold
    
    def cleanup(self):
        """Cleanup profiler resources."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

class RenderCache:
    """Intelligent render cache with LRU eviction."""
    
    def __init__(self, config: EnhancedRenderConfig):
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.logger = logging.getLogger(__name__)
        
    def _generate_cache_key(self, task: EnhancedRenderTask) -> str:
        """Generate cache key for render task."""
        # Create hash of scene data and render config
        cache_data = {
            'scene': task.scene_data,
            'config': task.render_config,
            'proxy': task.use_proxy
        }
        data_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, task: EnhancedRenderTask) -> Optional[RenderResult]:
        """Get cached render result."""
        if not self.config.enable_render_cache:
            return None
        
        cache_key = self._generate_cache_key(task)
        
        if cache_key in self.cache:
            # Update access statistics
            self.access_times[cache_key] = time.time()
            self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
            
            # Return cached result
            cached_result = self.cache[cache_key]
            cached_result.was_cached = True
            cached_result.task_id = task.task_id  # Update task ID
            
            return cached_result
        
        return None
    
    def put(self, task: EnhancedRenderTask, result: RenderResult):
        """Cache render result."""
        if not self.config.enable_render_cache or not result.success:
            return
        
        cache_key = self._generate_cache_key(task)
        
        # Check cache size limit
        if len(self.cache) >= self.config.cache_size:
            self._evict_lru()
        
        # Store result
        self.cache[cache_key] = result
        self.access_times[cache_key] = time.time()
        self.access_counts[cache_key] = 1
    
    def _evict_lru(self):
        """Evict least recently used cache entry."""
        if not self.access_times:
            return
        
        # Find LRU entry
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from all tracking structures
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.access_counts[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'capacity': self.config.cache_size,
            'utilization': len(self.cache) / self.config.cache_size,
            'total_accesses': sum(self.access_counts.values()),
            'unique_keys': len(self.cache)
        }

class ProxyRenderer:
    """Low-resolution proxy renderer for early training."""
    
    def __init__(self, config: EnhancedRenderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Proxy render settings
        self.proxy_width = int(config.width * config.proxy_scale_factor)
        self.proxy_height = int(config.height * config.proxy_scale_factor)
        
    def should_use_proxy(self, episode: int, task: EnhancedRenderTask) -> bool:
        """Determine if proxy renderer should be used."""
        if not self.config.enable_proxy_renderer:
            return False
        
        # Use proxy for early episodes or if explicitly requested
        return (episode < self.config.proxy_quality_threshold or 
                task.use_proxy or 
                task.priority > 1)  # Use proxy for lower priority tasks
    
    def render(self, task: EnhancedRenderTask) -> RenderResult:
        """Render using low-resolution proxy."""
        start_time = time.time()
        
        try:
            # Simulate fast proxy rendering
            # In real implementation, this would render at lower resolution
            time.sleep(0.01)  # Very fast proxy render
            
            render_time = time.time() - start_time
            
            return RenderResult(
                task_id=task.task_id,
                success=True,
                output_path=f"proxy_render_{task.task_id}.png",
                render_time=render_time,
                actual_render_time=render_time,
                was_proxy=True,
                worker_id="proxy",
                metadata={
                    'proxy_resolution': f"{self.proxy_width}x{self.proxy_height}",
                    'scale_factor': self.config.proxy_scale_factor
                }
            )
            
        except Exception as e:
            return RenderResult(
                task_id=task.task_id,
                success=False,
                error_message=f"Proxy render failed: {e}",
                was_proxy=True,
                worker_id="proxy"
            )

class RenderWorker:
    """Enhanced render worker with minimal setup overhead."""
    
    def __init__(self, worker_id: str, config: EnhancedRenderConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{worker_id}")
        
        # Pre-initialize resources to minimize setup overhead
        self.is_initialized = False
        self.render_context = None
        self.setup_time = 0.0
        
        # Performance tracking
        self.tasks_completed = 0
        self.total_render_time = 0.0
        
    def initialize(self):
        """Initialize worker resources once."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        
        try:
            # Pre-initialize rendering context
            self._setup_render_context()
            
            self.setup_time = time.time() - start_time
            self.is_initialized = True
            
            self.logger.info(f"Worker {self.worker_id} initialized in {self.setup_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} initialization failed: {e}")
            raise
    
    def _setup_render_context(self):
        """Setup rendering context with minimal overhead."""
        # Pre-allocate resources that can be reused
        self.render_context = {
            'initialized': True,
            'reusable_buffers': {},
            'cached_shaders': {},
            'temp_files': []
        }
    
    def render(self, task: EnhancedRenderTask) -> RenderResult:
        """Render task with optimized performance."""
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        actual_render_start = time.time()
        
        try:
            # Simulate rendering with reused context
            # In real implementation, this would use the pre-initialized context
            time.sleep(0.1)  # Simulate render time
            
            actual_render_time = time.time() - actual_render_start
            total_time = time.time() - start_time
            
            self.tasks_completed += 1
            self.total_render_time += actual_render_time
            
            return RenderResult(
                task_id=task.task_id,
                success=True,
                output_path=f"render_{task.task_id}.png",
                render_time=total_time,
                setup_time=0.0,  # Minimal since pre-initialized
                actual_render_time=actual_render_time,
                worker_id=self.worker_id,
                metadata={
                    'worker_tasks_completed': self.tasks_completed,
                    'worker_avg_render_time': self.total_render_time / self.tasks_completed
                }
            )
            
        except Exception as e:
            return RenderResult(
                task_id=task.task_id,
                success=False,
                error_message=f"Render failed: {e}",
                worker_id=self.worker_id
            )
    
    def cleanup(self):
        """Cleanup worker resources."""
        if self.render_context:
            # Cleanup temporary files and resources
            for temp_file in self.render_context.get('temp_files', []):
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except:
                    pass
        
        self.is_initialized = False

class DynamicWorkerPool:
    """Dynamic worker pool with intelligent scaling."""
    
    def __init__(self, config: EnhancedRenderConfig, profiler: PerformanceProfiler):
        self.config = config
        self.profiler = profiler
        self.logger = logging.getLogger(__name__)
        
        # Worker management
        self.workers = {}
        self.worker_threads = {}
        self.task_queue = queue.PriorityQueue(maxsize=config.max_queue_size)
        self.result_queue = queue.Queue()
        
        # Scaling control
        self.last_scale_time = time.time()
        self.scaling_lock = threading.Lock()
        
        # Initialize with minimum workers
        self._scale_to(config.initial_workers)
        
        # Start scaling monitor
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
    
    def _scaling_loop(self):
        """Background loop for dynamic worker scaling."""
        while self.scaling_active:
            try:
                time.sleep(self.config.worker_scale_interval)
                
                with self.scaling_lock:
                    current_time = time.time()
                    
                    # Only scale if enough time has passed
                    if current_time - self.last_scale_time < self.config.worker_scale_interval:
                        continue
                    
                    current_workers = len(self.workers)
                    
                    if self.profiler.should_scale_up() and current_workers < self.config.max_workers:
                        new_count = min(current_workers + 1, self.config.max_workers)
                        self._scale_to(new_count)
                        self.logger.info(f"Scaled up to {new_count} workers")
                        
                    elif self.profiler.should_scale_down() and current_workers > self.config.min_workers:
                        new_count = max(current_workers - 1, self.config.min_workers)
                        self._scale_to(new_count)
                        self.logger.info(f"Scaled down to {new_count} workers")
                    
                    self.last_scale_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Scaling loop error: {e}")
    
    def _scale_to(self, target_count: int):
        """Scale worker pool to target count."""
        current_count = len(self.workers)
        
        if target_count > current_count:
            # Add workers
            for i in range(current_count, target_count):
                worker_id = f"worker_{i}"
                worker = RenderWorker(worker_id, self.config)
                
                # Start worker thread
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(worker,),
                    daemon=True
                )
                thread.start()
                
                self.workers[worker_id] = worker
                self.worker_threads[worker_id] = thread
                
        elif target_count < current_count:
            # Remove workers
            workers_to_remove = list(self.workers.keys())[target_count:]
            
            for worker_id in workers_to_remove:
                # Signal worker to stop
                self.task_queue.put((0, time.time(), None))  # Poison pill
                
                # Cleanup worker
                worker = self.workers.pop(worker_id)
                worker.cleanup()
                
                # Remove thread reference
                self.worker_threads.pop(worker_id)
    
    def _worker_loop(self, worker: RenderWorker):
        """Main loop for worker thread."""
        try:
            worker.initialize()
            
            while True:
                try:
                    # Get task from queue
                    priority, timestamp, task = self.task_queue.get(timeout=1.0)
                    
                    # Check for poison pill (shutdown signal)
                    if task is None:
                        break
                    
                    # Calculate queue wait time
                    queue_wait_time = time.time() - timestamp
                    
                    # Render task
                    result = worker.render(task)
                    result.queue_wait_time = queue_wait_time
                    
                    # Record metrics
                    self.profiler.record_task_metrics(result)
                    
                    # Put result
                    self.result_queue.put(result)
                    
                    # Execute callback if provided
                    if task.callback:
                        try:
                            task.callback(result)
                        except Exception as e:
                            self.logger.warning(f"Callback error: {e}")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Worker {worker.worker_id} error: {e}")
                    
        finally:
            worker.cleanup()
    
    def submit_task(self, task: EnhancedRenderTask):
        """Submit task to worker pool."""
        try:
            self.task_queue.put((task.priority, task.timestamp, task), timeout=self.config.queue_timeout)
        except queue.Full:
            raise RuntimeError("Render queue is full")
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[RenderResult]:
        """Get completed render result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.task_queue.qsize()
    
    def cleanup(self):
        """Cleanup worker pool."""
        self.scaling_active = False
        
        # Stop all workers
        for _ in range(len(self.workers)):
            try:
                self.task_queue.put((0, time.time(), None))  # Poison pill
            except:
                pass
        
        # Wait for scaling thread
        if self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5.0)
        
        # Cleanup workers
        for worker in self.workers.values():
            worker.cleanup()

class EnhancedAsyncRenderingPipeline:
    """
    Enhanced async rendering pipeline with all optimizations.
    """
    
    def __init__(self, config: Optional[EnhancedRenderConfig] = None):
        self.config = config or EnhancedRenderConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.profiler = PerformanceProfiler(self.config)
        self.cache = RenderCache(self.config)
        self.proxy_renderer = ProxyRenderer(self.config)
        self.worker_pool = DynamicWorkerPool(self.config, self.profiler)
        
        # Pipeline state
        self.is_running = True
        self.total_tasks = 0
        self.completed_tasks = 0
        self.current_episode = 0
        
        # Start performance logging
        if self.config.enable_profiling:
            self._start_performance_logging()
        
        self.logger.info("Enhanced async rendering pipeline initialized")
    
    def _start_performance_logging(self):
        """Start background performance logging."""
        def log_performance():
            while self.is_running:
                try:
                    time.sleep(self.config.log_performance_interval)
                    
                    stats = self.get_performance_stats()
                    self.logger.info(f"Pipeline Performance: {json.dumps(stats, indent=2)}")
                    
                except Exception as e:
                    self.logger.warning(f"Performance logging error: {e}")
        
        self.perf_thread = threading.Thread(target=log_performance, daemon=True)
        self.perf_thread.start()
    
    def render_async(self, 
                    scene_data: Dict[str, Any],
                    render_config: Optional[Dict[str, Any]] = None,
                    priority: int = 1,
                    use_proxy: bool = False,
                    callback: Optional[Callable] = None) -> str:
        """
        Submit async render task.
        
        Args:
            scene_data: Scene data for rendering
            render_config: Render configuration
            priority: Task priority (0=highest, 2=lowest)
            use_proxy: Force use of proxy renderer
            callback: Completion callback
            
        Returns:
            Task ID for tracking
        """
        task_id = f"task_{self.total_tasks}_{int(time.time() * 1000)}"
        self.total_tasks += 1
        
        # Create render task
        task = EnhancedRenderTask(
            task_id=task_id,
            scene_data=scene_data,
            render_config=render_config or {},
            priority=priority,
            use_proxy=use_proxy,
            callback=callback
        )
        
        # Check cache first
        cached_result = self.cache.get(task)
        if cached_result:
            if callback:
                callback(cached_result)
            return task_id
        
        # Check if should use proxy
        if self.proxy_renderer.should_use_proxy(self.current_episode, task):
            result = self.proxy_renderer.render(task)
            self.cache.put(task, result)
            
            if callback:
                callback(result)
            
            return task_id
        
        # Submit to worker pool
        self.worker_pool.submit_task(task)
        
        return task_id
    
    def render_batch(self, 
                    tasks: List[Dict[str, Any]],
                    priority: int = 1) -> List[str]:
        """
        Submit batch of render tasks.
        
        Args:
            tasks: List of task specifications
            priority: Batch priority
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for task_spec in tasks:
            task_id = self.render_async(
                scene_data=task_spec['scene_data'],
                render_config=task_spec.get('render_config'),
                priority=priority,
                use_proxy=task_spec.get('use_proxy', False)
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[RenderResult]:
        """Get next completed render result."""
        result = self.worker_pool.get_result(timeout)
        
        if result:
            self.completed_tasks += 1
            
            # Cache successful results
            if result.success and not result.was_cached:
                # Reconstruct task for caching (simplified)
                dummy_task = EnhancedRenderTask(
                    task_id=result.task_id,
                    scene_data={},  # Would need to store original data
                    render_config={}
                )
                self.cache.put(dummy_task, result)
        
        return result
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> List[RenderResult]:
        """Wait for specific tasks to complete."""
        results = []
        remaining_ids = set(task_ids)
        start_time = time.time()
        
        while remaining_ids and (timeout is None or time.time() - start_time < timeout):
            result = self.get_result(timeout=1.0)
            
            if result and result.task_id in remaining_ids:
                results.append(result)
                remaining_ids.remove(result.task_id)
        
        return results
    
    def set_episode(self, episode: int):
        """Update current episode for proxy rendering decisions."""
        self.current_episode = episode
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        profiler_stats = self.profiler.get_performance_stats()
        cache_stats = self.cache.get_stats()
        
        return {
            'pipeline': {
                'total_tasks': self.total_tasks,
                'completed_tasks': self.completed_tasks,
                'queue_size': self.worker_pool.get_queue_size(),
                'worker_count': len(self.worker_pool.workers),
                'current_episode': self.current_episode
            },
            'performance': profiler_stats,
            'cache': cache_stats
        }
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        self.is_running = False
        
        # Cleanup components
        self.worker_pool.cleanup()
        self.profiler.cleanup()
        
        self.logger.info("Enhanced rendering pipeline cleanup completed")

# Factory function for easy creation
def create_enhanced_pipeline(
    min_workers: int = 2,
    max_workers: int = 8,
    enable_proxy_renderer: bool = True,
    enable_render_cache: bool = True,
    enable_profiling: bool = True
) -> EnhancedAsyncRenderingPipeline:
    """
    Factory function to create enhanced rendering pipeline.
    
    Args:
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers
        enable_proxy_renderer: Enable low-res proxy rendering
        enable_render_cache: Enable render result caching
        enable_profiling: Enable performance profiling
        
    Returns:
        Configured EnhancedAsyncRenderingPipeline
    """
    config = EnhancedRenderConfig(
        min_workers=min_workers,
        max_workers=max_workers,
        enable_proxy_renderer=enable_proxy_renderer,
        enable_render_cache=enable_render_cache,
        enable_profiling=enable_profiling
    )
    
    return EnhancedAsyncRenderingPipeline(config)

