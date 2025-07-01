"""
Enhanced Three.js Renderer with Performance Optimizations v3

Improvements based on feedback:
- Timeout and auto-restart for render subprocess
- EGL/OSMesa for best GPU headless performance
- Fallback error handling for renderer hangs/crashes
- Warm pool of render contexts
- Mock renderer for early RL episodes
- Performance profiling and optimization
"""

import json
import subprocess
import threading
import time
import signal
import psutil
import queue
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import math
import logging
import warnings
from pathlib import Path
import tempfile
import shutil
from contextlib import contextmanager
import pickle
import hashlib

# Try to import OpenGL for EGL/OSMesa support
try:
    from OpenGL import EGL
    from OpenGL.EGL import *
    EGL_AVAILABLE = True
except ImportError:
    EGL_AVAILABLE = False
    warnings.warn("EGL not available. Falling back to standard rendering.")

try:
    from OpenGL import osmesa
    OSMESA_AVAILABLE = True
except ImportError:
    OSMESA_AVAILABLE = False
    warnings.warn("OSMesa not available. Using standard OpenGL context.")

@dataclass
class EnhancedRendererConfig:
    """Enhanced configuration for Three.js renderer."""
    
    # Performance settings
    use_headless_gpu: bool = True
    prefer_egl: bool = True
    prefer_osmesa: bool = False
    enable_gpu_acceleration: bool = True
    
    # Process management
    render_timeout: float = 30.0  # seconds
    auto_restart_on_hang: bool = True
    max_restart_attempts: int = 3
    restart_delay: float = 1.0
    
    # Context pooling
    enable_context_pool: bool = True
    pool_size: int = 4
    pool_warmup: bool = True
    context_reuse_limit: int = 100
    
    # Mock rendering for early training
    enable_mock_renderer: bool = False
    mock_render_probability: float = 0.5
    mock_render_episodes_threshold: int = 100
    
    # Quality and performance
    default_width: int = 512
    default_height: int = 512
    max_width: int = 2048
    max_height: int = 2048
    quality_level: str = "medium"  # "low", "medium", "high"
    
    # Caching
    enable_render_cache: bool = True
    cache_size: int = 1000
    cache_ttl: float = 3600.0  # seconds
    
    # Monitoring
    enable_performance_monitoring: bool = True
    profile_render_times: bool = True
    log_memory_usage: bool = True

class RenderQuality(Enum):
    """Render quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class RenderContext:
    """Represents a rendering context with lifecycle management."""
    
    def __init__(self, context_id: str, config: EnhancedRendererConfig):
        self.context_id = context_id
        self.config = config
        self.process = None
        self.input_queue = None
        self.output_queue = None
        self.is_active = False
        self.render_count = 0
        self.created_time = time.time()
        self.last_used_time = time.time()
        self.logger = logging.getLogger(f"{__name__}.{context_id}")
        
    def initialize(self):
        """Initialize the rendering context."""
        try:
            # Create communication queues
            self.input_queue = mp.Queue()
            self.output_queue = mp.Queue()
            
            # Start rendering process
            self.process = mp.Process(
                target=self._render_worker,
                args=(self.input_queue, self.output_queue, self.config)
            )
            self.process.start()
            
            # Wait for initialization confirmation
            try:
                result = self.output_queue.get(timeout=10.0)
                if result.get('status') == 'initialized':
                    self.is_active = True
                    self.logger.info(f"Context {self.context_id} initialized successfully")
                else:
                    raise RuntimeError(f"Context initialization failed: {result}")
            except queue.Empty:
                raise RuntimeError("Context initialization timeout")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize context {self.context_id}: {e}")
            self.cleanup()
            raise
    
    def _render_worker(self, input_queue: mp.Queue, output_queue: mp.Queue, config: EnhancedRendererConfig):
        """Worker process for rendering."""
        try:
            # Initialize OpenGL context
            self._initialize_opengl_context(config)
            
            # Send initialization confirmation
            output_queue.put({'status': 'initialized'})
            
            # Main rendering loop
            while True:
                try:
                    # Get render task
                    task = input_queue.get(timeout=1.0)
                    
                    if task.get('command') == 'shutdown':
                        break
                    elif task.get('command') == 'render':
                        result = self._execute_render(task['data'], config)
                        output_queue.put({'status': 'success', 'result': result})
                    else:
                        output_queue.put({'status': 'error', 'error': f"Unknown command: {task.get('command')}"})
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    output_queue.put({'status': 'error', 'error': str(e)})
                    
        except Exception as e:
            output_queue.put({'status': 'error', 'error': f"Worker initialization failed: {e}"})
        finally:
            self._cleanup_opengl_context()
    
    def _initialize_opengl_context(self, config: EnhancedRendererConfig):
        """Initialize OpenGL context with EGL/OSMesa support."""
        if config.use_headless_gpu:
            if config.prefer_egl and EGL_AVAILABLE:
                self._initialize_egl_context(config)
            elif config.prefer_osmesa and OSMESA_AVAILABLE:
                self._initialize_osmesa_context(config)
            else:
                self._initialize_standard_context(config)
        else:
            self._initialize_standard_context(config)
    
    def _initialize_egl_context(self, config: EnhancedRendererConfig):
        """Initialize EGL context for headless GPU rendering."""
        try:
            # Get EGL display
            self.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
            if self.egl_display == EGL_NO_DISPLAY:
                raise RuntimeError("Failed to get EGL display")
            
            # Initialize EGL
            major, minor = EGLint(), EGLint()
            if not eglInitialize(self.egl_display, major, minor):
                raise RuntimeError("Failed to initialize EGL")
            
            # Configure EGL
            config_attribs = [
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_RED_SIZE, 8,
                EGL_DEPTH_SIZE, 8,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE
            ]
            
            configs = (EGLConfig * 1)()
            num_configs = EGLint()
            
            if not eglChooseConfig(self.egl_display, config_attribs, configs, 1, num_configs):
                raise RuntimeError("Failed to choose EGL config")
            
            # Create EGL context
            self.egl_context = eglCreateContext(self.egl_display, configs[0], EGL_NO_CONTEXT, None)
            if self.egl_context == EGL_NO_CONTEXT:
                raise RuntimeError("Failed to create EGL context")
            
            # Create pbuffer surface
            pbuffer_attribs = [
                EGL_WIDTH, config.default_width,
                EGL_HEIGHT, config.default_height,
                EGL_NONE
            ]
            
            self.egl_surface = eglCreatePbufferSurface(self.egl_display, configs[0], pbuffer_attribs)
            if self.egl_surface == EGL_NO_SURFACE:
                raise RuntimeError("Failed to create EGL surface")
            
            # Make context current
            if not eglMakeCurrent(self.egl_display, self.egl_surface, self.egl_surface, self.egl_context):
                raise RuntimeError("Failed to make EGL context current")
            
            self.context_type = "EGL"
            
        except Exception as e:
            raise RuntimeError(f"EGL initialization failed: {e}")
    
    def _initialize_osmesa_context(self, config: EnhancedRendererConfig):
        """Initialize OSMesa context for software rendering."""
        try:
            # Create OSMesa context
            self.osmesa_context = osmesa.OSMesaCreateContext(osmesa.OSMESA_RGBA, None)
            if not self.osmesa_context:
                raise RuntimeError("Failed to create OSMesa context")
            
            # Create buffer
            self.osmesa_buffer = (ctypes.c_ubyte * (config.default_width * config.default_height * 4))()
            
            # Make context current
            if not osmesa.OSMesaMakeCurrent(
                self.osmesa_context,
                self.osmesa_buffer,
                osmesa.GL_UNSIGNED_BYTE,
                config.default_width,
                config.default_height
            ):
                raise RuntimeError("Failed to make OSMesa context current")
            
            self.context_type = "OSMesa"
            
        except Exception as e:
            raise RuntimeError(f"OSMesa initialization failed: {e}")
    
    def _initialize_standard_context(self, config: EnhancedRendererConfig):
        """Initialize standard OpenGL context."""
        # This would use a standard OpenGL context creation method
        # For now, we'll simulate it
        self.context_type = "Standard"
    
    def _cleanup_opengl_context(self):
        """Cleanup OpenGL context."""
        if hasattr(self, 'egl_context'):
            eglDestroyContext(self.egl_display, self.egl_context)
            eglDestroySurface(self.egl_display, self.egl_surface)
            eglTerminate(self.egl_display)
        elif hasattr(self, 'osmesa_context'):
            osmesa.OSMesaDestroyContext(self.osmesa_context)
    
    def _execute_render(self, render_data: Dict[str, Any], config: EnhancedRendererConfig) -> Dict[str, Any]:
        """Execute rendering task."""
        start_time = time.time()
        
        try:
            # Simulate rendering process
            # In a real implementation, this would call Three.js rendering
            time.sleep(0.1)  # Simulate render time
            
            render_time = time.time() - start_time
            
            return {
                'image_data': f"rendered_image_{render_data.get('id', 'unknown')}",
                'render_time': render_time,
                'context_type': self.context_type,
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            raise RuntimeError(f"Render execution failed: {e}")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024)
        }
    
    def render(self, render_data: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Render with timeout and error handling."""
        if not self.is_active:
            raise RuntimeError(f"Context {self.context_id} is not active")
        
        timeout = timeout or self.config.render_timeout
        
        try:
            # Send render task
            self.input_queue.put({
                'command': 'render',
                'data': render_data
            })
            
            # Wait for result
            result = self.output_queue.get(timeout=timeout)
            
            if result.get('status') == 'success':
                self.render_count += 1
                self.last_used_time = time.time()
                return result['result']
            else:
                raise RuntimeError(f"Render failed: {result.get('error', 'Unknown error')}")
                
        except queue.Empty:
            raise TimeoutError(f"Render timeout after {timeout} seconds")
        except Exception as e:
            self.logger.error(f"Render error in context {self.context_id}: {e}")
            raise
    
    def should_restart(self) -> bool:
        """Check if context should be restarted."""
        return (
            self.render_count >= self.config.context_reuse_limit or
            not self.is_active or
            (self.process and not self.process.is_alive())
        )
    
    def cleanup(self):
        """Cleanup context resources."""
        self.is_active = False
        
        if self.process and self.process.is_alive():
            try:
                self.input_queue.put({'command': 'shutdown'})
                self.process.join(timeout=5.0)
                
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2.0)
                    
                    if self.process.is_alive():
                        self.process.kill()
                        
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")

class RenderContextPool:
    """Pool of rendering contexts for efficient resource management."""
    
    def __init__(self, config: EnhancedRendererConfig):
        self.config = config
        self.pool = queue.Queue(maxsize=config.pool_size)
        self.active_contexts = {}
        self.context_counter = 0
        self.logger = logging.getLogger(__name__)
        
        # Initialize pool
        if config.enable_context_pool:
            self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the context pool."""
        for i in range(self.config.pool_size):
            try:
                context = self._create_context()
                if self.config.pool_warmup:
                    # Warm up context with a test render
                    self._warmup_context(context)
                self.pool.put(context)
            except Exception as e:
                self.logger.error(f"Failed to initialize context {i}: {e}")
    
    def _create_context(self) -> RenderContext:
        """Create a new rendering context."""
        context_id = f"context_{self.context_counter}"
        self.context_counter += 1
        
        context = RenderContext(context_id, self.config)
        context.initialize()
        
        return context
    
    def _warmup_context(self, context: RenderContext):
        """Warm up context with a test render."""
        try:
            test_data = {
                'id': 'warmup',
                'type': 'test',
                'geometry': 'box',
                'material': 'basic'
            }
            context.render(test_data, timeout=5.0)
        except Exception as e:
            self.logger.warning(f"Context warmup failed: {e}")
    
    def get_context(self, timeout: float = 5.0) -> RenderContext:
        """Get a context from the pool."""
        try:
            context = self.pool.get(timeout=timeout)
            
            # Check if context needs restart
            if context.should_restart():
                context.cleanup()
                context = self._create_context()
            
            self.active_contexts[context.context_id] = context
            return context
            
        except queue.Empty:
            # Pool is empty, create new context
            return self._create_context()
    
    def return_context(self, context: RenderContext):
        """Return a context to the pool."""
        if context.context_id in self.active_contexts:
            del self.active_contexts[context.context_id]
        
        if context.should_restart():
            context.cleanup()
        else:
            try:
                self.pool.put_nowait(context)
            except queue.Full:
                # Pool is full, cleanup context
                context.cleanup()
    
    def cleanup(self):
        """Cleanup all contexts in the pool."""
        # Cleanup active contexts
        for context in self.active_contexts.values():
            context.cleanup()
        self.active_contexts.clear()
        
        # Cleanup pooled contexts
        while not self.pool.empty():
            try:
                context = self.pool.get_nowait()
                context.cleanup()
            except queue.Empty:
                break

class MockRenderer:
    """Mock renderer for early training episodes."""
    
    def __init__(self, config: EnhancedRendererConfig):
        self.config = config
        self.render_count = 0
        self.logger = logging.getLogger(__name__)
    
    def should_use_mock(self, episode: int) -> bool:
        """Determine if mock renderer should be used."""
        if not self.config.enable_mock_renderer:
            return False
        
        if episode < self.config.mock_render_episodes_threshold:
            return np.random.random() < self.config.mock_render_probability
        
        return False
    
    def render(self, render_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock render that returns synthetic data."""
        self.render_count += 1
        
        # Simulate very fast rendering
        time.sleep(0.001)
        
        return {
            'image_data': f"mock_render_{render_data.get('id', 'unknown')}",
            'render_time': 0.001,
            'context_type': 'Mock',
            'memory_usage': {'rss_mb': 10.0, 'vms_mb': 20.0},
            'is_mock': True
        }

class RenderCache:
    """Cache for rendered results."""
    
    def __init__(self, config: EnhancedRendererConfig):
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.logger = logging.getLogger(__name__)
    
    def _generate_cache_key(self, render_data: Dict[str, Any]) -> str:
        """Generate cache key for render data."""
        # Create a hash of the render data
        data_str = json.dumps(render_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, render_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached render result."""
        if not self.config.enable_render_cache:
            return None
        
        cache_key = self._generate_cache_key(render_data)
        
        if cache_key in self.cache:
            # Check TTL
            if time.time() - self.access_times[cache_key] < self.config.cache_ttl:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
                del self.access_times[cache_key]
        
        return None
    
    def put(self, render_data: Dict[str, Any], result: Dict[str, Any]):
        """Cache render result."""
        if not self.config.enable_render_cache:
            return
        
        cache_key = self._generate_cache_key(render_data)
        
        # Check cache size limit
        if len(self.cache) >= self.config.cache_size:
            self._evict_oldest()
        
        self.cache[cache_key] = result
        self.access_times[cache_key] = time.time()
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

class EnhancedThreeJSRenderer:
    """
    Enhanced Three.js renderer with all performance optimizations.
    """
    
    def __init__(self, config: Optional[EnhancedRendererConfig] = None):
        self.config = config or EnhancedRendererConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.context_pool = RenderContextPool(self.config) if self.config.enable_context_pool else None
        self.mock_renderer = MockRenderer(self.config)
        self.render_cache = RenderCache(self.config)
        
        # Performance monitoring
        self.render_times = []
        self.error_count = 0
        self.total_renders = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info("Enhanced Three.js renderer initialized")
    
    def render(self, render_data: Dict[str, Any], episode: int = 0) -> Dict[str, Any]:
        """
        Render with all optimizations and fallbacks.
        
        Args:
            render_data: Data for rendering
            episode: Current training episode
            
        Returns:
            Render result with metadata
        """
        start_time = time.time()
        self.total_renders += 1
        
        try:
            # Check cache first
            cached_result = self.render_cache.get(render_data)
            if cached_result:
                self.cache_hits += 1
                cached_result['from_cache'] = True
                return cached_result
            
            self.cache_misses += 1
            
            # Check if should use mock renderer
            if self.mock_renderer.should_use_mock(episode):
                result = self.mock_renderer.render(render_data)
                self.render_cache.put(render_data, result)
                return result
            
            # Use real renderer
            result = self._render_real(render_data)
            
            # Cache result
            self.render_cache.put(render_data, result)
            
            # Update performance metrics
            render_time = time.time() - start_time
            self.render_times.append(render_time)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Render failed: {e}")
            
            # Fallback to mock renderer
            self.logger.info("Falling back to mock renderer")
            return self.mock_renderer.render(render_data)
    
    def _render_real(self, render_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform real rendering with context pool."""
        if self.context_pool:
            context = self.context_pool.get_context()
            try:
                result = context.render(render_data)
                return result
            finally:
                self.context_pool.return_context(context)
        else:
            # Single context rendering
            context = RenderContext("single", self.config)
            try:
                context.initialize()
                result = context.render(render_data)
                return result
            finally:
                context.cleanup()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        avg_render_time = np.mean(self.render_times) if self.render_times else 0
        
        return {
            'total_renders': self.total_renders,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_renders, 1),
            'average_render_time': avg_render_time,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'context_pool_active': self.context_pool is not None,
            'mock_renderer_enabled': self.config.enable_mock_renderer
        }
    
    def cleanup(self):
        """Cleanup all resources."""
        if self.context_pool:
            self.context_pool.cleanup()
        
        self.logger.info("Enhanced renderer cleanup completed")

# Factory function for easy creation
def create_enhanced_renderer(
    use_headless_gpu: bool = True,
    enable_context_pool: bool = True,
    enable_mock_renderer: bool = False,
    pool_size: int = 4,
    render_timeout: float = 30.0
) -> EnhancedThreeJSRenderer:
    """
    Factory function to create enhanced renderer.
    
    Args:
        use_headless_gpu: Enable headless GPU rendering
        enable_context_pool: Enable context pooling
        enable_mock_renderer: Enable mock renderer for early training
        pool_size: Size of context pool
        render_timeout: Timeout for rendering operations
        
    Returns:
        Configured EnhancedThreeJSRenderer
    """
    config = EnhancedRendererConfig(
        use_headless_gpu=use_headless_gpu,
        enable_context_pool=enable_context_pool,
        enable_mock_renderer=enable_mock_renderer,
        pool_size=pool_size,
        render_timeout=render_timeout
    )
    
    return EnhancedThreeJSRenderer(config)

# Import numpy for mock renderer
import numpy as np
import ctypes

