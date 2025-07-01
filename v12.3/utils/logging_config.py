"""
Logging Configuration for RL-LLM System

This module provides comprehensive logging setup and utilities for the RL training
system, including structured logging, performance monitoring, and experiment tracking.
"""

import logging
import logging.handlers
import sys
import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import traceback
from contextlib import contextmanager


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ExperimentLogger:
    """Enhanced logger for experiment tracking and monitoring."""
    
    def __init__(self, name: str, log_dir: Union[str, Path], 
                 level: int = logging.INFO, structured: bool = False):
        """
        Initialize experiment logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            structured: Whether to use structured JSON logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.structured = structured
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers()
        
        # Metrics tracking
        self.metrics = {}
        self.step_count = 0
        
    def _setup_console_handler(self):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.structured:
            formatter = StructuredFormatter()
        else:
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log levels."""
        # Main log file
        main_log = self.log_dir / f"{self.name}.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log, maxBytes=10*1024*1024, backupCount=5
        )
        
        if self.structured:
            main_formatter = StructuredFormatter()
        else:
            main_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        main_handler.setFormatter(main_formatter)
        self.logger.addHandler(main_handler)
        
        # Error log file
        error_log = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_formatter)
        self.logger.addHandler(error_handler)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """
        Log metrics with optional step tracking.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # Update stored metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
        
        # Log metrics
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - Metrics: {metric_str}", 
                        extra={'step': step, 'metrics': metrics})
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}", 
                        extra={'hyperparameters': params})
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information."""
        self.logger.info(f"System Info: {json.dumps(info, indent=2)}", 
                        extra={'system_info': info})
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.logger.info(f"Model Info: {json.dumps(model_info, indent=2)}", 
                        extra={'model_info': model_info})
    
    def save_metrics(self, filepath: Optional[Union[str, Path]] = None):
        """Save metrics to file."""
        if filepath is None:
            filepath = self.log_dir / f"{self.name}_metrics.json"
        
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Saved metrics to {filepath}")
    
    @contextmanager
    def log_duration(self, operation_name: str):
        """Context manager to log operation duration."""
        start_time = datetime.now()
        self.logger.info(f"Starting {operation_name}")
        
        try:
            yield
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Completed {operation_name} in {duration:.2f}s", 
                           extra={'operation': operation_name, 'duration': duration})
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed {operation_name} after {duration:.2f}s: {e}", 
                            extra={'operation': operation_name, 'duration': duration, 'error': str(e)})
            raise


def setup_logging(config: Dict[str, Any]) -> ExperimentLogger:
    """
    Setup logging based on configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured experiment logger
    """
    log_config = config.get('logging', {})
    
    # Extract configuration
    name = log_config.get('name', 'rl_experiment')
    log_dir = log_config.get('log_dir', './logs')
    level_str = log_config.get('level', 'INFO')
    structured = log_config.get('structured', False)
    
    # Convert level string to logging constant
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # Create experiment logger
    exp_logger = ExperimentLogger(name, log_dir, level, structured)
    
    # Set root logger level to avoid duplicate logs
    logging.getLogger().setLevel(level)
    
    return exp_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance monitor."""
        self.logger = logger
        self.timers = {}
        self.counters = {}
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self.timers[name] = datetime.now()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return duration."""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = (datetime.now() - self.timers[name]).total_seconds()
        del self.timers[name]
        
        self.logger.debug(f"Timer '{name}': {duration:.4f}s")
        return duration
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter."""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)
    
    def reset_counter(self, name: str):
        """Reset a counter to zero."""
        self.counters[name] = 0
    
    def log_counters(self):
        """Log all counter values."""
        if self.counters:
            counter_str = ", ".join([f"{k}={v}" for k, v in self.counters.items()])
            self.logger.info(f"Counters: {counter_str}")
    
    @contextmanager
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)


# Global logger instance
_global_logger = None


def init_global_logger(config: Dict[str, Any]) -> ExperimentLogger:
    """Initialize global logger."""
    global _global_logger
    _global_logger = setup_logging(config)
    return _global_logger


def get_global_logger() -> Optional[ExperimentLogger]:
    """Get global logger instance."""
    return _global_logger


# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message to global logger."""
    if _global_logger:
        _global_logger.logger.info(message, extra=kwargs)


def log_error(message: str, **kwargs):
    """Log error message to global logger."""
    if _global_logger:
        _global_logger.logger.error(message, extra=kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message to global logger."""
    if _global_logger:
        _global_logger.logger.warning(message, extra=kwargs)


def log_debug(message: str, **kwargs):
    """Log debug message to global logger."""
    if _global_logger:
        _global_logger.logger.debug(message, extra=kwargs)


# Example configuration
DEFAULT_LOGGING_CONFIG = {
    'logging': {
        'name': 'rl_experiment',
        'log_dir': './logs',
        'level': 'INFO',
        'structured': False
    }
}

