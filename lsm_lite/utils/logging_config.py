"""
Enhanced logging configuration for LSM Lite with better debugging support.

This module provides comprehensive logging setup with different levels,
formatters, and handlers for better debugging and monitoring.
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json


class LSMFormatter(logging.Formatter):
    """Custom formatter for LSM Lite with enhanced information."""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        
        # Color codes for different log levels
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        # Base format
        base_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        
        if self.include_context:
            base_format = '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s'
        
        super().__init__(base_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """Format log record with colors and additional context."""
        # Add color for console output
        if hasattr(record, 'levelname') and record.levelname in self.colors:
            colored_levelname = f"{self.colors[record.levelname]}{record.levelname}{self.colors['RESET']}"
            record.levelname = colored_levelname
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            record.memory_mb = f"{memory_mb:.1f}MB"
        except:
            record.memory_mb = "N/A"
        
        # Add component context if available
        if hasattr(record, 'component'):
            record.name = f"{record.name}[{record.component}]"
        
        formatted = super().format(record)
        
        # Add memory info to the end
        if self.include_context:
            formatted += f" | Mem: {record.memory_mb}"
        
        return formatted


class ComponentFilter(logging.Filter):
    """Filter logs by component for focused debugging."""
    
    def __init__(self, components: Optional[list] = None, exclude_components: Optional[list] = None):
        super().__init__()
        self.components = components or []
        self.exclude_components = exclude_components or []
    
    def filter(self, record):
        """Filter based on component inclusion/exclusion."""
        logger_name = record.name.lower()
        
        # Check exclusions first
        if self.exclude_components:
            for exclude in self.exclude_components:
                if exclude.lower() in logger_name:
                    return False
        
        # Check inclusions
        if self.components:
            for component in self.components:
                if component.lower() in logger_name:
                    return True
            return False  # Not in included components
        
        return True  # No filtering applied


class PerformanceLogger:
    """Logger for performance metrics and timing information."""
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(f"lsm_lite.{name}")
        self.timers = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        import time
        self.timers[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str, log_level: int = logging.INFO):
        """End timing an operation and log the duration."""
        import time
        if operation in self.timers:
            duration = time.time() - self.timers[operation]
            self.logger.log(log_level, f"Operation '{operation}' completed in {duration:.3f}s")
            del self.timers[operation]
            return duration
        else:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return None
    
    def log_memory_usage(self, operation: str = "current"):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)
            memory_percent = process.memory_percent()
            
            self.logger.info(f"Memory usage for {operation}: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
            
            return {
                'memory_mb': memory_mb,
                'memory_percent': memory_percent
            }
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return None
    
    def log_gpu_usage(self):
        """Log GPU usage if available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                for i, gpu in enumerate(gpus):
                    memory_info = tf.config.experimental.get_memory_info(gpu.name)
                    current_mb = memory_info['current'] / (1024**2)
                    peak_mb = memory_info['peak'] / (1024**2)
                    
                    self.logger.info(f"GPU {i} memory: {current_mb:.1f}MB current, {peak_mb:.1f}MB peak")
            else:
                self.logger.info("No GPUs detected")
                
        except Exception as e:
            self.logger.debug(f"GPU monitoring failed: {e}")


class ErrorTracker:
    """Track and analyze errors for debugging."""
    
    def __init__(self, name: str = "error_tracker"):
        self.logger = logging.getLogger(f"lsm_lite.{name}")
        self.error_counts = {}
        self.error_history = []
        self.max_history = 100
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context and tracking."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        
        self.error_history.append(error_entry)
        
        # Maintain history size
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # Log the error
        context_str = f" | Context: {json.dumps(context)}" if context else ""
        self.logger.error(f"{error_type}: {error_message}{context_str}")
        
        # Log patterns if this error type is frequent
        if self.error_counts[error_type] > 3:
            self.logger.warning(f"Frequent error detected: {error_type} occurred {self.error_counts[error_type]} times")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of tracked errors."""
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': self.error_history[-10:],  # Last 10 errors
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }


def setup_logging(level: str = "INFO",
                 log_file: Optional[str] = None,
                 include_context: bool = True,
                 components: Optional[list] = None,
                 exclude_components: Optional[list] = None,
                 enable_performance_logging: bool = True,
                 enable_error_tracking: bool = True) -> Dict[str, Any]:
    """
    Setup comprehensive logging for LSM Lite.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        include_context: Whether to include detailed context in logs
        components: List of components to include in logging (None for all)
        exclude_components: List of components to exclude from logging
        enable_performance_logging: Whether to enable performance logging
        enable_error_tracking: Whether to enable error tracking
        
    Returns:
        Dictionary with logger instances and configuration
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger for LSM Lite
    root_logger = logging.getLogger('lsm_lite')
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = LSMFormatter(include_context=include_context)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Add component filter if specified
    if components or exclude_components:
        component_filter = ComponentFilter(components, exclude_components)
        console_handler.addFilter(component_filter)
    
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    file_handler = None
    if log_file:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Use rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB max, 5 backups
            )
            file_handler.setLevel(numeric_level)
            
            # File logs don't need colors
            file_formatter = LSMFormatter(include_context=include_context)
            file_handler.setFormatter(file_formatter)
            
            if components or exclude_components:
                file_handler.addFilter(component_filter)
            
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Failed to setup file logging: {e}")
    
    # Setup specialized loggers
    loggers = {'root': root_logger}
    
    if enable_performance_logging:
        perf_logger = PerformanceLogger()
        loggers['performance'] = perf_logger
    
    if enable_error_tracking:
        error_tracker = ErrorTracker()
        loggers['error_tracker'] = error_tracker
    
    # Log setup completion
    root_logger.info(f"Logging setup completed - Level: {level}, File: {log_file or 'None'}")
    
    if components:
        root_logger.info(f"Logging filtered to components: {components}")
    
    if exclude_components:
        root_logger.info(f"Logging excluding components: {exclude_components}")
    
    return {
        'loggers': loggers,
        'level': level,
        'log_file': log_file,
        'handlers': {
            'console': console_handler,
            'file': file_handler
        }
    }


def get_debug_logger(component: str) -> logging.Logger:
    """Get a debug logger for a specific component."""
    logger = logging.getLogger(f'lsm_lite.{component}')
    
    # Add component context to all log records
    class ComponentAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f"[{self.extra['component']}] {msg}", kwargs
    
    return ComponentAdapter(logger, {'component': component})


def log_system_info():
    """Log system information for debugging."""
    logger = logging.getLogger('lsm_lite.system')
    
    try:
        import platform
        import sys
        import tensorflow as tf
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"TensorFlow: {tf.__version__}")
        
        # GPU information
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPUs available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.name}")
        else:
            logger.info("No GPUs detected")
        
        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            logger.info(f"System Memory: {memory.total / (1024**3):.1f}GB total, "
                       f"{memory.available / (1024**3):.1f}GB available")
        except ImportError:
            logger.info("Memory information unavailable (psutil not installed)")
        
        logger.info("=== End System Information ===")
        
    except Exception as e:
        logger.error(f"Failed to log system information: {e}")


# Global instances for easy access
performance_logger = None
error_tracker = None


def get_performance_logger() -> Optional[PerformanceLogger]:
    """Get the global performance logger instance."""
    return performance_logger


def get_error_tracker() -> Optional[ErrorTracker]:
    """Get the global error tracker instance."""
    return error_tracker


# Context manager for timing operations
class TimedOperation:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: Optional[PerformanceLogger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_performance_logger()
        self.start_time = None
    
    def __enter__(self):
        if self.logger:
            self.logger.start_timer(self.operation_name)
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.end_timer(self.operation_name)
        elif self.start_time:
            import time
            duration = time.time() - self.start_time
            print(f"Operation '{self.operation_name}' completed in {duration:.3f}s")


# Initialize global loggers when module is imported
def _initialize_global_loggers():
    """Initialize global logger instances."""
    global performance_logger, error_tracker
    
    if performance_logger is None:
        performance_logger = PerformanceLogger()
    
    if error_tracker is None:
        error_tracker = ErrorTracker()


# Initialize on import
_initialize_global_loggers()