"""
Performance optimization utilities for dual CNN training.

This module provides memory profiling, batch processing optimizations,
and efficient tensor operations for the dual CNN architecture.
"""

import logging
import time
import gc
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from collections import deque
import threading
import os

logger = logging.getLogger(__name__)

# Try to import psutil, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, memory profiling will be limited")


@dataclass
class MemoryProfile:
    """Memory usage profile for training components."""
    component_name: str
    peak_memory_mb: float
    current_memory_mb: float
    gpu_memory_mb: float
    tensor_count: int
    largest_tensor_mb: float
    memory_growth_rate: float
    timestamp: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    batch_processing_time: float
    tensor_operation_time: float
    memory_allocation_time: float
    wave_storage_time: float
    attention_computation_time: float
    total_forward_pass_time: float
    memory_efficiency_score: float
    throughput_tokens_per_second: float


class MemoryProfiler:
    """Advanced memory profiler for dual CNN training."""
    
    def __init__(self, enable_gpu_profiling: bool = True):
        self.enable_gpu_profiling = enable_gpu_profiling
        self.profiles = {}
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory
        self.memory_timeline = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # GPU memory tracking
        self.gpu_available = self._check_gpu_availability()
        if self.gpu_available and enable_gpu_profiling:
            self._initialize_gpu_profiling()
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                result = {
                    'rss_mb': memory_info.rss / (1024**2),
                    'vms_mb': memory_info.vms / (1024**2),
                    'percent': process.memory_percent(),
                    'available_mb': psutil.virtual_memory().available / (1024**2)
                }
            else:
                # Fallback without psutil
                result = {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0, 'available_mb': 0.0}
            
            # Add GPU memory if available
            if self.gpu_available and self.enable_gpu_profiling:
                gpu_memory = self._get_gpu_memory()
                result.update(gpu_memory)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0, 'available_mb': 0.0}
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for profiling."""
        try:
            return len(tf.config.list_physical_devices('GPU')) > 0
        except Exception:
            return False
    
    def _initialize_gpu_profiling(self):
        """Initialize GPU memory profiling."""
        try:
            # Enable memory growth to avoid pre-allocation
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory profiling initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU profiling: {e}")
            self.gpu_available = False
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage."""
        try:
            if not self.gpu_available:
                return {'gpu_memory_mb': 0.0, 'gpu_utilization': 0.0}
            
            # Get GPU memory info
            gpu_memory_info = tf.config.experimental.get_memory_info('GPU:0')
            current_mb = gpu_memory_info['current'] / (1024**2)
            peak_mb = gpu_memory_info['peak'] / (1024**2)
            
            return {
                'gpu_memory_mb': current_mb,
                'gpu_peak_mb': peak_mb,
                'gpu_utilization': (current_mb / peak_mb * 100) if peak_mb > 0 else 0.0
            }
            
        except Exception as e:
            logger.debug(f"Failed to get GPU memory: {e}")
            return {'gpu_memory_mb': 0.0, 'gpu_utilization': 0.0}
    
    def profile_component(self, component_name: str) -> MemoryProfile:
        """Profile memory usage of a specific component."""
        with self._lock:
            current_memory = self._get_current_memory()
            
            # Calculate memory growth
            baseline_rss = self.baseline_memory.get('rss_mb', 0)
            current_rss = current_memory.get('rss_mb', 0)
            memory_growth = current_rss - baseline_rss
            
            # Count TensorFlow tensors
            tensor_count = len([obj for obj in gc.get_objects() 
                              if isinstance(obj, (tf.Tensor, tf.Variable))])
            
            # Find largest tensor
            largest_tensor_size = 0.0
            try:
                for obj in gc.get_objects():
                    if isinstance(obj, (tf.Tensor, tf.Variable)):
                        size_mb = obj.numpy().nbytes / (1024**2) if hasattr(obj, 'numpy') else 0
                        largest_tensor_size = max(largest_tensor_size, size_mb)
            except Exception:
                pass
            
            # Calculate growth rate
            growth_rate = 0.0
            if len(self.memory_timeline) > 1:
                recent_memories = [entry['rss_mb'] for entry in list(self.memory_timeline)[-10:]]
                if len(recent_memories) > 1:
                    growth_rate = (recent_memories[-1] - recent_memories[0]) / len(recent_memories)
            
            # Update timeline
            self.memory_timeline.append({
                'timestamp': time.time(),
                'rss_mb': current_rss,
                'component': component_name
            })
            
            # Update peak memory
            self.peak_memory = {
                key: max(self.peak_memory.get(key, 0), current_memory.get(key, 0))
                for key in current_memory.keys()
            }
            
            profile = MemoryProfile(
                component_name=component_name,
                peak_memory_mb=self.peak_memory.get('rss_mb', 0),
                current_memory_mb=current_rss,
                gpu_memory_mb=current_memory.get('gpu_memory_mb', 0),
                tensor_count=tensor_count,
                largest_tensor_mb=largest_tensor_size,
                memory_growth_rate=growth_rate,
                timestamp=time.time()
            )
            
            self.profiles[component_name] = profile
            return profile
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        current_memory = self._get_current_memory()
        
        return {
            'current_usage': current_memory,
            'peak_usage': self.peak_memory,
            'baseline_usage': self.baseline_memory,
            'memory_growth_mb': current_memory.get('rss_mb', 0) - self.baseline_memory.get('rss_mb', 0),
            'component_profiles': {name: profile.__dict__ for name, profile in self.profiles.items()},
            'timeline_length': len(self.memory_timeline),
            'gpu_available': self.gpu_available
        }
    
    def cleanup_memory(self) -> float:
        """Force garbage collection and return memory freed."""
        before_memory = self._get_current_memory().get('rss_mb', 0)
        
        # Force garbage collection
        gc.collect()
        
        # Clear TensorFlow session if needed
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        
        after_memory = self._get_current_memory().get('rss_mb', 0)
        memory_freed = before_memory - after_memory
        
        logger.info(f"Memory cleanup freed {memory_freed:.2f} MB")
        return memory_freed


class BatchOptimizer:
    """Optimized batch processing for large datasets."""
    
    def __init__(self, 
                 max_memory_mb: float = 1000.0,
                 adaptive_batch_size: bool = True,
                 prefetch_buffer_size: int = 2):
        self.max_memory_mb = max_memory_mb
        self.adaptive_batch_size = adaptive_batch_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.optimal_batch_size = None
        self.memory_profiler = MemoryProfiler()
        
        # Performance tracking
        self.batch_times = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
    
    def optimize_batch_size(self, 
                           dataset: tf.data.Dataset,
                           model_forward_fn: callable,
                           initial_batch_size: int = 32,
                           max_batch_size: int = 512) -> int:
        """Find optimal batch size through binary search."""
        logger.info("Optimizing batch size for memory and performance...")
        
        min_batch_size = 1
        max_batch_size = min(max_batch_size, 1024)  # Cap at reasonable limit
        optimal_size = initial_batch_size
        
        # Binary search for optimal batch size
        while min_batch_size <= max_batch_size:
            test_batch_size = (min_batch_size + max_batch_size) // 2
            
            try:
                # Test batch processing
                test_dataset = dataset.batch(test_batch_size).take(3)
                success, metrics = self._test_batch_size(test_dataset, model_forward_fn)
                
                if success and metrics['memory_mb'] <= self.max_memory_mb:
                    optimal_size = test_batch_size
                    min_batch_size = test_batch_size + 1
                    logger.debug(f"Batch size {test_batch_size} successful: "
                               f"{metrics['memory_mb']:.1f}MB, {metrics['time_per_sample']:.3f}s/sample")
                else:
                    max_batch_size = test_batch_size - 1
                    logger.debug(f"Batch size {test_batch_size} failed: "
                               f"memory={metrics.get('memory_mb', 'N/A')}MB")
                    
            except Exception as e:
                logger.debug(f"Batch size {test_batch_size} failed with error: {e}")
                max_batch_size = test_batch_size - 1
        
        self.optimal_batch_size = optimal_size
        logger.info(f"Optimal batch size determined: {optimal_size}")
        return optimal_size
    
    def _test_batch_size(self, test_dataset: tf.data.Dataset, model_forward_fn: callable) -> Tuple[bool, Dict]:
        """Test a specific batch size for memory and performance."""
        try:
            # Clear memory before test
            self.memory_profiler.cleanup_memory()
            
            start_memory = self.memory_profiler._get_current_memory()
            start_time = time.time()
            
            total_samples = 0
            for batch in test_dataset:
                batch_start = time.time()
                
                # Run forward pass
                _ = model_forward_fn(batch)
                
                batch_time = time.time() - batch_start
                total_samples += batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
                
                # Check memory usage
                current_memory = self.memory_profiler._get_current_memory()
                if current_memory.get('rss_mb', 0) > self.max_memory_mb:
                    return False, {'memory_mb': current_memory.get('rss_mb', 0)}
            
            total_time = time.time() - start_time
            end_memory = self.memory_profiler._get_current_memory()
            
            metrics = {
                'memory_mb': end_memory.get('rss_mb', 0) - start_memory.get('rss_mb', 0),
                'total_time': total_time,
                'time_per_sample': total_time / max(total_samples, 1),
                'samples_processed': total_samples
            }
            
            return True, metrics
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def create_optimized_dataset(self, 
                                dataset: tf.data.Dataset,
                                batch_size: Optional[int] = None,
                                shuffle_buffer_size: int = 10000,
                                num_parallel_calls: int = tf.data.AUTOTUNE) -> tf.data.Dataset:
        """Create optimized dataset with performance enhancements."""
        if batch_size is None:
            batch_size = self.optimal_batch_size or 32
        
        # Apply optimizations
        optimized_dataset = dataset
        
        # Shuffle with appropriate buffer size
        if shuffle_buffer_size > 0:
            optimized_dataset = optimized_dataset.shuffle(
                buffer_size=min(shuffle_buffer_size, 50000),  # Cap buffer size
                reshuffle_each_iteration=True
            )
        
        # Batch with drop remainder for consistent shapes
        optimized_dataset = optimized_dataset.batch(
            batch_size, 
            drop_remainder=True,
            num_parallel_calls=num_parallel_calls
        )
        
        # Prefetch for pipeline optimization
        optimized_dataset = optimized_dataset.prefetch(self.prefetch_buffer_size)
        
        # Cache small datasets in memory
        try:
            dataset_size = tf.data.experimental.cardinality(dataset).numpy()
            if dataset_size < 10000:  # Cache small datasets
                optimized_dataset = optimized_dataset.cache()
                logger.info("Dataset cached in memory for better performance")
        except Exception:
            pass  # Ignore if cardinality cannot be determined
        
        logger.info(f"Created optimized dataset with batch_size={batch_size}, "
                   f"prefetch={self.prefetch_buffer_size}")
        
        return optimized_dataset


class TensorOptimizer:
    """Efficient tensor operations for dual CNN coordination."""
    
    def __init__(self):
        self.mixed_precision_enabled = False
        self.xla_enabled = False
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup TensorFlow optimizations."""
        try:
            # Enable mixed precision if supported
            if self._check_mixed_precision_support():
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                self.mixed_precision_enabled = True
                logger.info("Mixed precision enabled for faster training")
            
            # Enable XLA compilation
            tf.config.optimizer.set_jit(True)
            self.xla_enabled = True
            logger.info("XLA compilation enabled")
            
        except Exception as e:
            logger.warning(f"Failed to enable some optimizations: {e}")
    
    def _check_mixed_precision_support(self) -> bool:
        """Check if mixed precision is supported."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
            
            # Check GPU compute capability
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            compute_capability = gpu_details.get('compute_capability')
            
            if compute_capability:
                major, minor = compute_capability
                return major >= 7 or (major == 6 and minor >= 1)
            
            return False
            
        except Exception:
            return False
    
    @tf.function(experimental_relax_shapes=True)
    def optimized_attention_computation(self, 
                                      query: tf.Tensor,
                                      key: tf.Tensor,
                                      value: tf.Tensor,
                                      mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Optimized multi-head attention computation."""
        # Use efficient attention implementation
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        
        # Compute attention scores efficiently
        scores = tf.matmul(query, key, transpose_b=True)
        
        # Scale scores
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_scores = scores / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_scores += (mask * -1e9)
        
        # Softmax with numerical stability
        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, value)
        
        return output
    
    @tf.function(experimental_relax_shapes=True)
    def optimized_wave_feature_extraction(self, 
                                        reservoir_states: tf.Tensor,
                                        attention_weights: tf.Tensor) -> tf.Tensor:
        """Optimized wave feature extraction from reservoir states."""
        # Efficient attention weighting
        # Average attention weights over heads for position-wise weighting
        position_weights = tf.reduce_mean(attention_weights, axis=[1, -1], keepdims=True)
        position_weights = tf.expand_dims(position_weights, axis=-1)
        
        # Apply attention weighting efficiently
        weighted_states = reservoir_states * position_weights
        
        return weighted_states
    
    @tf.function(experimental_relax_shapes=True)
    def optimized_dual_cnn_forward(self,
                                 first_cnn_input: tf.Tensor,
                                 second_cnn_input: tf.Tensor,
                                 first_cnn: tf.keras.Model,
                                 second_cnn: tf.keras.Model,
                                 coordination_weight: float = 0.3) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Optimized forward pass for dual CNN coordination."""
        # Parallel computation of both CNNs
        first_output = first_cnn(first_cnn_input, training=True)
        second_output = second_cnn(second_cnn_input, training=True)
        
        # Efficient weighted combination
        final_prediction_weight = 1.0 - coordination_weight
        combined_output = (coordination_weight * first_output + 
                          final_prediction_weight * second_output)
        
        return first_output, second_output, combined_output


class WaveStorageOptimizer:
    """Memory-optimized wave storage management."""
    
    def __init__(self, wave_storage):
        self.wave_storage = wave_storage
        self.compression_enabled = True
        self.adaptive_cleanup_enabled = True
        self.memory_threshold_mb = 500.0
        
        # Monitoring
        self.access_patterns = deque(maxlen=1000)
        self.cleanup_history = []
    
    def optimize_storage_memory(self) -> Dict[str, Any]:
        """Optimize wave storage memory usage."""
        stats_before = self.wave_storage.get_storage_stats()
        optimizations_applied = []
        
        # 1. Compress old waves
        if self.compression_enabled:
            compressed_count = self._compress_old_waves()
            if compressed_count > 0:
                optimizations_applied.append(f"compressed_{compressed_count}_waves")
        
        # 2. Adaptive cleanup based on access patterns
        if self.adaptive_cleanup_enabled:
            cleaned_count = self._adaptive_cleanup()
            if cleaned_count > 0:
                optimizations_applied.append(f"cleaned_{cleaned_count}_waves")
        
        # 3. Memory defragmentation
        defrag_savings = self._defragment_storage()
        if defrag_savings > 0:
            optimizations_applied.append(f"defragmented_{defrag_savings:.1f}MB")
        
        stats_after = self.wave_storage.get_storage_stats()
        
        return {
            'memory_before_mb': stats_before['memory_used_mb'],
            'memory_after_mb': stats_after['memory_used_mb'],
            'memory_saved_mb': stats_before['memory_used_mb'] - stats_after['memory_used_mb'],
            'optimizations_applied': optimizations_applied,
            'utilization_before': stats_before['utilization_percent'],
            'utilization_after': stats_after['utilization_percent']
        }
    
    def _compress_old_waves(self) -> int:
        """Compress old wave data to save memory."""
        compressed_count = 0
        try:
            # Use existing cleanup method as compression
            if hasattr(self.wave_storage, 'cleanup_old_waves'):
                keep_recent = max(50, self.wave_storage.window_size)
                compressed_count = self.wave_storage.cleanup_old_waves(keep_recent)
        except Exception as e:
            logger.warning(f"Wave compression failed: {e}")
        return compressed_count
    
    def _adaptive_cleanup(self) -> int:
        """Adaptive cleanup based on access patterns."""
        cleaned_count = 0
        try:
            if hasattr(self.wave_storage, 'cleanup_old_waves'):
                cleaned_count = self.wave_storage.cleanup_old_waves()
        except Exception as e:
            logger.warning(f"Adaptive cleanup failed: {e}")
        return cleaned_count
    
    def _defragment_storage(self) -> float:
        """Defragment storage to reduce memory fragmentation."""
        memory_saved = 0.0
        try:
            # Force garbage collection
            before_memory = 0.0
            if PSUTIL_AVAILABLE:
                before_memory = psutil.Process().memory_info().rss / (1024**2)
            gc.collect()
            if PSUTIL_AVAILABLE:
                after_memory = psutil.Process().memory_info().rss / (1024**2)
                memory_saved = before_memory - after_memory
        except Exception as e:
            logger.warning(f"Storage defragmentation failed: {e}")
        return memory_saved
    
    def track_access_pattern(self, position: int, operation: str):
        """Track wave access patterns for optimization."""
        self.access_patterns.append({
            'position': position,
            'operation': operation,
            'timestamp': time.time()
        })
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for storage optimization."""
        recommendations = []
        
        stats = self.wave_storage.get_storage_stats()
        
        # Memory usage recommendations
        if stats['utilization_percent'] > 90:
            recommendations.append("Consider increasing max_memory_mb or reducing window_size")
        
        if stats['memory_used_mb'] > self.memory_threshold_mb:
            recommendations.append("Enable more aggressive cleanup policies")
        
        # Access pattern recommendations
        if len(self.access_patterns) > 100:
            recent_accesses = list(self.access_patterns)[-100:]
            unique_positions = len(set(access['position'] for access in recent_accesses))
            
            if unique_positions < len(recent_accesses) * 0.3:
                recommendations.append("Consider reducing window_size due to low access diversity")
        
        return recommendations


class PerformanceMonitor:
    """Comprehensive performance monitoring for dual CNN training."""
    
    def __init__(self):
        self.memory_profiler = MemoryProfiler()
        self.batch_optimizer = BatchOptimizer()
        self.tensor_optimizer = TensorOptimizer()
        self.metrics_history = deque(maxlen=1000)
        self.start_time = time.time()
    
    def start_training_monitoring(self):
        """Start comprehensive training monitoring."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
    
    def record_batch_metrics(self, 
                           batch_size: int,
                           forward_time: float,
                           backward_time: float,
                           memory_usage: float,
                           wave_storage_ops: int = 0) -> PerformanceMetrics:
        """Record metrics for a training batch."""
        
        # Calculate derived metrics
        total_time = forward_time + backward_time
        tokens_per_second = batch_size / total_time if total_time > 0 else 0
        memory_efficiency = batch_size / memory_usage if memory_usage > 0 else 0
        
        metrics = PerformanceMetrics(
            batch_processing_time=total_time,
            tensor_operation_time=forward_time,
            memory_allocation_time=0.0,  # Would need specific measurement
            wave_storage_time=0.0,  # Would need specific measurement
            attention_computation_time=0.0,  # Would need specific measurement
            total_forward_pass_time=forward_time,
            memory_efficiency_score=memory_efficiency,
            throughput_tokens_per_second=tokens_per_second
        )
        
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'batch_size': batch_size,
            'memory_usage': memory_usage
        })
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'error': 'No metrics recorded'}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 batches
        
        # Calculate averages
        avg_batch_time = np.mean([m['metrics'].batch_processing_time for m in recent_metrics])
        avg_throughput = np.mean([m['metrics'].throughput_tokens_per_second for m in recent_metrics])
        avg_memory_efficiency = np.mean([m['metrics'].memory_efficiency_score for m in recent_metrics])
        
        # Memory summary
        memory_summary = self.memory_profiler.get_memory_summary()
        
        # Training duration
        training_duration = time.time() - self.start_time
        
        return {
            'training_duration_seconds': training_duration,
            'average_batch_time': avg_batch_time,
            'average_throughput_tokens_per_second': avg_throughput,
            'average_memory_efficiency': avg_memory_efficiency,
            'memory_summary': memory_summary,
            'total_batches_processed': len(self.metrics_history),
            'performance_trends': self._analyze_performance_trends(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.metrics_history) < 10:
            return {'insufficient_data': True}
        
        # Get recent metrics for trend analysis
        recent_metrics = list(self.metrics_history)[-50:]
        
        # Extract time series data
        batch_times = [m['metrics'].batch_processing_time for m in recent_metrics]
        throughputs = [m['metrics'].throughput_tokens_per_second for m in recent_metrics]
        memory_usage = [m['memory_usage'] for m in recent_metrics]
        
        # Calculate trends (simple linear regression slope)
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        
        return {
            'batch_time_trend': calculate_trend(batch_times),
            'throughput_trend': calculate_trend(throughputs),
            'memory_usage_trend': calculate_trend(memory_usage),
            'performance_stability': np.std(batch_times) / np.mean(batch_times) if batch_times else 0.0
        }
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify potential optimization opportunities."""
        opportunities = []
        
        if not self.metrics_history:
            return opportunities
        
        recent_metrics = list(self.metrics_history)[-20:]
        
        # Check for high memory usage
        avg_memory = np.mean([m['memory_usage'] for m in recent_metrics])
        if avg_memory > 1000:  # > 1GB
            opportunities.append("High memory usage detected - consider reducing batch size or enabling memory optimization")
        
        # Check for low throughput
        avg_throughput = np.mean([m['metrics'].throughput_tokens_per_second for m in recent_metrics])
        if avg_throughput < 10:  # < 10 tokens/second
            opportunities.append("Low throughput detected - consider batch size optimization or mixed precision training")
        
        # Check for memory efficiency
        avg_efficiency = np.mean([m['metrics'].memory_efficiency_score for m in recent_metrics])
        if avg_efficiency < 0.1:  # Low efficiency
            opportunities.append("Low memory efficiency - consider optimizing tensor operations or wave storage")
        
        # Check for performance instability
        batch_times = [m['metrics'].batch_processing_time for m in recent_metrics]
        if len(batch_times) > 5:
            stability = np.std(batch_times) / np.mean(batch_times)
            if stability > 0.3:  # High variance
                opportunities.append("Performance instability detected - consider profiling for bottlenecks")
        
        return opportunities
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report."""
        summary = self.get_performance_summary()
        
        if 'error' in summary:
            return "No performance data available for report generation."
        
        report = []
        report.append("=" * 60)
        report.append("Dual CNN Performance Optimization Report")
        report.append("=" * 60)
        report.append("")
        
        # Training Overview
        report.append("Training Overview:")
        report.append(f"  Duration: {summary['training_duration_seconds']:.1f} seconds")
        report.append(f"  Total Batches: {summary['total_batches_processed']}")
        report.append("")
        
        # Performance Metrics
        report.append("Performance Metrics:")
        report.append(f"  Average Batch Time: {summary['average_batch_time']:.3f} seconds")
        report.append(f"  Average Throughput: {summary['average_throughput_tokens_per_second']:.1f} tokens/second")
        report.append(f"  Memory Efficiency Score: {summary['average_memory_efficiency']:.3f}")
        report.append("")
        
        # Memory Usage
        memory_summary = summary['memory_summary']
        current_memory = memory_summary.get('current_usage', {})
        peak_memory = memory_summary.get('peak_usage', {})
        
        report.append("Memory Usage:")
        report.append(f"  Current Memory: {current_memory.get('rss_mb', 0):.1f} MB")
        report.append(f"  Peak Memory: {peak_memory.get('rss_mb', 0):.1f} MB")
        report.append(f"  Memory Growth: {memory_summary.get('memory_growth_mb', 0):.1f} MB")
        
        if memory_summary.get('gpu_available', False):
            report.append(f"  GPU Memory: {current_memory.get('gpu_memory_mb', 0):.1f} MB")
        report.append("")
        
        # Performance Trends
        trends = summary.get('performance_trends', {})
        if not trends.get('insufficient_data', False):
            report.append("Performance Trends:")
            report.append(f"  Batch Time Trend: {trends['batch_time_trend']:.6f} s/batch change")
            report.append(f"  Throughput Trend: {trends['throughput_trend']:.2f} tokens/s change")
            report.append(f"  Memory Trend: {trends['memory_usage_trend']:.2f} MB change")
            report.append(f"  Performance Stability: {trends['performance_stability']:.3f} (lower is better)")
            report.append("")
        
        # Optimization Opportunities
        opportunities = summary.get('optimization_opportunities', [])
        if opportunities:
            report.append("Optimization Recommendations:")
            for i, opportunity in enumerate(opportunities, 1):
                report.append(f"  {i}. {opportunity}")
            report.append("")
        
        # Component Profiles
        component_profiles = memory_summary.get('component_profiles', {})
        if component_profiles:
            report.append("Component Memory Profiles:")
            for component, profile in component_profiles.items():
                report.append(f"  {component}:")
                report.append(f"    Current: {profile['current_memory_mb']:.1f} MB")
                report.append(f"    Peak: {profile['peak_memory_mb']:.1f} MB")
                report.append(f"    Tensors: {profile['tensor_count']}")
            report.append("")
        
        report.append("=" * 60)
        report.append("End of Report")
        report.append("=" * 60)
        
        return "\n".join(report)