"""
Dual CNN training coordination for LSM models.

This module implements the DualCNNTrainer class that coordinates training
of both CNNs with rolling wave output storage, providing progress tracking
and metrics collection for the dual CNN architecture.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
import threading
from collections import deque

# Import dependencies with fallback handling
try:
    from ..utils.config import DualCNNConfig
except ImportError:
    from lsm_lite.utils.config import DualCNNConfig

try:
    from ..utils.error_handling import (
        handle_lsm_error, ErrorContext, ComputationError, 
        MemoryError, global_error_handler, ValidationUtils
    )
except ImportError:
    from lsm_lite.utils.error_handling import (
        handle_lsm_error, ErrorContext, ComputationError,
        MemoryError, global_error_handler, ValidationUtils
    )

try:
    from ..core.dual_cnn_pipeline import DualCNNPipeline, DualCNNTrainingError
except ImportError:
    from lsm_lite.core.dual_cnn_pipeline import DualCNNPipeline, DualCNNTrainingError

try:
    from ..core.rolling_wave_storage import RollingWaveStorage, WaveStorageError
except ImportError:
    from lsm_lite.core.rolling_wave_storage import RollingWaveStorage, WaveStorageError

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.current_memory = 0.0
        
    def update(self):
        """Update current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            self.current_memory = process.memory_info().rss / (1024**2)  # MB
            self.peak_memory = max(self.peak_memory, self.current_memory)
        except ImportError:
            # Fallback if psutil not available
            self.current_memory = 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            "current_mb": self.current_memory,
            "peak_mb": self.peak_memory
        }


@dataclass
class TrainingProgress:
    """Progress tracking for dual CNN training."""
    current_epoch: int
    total_epochs: int
    first_cnn_loss: float
    second_cnn_loss: float
    combined_loss: float
    wave_storage_utilization: float
    attention_entropy: float
    estimated_time_remaining: float
    learning_rate: float
    batch_processed: int
    total_batches: int


@dataclass
class WaveOutput:
    """Structure for rolling wave output data."""
    sequence_position: int
    wave_features: tf.Tensor
    attention_weights: tf.Tensor
    timestamp: float
    confidence_score: float


class DualCNNTrainer:
    """
    Coordinated training orchestrator for dual CNN architecture.
    
    This class manages the training loop for both CNNs with rolling wave storage,
    implementing coordinated loss computation, progress tracking, and metrics collection.
    """
    
    @handle_lsm_error
    def __init__(self, pipeline: DualCNNPipeline, config: DualCNNConfig):
        """
        Initialize the dual CNN trainer with enhanced error handling.
        
        Args:
            pipeline: Initialized DualCNNPipeline instance
            config: Dual CNN configuration
        """
        self.pipeline = pipeline
        self.config = config
        
        # Enhanced pipeline validation
        if not pipeline.is_initialized():
            context = ErrorContext(
                component="DualCNNTrainer",
                operation="initialization",
                config_values={"pipeline_status": pipeline.get_component_status()}
            )
            raise ComputationError(
                "Pipeline must be fully initialized before training",
                context=context
            )
        
        # Check for fallback modes
        fallback_status = pipeline.get_fallback_status()
        if fallback_status["fallback_mode_enabled"]:
            logger.warning("Pipeline is in fallback mode, some features may be limited")
        
        # Training components
        self.first_cnn_optimizer = None
        self.second_cnn_optimizer = None
        self.loss_fn = None
        self.metrics = {}
        
        # Training state
        self.training_history = {
            'first_cnn_loss': [],
            'second_cnn_loss': [],
            'combined_loss': [],
            'wave_storage_utilization': [],
            'attention_entropy': [],
            'learning_rate': [],
            'epoch_times': [],
            'memory_usage': [],
            'gradient_norms': []
        }
        self.current_epoch = 0
        self.is_training = False
        self._stop_training = False
        
        # Error handling and recovery
        self._training_errors = []
        self._recovery_attempts = 0
        self._max_recovery_attempts = 3
        self._use_single_cnn_fallback = fallback_status["single_cnn_fallback"]
        
        # Progress tracking
        self.progress_callbacks = []
        self.last_progress = None
        
        # Performance monitoring
        self._batch_times = deque(maxlen=100)
        self._memory_monitor = MemoryMonitor()
        
        # Import performance optimizers
        try:
            from ..utils.performance_optimizer import (
                PerformanceMonitor, BatchOptimizer, TensorOptimizer, WaveStorageOptimizer
            )
            self.performance_monitor = PerformanceMonitor()
            self.batch_optimizer = BatchOptimizer()
            self.tensor_optimizer = TensorOptimizer()
            self.wave_storage_optimizer = WaveStorageOptimizer(pipeline.wave_storage) if pipeline.wave_storage else None
            self._optimizations_enabled = True
            logger.info("Performance optimizations enabled")
        except ImportError as e:
            logger.warning(f"Performance optimizations not available: {e}")
            self.performance_monitor = None
            self.batch_optimizer = None
            self.tensor_optimizer = None
            self.wave_storage_optimizer = None
            self._optimizations_enabled = False
        
        # Initialize training components with error handling
        try:
            self._setup_training_components()
            logger.info("DualCNNTrainer initialized for coordinated training")
        except Exception as e:
            context = ErrorContext(
                component="DualCNNTrainer",
                operation="setup_training_components",
                config_values=self._safe_config_dict(self.config)
            )
            enhanced_error = global_error_handler.handle_error(e, context)
            raise enhanced_error
    
    def _setup_training_components(self):
        """Setup optimizers, loss functions, and metrics."""
        # Separate optimizers for each CNN
        self.first_cnn_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        self.second_cnn_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0
        )
        
        # Loss function for next-token prediction
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        
        # Initialize metrics tracking
        self.metrics = {
            'first_cnn_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'second_cnn_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'combined_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'first_cnn_top5': tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            'second_cnn_top5': tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        }
        
        logger.info("Training components initialized")
    
    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        """
        Add a progress callback function.
        
        Args:
            callback: Function that receives TrainingProgress updates
        """
        self.progress_callbacks.append(callback)
    
    @handle_lsm_error
    def train_dual_cnn(self, 
                      training_data: List[str],
                      epochs: Optional[int] = None,
                      batch_size: Optional[int] = None,
                      validation_data: Optional[List[str]] = None,
                      validation_split: float = 0.1,
                      enable_recovery: bool = True) -> Dict[str, Any]:
        """
        Train both CNNs with rolling wave coordination and enhanced error handling.
        
        Args:
            training_data: List of training text strings
            epochs: Number of training epochs (uses config default if None)
            batch_size: Training batch size (uses config default if None)
            validation_data: Optional validation data
            validation_split: Validation split ratio if validation_data is None
            enable_recovery: Whether to enable automatic error recovery
            
        Returns:
            Dictionary with training results and metrics
        """
        # Validate training data
        data_issues = ValidationUtils.validate_training_data(training_data)
        if data_issues:
            context = ErrorContext(
                component="DualCNNTrainer",
                operation="data_validation",
                config_values={"data_size": len(training_data)}
            )
            raise ComputationError(
                f"Training data validation failed: {'; '.join(data_issues)}",
                context=context
            )
        
        epochs = epochs or self.config.dual_training_epochs
        batch_size = batch_size or self.config.training_batch_size
        
        logger.info("Starting dual CNN training for %d epochs with batch size %d", 
                   epochs, batch_size)
        
        # Reset error tracking
        self._training_errors = []
        self._recovery_attempts = 0
        
        try:
            self.is_training = True
            self._stop_training = False
            start_time = time.time()
            
            # Prepare training data with validation
            train_dataset, val_dataset = self._prepare_training_data_with_validation(
                training_data, validation_data, validation_split, batch_size
            )
            
            # Calculate total batches for progress tracking
            total_batches = self._estimate_total_batches(train_dataset)
            
            # Training loop with error recovery
            for epoch in range(epochs):
                if self._stop_training:
                    logger.info("Training stopped early at epoch %d", epoch)
                    break
                
                try:
                    epoch_start_time = time.time()
                    self.current_epoch = epoch
                    
                    # Reset metrics for this epoch
                    for metric in self.metrics.values():
                        metric.reset_states()
                    
                    # Monitor memory before epoch
                    self._memory_monitor.update()
                    
                    # Train one epoch with error handling
                    epoch_results = self._train_epoch_with_recovery(
                        train_dataset, val_dataset, epoch, epochs, total_batches, enable_recovery
                    )
                    
                    # Update training history
                    self._update_training_history_enhanced(epoch_results)
                    
                    # Calculate epoch time
                    epoch_time = time.time() - epoch_start_time
                    self.training_history['epoch_times'].append(epoch_time)
                    
                    # Log epoch summary
                    self._log_epoch_summary_enhanced(epoch, epoch_results, epoch_time)
                    
                    # Check for early stopping conditions
                    if self._should_stop_early_enhanced(epoch_results):
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break
                    
                    # Check for training instability
                    if self._detect_training_instability(epoch_results):
                        if enable_recovery and self._attempt_training_recovery():
                            logger.info("Training recovery successful, continuing...")
                            continue
                        else:
                            logger.error("Training instability detected and recovery failed")
                            break
                
                except Exception as epoch_error:
                    logger.error(f"Epoch {epoch} failed: {epoch_error}")
                    self._training_errors.append((epoch, epoch_error))
                    
                    if enable_recovery and self._recovery_attempts < self._max_recovery_attempts:
                        if self._attempt_epoch_recovery(epoch_error):
                            logger.info(f"Epoch recovery successful for epoch {epoch}")
                            continue
                    
                    # Re-raise if recovery failed or disabled
                    context = ErrorContext(
                        component="DualCNNTrainer",
                        operation=f"train_epoch_{epoch}",
                        config_values={"epoch": epoch, "batch_size": batch_size}
                    )
                    enhanced_error = global_error_handler.handle_error(epoch_error, context)
                    raise enhanced_error
            
            # Calculate final training results
            total_time = time.time() - start_time
            final_results = self._calculate_final_results_enhanced(total_time)
            
            logger.info("Dual CNN training completed in %.2f seconds", total_time)
            return final_results
            
        except Exception as e:
            logger.error("Training failed: %s", str(e))
            
            # Attempt final recovery if enabled
            if enable_recovery and isinstance(e, (MemoryError, ComputationError)):
                if self._attempt_final_recovery():
                    logger.info("Final recovery successful, returning partial results")
                    return self._get_partial_results()
            
            # Create enhanced error for final failure
            context = ErrorContext(
                component="DualCNNTrainer",
                operation="train_dual_cnn",
                memory_usage=self._memory_monitor.current_memory,
                config_values={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "recovery_attempts": self._recovery_attempts
                }
            )
            
            if isinstance(e, (ComputationError, MemoryError, ConfigurationError)):
                raise e  # Already enhanced
            else:
                enhanced_error = global_error_handler.handle_error(e, context)
                raise enhanced_error
        finally:
            self.is_training = False
            self._cleanup_training_resources()
    
    def profile_memory_usage(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Profile memory usage during dual CNN training.
        
        Args:
            detailed: Whether to include detailed component-level profiling
            
        Returns:
            Dictionary with comprehensive memory usage statistics
        """
        if not self.performance_monitor:
            return {'error': 'Performance monitoring not available'}
        
        # Get current memory profile
        memory_summary = self.performance_monitor.memory_profiler.get_memory_summary()
        
        # Add dual CNN specific profiling
        dual_cnn_profile = {
            'pipeline_components': {},
            'wave_storage_stats': {},
            'training_state': {
                'is_training': self.is_training,
                'current_epoch': self.current_epoch,
                'training_errors': len(self._training_errors),
                'recovery_attempts': self._recovery_attempts
            }
        }
        
        # Profile pipeline components
        if self.pipeline:
            components = ['tokenizer', 'embedder', 'reservoir', 'first_cnn', 'second_cnn', 'wave_storage']
            for component_name in components:
                component = getattr(self.pipeline, component_name, None)
                if component is not None:
                    try:
                        # Profile component memory
                        profile = self.performance_monitor.memory_profiler.profile_component(f"pipeline_{component_name}")
                        dual_cnn_profile['pipeline_components'][component_name] = {
                            'current_memory_mb': profile.current_memory_mb,
                            'tensor_count': profile.tensor_count,
                            'largest_tensor_mb': profile.largest_tensor_mb
                        }
                        
                        # Add component-specific stats
                        if component_name == 'wave_storage' and hasattr(component, 'get_storage_stats'):
                            dual_cnn_profile['wave_storage_stats'] = component.get_storage_stats()
                            
                    except Exception as e:
                        dual_cnn_profile['pipeline_components'][component_name] = {'error': str(e)}
        
        # Combine with general memory summary
        result = {
            'general_memory': memory_summary,
            'dual_cnn_specific': dual_cnn_profile,
            'optimization_recommendations': self._generate_memory_optimization_recommendations(memory_summary, dual_cnn_profile)
        }
        
        if detailed:
            result['detailed_analysis'] = self._detailed_memory_analysis()
        
        return result
    
    def _generate_memory_optimization_recommendations(self, 
                                                    memory_summary: Dict[str, Any],
                                                    dual_cnn_profile: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # Check overall memory usage
        current_memory = memory_summary.get('current_usage', {}).get('rss_mb', 0)
        if current_memory > 2000:  # > 2GB
            recommendations.append("High memory usage detected. Consider reducing batch size or enabling gradient checkpointing.")
        
        # Check wave storage efficiency
        wave_stats = dual_cnn_profile.get('wave_storage_stats', {})
        if wave_stats:
            utilization = wave_stats.get('utilization_percent', 0)
            if utilization > 90:
                recommendations.append("Wave storage is near capacity. Consider increasing memory limit or enabling more aggressive cleanup.")
            elif utilization < 20:
                recommendations.append("Wave storage is underutilized. Consider reducing memory allocation for better efficiency.")
        
        # Check for memory growth
        memory_growth = memory_summary.get('memory_growth_mb', 0)
        if memory_growth > 500:  # > 500MB growth
            recommendations.append("Significant memory growth detected. Check for memory leaks or enable periodic cleanup.")
        
        # Check component memory distribution
        components = dual_cnn_profile.get('pipeline_components', {})
        total_component_memory = sum(
            comp.get('current_memory_mb', 0) for comp in components.values() 
            if isinstance(comp, dict) and 'current_memory_mb' in comp
        )
        
        if total_component_memory > current_memory * 0.8:
            recommendations.append("Pipeline components using high proportion of memory. Consider model compression or quantization.")
        
        # Check for tensor accumulation
        total_tensors = sum(
            comp.get('tensor_count', 0) for comp in components.values()
            if isinstance(comp, dict) and 'tensor_count' in comp
        )
        
        if total_tensors > 1000:
            recommendations.append("High tensor count detected. Consider enabling automatic tensor cleanup or reducing model complexity.")
        
        return recommendations
    
    def _detailed_memory_analysis(self) -> Dict[str, Any]:
        """Perform detailed memory analysis."""
        analysis = {
            'tensor_analysis': {},
            'gradient_analysis': {},
            'cache_analysis': {},
            'fragmentation_analysis': {}
        }
        
        try:
            # Analyze TensorFlow tensors
            import gc
            tensors = [obj for obj in gc.get_objects() if isinstance(obj, (tf.Tensor, tf.Variable))]
            
            tensor_sizes = []
            tensor_dtypes = {}
            
            for tensor in tensors:
                try:
                    if hasattr(tensor, 'numpy'):
                        size_mb = tensor.numpy().nbytes / (1024**2)
                        tensor_sizes.append(size_mb)
                        
                        dtype = str(tensor.dtype)
                        tensor_dtypes[dtype] = tensor_dtypes.get(dtype, 0) + 1
                except Exception:
                    continue
            
            analysis['tensor_analysis'] = {
                'total_tensors': len(tensors),
                'total_size_mb': sum(tensor_sizes),
                'average_size_mb': np.mean(tensor_sizes) if tensor_sizes else 0,
                'largest_tensor_mb': max(tensor_sizes) if tensor_sizes else 0,
                'dtype_distribution': tensor_dtypes
            }
            
            # Analyze gradients if training
            if self.is_training and hasattr(self, 'first_cnn_optimizer'):
                try:
                    first_cnn_vars = self.pipeline.first_cnn.trainable_variables if self.pipeline.first_cnn else []
                    second_cnn_vars = self.pipeline.second_cnn.trainable_variables if self.pipeline.second_cnn else []
                    
                    analysis['gradient_analysis'] = {
                        'first_cnn_parameters': len(first_cnn_vars),
                        'second_cnn_parameters': len(second_cnn_vars),
                        'total_parameters': len(first_cnn_vars) + len(second_cnn_vars)
                    }
                except Exception as e:
                    analysis['gradient_analysis'] = {'error': str(e)}
            
            # Analyze TensorFlow cache
            try:
                # This is a simplified cache analysis
                analysis['cache_analysis'] = {
                    'keras_backend_cleared': False,  # Would need to track this
                    'graph_cache_size': 'unknown'  # Would need TF internals access
                }
            except Exception as e:
                analysis['cache_analysis'] = {'error': str(e)}
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage with advanced techniques.
        
        Returns:
            Dictionary with optimization results
        """
        optimization_results = {
            'optimizations_applied': [],
            'memory_before_mb': 0.0,
            'memory_after_mb': 0.0,
            'memory_saved_mb': 0.0
        }
        
        try:
            # Get baseline memory
            if self.performance_monitor:
                baseline_memory = self.performance_monitor.memory_profiler._get_current_memory()
                optimization_results['memory_before_mb'] = baseline_memory.get('rss_mb', 0)
            
            # 1. Optimize wave storage
            if self.wave_storage_optimizer:
                try:
                    wave_results = self.wave_storage_optimizer.optimize_storage_memory()
                    if wave_results['memory_saved_mb'] > 0:
                        optimization_results['optimizations_applied'].extend(wave_results['optimizations_applied'])
                        logger.info(f"Wave storage optimization saved {wave_results['memory_saved_mb']:.1f}MB")
                except Exception as e:
                    logger.warning(f"Wave storage optimization failed: {e}")
            
            # 2. Clean up TensorFlow memory
            try:
                if self.performance_monitor:
                    memory_freed = self.performance_monitor.memory_profiler.cleanup_memory()
                    if memory_freed > 0:
                        optimization_results['optimizations_applied'].append(f"tensorflow_cleanup_{memory_freed:.1f}MB")
            except Exception as e:
                logger.warning(f"TensorFlow memory cleanup failed: {e}")
            
            # 3. Optimize gradient accumulation
            try:
                if self.is_training:
                    # Clear gradient tapes and intermediate computations
                    tf.keras.backend.clear_session()
                    optimization_results['optimizations_applied'].append("gradient_cleanup")
            except Exception as e:
                logger.warning(f"Gradient cleanup failed: {e}")
            
            # 4. Optimize tensor operations
            if self.tensor_optimizer and self._optimizations_enabled:
                try:
                    # Enable memory growth for GPU if available
                    gpus = tf.config.list_physical_devices('GPU')
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    if gpus:
                        optimization_results['optimizations_applied'].append("gpu_memory_growth_enabled")
                except Exception as e:
                    logger.warning(f"GPU memory optimization failed: {e}")
            
            # 5. Force garbage collection
            try:
                import gc
                collected = gc.collect()
                if collected > 0:
                    optimization_results['optimizations_applied'].append(f"garbage_collection_{collected}_objects")
            except Exception as e:
                logger.warning(f"Garbage collection failed: {e}")
            
            # Get final memory
            if self.performance_monitor:
                final_memory = self.performance_monitor.memory_profiler._get_current_memory()
                optimization_results['memory_after_mb'] = final_memory.get('rss_mb', 0)
                optimization_results['memory_saved_mb'] = (
                    optimization_results['memory_before_mb'] - optimization_results['memory_after_mb']
                )
            
            logger.info(f"Memory optimization completed. Saved {optimization_results['memory_saved_mb']:.1f}MB")
            
        except Exception as e:
            optimization_results['error'] = str(e)
            logger.error(f"Memory optimization failed: {e}")
        
        return optimization_results
    
    def _prepare_training_data_with_validation(self, 
                              training_data: List[str],
                              validation_data: Optional[List[str]],
                              validation_split: float,
                              batch_size: int) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        """Prepare training and validation datasets with optimizations."""
        logger.info("Preparing training data from %d samples", len(training_data))
        
        # Split data if no validation data provided
        if validation_data is None and validation_split > 0:
            split_idx = int(len(training_data) * (1 - validation_split))
            validation_data = training_data[split_idx:]
            training_data = training_data[:split_idx]
        
        # Create base datasets with large dataset optimizations
        train_dataset = self._create_optimized_dataset_for_large_data(training_data, batch_size, shuffle=True)
        val_dataset = None
        if validation_data:
            val_dataset = self._create_optimized_dataset_for_large_data(validation_data, batch_size, shuffle=False)
        
        # Apply advanced optimizations if available
        if self._optimizations_enabled and self.batch_optimizer:
            try:
                # Optimize batch size for memory and performance
                optimal_batch_size = self.batch_optimizer.optimize_batch_size(
                    train_dataset.unbatch(),
                    self._create_dummy_forward_fn(),
                    initial_batch_size=batch_size,
                    max_batch_size=min(batch_size * 4, 256)  # Cap at reasonable limit
                )
                
                if optimal_batch_size != batch_size:
                    logger.info(f"Optimized batch size from {batch_size} to {optimal_batch_size}")
                    train_dataset = self.batch_optimizer.create_optimized_dataset(
                        train_dataset.unbatch(), optimal_batch_size
                    )
                    if val_dataset:
                        val_dataset = self.batch_optimizer.create_optimized_dataset(
                            val_dataset.unbatch(), optimal_batch_size, shuffle_buffer_size=0
                        )
                
            except Exception as e:
                logger.warning(f"Batch optimization failed, using original settings: {e}")
        
        logger.info("Prepared training dataset with %d samples, validation: %s",
                   len(training_data), "Yes" if val_dataset else "No")
        
        return train_dataset, val_dataset
    
    def _create_optimized_dataset_for_large_data(self, texts: List[str], batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset for large datasets."""
        logger.info(f"Creating optimized dataset for {len(texts)} samples")
        
        # For very large datasets, use streaming approach
        if len(texts) > 10000:
            return self._create_streaming_dataset(texts, batch_size, shuffle)
        else:
            return self._create_dataset(texts, batch_size, shuffle)
    
    def _create_streaming_dataset(self, texts: List[str], batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """Create streaming dataset for very large datasets to avoid memory issues."""
        
        def text_generator():
            """Generator function for streaming text data."""
            indices = list(range(len(texts)))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                text = texts[idx]
                if not text or not text.strip():
                    continue
                
                try:
                    # Tokenize text
                    tokenized = self.pipeline.tokenizer.tokenize([text], padding=True, truncation=True)
                    tokens = tokenized['input_ids'][0].numpy().tolist()
                    
                    # Create input-target pairs for language modeling
                    for i in range(len(tokens) - 1):
                        if tokens[i] != 0:  # Skip padding tokens
                            # Create context window
                            start_idx = max(0, i - self.config.embedder_max_length + 1)
                            input_seq = tokens[start_idx:i+1]
                            target_token = tokens[i+1]
                            
                            # Pad input sequence
                            if len(input_seq) < self.config.embedder_max_length:
                                padding = [0] * (self.config.embedder_max_length - len(input_seq))
                                input_seq = padding + input_seq
                            
                            yield (input_seq, target_token)
                            
                except Exception as e:
                    logger.debug(f"Failed to process text: {e}")
                    continue
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            text_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.config.embedder_max_length,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        # Apply optimizations for streaming
        if shuffle:
            # Use smaller shuffle buffer for streaming to save memory
            shuffle_buffer = min(1000, len(texts) // 10)
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Created streaming dataset with batch_size={batch_size}")
        return dataset
    
    def optimize_batch_processing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Apply advanced batch processing optimizations for large datasets.
        
        Args:
            dataset: Input dataset to optimize
            
        Returns:
            Optimized dataset with performance enhancements
        """
        if not self._optimizations_enabled or not self.batch_optimizer:
            return dataset
        
        try:
            # Apply dataset optimizations
            optimized_dataset = dataset
            
            # Enable dataset caching for small datasets
            try:
                cardinality = tf.data.experimental.cardinality(dataset).numpy()
                if 0 < cardinality < 1000:  # Cache small datasets
                    optimized_dataset = optimized_dataset.cache()
                    logger.info("Dataset cached for better performance")
            except Exception:
                pass  # Ignore if cardinality cannot be determined
            
            # Apply parallel processing
            optimized_dataset = optimized_dataset.map(
                self._preprocess_batch,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False  # Allow non-deterministic ordering for speed
            )
            
            # Optimize prefetching
            optimized_dataset = optimized_dataset.prefetch(tf.data.AUTOTUNE)
            
            logger.info("Applied advanced batch processing optimizations")
            return optimized_dataset
            
        except Exception as e:
            logger.warning(f"Batch processing optimization failed: {e}")
            return dataset
    
    def _preprocess_batch(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess batch with optimizations."""
        # Apply any batch-level preprocessing here
        # For now, just return as-is, but this could include:
        # - Data augmentation
        # - Normalization
        # - Feature engineering
        return input_batch, target_batch
    
    def _create_dummy_forward_fn(self):
        """Create a dummy forward function for batch size optimization."""
        def dummy_forward(batch):
            try:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_batch, target_batch = batch[0], batch[1]
                else:
                    # Handle single tensor case
                    input_batch = batch
                    target_batch = tf.zeros((tf.shape(batch)[0],), dtype=tf.int32)
                
                # Simulate forward pass through pipeline
                embedded = self.pipeline.embedder(input_batch)
                reservoir_states, attention_weights = self.pipeline.reservoir(embedded, training=False)
                first_cnn_output = self.pipeline.first_cnn(reservoir_states, training=False)
                
                return first_cnn_output
            except Exception as e:
                logger.debug(f"Dummy forward pass failed: {e}")
                return tf.zeros((1, self.pipeline.tokenizer.vocab_size))
        
        return dummy_forward
    
    def _create_dataset(self, texts: List[str], batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset from text data."""
        # Tokenize all texts
        input_sequences = []
        target_sequences = []
        
        for text in texts:
            if not text or not text.strip():
                continue
            
            # Tokenize text
            tokenized = self.pipeline.tokenizer.tokenize([text], padding=True, truncation=True)
            tokens = tokenized['input_ids'][0].numpy().tolist()
            
            # Create input-target pairs for language modeling
            for i in range(len(tokens) - 1):
                if tokens[i] != 0:  # Skip padding tokens
                    # Create context window
                    start_idx = max(0, i - self.config.embedder_max_length + 1)
                    input_seq = tokens[start_idx:i+1]
                    target_token = tokens[i+1]
                    
                    # Pad input sequence
                    if len(input_seq) < self.config.embedder_max_length:
                        padding = [0] * (self.config.embedder_max_length - len(input_seq))
                        input_seq = padding + input_seq
                    
                    input_sequences.append(input_seq)
                    target_sequences.append(target_token)
        
        if not input_sequences:
            raise ValueError("No valid training sequences created")
        
        # Convert to tensors
        input_tensor = tf.constant(input_sequences, dtype=tf.int32)
        target_tensor = tf.constant(target_sequences, dtype=tf.int32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(input_sequences)))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _train_epoch_with_recovery(self, 
                    train_dataset: tf.data.Dataset,
                    val_dataset: Optional[tf.data.Dataset],
                    epoch: int,
                    total_epochs: int,
                    total_batches: int,
                    enable_recovery: bool) -> Dict[str, float]:
        """Train one epoch with recovery and optimization."""
        return self._train_epoch(train_dataset, val_dataset, epoch, total_epochs, total_batches)
    
    def _train_epoch(self, 
                    train_dataset: tf.data.Dataset,
                    val_dataset: Optional[tf.data.Dataset],
                    epoch: int,
                    total_epochs: int,
                    total_batches: int) -> Dict[str, float]:
        """Train one epoch with dual CNN coordination and performance monitoring."""
        epoch_losses = {
            'first_cnn_loss': 0.0,
            'second_cnn_loss': 0.0,
            'combined_loss': 0.0
        }
        
        batch_count = 0
        epoch_start_time = time.time()
        
        # Start performance monitoring for this epoch
        if self.performance_monitor:
            self.performance_monitor.start_training_monitoring()
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_dataset):
            if self._stop_training:
                break
            
            batch_start_time = time.time()
            
            # Monitor memory before batch
            if self._memory_monitor:
                self._memory_monitor.update()
            
            # Train on batch with dual CNN coordination and optimizations
            batch_losses = self._train_batch_optimized(input_batch, target_batch)
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            self._batch_times.append(batch_time)
            
            # Record performance metrics
            if self.performance_monitor:
                current_memory = self._memory_monitor.current_memory if self._memory_monitor else 0.0
                batch_size = input_batch.shape[0] if hasattr(input_batch, 'shape') else 32
                
                self.performance_monitor.record_batch_metrics(
                    batch_size=batch_size,
                    forward_time=batch_time * 0.6,  # Estimate
                    backward_time=batch_time * 0.4,  # Estimate
                    memory_usage=current_memory
                )
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += float(batch_losses[key])
            
            batch_count += 1
            
            # Optimize wave storage periodically
            if (self.wave_storage_optimizer and 
                batch_idx % 50 == 0 and batch_idx > 0):
                try:
                    optimization_results = self.wave_storage_optimizer.optimize_storage_memory()
                    if optimization_results['memory_saved_mb'] > 10:
                        logger.info(f"Wave storage optimization saved "
                                  f"{optimization_results['memory_saved_mb']:.1f}MB")
                except Exception as e:
                    logger.debug(f"Wave storage optimization failed: {e}")
            
            # Update progress
            if batch_idx % 10 == 0:  # Update every 10 batches
                self._update_progress(epoch, total_epochs, batch_idx, total_batches, batch_losses)
            
            # Log batch progress with performance info
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_batch_time = np.mean(list(self._batch_times)[-10:])
                current_memory_mb = self._memory_monitor.current_memory if self._memory_monitor else 0
                
                logger.info("Epoch %d, Batch %d/%d - Combined Loss: %.4f, "
                           "Avg Batch Time: %.3fs, Memory: %.1fMB",
                           epoch + 1, batch_idx, total_batches, 
                           float(batch_losses['combined_loss']),
                           avg_batch_time, current_memory_mb)
        
        # Average losses over batches
        if batch_count > 0:
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
        
        # Validate if validation data available
        if val_dataset is not None:
            val_losses = self._validate_epoch(val_dataset)
            epoch_losses.update(val_losses)
        
        # Add performance metrics to epoch results
        epoch_time = time.time() - epoch_start_time
        epoch_losses['epoch_time'] = epoch_time
        epoch_losses['avg_batch_time'] = np.mean(self._batch_times) if self._batch_times else 0.0
        epoch_losses['memory_usage_mb'] = self._memory_monitor.current_memory if self._memory_monitor else 0.0
        
        return epoch_losses
    
    def _train_batch_optimized(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Train a batch with performance optimizations."""
        # Profile memory before batch processing
        if self.performance_monitor:
            self.performance_monitor.memory_profiler.profile_component("batch_start")
        
        if self._optimizations_enabled and self.tensor_optimizer:
            result = self._train_batch_with_tensor_optimization(input_batch, target_batch)
        else:
            result = self._train_batch(input_batch, target_batch)
        
        # Profile memory after batch processing
        if self.performance_monitor:
            self.performance_monitor.memory_profiler.profile_component("batch_end")
        
        return result
    
    def _prepare_second_cnn_input(self, wave_features: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        """Prepare input for second CNN from wave features."""
        try:
            # Reshape wave features for second CNN
            # Expected shape: (batch_size, window_size, feature_dim)
            target_shape = (batch_size, self.config.wave_window_size, self.config.wave_feature_dim)
            
            # If wave_features doesn't match expected shape, reshape or pad
            current_shape = tf.shape(wave_features)
            
            if len(wave_features.shape) == 2:
                # If 2D, add sequence dimension
                wave_features = tf.expand_dims(wave_features, axis=1)
            
            # Pad or truncate to match window size
            seq_len = tf.shape(wave_features)[1]
            if seq_len < self.config.wave_window_size:
                # Pad with zeros
                padding = [[0, 0], [0, self.config.wave_window_size - seq_len], [0, 0]]
                wave_features = tf.pad(wave_features, padding)
            elif seq_len > self.config.wave_window_size:
                # Truncate
                wave_features = wave_features[:, :self.config.wave_window_size, :]
            
            # Ensure correct feature dimension
            feature_dim = tf.shape(wave_features)[2]
            if feature_dim != self.config.wave_feature_dim:
                # Project to correct dimension
                wave_features = tf.layers.dense(wave_features, self.config.wave_feature_dim)
            
            return wave_features
            
        except Exception as e:
            logger.debug(f"Failed to prepare second CNN input: {e}")
            # Fallback: create zero tensor with correct shape
            return tf.zeros((batch_size, self.config.wave_window_size, self.config.wave_feature_dim))
    
    def _train_batch(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Standard batch training without tensor optimizations."""
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass through pipeline
            embedded = self.pipeline.embedder(input_batch)
            reservoir_states, attention_weights = self.pipeline.reservoir(embedded, training=True)
            
            # Store wave features if wave storage is available
            if self.pipeline.wave_storage:
                try:
                    # Extract wave features from reservoir states
                    wave_features = tf.reduce_mean(reservoir_states, axis=1)  # Average over sequence
                    
                    # Store waves for each sample in batch
                    for i in range(tf.shape(input_batch)[0]):
                        wave = wave_features[i]
                        confidence = tf.reduce_mean(attention_weights[i]) if attention_weights is not None else 1.0
                        self.pipeline.wave_storage.store_wave(
                            wave, 
                            sequence_position=i,
                            confidence_score=float(confidence)
                        )
                except Exception as e:
                    logger.debug(f"Wave storage failed: {e}")
            
            # First CNN forward pass
            first_cnn_output = self.pipeline.first_cnn(reservoir_states, training=True)
            
            # Second CNN forward pass (if available)
            if self.pipeline.second_cnn and not self._use_single_cnn_fallback:
                try:
                    # Get wave sequence for second CNN
                    wave_sequence = self.pipeline.wave_storage.get_rolling_window(
                        center_pos=tf.shape(input_batch)[0] // 2
                    ) if self.pipeline.wave_storage else reservoir_states
                    
                    second_cnn_output = self.pipeline.second_cnn(wave_sequence, training=True)
                except Exception as e:
                    logger.debug(f"Second CNN failed, using first CNN only: {e}")
                    second_cnn_output = first_cnn_output
                    self._use_single_cnn_fallback = True
            else:
                second_cnn_output = first_cnn_output
            
            # Compute losses
            first_cnn_loss = tf.reduce_mean(self.loss_fn(target_batch, first_cnn_output))
            second_cnn_loss = tf.reduce_mean(self.loss_fn(target_batch, second_cnn_output))
            
            # Combined loss with coordination weighting
            coordination_weight = self.config.wave_coordination_weight
            final_weight = self.config.final_prediction_weight
            combined_output = (coordination_weight * first_cnn_output + 
                             final_weight * second_cnn_output)
            combined_loss = tf.reduce_mean(self.loss_fn(target_batch, combined_output))
        
        # Apply gradients
        self._apply_optimized_gradients(tape, first_cnn_loss, second_cnn_loss)
        
        # Update metrics
        self.metrics['first_cnn_accuracy'].update_state(target_batch, first_cnn_output)
        self.metrics['second_cnn_accuracy'].update_state(target_batch, second_cnn_output)
        self.metrics['combined_accuracy'].update_state(target_batch, combined_output)
        
        del tape  # Clean up persistent tape
        
        return {
            'first_cnn_loss': first_cnn_loss,
            'second_cnn_loss': second_cnn_loss,
            'combined_loss': combined_loss
        }
    
    @tf.function(experimental_relax_shapes=True)
    def _train_batch_with_tensor_optimization(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Optimized batch training with efficient tensor operations."""
        batch_size = tf.shape(input_batch)[0]
        
        # Forward pass through pipeline components with optimizations
        with tf.GradientTape(persistent=True) as tape:
            # Embed inputs
            embedded = self.pipeline.embedder(input_batch)
            
            # Process through attentive reservoir
            reservoir_states, attention_weights = self.pipeline.reservoir(embedded, training=True)
            
            # Optimized wave feature extraction
            wave_features = self.tensor_optimizer.optimized_wave_feature_extraction(
                reservoir_states, attention_weights
            )
            
            # Prepare inputs for both CNNs
            first_cnn_input = reservoir_states
            second_cnn_input = self._prepare_second_cnn_input(wave_features, batch_size)
            
            # Optimized dual CNN forward pass
            first_cnn_output, second_cnn_output, combined_output = self.tensor_optimizer.optimized_dual_cnn_forward(
                first_cnn_input, second_cnn_input,
                self.pipeline.first_cnn, self.pipeline.second_cnn,
                self.config.wave_coordination_weight
            )
            
            # Compute losses
            first_cnn_loss = tf.reduce_mean(self.loss_fn(target_batch, first_cnn_output))
            second_cnn_loss = tf.reduce_mean(self.loss_fn(target_batch, second_cnn_output))
            combined_loss = tf.reduce_mean(self.loss_fn(target_batch, combined_output))
        
        # Compute and apply gradients efficiently
        self._apply_optimized_gradients(tape, first_cnn_loss, second_cnn_loss)
        
        # Update metrics
        self.metrics['first_cnn_accuracy'].update_state(target_batch, first_cnn_output)
        self.metrics['second_cnn_accuracy'].update_state(target_batch, second_cnn_output)
        self.metrics['combined_accuracy'].update_state(target_batch, combined_output)
        
        del tape  # Clean up persistent tape
        
        return {
            'first_cnn_loss': first_cnn_loss,
            'second_cnn_loss': second_cnn_loss,
            'combined_loss': combined_loss
        }
    
    def _apply_optimized_gradients(self, tape, first_cnn_loss, second_cnn_loss):
        """Apply gradients with optimization techniques."""
        # Get trainable variables
        first_cnn_vars = self.pipeline.first_cnn.trainable_variables
        second_cnn_vars = self.pipeline.second_cnn.trainable_variables
        
        # Compute gradients efficiently
        first_cnn_grads = tape.gradient(first_cnn_loss, first_cnn_vars)
        second_cnn_grads = tape.gradient(second_cnn_loss, second_cnn_vars)
        
        # Apply gradient clipping for stability
        first_cnn_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else None 
                          for grad in first_cnn_grads]
        second_cnn_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else None 
                           for grad in second_cnn_grads]
        
        # Apply gradients
        self.first_cnn_optimizer.apply_gradients(zip(first_cnn_grads, first_cnn_vars))
        self.second_cnn_optimizer.apply_gradients(zip(second_cnn_grads, second_cnn_vars))
        
        # Track gradient norms for monitoring
        first_grad_norm = tf.reduce_mean([tf.norm(grad) for grad in first_cnn_grads if grad is not None])
        second_grad_norm = tf.reduce_mean([tf.norm(grad) for grad in second_cnn_grads if grad is not None])
        
        self.training_history['gradient_norms'].append({
            'first_cnn': float(first_grad_norm),
            'second_cnn': float(second_grad_norm)
        })
        if first_cnn_grads:
            # Filter out None gradients and apply clipping
            filtered_first_grads = []
            filtered_first_vars = []
            
            for grad, var in zip(first_cnn_grads, first_cnn_vars + reservoir_vars):
                if grad is not None:
                    # Gradient clipping
                    clipped_grad = tf.clip_by_norm(grad, 1.0)
                    filtered_first_grads.append(clipped_grad)
                    filtered_first_vars.append(var)
            
            if filtered_first_grads:
                self.first_cnn_optimizer.apply_gradients(zip(filtered_first_grads, filtered_first_vars))
        
        if second_cnn_grads:
            # Filter out None gradients and apply clipping
            filtered_second_grads = []
            filtered_second_vars = []
            
            for grad, var in zip(second_cnn_grads, second_cnn_vars + reservoir_vars):
                if grad is not None:
                    # Gradient clipping
                    clipped_grad = tf.clip_by_norm(grad, 1.0)
                    filtered_second_grads.append(clipped_grad)
                    filtered_second_vars.append(var)
            
            if filtered_second_grads:
                self.second_cnn_optimizer.apply_gradients(zip(filtered_second_grads, filtered_second_vars))
    
    @tf.function
    def _train_batch(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Train a single batch with dual CNN coordination."""
        batch_size = tf.shape(input_batch)[0]
        
        # Forward pass through pipeline components
        with tf.GradientTape(persistent=True) as tape:
            # Embed inputs
            embedded = self.pipeline.embedder(input_batch)
            
            # Process through attentive reservoir
            reservoir_states, attention_weights = self.pipeline.reservoir(embedded, training=True)
            
            # First CNN: Next-token prediction
            first_cnn_output = self.pipeline.first_cnn(reservoir_states, training=True)
            first_cnn_loss = self.loss_fn(target_batch, first_cnn_output)
            first_cnn_loss = tf.reduce_mean(first_cnn_loss)
            
            # Store wave outputs for second CNN
            wave_features = self._extract_wave_features(reservoir_states, attention_weights)
            
            # Second CNN: Final prediction using wave features
            second_cnn_input = self._prepare_second_cnn_input(wave_features, batch_size)
            second_cnn_output = self.pipeline.second_cnn(second_cnn_input, training=True)
            second_cnn_loss = self.loss_fn(target_batch, second_cnn_output)
            second_cnn_loss = tf.reduce_mean(second_cnn_loss)
            
            # Combined loss with weighting
            combined_loss = (self.config.wave_coordination_weight * first_cnn_loss + 
                           self.config.final_prediction_weight * second_cnn_loss)
        
        # Compute gradients
        first_cnn_vars = self.pipeline.first_cnn.trainable_variables
        second_cnn_vars = self.pipeline.second_cnn.trainable_variables
        reservoir_vars = self.pipeline.reservoir.trainable_variables
        
        first_cnn_grads = tape.gradient(first_cnn_loss, first_cnn_vars + reservoir_vars)
        second_cnn_grads = tape.gradient(second_cnn_loss, second_cnn_vars + reservoir_vars)
        
        # Apply gradients (filter out None gradients)
        if first_cnn_grads:
            filtered_first_grads = [(g, v) for g, v in zip(first_cnn_grads, first_cnn_vars + reservoir_vars) if g is not None]
            if filtered_first_grads:
                self.first_cnn_optimizer.apply_gradients(filtered_first_grads)
        
        if second_cnn_grads:
            filtered_second_grads = [(g, v) for g, v in zip(second_cnn_grads, second_cnn_vars + reservoir_vars) if g is not None]
            if filtered_second_grads:
                self.second_cnn_optimizer.apply_gradients(filtered_second_grads)
        
        # Update metrics
        self.metrics['first_cnn_accuracy'].update_state(target_batch, first_cnn_output)
        self.metrics['second_cnn_accuracy'].update_state(target_batch, second_cnn_output)
        
        # Combined prediction (weighted average)
        combined_output = (self.config.wave_coordination_weight * first_cnn_output + 
                          self.config.final_prediction_weight * second_cnn_output)
        self.metrics['combined_accuracy'].update_state(target_batch, combined_output)
        
        del tape  # Clean up persistent tape
        
        return {
            'first_cnn_loss': first_cnn_loss,
            'second_cnn_loss': second_cnn_loss,
            'combined_loss': combined_loss
        }
    
    def _extract_wave_features(self, reservoir_states: tf.Tensor, attention_weights: tf.Tensor) -> tf.Tensor:
        """Extract wave features from reservoir states and attention."""
        # Combine reservoir states with attention information
        # Shape: (batch, sequence, reservoir_size) -> (batch, sequence, wave_feature_dim)
        
        # Apply attention weighting to reservoir states
        # Average attention weights over heads and target positions to get per-position weights
        attended_weights = tf.reduce_mean(attention_weights, axis=[1, -1])  # (batch, sequence)
        attended_weights = tf.expand_dims(attended_weights, axis=-1)  # (batch, sequence, 1)
        
        # Weight reservoir states by attention
        weighted_states = reservoir_states * attended_weights
        
        # Project to wave feature dimension if needed
        if weighted_states.shape[-1] != self.config.wave_feature_dim:
            projection_layer = tf.keras.layers.Dense(self.config.wave_feature_dim)
            wave_features = projection_layer(weighted_states)
        else:
            wave_features = weighted_states
        
        return wave_features
    
    def _prepare_second_cnn_input(self, wave_features: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        """Prepare input for second CNN from wave features."""
        # Reshape wave features to match second CNN input requirements
        # Expected shape: (batch, window_size, feature_dim)
        
        sequence_length = tf.shape(wave_features)[1]
        feature_dim = tf.shape(wave_features)[2]
        
        # If sequence is longer than window size, take the last window_size elements
        if sequence_length > self.config.wave_window_size:
            wave_features = wave_features[:, -self.config.wave_window_size:, :]
        elif sequence_length < self.config.wave_window_size:
            # Pad with zeros if sequence is shorter
            padding_size = self.config.wave_window_size - sequence_length
            padding = tf.zeros((batch_size, padding_size, feature_dim))
            wave_features = tf.concat([padding, wave_features], axis=1)
        
        return wave_features
    
    def _validate_epoch(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Validate model performance on validation dataset."""
        val_losses = {
            'val_first_cnn_loss': 0.0,
            'val_second_cnn_loss': 0.0,
            'val_combined_loss': 0.0
        }
        
        batch_count = 0
        
        for input_batch, target_batch in val_dataset:
            # Forward pass without training
            embedded = self.pipeline.embedder(input_batch)
            reservoir_states, attention_weights = self.pipeline.reservoir(embedded, training=False)
            
            # First CNN validation
            first_cnn_output = self.pipeline.first_cnn(reservoir_states, training=False)
            first_cnn_loss = tf.reduce_mean(self.loss_fn(target_batch, first_cnn_output))
            
            # Second CNN validation
            wave_features = self._extract_wave_features(reservoir_states, attention_weights)
            second_cnn_input = self._prepare_second_cnn_input(wave_features, tf.shape(input_batch)[0])
            second_cnn_output = self.pipeline.second_cnn(second_cnn_input, training=False)
            second_cnn_loss = tf.reduce_mean(self.loss_fn(target_batch, second_cnn_output))
            
            # Combined validation loss
            combined_loss = (self.config.wave_coordination_weight * first_cnn_loss + 
                           self.config.final_prediction_weight * second_cnn_loss)
            
            val_losses['val_first_cnn_loss'] += float(first_cnn_loss)
            val_losses['val_second_cnn_loss'] += float(second_cnn_loss)
            val_losses['val_combined_loss'] += float(combined_loss)
            
            batch_count += 1
        
        # Average validation losses
        if batch_count > 0:
            for key in val_losses:
                val_losses[key] /= batch_count
        
        return val_losses
    
    def _update_progress(self, epoch: int, total_epochs: int, batch_idx: int, 
                        total_batches: int, batch_losses: Dict[str, tf.Tensor]):
        """Update and broadcast training progress."""
        # Calculate wave storage utilization
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        wave_utilization = storage_stats['utilization_percent']
        
        # Calculate attention entropy if available
        attention_entropy = 0.0
        if hasattr(self.pipeline.reservoir, 'compute_attention_entropy'):
            entropy_tensor = self.pipeline.reservoir.compute_attention_entropy()
            if entropy_tensor is not None:
                attention_entropy = float(tf.reduce_mean(entropy_tensor))
        
        # Estimate remaining time
        if hasattr(self, '_batch_times'):
            avg_batch_time = np.mean(self._batch_times[-10:])  # Average of last 10 batches
            remaining_batches = (total_epochs - epoch - 1) * total_batches + (total_batches - batch_idx)
            estimated_time_remaining = avg_batch_time * remaining_batches
        else:
            estimated_time_remaining = 0.0
        
        # Create progress object
        progress = TrainingProgress(
            current_epoch=epoch + 1,
            total_epochs=total_epochs,
            first_cnn_loss=float(batch_losses['first_cnn_loss']),
            second_cnn_loss=float(batch_losses['second_cnn_loss']),
            combined_loss=float(batch_losses['combined_loss']),
            wave_storage_utilization=wave_utilization,
            attention_entropy=attention_entropy,
            estimated_time_remaining=estimated_time_remaining,
            learning_rate=float(self.first_cnn_optimizer.learning_rate),
            batch_processed=batch_idx + 1,
            total_batches=total_batches
        )
        
        self.last_progress = progress
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)
    
    def _update_training_history(self, epoch_results: Dict[str, float]):
        """Update training history with epoch results."""
        self.training_history['first_cnn_loss'].append(epoch_results['first_cnn_loss'])
        self.training_history['second_cnn_loss'].append(epoch_results['second_cnn_loss'])
        self.training_history['combined_loss'].append(epoch_results['combined_loss'])
        
        # Add wave storage utilization
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        self.training_history['wave_storage_utilization'].append(storage_stats['utilization_percent'])
        
        # Add attention entropy
        if hasattr(self.pipeline.reservoir, 'compute_attention_entropy'):
            entropy_tensor = self.pipeline.reservoir.compute_attention_entropy()
            if entropy_tensor is not None:
                entropy_value = float(tf.reduce_mean(entropy_tensor))
                self.training_history['attention_entropy'].append(entropy_value)
            else:
                self.training_history['attention_entropy'].append(0.0)
        else:
            self.training_history['attention_entropy'].append(0.0)
        
        # Add learning rate
        self.training_history['learning_rate'].append(float(self.first_cnn_optimizer.learning_rate))
    
    def _log_epoch_summary(self, epoch: int, epoch_results: Dict[str, float], epoch_time: float):
        """Log comprehensive epoch summary."""
        logger.info("=== Epoch %d Summary ===", epoch + 1)
        logger.info("Time: %.2f seconds", epoch_time)
        logger.info("First CNN Loss: %.4f", epoch_results['first_cnn_loss'])
        logger.info("Second CNN Loss: %.4f", epoch_results['second_cnn_loss'])
        logger.info("Combined Loss: %.4f", epoch_results['combined_loss'])
        
        # Log validation results if available
        if 'val_combined_loss' in epoch_results:
            logger.info("Validation Combined Loss: %.4f", epoch_results['val_combined_loss'])
        
        # Log metrics
        logger.info("First CNN Accuracy: %.4f", float(self.metrics['first_cnn_accuracy'].result()))
        logger.info("Second CNN Accuracy: %.4f", float(self.metrics['second_cnn_accuracy'].result()))
        logger.info("Combined Accuracy: %.4f", float(self.metrics['combined_accuracy'].result()))
        
        # Log wave storage stats
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        logger.info("Wave Storage Utilization: %.1f%%", storage_stats['utilization_percent'])
        
        logger.info("========================")
    
    def _should_stop_early(self, epoch_results: Dict[str, float]) -> bool:
        """Check if training should stop early based on results."""
        # Simple early stopping based on validation loss if available
        if 'val_combined_loss' in epoch_results and len(self.training_history['combined_loss']) >= 5:
            recent_losses = self.training_history['combined_loss'][-5:]
            # Stop if validation loss is worse than all recent training losses
            if all(loss < epoch_results['val_combined_loss'] for loss in recent_losses):
                return True
        
        return False
    
    def _calculate_final_results(self, total_time: float) -> Dict[str, Any]:
        """Calculate comprehensive final training results."""
        final_metrics = {}
        
        # Training metrics
        for metric_name, metric in self.metrics.items():
            final_metrics[metric_name] = float(metric.result())
        
        # Loss progression
        if self.training_history['combined_loss']:
            final_metrics['initial_loss'] = self.training_history['combined_loss'][0]
            final_metrics['final_loss'] = self.training_history['combined_loss'][-1]
            final_metrics['loss_improvement'] = (
                final_metrics['initial_loss'] - final_metrics['final_loss']
            )
        
        # Wave storage statistics
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        final_metrics['final_wave_utilization'] = storage_stats['utilization_percent']
        final_metrics['wave_memory_used_mb'] = storage_stats['memory_used_mb']
        
        # Attention analysis
        if self.training_history['attention_entropy']:
            final_metrics['avg_attention_entropy'] = np.mean(self.training_history['attention_entropy'])
            final_metrics['final_attention_entropy'] = self.training_history['attention_entropy'][-1]
        
        # Training efficiency
        final_metrics['total_training_time'] = total_time
        final_metrics['epochs_completed'] = len(self.training_history['combined_loss'])
        final_metrics['avg_epoch_time'] = np.mean(self.training_history['epoch_times'])
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics,
            'pipeline_status': self.pipeline.get_component_status()
        }
    
    def stop_training(self):
        """Stop training gracefully."""
        self._stop_training = True
        logger.info("Training stop requested")
    
    def get_current_progress(self) -> Optional[TrainingProgress]:
        """Get the current training progress."""
        return self.last_progress
    
    def _prepare_training_data_with_validation(self, 
                                             training_data: List[str],
                                             validation_data: Optional[List[str]],
                                             validation_split: float,
                                             batch_size: int) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        """Prepare training data with enhanced validation and error handling."""
        logger.info("Preparing training data from %d samples", len(training_data))
        
        try:
            # Split data if no validation data provided
            if validation_data is None and validation_split > 0:
                split_idx = int(len(training_data) * (1 - validation_split))
                validation_data = training_data[split_idx:]
                training_data = training_data[:split_idx]
            
            # Create training dataset with error handling
            train_dataset = self._create_dataset_with_validation(training_data, batch_size, shuffle=True)
            
            # Create validation dataset if available
            val_dataset = None
            if validation_data:
                val_dataset = self._create_dataset_with_validation(validation_data, batch_size, shuffle=False)
            
            logger.info("Prepared training dataset with %d samples, validation: %s",
                       len(training_data), "Yes" if val_dataset else "No")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            context = ErrorContext(
                component="DualCNNTrainer",
                operation="prepare_training_data",
                config_values={
                    "data_size": len(training_data),
                    "batch_size": batch_size,
                    "validation_split": validation_split
                }
            )
            raise ComputationError(f"Failed to prepare training data: {str(e)}", context=context)
    
    def _create_dataset_with_validation(self, texts: List[str], batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset with enhanced validation and error handling."""
        input_sequences = []
        target_sequences = []
        failed_tokenizations = 0
        
        for text in texts:
            if not text or not text.strip():
                continue
            
            try:
                # Tokenize text with error handling
                tokenized = self.pipeline.tokenizer.tokenize([text], padding=True, truncation=True)
                tokens = tokenized['input_ids'][0].numpy().tolist()
                
                # Create input-target pairs for language modeling
                for i in range(len(tokens) - 1):
                    if tokens[i] != 0:  # Skip padding tokens
                        # Create context window
                        start_idx = max(0, i - self.config.embedder_max_length + 1)
                        input_seq = tokens[start_idx:i+1]
                        target_token = tokens[i+1]
                        
                        # Pad input sequence
                        if len(input_seq) < self.config.embedder_max_length:
                            padding = [0] * (self.config.embedder_max_length - len(input_seq))
                            input_seq = padding + input_seq
                        
                        input_sequences.append(input_seq)
                        target_sequences.append(target_token)
                        
            except Exception as e:
                failed_tokenizations += 1
                logger.debug(f"Failed to tokenize text: {e}")
                continue
        
        # Check tokenization success rate
        if failed_tokenizations > len(texts) * 0.5:
            logger.warning(f"High tokenization failure rate: {failed_tokenizations}/{len(texts)}")
        
        if not input_sequences:
            raise ValueError("No valid training sequences created from input data")
        
        # Convert to tensors with validation
        try:
            input_tensor = tf.constant(input_sequences, dtype=tf.int32)
            target_tensor = tf.constant(target_sequences, dtype=tf.int32)
            
            # Validate tensor shapes
            ValidationUtils.validate_tensor_shape(
                input_tensor, 
                (None, self.config.embedder_max_length), 
                "input_sequences"
            )
            
        except Exception as e:
            raise ComputationError(f"Failed to create tensors from sequences: {str(e)}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(input_sequences)))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _estimate_total_batches(self, dataset: tf.data.Dataset) -> int:
        """Estimate total number of batches in dataset."""
        try:
            # Try to get cardinality
            cardinality = dataset.cardinality().numpy()
            if cardinality != tf.data.UNKNOWN_CARDINALITY:
                return int(cardinality)
        except:
            pass
        
        # Fallback: count batches (expensive but accurate)
        try:
            count = 0
            for _ in dataset.take(1000):  # Limit counting to avoid hanging
                count += 1
            return max(count, 1)  # Ensure at least 1 batch
        except:
            return 1  # Final fallback
    
    def _train_epoch_with_recovery(self, 
                                  train_dataset: tf.data.Dataset,
                                  val_dataset: Optional[tf.data.Dataset],
                                  epoch: int,
                                  total_epochs: int,
                                  total_batches: int,
                                  enable_recovery: bool) -> Dict[str, float]:
        """Train one epoch with error recovery mechanisms."""
        epoch_losses = {
            'first_cnn_loss': 0.0,
            'second_cnn_loss': 0.0,
            'combined_loss': 0.0
        }
        
        batch_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_dataset):
            if self._stop_training:
                break
            
            try:
                batch_start_time = time.time()
                
                # Monitor memory before batch
                self._memory_monitor.update()
                
                # Check for memory pressure
                if self._memory_monitor.current_memory > 0.9 * self.config.max_memory_usage_gb * 1024:
                    logger.warning("High memory usage detected, triggering cleanup")
                    self._emergency_memory_cleanup()
                
                # Train on batch with dual CNN coordination
                batch_losses = self._train_batch_with_validation(input_batch, target_batch)
                
                # Check for NaN/Inf losses
                if self._check_loss_validity(batch_losses):
                    # Accumulate losses
                    for key in epoch_losses:
                        epoch_losses[key] += batch_losses[key]
                    
                    batch_count += 1
                    consecutive_failures = 0
                    
                    # Track batch time
                    batch_time = time.time() - batch_start_time
                    self._batch_times.append(batch_time)
                    
                else:
                    consecutive_failures += 1
                    logger.warning(f"Invalid loss detected in batch {batch_idx}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        raise ComputationError("Too many consecutive invalid losses detected")
                
                # Update progress
                if batch_idx % 10 == 0:
                    self._update_progress_enhanced(epoch, total_epochs, batch_idx, total_batches, batch_losses)
                
                # Log batch progress
                if batch_idx % 100 == 0 and batch_idx > 0:
                    logger.info("Epoch %d, Batch %d/%d - Combined Loss: %.4f, Memory: %.1f MB",
                               epoch + 1, batch_idx, total_batches, 
                               batch_losses['combined_loss'], self._memory_monitor.current_memory)
                
            except Exception as batch_error:
                consecutive_failures += 1
                logger.error(f"Batch {batch_idx} failed: {batch_error}")
                
                if enable_recovery and consecutive_failures < max_consecutive_failures:
                    if self._attempt_batch_recovery(batch_error):
                        logger.info(f"Batch recovery successful for batch {batch_idx}")
                        continue
                
                # Re-raise if recovery failed or too many failures
                if consecutive_failures >= max_consecutive_failures:
                    raise ComputationError(f"Too many consecutive batch failures: {consecutive_failures}")
                else:
                    raise batch_error
        
        # Average losses over batches
        if batch_count > 0:
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
        else:
            logger.warning("No successful batches in epoch")
            # Return zero losses to avoid division by zero
            epoch_losses = {key: 0.0 for key in epoch_losses}
        
        # Validate if validation data available
        if val_dataset is not None:
            try:
                val_losses = self._validate_epoch_with_recovery(val_dataset)
                epoch_losses.update(val_losses)
            except Exception as val_error:
                logger.warning(f"Validation failed: {val_error}")
                # Continue without validation results
        
        return epoch_losses
    
    def _train_batch_with_validation(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Train a single batch with enhanced validation and error handling."""
        # Validate input shapes
        try:
            ValidationUtils.validate_tensor_shape(
                input_batch, 
                (None, self.config.embedder_max_length), 
                "input_batch"
            )
            ValidationUtils.validate_tensor_shape(
                target_batch, 
                (None,), 
                "target_batch"
            )
        except Exception as e:
            raise ComputationError(f"Batch validation failed: {str(e)}")
        
        batch_size = tf.shape(input_batch)[0]
        
        # Forward pass through pipeline components with error handling
        with tf.GradientTape(persistent=True) as tape:
            try:
                # Embed inputs
                embedded = self.pipeline.embedder(input_batch)
                
                # Process through attentive reservoir
                if isinstance(self.pipeline.reservoir, AttentiveReservoir):
                    reservoir_states, attention_weights = self.pipeline.reservoir(embedded, training=True)
                else:
                    # Fallback for standard reservoir
                    reservoir_states = self.pipeline.reservoir(embedded, training=True)
                    attention_weights = None
                
                # First CNN: Next-token prediction
                first_cnn_output = self.pipeline.first_cnn(reservoir_states, training=True)
                first_cnn_loss = self.loss_fn(target_batch, first_cnn_output)
                first_cnn_loss = tf.reduce_mean(first_cnn_loss)
                
                # Second CNN processing (if available and not in fallback mode)
                if self.pipeline.second_cnn is not None and not self._use_single_cnn_fallback:
                    # Extract wave features
                    wave_features = self._extract_wave_features_safe(reservoir_states, attention_weights)
                    
                    # Prepare second CNN input
                    second_cnn_input = self._prepare_second_cnn_input_safe(wave_features, batch_size)
                    
                    # Second CNN prediction
                    second_cnn_output = self.pipeline.second_cnn(second_cnn_input, training=True)
                    second_cnn_loss = self.loss_fn(target_batch, second_cnn_output)
                    second_cnn_loss = tf.reduce_mean(second_cnn_loss)
                    
                    # Combined loss with weighting
                    combined_loss = (self.config.wave_coordination_weight * first_cnn_loss + 
                                   self.config.final_prediction_weight * second_cnn_loss)
                else:
                    # Single CNN fallback mode
                    second_cnn_loss = tf.constant(0.0)
                    combined_loss = first_cnn_loss
                
            except Exception as forward_error:
                raise ComputationError(f"Forward pass failed: {str(forward_error)}")
        
        # Compute gradients with error handling
        try:
            first_cnn_vars = self.pipeline.first_cnn.trainable_variables
            reservoir_vars = self.pipeline.reservoir.trainable_variables
            
            first_cnn_grads = tape.gradient(first_cnn_loss, first_cnn_vars + reservoir_vars)
            
            # Apply first CNN gradients
            if first_cnn_grads:
                filtered_first_grads = [(g, v) for g, v in zip(first_cnn_grads, first_cnn_vars + reservoir_vars) if g is not None]
                if filtered_first_grads:
                    # Check gradient norms
                    grad_norm = tf.linalg.global_norm([g for g, v in filtered_first_grads])
                    self.training_history['gradient_norms'].append(float(grad_norm))
                    
                    if grad_norm > 10.0:  # Large gradient threshold
                        logger.warning(f"Large gradient norm detected: {grad_norm:.2f}")
                    
                    self.first_cnn_optimizer.apply_gradients(filtered_first_grads)
            
            # Second CNN gradients (if applicable)
            if self.pipeline.second_cnn is not None and not self._use_single_cnn_fallback:
                second_cnn_vars = self.pipeline.second_cnn.trainable_variables
                second_cnn_grads = tape.gradient(second_cnn_loss, second_cnn_vars + reservoir_vars)
                
                if second_cnn_grads:
                    filtered_second_grads = [(g, v) for g, v in zip(second_cnn_grads, second_cnn_vars + reservoir_vars) if g is not None]
                    if filtered_second_grads:
                        self.second_cnn_optimizer.apply_gradients(filtered_second_grads)
            
        except Exception as grad_error:
            raise ComputationError(f"Gradient computation/application failed: {str(grad_error)}")
        
        # Update metrics with error handling
        try:
            self.metrics['first_cnn_accuracy'].update_state(target_batch, first_cnn_output)
            
            if self.pipeline.second_cnn is not None and not self._use_single_cnn_fallback:
                self.metrics['second_cnn_accuracy'].update_state(target_batch, second_cnn_output)
                
                # Combined prediction (weighted average)
                combined_output = (self.config.wave_coordination_weight * first_cnn_output + 
                                  self.config.final_prediction_weight * second_cnn_output)
                self.metrics['combined_accuracy'].update_state(target_batch, combined_output)
            
        except Exception as metric_error:
            logger.warning(f"Metric update failed: {metric_error}")
        
        del tape  # Clean up persistent tape
        
        return {
            'first_cnn_loss': first_cnn_loss,
            'second_cnn_loss': second_cnn_loss,
            'combined_loss': combined_loss
        }
    
    def _extract_wave_features_safe(self, reservoir_states: tf.Tensor, attention_weights: Optional[tf.Tensor]) -> tf.Tensor:
        """Safely extract wave features with fallback handling."""
        try:
            if attention_weights is not None:
                # Use attention-weighted features
                attended_weights = tf.reduce_mean(attention_weights, axis=[1, -1])
                attended_weights = tf.expand_dims(attended_weights, axis=-1)
                weighted_states = reservoir_states * attended_weights
            else:
                # Fallback: use reservoir states directly
                weighted_states = reservoir_states
            
            # Project to wave feature dimension if needed
            if weighted_states.shape[-1] != self.config.wave_feature_dim:
                # Simple linear projection (should be learned in practice)
                projection_layer = tf.keras.layers.Dense(self.config.wave_feature_dim)
                wave_features = projection_layer(weighted_states)
            else:
                wave_features = weighted_states
            
            return wave_features
            
        except Exception as e:
            logger.warning(f"Wave feature extraction failed, using reservoir states: {e}")
            # Fallback: return reservoir states (may need reshaping)
            if reservoir_states.shape[-1] == self.config.wave_feature_dim:
                return reservoir_states
            else:
                # Simple truncation/padding
                if reservoir_states.shape[-1] > self.config.wave_feature_dim:
                    return reservoir_states[..., :self.config.wave_feature_dim]
                else:
                    padding_size = self.config.wave_feature_dim - reservoir_states.shape[-1]
                    padding = tf.zeros((*reservoir_states.shape[:-1], padding_size))
                    return tf.concat([reservoir_states, padding], axis=-1)
    
    def _prepare_second_cnn_input_safe(self, wave_features: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        """Safely prepare input for second CNN with error handling."""
        try:
            sequence_length = tf.shape(wave_features)[1]
            feature_dim = tf.shape(wave_features)[2]
            
            # Adjust sequence length to match window size
            if sequence_length > self.config.wave_window_size:
                wave_features = wave_features[:, -self.config.wave_window_size:, :]
            elif sequence_length < self.config.wave_window_size:
                padding_size = self.config.wave_window_size - sequence_length
                padding = tf.zeros((batch_size, padding_size, feature_dim))
                wave_features = tf.concat([padding, wave_features], axis=1)
            
            return wave_features
            
        except Exception as e:
            logger.warning(f"Second CNN input preparation failed: {e}")
            # Fallback: create zero tensor with correct shape
            return tf.zeros((batch_size, self.config.wave_window_size, self.config.wave_feature_dim))
    
    def _check_loss_validity(self, losses: Dict[str, tf.Tensor]) -> bool:
        """Check if losses are valid (not NaN or Inf)."""
        for loss_name, loss_value in losses.items():
            if tf.math.is_nan(loss_value) or tf.math.is_inf(loss_value):
                logger.warning(f"Invalid loss detected: {loss_name} = {loss_value}")
                return False
        return True
    
    def _detect_training_instability(self, epoch_results: Dict[str, float]) -> bool:
        """Detect training instability patterns."""
        if len(self.training_history['combined_loss']) < 3:
            return False
        
        recent_losses = self.training_history['combined_loss'][-3:]
        
        # Check for exploding loss
        if any(loss > 100.0 for loss in recent_losses):
            logger.warning("Exploding loss detected")
            return True
        
        # Check for rapid loss increase
        if len(recent_losses) >= 2:
            loss_increase = recent_losses[-1] / recent_losses[-2]
            if loss_increase > 2.0:
                logger.warning(f"Rapid loss increase detected: {loss_increase:.2f}x")
                return True
        
        return False
    
    def _attempt_training_recovery(self) -> bool:
        """Attempt to recover from training instability."""
        try:
            self._recovery_attempts += 1
            logger.info(f"Attempting training recovery (attempt {self._recovery_attempts})")
            
            # Reduce learning rate
            current_lr = float(self.first_cnn_optimizer.learning_rate)
            new_lr = current_lr * 0.5
            self.first_cnn_optimizer.learning_rate.assign(new_lr)
            
            if self.second_cnn_optimizer:
                self.second_cnn_optimizer.learning_rate.assign(new_lr)
            
            logger.info(f"Reduced learning rate from {current_lr:.6f} to {new_lr:.6f}")
            
            # Clear gradients
            tf.keras.backend.clear_session()
            
            return True
            
        except Exception as e:
            logger.error(f"Training recovery failed: {e}")
            return False
    
    def _attempt_epoch_recovery(self, error: Exception) -> bool:
        """Attempt to recover from epoch-level errors."""
        try:
            self._recovery_attempts += 1
            
            # Memory-related recovery
            if "memory" in str(error).lower() or "oom" in str(error).lower():
                return self._recover_from_memory_error()
            
            # Computation-related recovery
            if "shape" in str(error).lower() or "dimension" in str(error).lower():
                return self._recover_from_shape_error()
            
            # Generic recovery
            return self._generic_recovery()
            
        except Exception as recovery_error:
            logger.error(f"Epoch recovery failed: {recovery_error}")
            return False
    
    def _attempt_batch_recovery(self, error: Exception) -> bool:
        """Attempt to recover from batch-level errors."""
        try:
            # Skip problematic batch and continue
            logger.info("Skipping problematic batch")
            return True
            
        except Exception as recovery_error:
            logger.error(f"Batch recovery failed: {recovery_error}")
            return False
    
    def _recover_from_memory_error(self) -> bool:
        """Recover from memory-related errors."""
        try:
            logger.info("Attempting memory error recovery")
            
            # Emergency memory cleanup
            self._emergency_memory_cleanup()
            
            # Reduce batch size if possible
            if hasattr(self.config, 'training_batch_size') and self.config.training_batch_size > 1:
                self.config.training_batch_size = max(1, self.config.training_batch_size // 2)
                logger.info(f"Reduced batch size to {self.config.training_batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _recover_from_shape_error(self) -> bool:
        """Recover from shape/dimension errors."""
        try:
            logger.info("Attempting shape error recovery")
            
            # Reset pipeline components if needed
            # This is a placeholder - specific recovery would depend on the error
            return False  # Shape errors usually require manual intervention
            
        except Exception as e:
            logger.error(f"Shape recovery failed: {e}")
            return False
    
    def _generic_recovery(self) -> bool:
        """Generic recovery attempt."""
        try:
            logger.info("Attempting generic recovery")
            
            # Clear TensorFlow session
            tf.keras.backend.clear_session()
            
            # Reduce learning rate
            if hasattr(self.first_cnn_optimizer, 'learning_rate'):
                current_lr = float(self.first_cnn_optimizer.learning_rate)
                new_lr = current_lr * 0.8
                self.first_cnn_optimizer.learning_rate.assign(new_lr)
            
            return True
            
        except Exception as e:
            logger.error(f"Generic recovery failed: {e}")
            return False
    
    def _emergency_memory_cleanup(self):
        """Perform emergency memory cleanup."""
        try:
            import gc
            
            # Clear Python garbage
            gc.collect()
            
            # Clear TensorFlow memory
            tf.keras.backend.clear_session()
            
            # Clear wave storage if available
            if hasattr(self.pipeline, 'wave_storage') and self.pipeline.wave_storage:
                self.pipeline.wave_storage.cleanup_old_waves(keep_recent=10)
            
            logger.info("Emergency memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Emergency memory cleanup failed: {e}")
    
    def _attempt_final_recovery(self) -> bool:
        """Attempt final recovery before complete failure."""
        try:
            logger.info("Attempting final recovery")
            
            # Enable single CNN fallback
            self._use_single_cnn_fallback = True
            
            # Perform cleanup
            self._emergency_memory_cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Final recovery failed: {e}")
            return False
    
    def _get_partial_results(self) -> Dict[str, Any]:
        """Get partial results when training fails but some progress was made."""
        return {
            'training_history': self.training_history,
            'final_metrics': {
                'epochs_completed': len(self.training_history['combined_loss']),
                'recovery_attempts': self._recovery_attempts,
                'training_errors': len(self._training_errors)
            },
            'pipeline_status': self.pipeline.get_component_status(),
            'fallback_status': self.pipeline.get_fallback_status(),
            'partial_results': True
        }
    
    def _cleanup_training_resources(self):
        """Clean up training resources after completion or failure."""
        try:
            # Clear batch times
            self._batch_times.clear()
            
            # Reset memory monitor
            self._memory_monitor = MemoryMonitor()
            
            # Clear training errors
            self._training_errors = []
            
            logger.debug("Training resource cleanup completed")
            
        except Exception as e:
            logger.warning(f"Training resource cleanup failed: {e}")
    
    def _update_progress_enhanced(self, epoch: int, total_epochs: int, batch_idx: int, 
                                total_batches: int, batch_losses: Dict[str, tf.Tensor]):
        """Enhanced progress update with additional monitoring."""
        # Update memory usage
        self._memory_monitor.update()
        
        # Calculate wave storage utilization
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        wave_utilization = storage_stats['utilization_percent']
        
        # Calculate attention entropy if available
        attention_entropy = 0.0
        if hasattr(self.pipeline.reservoir, 'compute_attention_entropy'):
            entropy_tensor = self.pipeline.reservoir.compute_attention_entropy()
            if entropy_tensor is not None:
                attention_entropy = float(tf.reduce_mean(entropy_tensor))
        
        # Estimate remaining time
        if self._batch_times:
            avg_batch_time = np.mean(list(self._batch_times)[-10:])
            remaining_batches = (total_epochs - epoch - 1) * total_batches + (total_batches - batch_idx)
            estimated_time_remaining = avg_batch_time * remaining_batches
        else:
            estimated_time_remaining = 0.0
        
        # Create enhanced progress object
        progress = TrainingProgress(
            current_epoch=epoch + 1,
            total_epochs=total_epochs,
            first_cnn_loss=float(batch_losses['first_cnn_loss']),
            second_cnn_loss=float(batch_losses['second_cnn_loss']),
            combined_loss=float(batch_losses['combined_loss']),
            wave_storage_utilization=wave_utilization,
            attention_entropy=attention_entropy,
            estimated_time_remaining=estimated_time_remaining,
            learning_rate=float(self.first_cnn_optimizer.learning_rate),
            batch_processed=batch_idx + 1,
            total_batches=total_batches
        )
        
        self.last_progress = progress
        
        # Call progress callbacks with error handling
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)
    
    def _update_training_history_enhanced(self, epoch_results: Dict[str, float]):
        """Enhanced training history update with additional metrics."""
        # Standard metrics
        self.training_history['first_cnn_loss'].append(epoch_results['first_cnn_loss'])
        self.training_history['second_cnn_loss'].append(epoch_results['second_cnn_loss'])
        self.training_history['combined_loss'].append(epoch_results['combined_loss'])
        
        # Memory usage
        memory_stats = self._memory_monitor.get_stats()
        self.training_history['memory_usage'].append(memory_stats['current_mb'])
        
        # Wave storage utilization
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        self.training_history['wave_storage_utilization'].append(storage_stats['utilization_percent'])
        
        # Attention entropy
        if hasattr(self.pipeline.reservoir, 'compute_attention_entropy'):
            entropy_tensor = self.pipeline.reservoir.compute_attention_entropy()
            if entropy_tensor is not None:
                entropy_value = float(tf.reduce_mean(entropy_tensor))
                self.training_history['attention_entropy'].append(entropy_value)
            else:
                self.training_history['attention_entropy'].append(0.0)
        else:
            self.training_history['attention_entropy'].append(0.0)
        
        # Learning rate
        self.training_history['learning_rate'].append(float(self.first_cnn_optimizer.learning_rate))
    
    def _log_epoch_summary_enhanced(self, epoch: int, epoch_results: Dict[str, float], epoch_time: float):
        """Enhanced epoch summary logging with additional information."""
        logger.info("=== Epoch %d Summary ===", epoch + 1)
        logger.info("Time: %.2f seconds", epoch_time)
        logger.info("First CNN Loss: %.4f", epoch_results['first_cnn_loss'])
        logger.info("Second CNN Loss: %.4f", epoch_results['second_cnn_loss'])
        logger.info("Combined Loss: %.4f", epoch_results['combined_loss'])
        
        # Log validation results if available
        if 'val_combined_loss' in epoch_results:
            logger.info("Validation Combined Loss: %.4f", epoch_results['val_combined_loss'])
        
        # Log metrics
        logger.info("First CNN Accuracy: %.4f", float(self.metrics['first_cnn_accuracy'].result()))
        if not self._use_single_cnn_fallback:
            logger.info("Second CNN Accuracy: %.4f", float(self.metrics['second_cnn_accuracy'].result()))
            logger.info("Combined Accuracy: %.4f", float(self.metrics['combined_accuracy'].result()))
        
        # Log system metrics
        memory_stats = self._memory_monitor.get_stats()
        logger.info("Memory Usage: %.1f MB (Peak: %.1f MB)", 
                   memory_stats['current_mb'], memory_stats['peak_mb'])
        
        # Log wave storage stats
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        logger.info("Wave Storage Utilization: %.1f%%", storage_stats['utilization_percent'])
        
        # Log recovery information if applicable
        if self._recovery_attempts > 0:
            logger.info("Recovery Attempts: %d", self._recovery_attempts)
        
        if self._use_single_cnn_fallback:
            logger.info("Mode: Single CNN Fallback")
        
        logger.info("========================")
    
    def _should_stop_early_enhanced(self, epoch_results: Dict[str, float]) -> bool:
        """Enhanced early stopping with additional criteria."""
        # Standard early stopping based on validation loss
        if 'val_combined_loss' in epoch_results and len(self.training_history['combined_loss']) >= 5:
            recent_losses = self.training_history['combined_loss'][-5:]
            if all(loss < epoch_results['val_combined_loss'] for loss in recent_losses):
                return True
        
        # Stop if loss becomes NaN or Inf
        if (np.isnan(epoch_results['combined_loss']) or 
            np.isinf(epoch_results['combined_loss'])):
            logger.error("Loss became NaN or Inf, stopping training")
            return True
        
        # Stop if too many recovery attempts
        if self._recovery_attempts >= self._max_recovery_attempts:
            logger.warning("Maximum recovery attempts reached, stopping training")
            return True
        
        # Stop if memory usage is too high consistently
        if len(self.training_history['memory_usage']) >= 3:
            recent_memory = self.training_history['memory_usage'][-3:]
            if all(mem > 0.95 * self.config.max_memory_usage_gb * 1024 for mem in recent_memory):
                logger.warning("Consistently high memory usage, stopping training")
                return True
        
        return False
    
    def _validate_epoch_with_recovery(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Validate epoch with error recovery."""
        try:
            return self._validate_epoch(val_dataset)
        except Exception as e:
            logger.warning(f"Validation failed, attempting recovery: {e}")
            
            # Try with smaller validation batches
            try:
                val_losses = {'val_combined_loss': 0.0}
                batch_count = 0
                
                for input_batch, target_batch in val_dataset.take(10):  # Limit validation batches
                    try:
                        # Simple forward pass without training
                        embedded = self.pipeline.embedder(input_batch)
                        
                        if isinstance(self.pipeline.reservoir, AttentiveReservoir):
                            reservoir_states, _ = self.pipeline.reservoir(embedded, training=False)
                        else:
                            reservoir_states = self.pipeline.reservoir(embedded, training=False)
                        
                        first_cnn_output = self.pipeline.first_cnn(reservoir_states, training=False)
                        first_cnn_loss = tf.reduce_mean(self.loss_fn(target_batch, first_cnn_output))
                        
                        val_losses['val_combined_loss'] += float(first_cnn_loss)
                        batch_count += 1
                        
                    except Exception as batch_error:
                        logger.debug(f"Validation batch failed: {batch_error}")
                        continue
                
                if batch_count > 0:
                    val_losses['val_combined_loss'] /= batch_count
                    return val_losses
                else:
                    return {}
                    
            except Exception as recovery_error:
                logger.warning(f"Validation recovery failed: {recovery_error}")
                return {}
    
    def _calculate_final_results_enhanced(self, total_time: float) -> Dict[str, Any]:
        """Calculate enhanced final training results with additional metrics."""
        final_metrics = {}
        
        # Training metrics
        for metric_name, metric in self.metrics.items():
            try:
                final_metrics[metric_name] = float(metric.result())
            except:
                final_metrics[metric_name] = 0.0
        
        # Loss progression
        if self.training_history['combined_loss']:
            final_metrics['initial_loss'] = self.training_history['combined_loss'][0]
            final_metrics['final_loss'] = self.training_history['combined_loss'][-1]
            final_metrics['loss_improvement'] = (
                final_metrics['initial_loss'] - final_metrics['final_loss']
            )
            final_metrics['best_loss'] = min(self.training_history['combined_loss'])
        
        # Memory statistics
        if self.training_history['memory_usage']:
            final_metrics['peak_memory_mb'] = max(self.training_history['memory_usage'])
            final_metrics['avg_memory_mb'] = np.mean(self.training_history['memory_usage'])
        
        # Wave storage statistics
        storage_stats = self.pipeline.wave_storage.get_storage_stats()
        final_metrics['final_wave_utilization'] = storage_stats['utilization_percent']
        final_metrics['wave_memory_used_mb'] = storage_stats['memory_used_mb']
        
        # Attention analysis
        if self.training_history['attention_entropy']:
            final_metrics['avg_attention_entropy'] = np.mean(self.training_history['attention_entropy'])
            final_metrics['final_attention_entropy'] = self.training_history['attention_entropy'][-1]
        
        # Training efficiency
        final_metrics['total_training_time'] = total_time
        final_metrics['epochs_completed'] = len(self.training_history['combined_loss'])
        final_metrics['avg_epoch_time'] = np.mean(self.training_history['epoch_times'])
        
        # Error and recovery statistics
        final_metrics['recovery_attempts'] = self._recovery_attempts
        final_metrics['training_errors'] = len(self._training_errors)
        final_metrics['fallback_mode_used'] = self._use_single_cnn_fallback
        
        # Gradient statistics
        if self.training_history['gradient_norms']:
            final_metrics['avg_gradient_norm'] = np.mean(self.training_history['gradient_norms'])
            final_metrics['max_gradient_norm'] = max(self.training_history['gradient_norms'])
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics,
            'pipeline_status': self.pipeline.get_component_status(),
            'fallback_status': self.pipeline.get_fallback_status(),
            'error_summary': {
                'total_errors': len(self._training_errors),
                'recovery_attempts': self._recovery_attempts,
                'successful_recovery': self._recovery_attempts < self._max_recovery_attempts
            }
        }
    
    def save_training_state(self, filepath: str):
        """Save current training state for resuming later."""
        state = {
            'current_epoch': self.current_epoch,
            'training_history': self.training_history,
            'config': self.config.__dict__,
            'optimizer_states': {
                'first_cnn': self.first_cnn_optimizer.get_config(),
                'second_cnn': self.second_cnn_optimizer.get_config() if self.second_cnn_optimizer else None
            },
            'recovery_state': {
                'recovery_attempts': self._recovery_attempts,
                'use_single_cnn_fallback': self._use_single_cnn_fallback,
                'training_errors': [(str(e[0]), str(e[1])) for e in self._training_errors]
            },
            'fallback_status': self.pipeline.get_fallback_status()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Training state saved to {filepath}")
        
        logger.info("Training state saved to: %s", filepath)
    
    def load_training_state(self, filepath: str):
        """Load training state to resume training."""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_epoch = state['current_epoch']
        self.training_history = state['training_history']
        
        logger.info("Training state loaded from: %s", filepath)
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        status = "training" if self.is_training else "idle"
        return f"DualCNNTrainer(status={status}, epoch={self.current_epoch})"
    
    def _estimate_total_batches(self, dataset: tf.data.Dataset) -> int:
        """Estimate total number of batches in dataset."""
        try:
            cardinality = tf.data.experimental.cardinality(dataset).numpy()
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
                # Fallback: count batches (expensive but accurate)
                return sum(1 for _ in dataset.take(1000))  # Sample first 1000 batches
            return int(cardinality)
        except Exception:
            return 100  # Default fallback
    
    def _detect_training_instability(self, epoch_results: Dict[str, float]) -> bool:
        """Detect training instability from metrics."""
        # Check for NaN or infinite losses
        for key in ['first_cnn_loss', 'second_cnn_loss', 'combined_loss']:
            loss_value = epoch_results.get(key, 0)
            if np.isnan(loss_value) or np.isinf(loss_value):
                logger.error(f"Training instability detected: {key} = {loss_value}")
                return True
        
        # Check for exploding gradients (indicated by sudden loss spikes)
        if len(self.training_history['combined_loss']) >= 3:
            recent_losses = self.training_history['combined_loss'][-3:]
            if recent_losses[-1] > recent_losses[-2] * 2:  # Loss doubled
                logger.warning("Potential exploding gradients detected")
                return True
        
        return False
    
    def _attempt_training_recovery(self) -> bool:
        """Attempt to recover from training instability."""
        try:
            self._recovery_attempts += 1
            logger.info(f"Attempting training recovery (attempt {self._recovery_attempts})")
            
            # Reduce learning rate
            new_lr = self.first_cnn_optimizer.learning_rate * 0.5
            self.first_cnn_optimizer.learning_rate.assign(new_lr)
            self.second_cnn_optimizer.learning_rate.assign(new_lr)
            
            # Clear memory
            if self._memory_monitor:
                self._memory_monitor.cleanup_memory()
            
            # Optimize wave storage
            if self.wave_storage_optimizer:
                self.wave_storage_optimizer.optimize_storage_memory()
            
            logger.info(f"Recovery applied: reduced learning rate to {new_lr}")
            return True
            
        except Exception as e:
            logger.error(f"Training recovery failed: {e}")
            return False
    
    def _attempt_epoch_recovery(self, epoch_error: Exception) -> bool:
        """Attempt to recover from epoch-level errors."""
        try:
            logger.info("Attempting epoch recovery...")
            
            # Clear TensorFlow session
            tf.keras.backend.clear_session()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset metrics
            for metric in self.metrics.values():
                metric.reset_states()
            
            return True
            
        except Exception as e:
            logger.error(f"Epoch recovery failed: {e}")
            return False
    
    def _attempt_final_recovery(self) -> bool:
        """Attempt final recovery before complete failure."""
        try:
            logger.info("Attempting final recovery...")
            
            # Enable single CNN fallback if available
            if hasattr(self.pipeline, '_use_single_cnn_fallback'):
                self.pipeline._use_single_cnn_fallback = True
                logger.info("Enabled single CNN fallback mode")
                return True
            
            return False
            
        except Exception:
            return False
    
    def _get_partial_results(self) -> Dict[str, Any]:
        """Get partial results when training fails."""
        return {
            'status': 'partial_completion',
            'completed_epochs': self.current_epoch,
            'training_history': self.training_history,
            'error_count': len(self._training_errors),
            'recovery_attempts': self._recovery_attempts,
            'performance_summary': self.performance_monitor.get_performance_summary() if self.performance_monitor else {}
        }
    
    def _cleanup_training_resources(self):
        """Clean up training resources and memory."""
        try:
            # Clear TensorFlow session
            tf.keras.backend.clear_session()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clean up wave storage
            if self.pipeline.wave_storage:
                self.pipeline.wave_storage.clear_storage()
            
            logger.info("Training resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {e}") 
   
    def _safe_config_dict(self, config) -> Dict[str, Any]:
        """Safely extract configuration values for error reporting."""
        safe_dict = {}
        if not hasattr(config, '__dict__'):
            return safe_dict
        
        for key, value in config.__dict__.items():
            try:
                # Only include simple types that can be safely formatted
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_dict[key] = value
                elif isinstance(value, (list, tuple)):
                    # Include lists/tuples of simple types
                    if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                        safe_dict[key] = value
                    else:
                        safe_dict[key] = f"<{type(value).__name__} with {len(value)} items>"
                elif isinstance(value, dict):
                    safe_dict[key] = f"<dict with {len(value)} keys>"
                else:
                    safe_dict[key] = f"<{type(value).__name__} object>"
            except Exception:
                safe_dict[key] = "<formatting error>"
        
        return safe_dict