"""
Rolling wave storage for dual CNN pipeline.

This module implements efficient storage and retrieval of rolling wave outputs
from the first CNN for use by the second CNN in the dual CNN architecture.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, List, Dict, Any
import logging
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


class WaveStorageError(Exception):
    """Exception raised for wave storage related errors."""
    
    def __init__(self, operation: str, details: str):
        super().__init__(f"Wave storage {operation} failed: {details}")
        self.operation = operation
        self.details = details


class RollingWaveStorage:
    """
    Manages efficient storage and retrieval of rolling wave outputs from the first CNN.
    
    Uses a circular buffer approach for memory-efficient storage of wave sequences
    with configurable window sizes and overlap handling.
    """
    
    def __init__(self, 
                 max_sequence_length: int, 
                 feature_dim: int,
                 window_size: int = 50, 
                 overlap: int = 10,
                 max_memory_mb: float = 100.0):
        """
        Initialize the rolling wave storage.
        
        Args:
            max_sequence_length: Maximum length of sequences to store
            feature_dim: Dimension of wave feature vectors
            window_size: Size of rolling window for wave outputs
            overlap: Number of overlapping elements between windows
            max_memory_mb: Maximum memory usage in MB for storage
        """
        self.max_sequence_length = max_sequence_length
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.overlap = overlap
        self.max_memory_mb = max_memory_mb
        
        # Calculate maximum number of waves we can store based on memory limit
        bytes_per_wave = feature_dim * 4  # float32 = 4 bytes
        max_waves = int((max_memory_mb * 1024 * 1024) / bytes_per_wave)
        self.max_waves = min(max_waves, max_sequence_length)
        
        # Initialize circular buffer storage
        self._buffer = np.zeros((self.max_waves, feature_dim), dtype=np.float32)
        self._positions = np.zeros(self.max_waves, dtype=np.int32)
        self._timestamps = np.zeros(self.max_waves, dtype=np.float64)
        self._confidence_scores = np.zeros(self.max_waves, dtype=np.float32)
        
        # Buffer management
        self._write_index = 0
        self._stored_count = 0
        self._sequence_start = 0
        self._lock = threading.Lock()
        
        logger.info(f"Initialized RollingWaveStorage: max_waves={self.max_waves}, "
                   f"feature_dim={feature_dim}, window_size={window_size}")
    
    def store_wave(self, 
                   wave_output: tf.Tensor, 
                   sequence_position: int,
                   confidence_score: float = 1.0,
                   timestamp: Optional[float] = None) -> None:
        """
        Store a wave output in the circular buffer.
        
        Args:
            wave_output: Wave feature tensor of shape (feature_dim,)
            sequence_position: Position in the sequence
            confidence_score: Confidence score for this wave output
            timestamp: Optional timestamp, uses current time if None
            
        Raises:
            WaveStorageError: If storage operation fails
        """
        try:
            with self._lock:
                # Convert tensor to numpy if needed
                if isinstance(wave_output, tf.Tensor):
                    wave_features = wave_output.numpy()
                else:
                    wave_features = np.array(wave_output)
                
                # Validate input dimensions
                if wave_features.shape[-1] != self.feature_dim:
                    raise WaveStorageError(
                        "store_wave",
                        f"Feature dimension mismatch: expected {self.feature_dim}, "
                        f"got {wave_features.shape[-1]}"
                    )
                
                # Flatten if needed
                if len(wave_features.shape) > 1:
                    wave_features = wave_features.flatten()[:self.feature_dim]
                
                # Store in circular buffer
                self._buffer[self._write_index] = wave_features
                self._positions[self._write_index] = sequence_position
                self._confidence_scores[self._write_index] = confidence_score
                self._timestamps[self._write_index] = timestamp or time.time()
                
                # Update indices
                self._write_index = (self._write_index + 1) % self.max_waves
                self._stored_count = min(self._stored_count + 1, self.max_waves)
                
                logger.debug(f"Stored wave at position {sequence_position}, "
                           f"buffer utilization: {self._stored_count}/{self.max_waves}")
                
        except Exception as e:
            raise WaveStorageError("store_wave", str(e))
    
    def get_wave_sequence(self, 
                         start_pos: int, 
                         length: int,
                         include_metadata: bool = False) -> tf.Tensor:
        """
        Retrieve a sequence of wave outputs.
        
        Args:
            start_pos: Starting position in the sequence
            length: Number of wave outputs to retrieve
            include_metadata: Whether to include position and confidence data
            
        Returns:
            Tensor of shape (length, feature_dim) or (length, feature_dim + 2) if metadata included
            
        Raises:
            WaveStorageError: If retrieval operation fails
        """
        try:
            with self._lock:
                if self._stored_count == 0:
                    raise WaveStorageError("get_wave_sequence", "No waves stored")
                
                # Find waves within the requested range
                valid_indices = []
                for i in range(self._stored_count):
                    buffer_idx = (self._write_index - self._stored_count + i) % self.max_waves
                    pos = self._positions[buffer_idx]
                    if start_pos <= pos < start_pos + length:
                        valid_indices.append(buffer_idx)
                
                if not valid_indices:
                    # Return zeros if no valid waves found
                    logger.warning(f"No waves found for range [{start_pos}, {start_pos + length})")
                    output_dim = self.feature_dim + 2 if include_metadata else self.feature_dim
                    return tf.zeros((length, output_dim), dtype=tf.float32)
                
                # Sort by position
                valid_indices.sort(key=lambda idx: self._positions[idx])
                
                # Build output sequence
                output_features = []
                for i in range(length):
                    target_pos = start_pos + i
                    
                    # Find closest wave
                    best_idx = None
                    min_distance = float('inf')
                    
                    for idx in valid_indices:
                        distance = abs(self._positions[idx] - target_pos)
                        if distance < min_distance:
                            min_distance = distance
                            best_idx = idx
                    
                    if best_idx is not None:
                        features = self._buffer[best_idx].copy()
                        if include_metadata:
                            metadata = np.array([
                                self._positions[best_idx],
                                self._confidence_scores[best_idx]
                            ])
                            features = np.concatenate([features, metadata])
                        output_features.append(features)
                    else:
                        # Fill with zeros if no suitable wave found
                        output_dim = self.feature_dim + 2 if include_metadata else self.feature_dim
                        output_features.append(np.zeros(output_dim))
                
                result = tf.constant(output_features, dtype=tf.float32)
                logger.debug(f"Retrieved wave sequence: start_pos={start_pos}, "
                           f"length={length}, found={len(valid_indices)} waves")
                
                return result
                
        except Exception as e:
            raise WaveStorageError("get_wave_sequence", str(e))
    
    def get_rolling_window(self, 
                          center_pos: int, 
                          include_overlap: bool = False) -> tf.Tensor:
        """
        Get a rolling window of waves centered around a position.
        
        Args:
            center_pos: Center position for the window
            include_overlap: Whether to include overlap regions
            
        Returns:
            Tensor of shape (window_size, feature_dim) or (window_size + 2*overlap, feature_dim)
        """
        window_size = self.window_size
        if include_overlap:
            window_size += 2 * self.overlap
        
        start_pos = center_pos - window_size // 2
        return self.get_wave_sequence(start_pos, window_size)
    
    def clear_storage(self) -> None:
        """Reset storage for new sequences."""
        try:
            with self._lock:
                self._write_index = 0
                self._stored_count = 0
                self._sequence_start = 0
                
                # Clear buffers
                self._buffer.fill(0)
                self._positions.fill(0)
                self._timestamps.fill(0)
                self._confidence_scores.fill(0)
                
                logger.info("Cleared wave storage")
                
        except Exception as e:
            raise WaveStorageError("clear_storage", str(e))
    
    def get_storage_stats(self) -> dict:
        """
        Get current storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        with self._lock:
            memory_used_mb = (self._stored_count * self.feature_dim * 4) / (1024 * 1024)
            utilization = self._stored_count / self.max_waves if self.max_waves > 0 else 0
            
            return {
                'stored_count': self._stored_count,
                'max_capacity': self.max_waves,
                'utilization_percent': utilization * 100,
                'memory_used_mb': memory_used_mb,
                'memory_limit_mb': self.max_memory_mb,
                'feature_dim': self.feature_dim,
                'window_size': self.window_size,
                'overlap': self.overlap
            }
    
    def cleanup_old_waves(self, keep_recent: int = None) -> int:
        """
        Clean up old waves to free memory.
        
        Args:
            keep_recent: Number of recent waves to keep, defaults to window_size
            
        Returns:
            Number of waves removed
        """
        if keep_recent is None:
            keep_recent = self.window_size
            
        try:
            with self._lock:
                if self._stored_count <= keep_recent:
                    return 0
                
                waves_to_remove = self._stored_count - keep_recent
                
                # Shift the buffer to keep only recent waves
                for i in range(keep_recent):
                    old_idx = (self._write_index - keep_recent + i) % self.max_waves
                    new_idx = i
                    
                    if old_idx != new_idx:
                        self._buffer[new_idx] = self._buffer[old_idx]
                        self._positions[new_idx] = self._positions[old_idx]
                        self._timestamps[new_idx] = self._timestamps[old_idx]
                        self._confidence_scores[new_idx] = self._confidence_scores[old_idx]
                
                # Update indices
                self._write_index = keep_recent % self.max_waves
                self._stored_count = keep_recent
                
                logger.info(f"Cleaned up {waves_to_remove} old waves, "
                           f"kept {keep_recent} recent waves")
                
                return waves_to_remove
                
        except Exception as e:
            raise WaveStorageError("cleanup_old_waves", str(e))
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage with advanced techniques.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            with self._lock:
                stats_before = self.get_storage_stats()
                optimizations_applied = []
                
                # 1. Compress low-confidence waves
                compressed_count = self._compress_low_confidence_waves()
                if compressed_count > 0:
                    optimizations_applied.append(f"compressed_{compressed_count}_low_confidence_waves")
                
                # 2. Remove duplicate or near-duplicate waves
                deduplicated_count = self._remove_duplicate_waves()
                if deduplicated_count > 0:
                    optimizations_applied.append(f"removed_{deduplicated_count}_duplicate_waves")
                
                # 3. Quantize wave features to reduce precision
                quantized_savings = self._quantize_wave_features()
                if quantized_savings > 0:
                    optimizations_applied.append(f"quantized_features_saved_{quantized_savings:.1f}MB")
                
                # 4. Adaptive cleanup based on access patterns
                adaptive_cleanup_count = self._adaptive_cleanup_by_usage()
                if adaptive_cleanup_count > 0:
                    optimizations_applied.append(f"adaptive_cleanup_{adaptive_cleanup_count}_waves")
                
                stats_after = self.get_storage_stats()
                
                return {
                    'memory_before_mb': stats_before['memory_used_mb'],
                    'memory_after_mb': stats_after['memory_used_mb'],
                    'memory_saved_mb': stats_before['memory_used_mb'] - stats_after['memory_used_mb'],
                    'optimizations_applied': optimizations_applied,
                    'utilization_before': stats_before['utilization_percent'],
                    'utilization_after': stats_after['utilization_percent']
                }
                
        except Exception as e:
            raise WaveStorageError("optimize_memory_usage", str(e))
    
    def _compress_low_confidence_waves(self) -> int:
        """Compress waves with low confidence scores."""
        compressed_count = 0
        confidence_threshold = 0.5
        
        try:
            for i in range(self._stored_count):
                buffer_idx = (self._write_index - self._stored_count + i) % self.max_waves
                
                if self._confidence_scores[buffer_idx] < confidence_threshold:
                    # Simple compression: reduce precision of low-confidence waves
                    self._buffer[buffer_idx] = np.round(self._buffer[buffer_idx], decimals=2)
                    compressed_count += 1
            
            return compressed_count
            
        except Exception as e:
            logger.warning(f"Wave compression failed: {e}")
            return 0
    
    def _remove_duplicate_waves(self) -> int:
        """Remove duplicate or near-duplicate waves."""
        removed_count = 0
        similarity_threshold = 0.95
        
        try:
            if self._stored_count < 2:
                return 0
            
            # Track waves to remove
            waves_to_remove = set()
            
            for i in range(self._stored_count - 1):
                if i in waves_to_remove:
                    continue
                    
                idx_i = (self._write_index - self._stored_count + i) % self.max_waves
                wave_i = self._buffer[idx_i]
                
                for j in range(i + 1, self._stored_count):
                    if j in waves_to_remove:
                        continue
                        
                    idx_j = (self._write_index - self._stored_count + j) % self.max_waves
                    wave_j = self._buffer[idx_j]
                    
                    # Calculate cosine similarity
                    similarity = np.dot(wave_i, wave_j) / (np.linalg.norm(wave_i) * np.linalg.norm(wave_j))
                    
                    if similarity > similarity_threshold:
                        # Keep the one with higher confidence
                        if self._confidence_scores[idx_i] >= self._confidence_scores[idx_j]:
                            waves_to_remove.add(j)
                        else:
                            waves_to_remove.add(i)
                            break
            
            # Remove duplicate waves by compacting the buffer
            if waves_to_remove:
                new_write_index = 0
                for i in range(self._stored_count):
                    if i not in waves_to_remove:
                        old_idx = (self._write_index - self._stored_count + i) % self.max_waves
                        
                        if new_write_index != old_idx:
                            self._buffer[new_write_index] = self._buffer[old_idx]
                            self._positions[new_write_index] = self._positions[old_idx]
                            self._timestamps[new_write_index] = self._timestamps[old_idx]
                            self._confidence_scores[new_write_index] = self._confidence_scores[old_idx]
                        
                        new_write_index += 1
                
                removed_count = len(waves_to_remove)
                self._stored_count -= removed_count
                self._write_index = new_write_index % self.max_waves
            
            return removed_count
            
        except Exception as e:
            logger.warning(f"Duplicate removal failed: {e}")
            return 0
    
    def _quantize_wave_features(self) -> float:
        """Quantize wave features to reduce memory usage."""
        try:
            # Convert from float32 to float16 for older waves
            memory_saved = 0.0
            quantization_age_threshold = 60.0  # 1 minute
            current_time = time.time()
            
            for i in range(self._stored_count):
                buffer_idx = (self._write_index - self._stored_count + i) % self.max_waves
                wave_age = current_time - self._timestamps[buffer_idx]
                
                if wave_age > quantization_age_threshold:
                    # Quantize to lower precision
                    original_size = self._buffer[buffer_idx].nbytes
                    self._buffer[buffer_idx] = self._buffer[buffer_idx].astype(np.float16).astype(np.float32)
                    memory_saved += (original_size - self._buffer[buffer_idx].nbytes) / (1024**2)
            
            return memory_saved
            
        except Exception as e:
            logger.warning(f"Wave quantization failed: {e}")
            return 0.0
    
    def _adaptive_cleanup_by_usage(self) -> int:
        """Adaptive cleanup based on wave usage patterns."""
        try:
            # This is a simplified implementation
            # In practice, you'd track access patterns and remove least-used waves
            
            if self._stored_count < self.max_waves * 0.8:
                return 0  # No need for cleanup yet
            
            # Remove oldest 20% of waves when storage is 80% full
            waves_to_remove = int(self._stored_count * 0.2)
            return self.cleanup_old_waves(self._stored_count - waves_to_remove)
            
        except Exception as e:
            logger.warning(f"Adaptive cleanup failed: {e}")
            return 0
    
    def get_memory_efficiency_stats(self) -> Dict[str, Any]:
        """Get detailed memory efficiency statistics."""
        try:
            with self._lock:
                if self._stored_count == 0:
                    return {'error': 'No waves stored'}
                
                # Calculate statistics
                confidence_scores = [self._confidence_scores[
                    (self._write_index - self._stored_count + i) % self.max_waves
                ] for i in range(self._stored_count)]
                
                wave_ages = [time.time() - self._timestamps[
                    (self._write_index - self._stored_count + i) % self.max_waves
                ] for i in range(self._stored_count)]
                
                return {
                    'average_confidence': np.mean(confidence_scores),
                    'min_confidence': np.min(confidence_scores),
                    'max_confidence': np.max(confidence_scores),
                    'average_age_seconds': np.mean(wave_ages),
                    'oldest_wave_age_seconds': np.max(wave_ages),
                    'newest_wave_age_seconds': np.min(wave_ages),
                    'memory_fragmentation_ratio': self._calculate_fragmentation_ratio(),
                    'compression_potential': self._estimate_compression_potential()
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        try:
            if self._stored_count == 0:
                return 0.0
            
            # Simple fragmentation estimate based on buffer utilization
            utilization = self._stored_count / self.max_waves
            fragmentation = 1.0 - utilization
            
            return fragmentation
            
        except Exception:
            return 0.0
    
    def _estimate_compression_potential(self) -> float:
        """Estimate potential memory savings from compression."""
        try:
            if self._stored_count == 0:
                return 0.0
            
            # Estimate based on low-confidence waves and duplicates
            low_confidence_count = sum(1 for i in range(self._stored_count)
                                     if self._confidence_scores[
                                         (self._write_index - self._stored_count + i) % self.max_waves
                                     ] < 0.5)
            
            # Rough estimate: 50% compression for low-confidence waves
            compression_potential = (low_confidence_count / self._stored_count) * 0.5
            
            return compression_potential
            
        except Exception:
            return 0.0
    
    def __len__(self) -> int:
        """Return number of stored waves."""
        return self._stored_count
    
    def __repr__(self) -> str:
        """String representation of the storage."""
        stats = self.get_storage_stats()
        return (f"RollingWaveStorage(stored={stats['stored_count']}, "
                f"capacity={stats['max_capacity']}, "
                f"utilization={stats['utilization_percent']:.1f}%)")