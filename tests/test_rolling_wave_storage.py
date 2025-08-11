"""
Unit tests for RollingWaveStorage component.

Tests cover circular buffer functionality, memory management, wave storage and retrieval,
and error handling scenarios.
"""

import pytest
import numpy as np
import tensorflow as tf
import threading
import time
from unittest.mock import patch

from lsm_lite.core.rolling_wave_storage import RollingWaveStorage, WaveStorageError


class TestRollingWaveStorage:
    """Test suite for RollingWaveStorage component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_dim = 64
        self.max_sequence_length = 100
        self.window_size = 10
        self.overlap = 2
        self.storage = RollingWaveStorage(
            max_sequence_length=self.max_sequence_length,
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            overlap=self.overlap,
            max_memory_mb=1.0  # Small memory limit for testing
        )
    
    def test_initialization(self):
        """Test proper initialization of RollingWaveStorage."""
        assert self.storage.max_sequence_length == self.max_sequence_length
        assert self.storage.feature_dim == self.feature_dim
        assert self.storage.window_size == self.window_size
        assert self.storage.overlap == self.overlap
        assert len(self.storage) == 0
        
        # Check that buffer is properly sized
        stats = self.storage.get_storage_stats()
        assert stats['max_capacity'] > 0
        assert stats['feature_dim'] == self.feature_dim
        assert stats['utilization_percent'] == 0.0
    
    def test_store_wave_basic(self):
        """Test basic wave storage functionality."""
        # Create test wave
        wave_output = tf.random.normal((self.feature_dim,))
        sequence_position = 5
        confidence_score = 0.8
        
        # Store wave
        self.storage.store_wave(wave_output, sequence_position, confidence_score)
        
        # Check storage stats
        assert len(self.storage) == 1
        stats = self.storage.get_storage_stats()
        assert stats['stored_count'] == 1
        assert stats['utilization_percent'] > 0
    
    def test_store_wave_numpy_input(self):
        """Test storing wave with numpy array input."""
        wave_output = np.random.randn(self.feature_dim).astype(np.float32)
        sequence_position = 3
        
        self.storage.store_wave(wave_output, sequence_position)
        assert len(self.storage) == 1
    
    def test_store_wave_invalid_dimensions(self):
        """Test error handling for invalid wave dimensions."""
        # Wrong feature dimension
        wave_output = tf.random.normal((self.feature_dim + 10,))
        
        with pytest.raises(WaveStorageError) as exc_info:
            self.storage.store_wave(wave_output, 0)
        
        assert "Feature dimension mismatch" in str(exc_info.value)
        assert exc_info.value.operation == "store_wave"
    
    def test_store_multiple_waves(self):
        """Test storing multiple waves in sequence."""
        num_waves = 5
        
        for i in range(num_waves):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i, confidence_score=0.9)
        
        assert len(self.storage) == num_waves
        stats = self.storage.get_storage_stats()
        assert stats['stored_count'] == num_waves
    
    def test_circular_buffer_overflow(self):
        """Test circular buffer behavior when exceeding capacity."""
        # Fill beyond capacity
        max_capacity = self.storage.get_storage_stats()['max_capacity']
        
        for i in range(max_capacity + 5):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        # Should not exceed max capacity
        assert len(self.storage) == max_capacity
        stats = self.storage.get_storage_stats()
        assert stats['stored_count'] == max_capacity
        assert stats['utilization_percent'] == 100.0
    
    def test_get_wave_sequence_basic(self):
        """Test basic wave sequence retrieval."""
        # Store some waves
        num_waves = 5
        stored_waves = []
        
        for i in range(num_waves):
            wave_output = tf.random.normal((self.feature_dim,))
            stored_waves.append(wave_output.numpy())
            self.storage.store_wave(wave_output, i)
        
        # Retrieve sequence
        sequence = self.storage.get_wave_sequence(start_pos=1, length=3)
        
        assert sequence.shape == (3, self.feature_dim)
        assert isinstance(sequence, tf.Tensor)
    
    def test_get_wave_sequence_with_metadata(self):
        """Test wave sequence retrieval with metadata."""
        # Store waves with different confidence scores
        for i in range(3):
            wave_output = tf.random.normal((self.feature_dim,))
            confidence = 0.5 + i * 0.2
            self.storage.store_wave(wave_output, i, confidence_score=confidence)
        
        # Retrieve with metadata
        sequence = self.storage.get_wave_sequence(
            start_pos=0, length=3, include_metadata=True
        )
        
        assert sequence.shape == (3, self.feature_dim + 2)
        
        # Check that metadata is included (positions and confidence scores)
        positions = sequence[:, -2].numpy()
        confidences = sequence[:, -1].numpy()
        
        assert len(positions) == 3
        assert len(confidences) == 3
    
    def test_get_wave_sequence_empty_storage(self):
        """Test retrieving from empty storage."""
        with pytest.raises(WaveStorageError) as exc_info:
            self.storage.get_wave_sequence(0, 5)
        
        assert "No waves stored" in str(exc_info.value)
        assert exc_info.value.operation == "get_wave_sequence"
    
    def test_get_wave_sequence_missing_range(self):
        """Test retrieving waves from a range with no stored data."""
        # Store waves at positions 0-2
        for i in range(3):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        # Try to retrieve from positions 10-15 (no data)
        sequence = self.storage.get_wave_sequence(start_pos=10, length=5)
        
        # Should return zeros
        assert sequence.shape == (5, self.feature_dim)
        assert tf.reduce_sum(sequence).numpy() == 0.0
    
    def test_get_rolling_window(self):
        """Test rolling window retrieval."""
        # Store waves around center position
        center_pos = 5
        for i in range(10):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        # Get rolling window
        window = self.storage.get_rolling_window(center_pos)
        
        expected_size = self.window_size
        assert window.shape == (expected_size, self.feature_dim)
    
    def test_get_rolling_window_with_overlap(self):
        """Test rolling window with overlap."""
        # Store waves
        for i in range(15):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        # Get window with overlap
        window = self.storage.get_rolling_window(center_pos=7, include_overlap=True)
        
        expected_size = self.window_size + 2 * self.overlap
        assert window.shape == (expected_size, self.feature_dim)
    
    def test_clear_storage(self):
        """Test storage clearing functionality."""
        # Store some waves
        for i in range(5):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        assert len(self.storage) == 5
        
        # Clear storage
        self.storage.clear_storage()
        
        assert len(self.storage) == 0
        stats = self.storage.get_storage_stats()
        assert stats['stored_count'] == 0
        assert stats['utilization_percent'] == 0.0
    
    def test_cleanup_old_waves(self):
        """Test cleanup of old waves."""
        # Store many waves
        num_waves = 20
        for i in range(num_waves):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        initial_count = len(self.storage)
        keep_recent = 5
        
        # Cleanup old waves
        removed_count = self.storage.cleanup_old_waves(keep_recent=keep_recent)
        
        assert len(self.storage) == keep_recent
        assert removed_count == initial_count - keep_recent
    
    def test_cleanup_old_waves_default(self):
        """Test cleanup with default keep_recent parameter."""
        # Store more waves than window size
        num_waves = self.window_size + 5
        for i in range(num_waves):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        # Cleanup with default (should keep window_size waves)
        removed_count = self.storage.cleanup_old_waves()
        
        assert len(self.storage) == self.window_size
        assert removed_count == 5
    
    def test_cleanup_no_waves_to_remove(self):
        """Test cleanup when no waves need to be removed."""
        # Store fewer waves than keep_recent
        for i in range(3):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        removed_count = self.storage.cleanup_old_waves(keep_recent=5)
        
        assert removed_count == 0
        assert len(self.storage) == 3
    
    def test_thread_safety(self):
        """Test thread safety of storage operations."""
        num_threads = 5
        waves_per_thread = 10
        threads = []
        
        def store_waves(thread_id):
            for i in range(waves_per_thread):
                wave_output = tf.random.normal((self.feature_dim,))
                position = thread_id * waves_per_thread + i
                self.storage.store_wave(wave_output, position)
        
        # Start threads
        for i in range(num_threads):
            thread = threading.Thread(target=store_waves, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check that all waves were stored (up to capacity limit)
        expected_waves = min(num_threads * waves_per_thread, 
                           self.storage.get_storage_stats()['max_capacity'])
        assert len(self.storage) == expected_waves
    
    def test_memory_management(self):
        """Test memory usage tracking."""
        stats_before = self.storage.get_storage_stats()
        memory_before = stats_before['memory_used_mb']
        
        # Store some waves
        for i in range(10):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        stats_after = self.storage.get_storage_stats()
        memory_after = stats_after['memory_used_mb']
        
        # Memory usage should increase
        assert memory_after > memory_before
        assert memory_after <= stats_after['memory_limit_mb']
    
    def test_storage_stats(self):
        """Test storage statistics reporting."""
        stats = self.storage.get_storage_stats()
        
        required_keys = [
            'stored_count', 'max_capacity', 'utilization_percent',
            'memory_used_mb', 'memory_limit_mb', 'feature_dim',
            'window_size', 'overlap'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['feature_dim'] == self.feature_dim
        assert stats['window_size'] == self.window_size
        assert stats['overlap'] == self.overlap
    
    def test_repr(self):
        """Test string representation."""
        # Store some waves
        for i in range(3):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        repr_str = repr(self.storage)
        
        assert "RollingWaveStorage" in repr_str
        assert "stored=3" in repr_str
        assert "capacity=" in repr_str
        assert "utilization=" in repr_str
    
    def test_error_handling_store_wave(self):
        """Test error handling in store_wave method."""
        # Test with invalid input type
        with patch.object(self.storage, '_lock') as mock_lock:
            mock_lock.__enter__.side_effect = Exception("Lock error")
            
            with pytest.raises(WaveStorageError) as exc_info:
                wave_output = tf.random.normal((self.feature_dim,))
                self.storage.store_wave(wave_output, 0)
            
            assert exc_info.value.operation == "store_wave"
    
    def test_error_handling_get_wave_sequence(self):
        """Test error handling in get_wave_sequence method."""
        # Store a wave first
        wave_output = tf.random.normal((self.feature_dim,))
        self.storage.store_wave(wave_output, 0)
        
        # Test with lock error
        with patch.object(self.storage, '_lock') as mock_lock:
            mock_lock.__enter__.side_effect = Exception("Lock error")
            
            with pytest.raises(WaveStorageError) as exc_info:
                self.storage.get_wave_sequence(0, 1)
            
            assert exc_info.value.operation == "get_wave_sequence"
    
    def test_error_handling_clear_storage(self):
        """Test error handling in clear_storage method."""
        with patch.object(self.storage, '_lock') as mock_lock:
            mock_lock.__enter__.side_effect = Exception("Lock error")
            
            with pytest.raises(WaveStorageError) as exc_info:
                self.storage.clear_storage()
            
            assert exc_info.value.operation == "clear_storage"
    
    def test_error_handling_cleanup_old_waves(self):
        """Test error handling in cleanup_old_waves method."""
        # Store some waves first
        for i in range(5):
            wave_output = tf.random.normal((self.feature_dim,))
            self.storage.store_wave(wave_output, i)
        
        with patch.object(self.storage, '_lock') as mock_lock:
            mock_lock.__enter__.side_effect = Exception("Lock error")
            
            with pytest.raises(WaveStorageError) as exc_info:
                self.storage.cleanup_old_waves()
            
            assert exc_info.value.operation == "cleanup_old_waves"


class TestWaveStorageError:
    """Test suite for WaveStorageError exception."""
    
    def test_wave_storage_error_creation(self):
        """Test WaveStorageError exception creation."""
        operation = "test_operation"
        details = "Test error details"
        
        error = WaveStorageError(operation, details)
        
        assert error.operation == operation
        assert error.details == details
        assert operation in str(error)
        assert details in str(error)
    
    def test_wave_storage_error_inheritance(self):
        """Test that WaveStorageError inherits from Exception."""
        error = WaveStorageError("test", "details")
        assert isinstance(error, Exception)


class TestRollingWaveStorageIntegration:
    """Integration tests for RollingWaveStorage with TensorFlow operations."""
    
    def test_tensorflow_integration(self):
        """Test integration with TensorFlow operations."""
        storage = RollingWaveStorage(
            max_sequence_length=50,
            feature_dim=32,
            window_size=5,
            overlap=1
        )
        
        # Create a simple TensorFlow model that generates wave outputs
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='tanh')
        ])
        
        # Generate and store wave outputs
        batch_size = 10
        input_data = tf.random.normal((batch_size, 16))
        wave_outputs = model(input_data)
        
        for i, wave in enumerate(wave_outputs):
            storage.store_wave(wave, i)
        
        # Retrieve sequences
        sequence = storage.get_wave_sequence(2, 5)
        assert sequence.shape == (5, 32)
        
        # Test rolling window
        window = storage.get_rolling_window(5)
        assert window.shape == (5, 32)
    
    def test_large_scale_storage(self):
        """Test storage with larger scale data."""
        storage = RollingWaveStorage(
            max_sequence_length=1000,
            feature_dim=128,
            window_size=20,
            overlap=5,
            max_memory_mb=10.0
        )
        
        # Store many waves
        num_waves = 500
        for i in range(num_waves):
            wave_output = tf.random.normal((128,))
            storage.store_wave(wave_output, i, confidence_score=np.random.rand())
        
        # Test retrieval of various sequences
        for start_pos in [0, 100, 200, 300, 400]:
            sequence = storage.get_wave_sequence(start_pos, 50)
            assert sequence.shape == (50, 128)
        
        # Test cleanup
        initial_count = len(storage)
        removed = storage.cleanup_old_waves(keep_recent=100)
        assert len(storage) == 100
        assert removed == initial_count - 100


if __name__ == "__main__":
    pytest.main([__file__])