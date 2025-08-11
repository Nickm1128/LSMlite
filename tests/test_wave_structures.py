"""
Unit tests for wave output data structures.

This module tests the WaveOutput, TrainingProgress, and WaveOutputBatch classes
for proper functionality, validation, and serialization.
"""

import os
import json
import pickle
import tempfile
import unittest
from datetime import datetime
import numpy as np

# Import the classes to test
from lsm_lite.data.wave_structures import WaveOutput, TrainingProgress, WaveOutputBatch


class TestWaveOutput(unittest.TestCase):
    """Test cases for WaveOutput dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_wave_features = np.random.rand(10, 256).astype(np.float32)
        self.sample_attention_weights = np.random.rand(8, 10).astype(np.float32)
        self.sample_timestamp = datetime.now().timestamp()
        
        self.valid_wave_output = WaveOutput(
            sequence_position=5,
            wave_features=self.sample_wave_features,
            attention_weights=self.sample_attention_weights,
            timestamp=self.sample_timestamp,
            confidence_score=0.85,
            sequence_id="test_seq_001",
            batch_index=2,
            epoch=3,
            loss_value=0.123,
            gradient_norm=1.45
        )
    
    def test_valid_initialization(self):
        """Test valid WaveOutput initialization."""
        wave_output = WaveOutput(
            sequence_position=0,
            wave_features=np.array([[1.0, 2.0, 3.0]]),
            attention_weights=np.array([[0.5, 0.3, 0.2]]),
            timestamp=123456.789,
            confidence_score=0.9
        )
        
        self.assertEqual(wave_output.sequence_position, 0)
        self.assertEqual(wave_output.confidence_score, 0.9)
        self.assertEqual(wave_output.timestamp, 123456.789)
        self.assertIsInstance(wave_output.wave_features, np.ndarray)
        self.assertIsInstance(wave_output.attention_weights, np.ndarray)
    
    def test_invalid_sequence_position(self):
        """Test validation of sequence position."""
        with self.assertRaises(ValueError):
            WaveOutput(
                sequence_position=-1,
                wave_features=np.array([[1.0]]),
                attention_weights=np.array([[1.0]]),
                timestamp=123456.789,
                confidence_score=0.9
            )
    
    def test_invalid_confidence_score(self):
        """Test validation of confidence score."""
        # Test confidence score > 1.0
        with self.assertRaises(ValueError):
            WaveOutput(
                sequence_position=0,
                wave_features=np.array([[1.0]]),
                attention_weights=np.array([[1.0]]),
                timestamp=123456.789,
                confidence_score=1.5
            )
        
        # Test confidence score < 0.0
        with self.assertRaises(ValueError):
            WaveOutput(
                sequence_position=0,
                wave_features=np.array([[1.0]]),
                attention_weights=np.array([[1.0]]),
                timestamp=123456.789,
                confidence_score=-0.1
            )
    
    def test_invalid_timestamp(self):
        """Test validation of timestamp."""
        with self.assertRaises(ValueError):
            WaveOutput(
                sequence_position=0,
                wave_features=np.array([[1.0]]),
                attention_weights=np.array([[1.0]]),
                timestamp=-1.0,
                confidence_score=0.9
            )
    
    def test_invalid_array_dimensions(self):
        """Test validation of array dimensions."""
        # Test 0-dimensional wave features
        with self.assertRaises(ValueError):
            WaveOutput(
                sequence_position=0,
                wave_features=np.array(1.0),  # 0-dimensional
                attention_weights=np.array([[1.0]]),
                timestamp=123456.789,
                confidence_score=0.9
            )
        
        # Test 0-dimensional attention weights
        with self.assertRaises(ValueError):
            WaveOutput(
                sequence_position=0,
                wave_features=np.array([[1.0]]),
                attention_weights=np.array(1.0),  # 0-dimensional
                timestamp=123456.789,
                confidence_score=0.9
            )
    
    def test_properties(self):
        """Test computed properties."""
        self.assertEqual(self.valid_wave_output.feature_dim, 256)
        self.assertEqual(self.valid_wave_output.attention_heads, 8)
        self.assertGreater(self.valid_wave_output.memory_usage_bytes, 0)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        # Test with arrays included
        data_with_arrays = self.valid_wave_output.to_dict(include_arrays=True)
        self.assertIn('wave_features', data_with_arrays)
        self.assertIn('attention_weights', data_with_arrays)
        self.assertIn('wave_features_shape', data_with_arrays)
        self.assertIn('attention_weights_shape', data_with_arrays)
        self.assertEqual(data_with_arrays['sequence_position'], 5)
        self.assertEqual(data_with_arrays['confidence_score'], 0.85)
        
        # Test without arrays
        data_without_arrays = self.valid_wave_output.to_dict(include_arrays=False)
        self.assertNotIn('wave_features', data_without_arrays)
        self.assertNotIn('attention_weights', data_without_arrays)
        self.assertIn('feature_dim', data_without_arrays)
        self.assertIn('attention_heads', data_without_arrays)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        original_data = self.valid_wave_output.to_dict()
        reconstructed = WaveOutput.from_dict(original_data)
        
        self.assertEqual(reconstructed.sequence_position, self.valid_wave_output.sequence_position)
        self.assertEqual(reconstructed.confidence_score, self.valid_wave_output.confidence_score)
        self.assertEqual(reconstructed.timestamp, self.valid_wave_output.timestamp)
        self.assertEqual(reconstructed.sequence_id, self.valid_wave_output.sequence_id)
        self.assertEqual(reconstructed.batch_index, self.valid_wave_output.batch_index)
        self.assertEqual(reconstructed.epoch, self.valid_wave_output.epoch)
        
        # Check array shapes
        self.assertEqual(reconstructed.wave_features.shape, self.valid_wave_output.wave_features.shape)
        self.assertEqual(reconstructed.attention_weights.shape, self.valid_wave_output.attention_weights.shape)
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to JSON
            self.valid_wave_output.save_json(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load from JSON
            loaded_wave_output = WaveOutput.load_json(temp_path)
            
            # Verify data integrity
            self.assertEqual(loaded_wave_output.sequence_position, self.valid_wave_output.sequence_position)
            self.assertEqual(loaded_wave_output.confidence_score, self.valid_wave_output.confidence_score)
            self.assertEqual(loaded_wave_output.sequence_id, self.valid_wave_output.sequence_id)
            
            # Check array shapes (values might have small floating point differences)
            self.assertEqual(loaded_wave_output.wave_features.shape, self.valid_wave_output.wave_features.shape)
            self.assertEqual(loaded_wave_output.attention_weights.shape, self.valid_wave_output.attention_weights.shape)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_pickle_serialization(self):
        """Test pickle serialization and deserialization."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to pickle
            self.valid_wave_output.save_pickle(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load from pickle
            loaded_wave_output = WaveOutput.load_pickle(temp_path)
            
            # Verify exact data integrity (pickle preserves exact values)
            self.assertEqual(loaded_wave_output.sequence_position, self.valid_wave_output.sequence_position)
            self.assertEqual(loaded_wave_output.confidence_score, self.valid_wave_output.confidence_score)
            self.assertEqual(loaded_wave_output.sequence_id, self.valid_wave_output.sequence_id)
            
            # Check arrays are exactly equal
            np.testing.assert_array_equal(loaded_wave_output.wave_features, self.valid_wave_output.wave_features)
            np.testing.assert_array_equal(loaded_wave_output.attention_weights, self.valid_wave_output.attention_weights)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_string_representation(self):
        """Test string representation."""
        str_repr = str(self.valid_wave_output)
        self.assertIn("WaveOutput", str_repr)
        self.assertIn("pos=5", str_repr)
        self.assertIn("confidence=0.850", str_repr)
        self.assertIn("features=(10, 256)", str_repr)
        self.assertIn("attention=(8, 10)", str_repr)


class TestTrainingProgress(unittest.TestCase):
    """Test cases for TrainingProgress dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.current_time = datetime.now().timestamp()
        
        self.valid_progress = TrainingProgress(
            current_epoch=2,
            total_epochs=10,
            current_batch=15,
            total_batches=100,
            first_cnn_loss=0.456,
            second_cnn_loss=0.321,
            combined_loss=0.389,
            wave_storage_utilization=0.75,
            attention_entropy=2.34,
            epoch_start_time=self.current_time - 300,  # 5 minutes ago
            batch_start_time=self.current_time - 30,   # 30 seconds ago
            estimated_time_remaining=1800,  # 30 minutes
            first_cnn_accuracy=0.82,
            second_cnn_accuracy=0.78,
            learning_rate=0.001,
            validation_loss=0.412,
            validation_accuracy=0.80,
            training_stage="training"
        )
    
    def test_valid_initialization(self):
        """Test valid TrainingProgress initialization."""
        progress = TrainingProgress(
            current_epoch=0,
            total_epochs=5,
            current_batch=0,
            total_batches=50,
            first_cnn_loss=1.0,
            second_cnn_loss=1.2,
            combined_loss=1.1,
            wave_storage_utilization=0.5,
            attention_entropy=1.5,
            epoch_start_time=self.current_time,
            batch_start_time=self.current_time,
            estimated_time_remaining=3600
        )
        
        self.assertEqual(progress.current_epoch, 0)
        self.assertEqual(progress.total_epochs, 5)
        self.assertEqual(progress.training_stage, "training")
    
    def test_invalid_epoch_values(self):
        """Test validation of epoch values."""
        # Test negative current epoch
        with self.assertRaises(ValueError):
            TrainingProgress(
                current_epoch=-1,
                total_epochs=5,
                current_batch=0,
                total_batches=50,
                first_cnn_loss=1.0,
                second_cnn_loss=1.2,
                combined_loss=1.1,
                wave_storage_utilization=0.5,
                attention_entropy=1.5,
                epoch_start_time=self.current_time,
                batch_start_time=self.current_time,
                estimated_time_remaining=3600
            )
        
        # Test current epoch > total epochs
        with self.assertRaises(ValueError):
            TrainingProgress(
                current_epoch=6,
                total_epochs=5,
                current_batch=0,
                total_batches=50,
                first_cnn_loss=1.0,
                second_cnn_loss=1.2,
                combined_loss=1.1,
                wave_storage_utilization=0.5,
                attention_entropy=1.5,
                epoch_start_time=self.current_time,
                batch_start_time=self.current_time,
                estimated_time_remaining=3600
            )
    
    def test_invalid_batch_values(self):
        """Test validation of batch values."""
        # Test negative current batch
        with self.assertRaises(ValueError):
            TrainingProgress(
                current_epoch=0,
                total_epochs=5,
                current_batch=-1,
                total_batches=50,
                first_cnn_loss=1.0,
                second_cnn_loss=1.2,
                combined_loss=1.1,
                wave_storage_utilization=0.5,
                attention_entropy=1.5,
                epoch_start_time=self.current_time,
                batch_start_time=self.current_time,
                estimated_time_remaining=3600
            )
        
        # Test current batch > total batches
        with self.assertRaises(ValueError):
            TrainingProgress(
                current_epoch=0,
                total_epochs=5,
                current_batch=51,
                total_batches=50,
                first_cnn_loss=1.0,
                second_cnn_loss=1.2,
                combined_loss=1.1,
                wave_storage_utilization=0.5,
                attention_entropy=1.5,
                epoch_start_time=self.current_time,
                batch_start_time=self.current_time,
                estimated_time_remaining=3600
            )
    
    def test_invalid_loss_values(self):
        """Test validation of loss values."""
        # Test negative first CNN loss
        with self.assertRaises(ValueError):
            TrainingProgress(
                current_epoch=0,
                total_epochs=5,
                current_batch=0,
                total_batches=50,
                first_cnn_loss=-0.1,
                second_cnn_loss=1.2,
                combined_loss=1.1,
                wave_storage_utilization=0.5,
                attention_entropy=1.5,
                epoch_start_time=self.current_time,
                batch_start_time=self.current_time,
                estimated_time_remaining=3600
            )
    
    def test_invalid_utilization_values(self):
        """Test validation of utilization values."""
        # Test wave storage utilization > 1.0
        with self.assertRaises(ValueError):
            TrainingProgress(
                current_epoch=0,
                total_epochs=5,
                current_batch=0,
                total_batches=50,
                first_cnn_loss=1.0,
                second_cnn_loss=1.2,
                combined_loss=1.1,
                wave_storage_utilization=1.5,
                attention_entropy=1.5,
                epoch_start_time=self.current_time,
                batch_start_time=self.current_time,
                estimated_time_remaining=3600
            )
    
    def test_invalid_training_stage(self):
        """Test validation of training stage."""
        with self.assertRaises(ValueError):
            TrainingProgress(
                current_epoch=0,
                total_epochs=5,
                current_batch=0,
                total_batches=50,
                first_cnn_loss=1.0,
                second_cnn_loss=1.2,
                combined_loss=1.1,
                wave_storage_utilization=0.5,
                attention_entropy=1.5,
                epoch_start_time=self.current_time,
                batch_start_time=self.current_time,
                estimated_time_remaining=3600,
                training_stage="invalid_stage"
            )
    
    def test_progress_properties(self):
        """Test computed progress properties."""
        # Test epoch progress
        expected_epoch_progress = 2 / 10  # current_epoch / total_epochs
        self.assertAlmostEqual(self.valid_progress.epoch_progress, expected_epoch_progress)
        
        # Test batch progress
        expected_batch_progress = 15 / 100  # current_batch / total_batches
        self.assertAlmostEqual(self.valid_progress.batch_progress, expected_batch_progress)
        
        # Test overall progress
        expected_overall = (2 + 0.15) / 10  # (completed_epochs + batch_progress) / total_epochs
        self.assertAlmostEqual(self.valid_progress.overall_progress, expected_overall)
        
        # Test elapsed time (should be positive)
        self.assertGreater(self.valid_progress.elapsed_time, 0)
        
        # Test is_improving (should be True with validation data)
        self.assertTrue(self.valid_progress.is_improving)
    
    def test_update_methods(self):
        """Test update methods."""
        # Test update_batch
        self.valid_progress.update_batch(
            batch_idx=20,
            first_loss=0.4,
            second_loss=0.3,
            combined_loss=0.35,
            wave_utilization=0.8,
            attention_entropy=2.1
        )
        
        self.assertEqual(self.valid_progress.current_batch, 20)
        self.assertEqual(self.valid_progress.first_cnn_loss, 0.4)
        self.assertEqual(self.valid_progress.second_cnn_loss, 0.3)
        self.assertEqual(self.valid_progress.combined_loss, 0.35)
        self.assertEqual(self.valid_progress.wave_storage_utilization, 0.8)
        self.assertEqual(self.valid_progress.attention_entropy, 2.1)
        
        # Test update_epoch
        self.valid_progress.update_epoch(3)
        self.assertEqual(self.valid_progress.current_epoch, 3)
        self.assertEqual(self.valid_progress.current_batch, 0)
        
        # Test update_validation
        self.valid_progress.update_validation(0.35, 0.85)
        self.assertEqual(self.valid_progress.validation_loss, 0.35)
        self.assertEqual(self.valid_progress.validation_accuracy, 0.85)
        
        # Test mark_completed
        self.valid_progress.mark_completed()
        self.assertEqual(self.valid_progress.training_stage, "completed")
        self.assertEqual(self.valid_progress.estimated_time_remaining, 0.0)
        
        # Test mark_failed
        error_msg = "Test error"
        self.valid_progress.mark_failed(error_msg)
        self.assertEqual(self.valid_progress.training_stage, "failed")
        self.assertEqual(self.valid_progress.error_message, error_msg)
        self.assertEqual(self.valid_progress.estimated_time_remaining, 0.0)
    
    def test_serialization(self):
        """Test dictionary conversion and JSON serialization."""
        # Test to_dict
        data = self.valid_progress.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['current_epoch'], 2)
        self.assertEqual(data['total_epochs'], 10)
        self.assertEqual(data['training_stage'], "training")
        
        # Test from_dict
        reconstructed = TrainingProgress.from_dict(data)
        self.assertEqual(reconstructed.current_epoch, self.valid_progress.current_epoch)
        self.assertEqual(reconstructed.total_epochs, self.valid_progress.total_epochs)
        self.assertEqual(reconstructed.training_stage, self.valid_progress.training_stage)
        
        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to JSON
            self.valid_progress.save_json(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load from JSON
            loaded_progress = TrainingProgress.load_json(temp_path)
            
            # Verify data integrity
            self.assertEqual(loaded_progress.current_epoch, self.valid_progress.current_epoch)
            self.assertEqual(loaded_progress.total_epochs, self.valid_progress.total_epochs)
            self.assertEqual(loaded_progress.training_stage, self.valid_progress.training_stage)
            self.assertEqual(loaded_progress.first_cnn_loss, self.valid_progress.first_cnn_loss)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_progress_summary(self):
        """Test progress summary generation."""
        summary = self.valid_progress.get_progress_summary()
        self.assertIn("Training Progress:", summary)
        self.assertIn("Epoch: 2/10", summary)
        self.assertIn("Batch: 15/100", summary)
        self.assertIn("Losses:", summary)
        self.assertIn("Wave Storage:", summary)
        self.assertIn("Attention Entropy:", summary)
        self.assertIn("ETA:", summary)
        self.assertIn("Validation Loss:", summary)
        self.assertIn("Status: training", summary)
    
    def test_string_representation(self):
        """Test string representation."""
        str_repr = str(self.valid_progress)
        self.assertIn("TrainingProgress", str_repr)
        self.assertIn("epoch=2/10", str_repr)
        self.assertIn("batch=15/100", str_repr)
        self.assertIn("status=training", str_repr)


class TestWaveOutputBatch(unittest.TestCase):
    """Test cases for WaveOutputBatch class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch = WaveOutputBatch(max_size=5)
        
        # Create sample wave outputs
        self.wave_outputs = []
        for i in range(3):
            wave_output = WaveOutput(
                sequence_position=i,
                wave_features=np.random.rand(5, 10).astype(np.float32),
                attention_weights=np.random.rand(4, 5).astype(np.float32),
                timestamp=datetime.now().timestamp() + i,
                confidence_score=0.8 + i * 0.05,
                sequence_id=f"seq_{i:03d}"
            )
            self.wave_outputs.append(wave_output)
            self.batch.add(wave_output)
    
    def test_initialization(self):
        """Test WaveOutputBatch initialization."""
        batch = WaveOutputBatch(max_size=10)
        self.assertEqual(batch.max_size, 10)
        self.assertEqual(len(batch), 0)
        self.assertEqual(batch.get_memory_usage(), 0)
    
    def test_add_wave_output(self):
        """Test adding wave outputs to batch."""
        self.assertEqual(len(self.batch), 3)
        
        # Add more wave outputs to test capacity limit
        for i in range(3, 8):  # Add 5 more (total would be 8, but max is 5)
            wave_output = WaveOutput(
                sequence_position=i,
                wave_features=np.random.rand(5, 10).astype(np.float32),
                attention_weights=np.random.rand(4, 5).astype(np.float32),
                timestamp=datetime.now().timestamp() + i,
                confidence_score=0.9,
                sequence_id=f"seq_{i:03d}"
            )
            self.batch.add(wave_output)
        
        # Should still be at max capacity
        self.assertEqual(len(self.batch), 5)
        
        # The first wave outputs should have been removed
        positions = [wo.sequence_position for wo in self.batch.wave_outputs]
        self.assertEqual(positions, [3, 4, 5, 6, 7])
    
    def test_get_by_position(self):
        """Test retrieving wave output by position."""
        # Test existing position
        wave_output = self.batch.get_by_position(1)
        self.assertIsNotNone(wave_output)
        self.assertEqual(wave_output.sequence_position, 1)
        
        # Test non-existing position
        wave_output = self.batch.get_by_position(99)
        self.assertIsNone(wave_output)
    
    def test_get_range(self):
        """Test retrieving wave outputs in a range."""
        # Test valid range
        range_outputs = self.batch.get_range(0, 2)
        self.assertEqual(len(range_outputs), 2)
        positions = [wo.sequence_position for wo in range_outputs]
        self.assertEqual(positions, [0, 1])
        
        # Test range with no matches
        range_outputs = self.batch.get_range(10, 20)
        self.assertEqual(len(range_outputs), 0)
        
        # Test partial range
        range_outputs = self.batch.get_range(1, 5)
        self.assertEqual(len(range_outputs), 2)  # positions 1 and 2
    
    def test_clear(self):
        """Test clearing the batch."""
        self.assertEqual(len(self.batch), 3)
        self.batch.clear()
        self.assertEqual(len(self.batch), 0)
        self.assertEqual(self.batch.get_memory_usage(), 0)
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        memory_usage = self.batch.get_memory_usage()
        self.assertGreater(memory_usage, 0)
        
        # Memory usage should be sum of individual wave outputs
        expected_usage = sum(wo.memory_usage_bytes for wo in self.batch.wave_outputs)
        self.assertEqual(memory_usage, expected_usage)
    
    def test_batch_serialization_pickle(self):
        """Test batch serialization with pickle format."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save batch
            self.batch.save_batch(temp_path, format='pickle')
            self.assertTrue(os.path.exists(temp_path))
            
            # Load batch
            loaded_batch = WaveOutputBatch.load_batch(temp_path, format='pickle')
            
            # Verify data integrity
            self.assertEqual(len(loaded_batch), len(self.batch))
            
            for original, loaded in zip(self.batch.wave_outputs, loaded_batch.wave_outputs):
                self.assertEqual(original.sequence_position, loaded.sequence_position)
                self.assertEqual(original.confidence_score, loaded.confidence_score)
                self.assertEqual(original.sequence_id, loaded.sequence_id)
                np.testing.assert_array_equal(original.wave_features, loaded.wave_features)
                np.testing.assert_array_equal(original.attention_weights, loaded.attention_weights)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_batch_serialization_json(self):
        """Test batch serialization with JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save batch
            self.batch.save_batch(temp_path, format='json')
            self.assertTrue(os.path.exists(temp_path))
            
            # Load batch
            loaded_batch = WaveOutputBatch.load_batch(temp_path, format='json')
            
            # Verify data integrity
            self.assertEqual(len(loaded_batch), len(self.batch))
            self.assertEqual(loaded_batch.max_size, self.batch.max_size)
            
            for original, loaded in zip(self.batch.wave_outputs, loaded_batch.wave_outputs):
                self.assertEqual(original.sequence_position, loaded.sequence_position)
                self.assertEqual(original.confidence_score, loaded.confidence_score)
                self.assertEqual(original.sequence_id, loaded.sequence_id)
                # Arrays might have small floating point differences in JSON
                np.testing.assert_allclose(original.wave_features, loaded.wave_features, rtol=1e-6)
                np.testing.assert_allclose(original.attention_weights, loaded.attention_weights, rtol=1e-6)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_invalid_serialization_format(self):
        """Test invalid serialization format."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.batch.save_batch(temp_path, format='invalid_format')
            
            with self.assertRaises(ValueError):
                WaveOutputBatch.load_batch(temp_path, format='invalid_format')
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_iteration(self):
        """Test iteration over wave outputs."""
        positions = []
        for wave_output in self.batch:
            positions.append(wave_output.sequence_position)
        
        self.assertEqual(positions, [0, 1, 2])
    
    def test_string_representation(self):
        """Test string representation."""
        str_repr = str(self.batch)
        self.assertIn("WaveOutputBatch", str_repr)
        self.assertIn("size=3/5", str_repr)
        self.assertIn("memory=", str_repr)


class TestIntegration(unittest.TestCase):
    """Integration tests for wave structures working together."""
    
    def test_wave_output_in_training_progress(self):
        """Test using WaveOutput data in TrainingProgress context."""
        # Create a wave output
        wave_output = WaveOutput(
            sequence_position=10,
            wave_features=np.random.rand(8, 128).astype(np.float32),
            attention_weights=np.random.rand(4, 8).astype(np.float32),
            timestamp=datetime.now().timestamp(),
            confidence_score=0.92,
            loss_value=0.234
        )
        
        # Create training progress that could use wave output data
        progress = TrainingProgress(
            current_epoch=1,
            total_epochs=5,
            current_batch=10,
            total_batches=50,
            first_cnn_loss=wave_output.loss_value,  # Use loss from wave output
            second_cnn_loss=0.198,
            combined_loss=0.216,
            wave_storage_utilization=0.6,
            attention_entropy=2.45,
            epoch_start_time=wave_output.timestamp - 100,
            batch_start_time=wave_output.timestamp - 10,
            estimated_time_remaining=1200
        )
        
        # Verify the integration works
        self.assertEqual(progress.first_cnn_loss, wave_output.loss_value)
        self.assertLess(progress.batch_start_time, wave_output.timestamp)
    
    def test_batch_with_training_progress(self):
        """Test WaveOutputBatch in context of training progress."""
        batch = WaveOutputBatch(max_size=100)
        
        # Simulate adding wave outputs during training
        for epoch in range(2):
            for batch_idx in range(10):
                wave_output = WaveOutput(
                    sequence_position=epoch * 10 + batch_idx,
                    wave_features=np.random.rand(5, 64).astype(np.float32),
                    attention_weights=np.random.rand(2, 5).astype(np.float32),
                    timestamp=datetime.now().timestamp(),
                    confidence_score=0.8 + np.random.rand() * 0.2,
                    epoch=epoch,
                    batch_index=batch_idx
                )
                batch.add(wave_output)
        
        # Verify batch contains expected data
        self.assertEqual(len(batch), 20)
        
        # Check that we can retrieve wave outputs by epoch
        epoch_0_outputs = [wo for wo in batch if wo.epoch == 0]
        epoch_1_outputs = [wo for wo in batch if wo.epoch == 1]
        
        self.assertEqual(len(epoch_0_outputs), 10)
        self.assertEqual(len(epoch_1_outputs), 10)
        
        # Create corresponding training progress
        progress = TrainingProgress(
            current_epoch=1,
            total_epochs=2,
            current_batch=9,
            total_batches=10,
            first_cnn_loss=0.3,
            second_cnn_loss=0.25,
            combined_loss=0.275,
            wave_storage_utilization=len(batch) / batch.max_size,
            attention_entropy=1.8,
            epoch_start_time=datetime.now().timestamp() - 300,
            batch_start_time=datetime.now().timestamp() - 30,
            estimated_time_remaining=60
        )
        
        # Verify integration
        self.assertEqual(progress.wave_storage_utilization, 0.2)  # 20/100
        self.assertEqual(progress.current_epoch, 1)


if __name__ == '__main__':
    unittest.main()