"""
Tests for DualCNNTrainer class.

This module tests the coordinated training functionality for dual CNN architecture,
including training loop coordination, progress tracking, and metrics collection.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

# Import the classes to test
from lsm_lite.training.dual_cnn_trainer import DualCNNTrainer, TrainingProgress, WaveOutput
from lsm_lite.utils.config import DualCNNConfig
from lsm_lite.core.dual_cnn_pipeline import DualCNNPipeline


class TestDualCNNTrainer:
    """Test cases for DualCNNTrainer class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock DualCNNConfig for testing."""
        return DualCNNConfig(
            embedder_fit_samples=100,
            embedder_batch_size=16,
            embedder_max_length=32,
            reservoir_size=64,
            attention_heads=4,
            attention_dim=16,
            wave_window_size=10,
            wave_overlap=2,
            max_wave_storage=50,
            wave_feature_dim=64,
            dual_training_epochs=2,
            training_batch_size=8,
            learning_rate=0.001,
            wave_coordination_weight=0.3,
            final_prediction_weight=0.7
        )
    
    @pytest.fixture
    def mock_pipeline(self, mock_config):
        """Create a mock DualCNNPipeline for testing."""
        pipeline = Mock(spec=DualCNNPipeline)
        pipeline.config = mock_config
        pipeline.is_initialized.return_value = True
        
        # Mock tokenizer
        pipeline.tokenizer = Mock()
        pipeline.tokenizer.tokenize.return_value = {
            'input_ids': [tf.constant([1, 2, 3, 4, 5, 0, 0, 0], dtype=tf.int32)]
        }
        pipeline.tokenizer.vocab_size = 1000
        
        # Mock embedder
        pipeline.embedder = Mock()
        def embedder_side_effect(inputs):
            batch_size = tf.shape(inputs)[0]
            return tf.random.normal((batch_size, 32, 64))
        pipeline.embedder.side_effect = embedder_side_effect
        
        # Mock reservoir
        pipeline.reservoir = Mock()
        def reservoir_side_effect(inputs, training=None):
            batch_size = tf.shape(inputs)[0]
            reservoir_output = tf.random.normal((batch_size, 32, 64))
            attention_weights = tf.random.normal((batch_size, 4, 32, 32))
            return (reservoir_output, attention_weights)
        pipeline.reservoir.side_effect = reservoir_side_effect
        pipeline.reservoir.trainable_variables = [tf.Variable(tf.random.normal((64, 64)))]
        pipeline.reservoir.compute_attention_entropy.return_value = tf.constant([2.5, 2.3, 2.7])
        
        # Mock first CNN
        pipeline.first_cnn = Mock()
        def first_cnn_side_effect(inputs, training=None):
            batch_size = tf.shape(inputs)[0]
            return tf.random.normal((batch_size, 1000))
        pipeline.first_cnn.side_effect = first_cnn_side_effect
        pipeline.first_cnn.trainable_variables = [tf.Variable(tf.random.normal((64, 1000)))]
        
        # Mock second CNN
        pipeline.second_cnn = Mock()
        def second_cnn_side_effect(inputs, training=None):
            batch_size = tf.shape(inputs)[0]
            return tf.random.normal((batch_size, 1000))
        pipeline.second_cnn.side_effect = second_cnn_side_effect
        pipeline.second_cnn.trainable_variables = [tf.Variable(tf.random.normal((64, 1000)))]
        
        # Mock wave storage
        pipeline.wave_storage = Mock()
        pipeline.wave_storage.get_storage_stats.return_value = {
            'stored_count': 25,
            'max_capacity': 50,
            'utilization_percent': 50.0,
            'memory_used_mb': 1.5,
            'memory_limit_mb': 10.0
        }
        
        # Mock component status
        pipeline.get_component_status.return_value = {
            'tokenizer': True,
            'embedder': True,
            'reservoir': True,
            'wave_storage': True,
            'first_cnn': True,
            'second_cnn': True,
            'fully_initialized': True
        }
        
        return pipeline
    
    @pytest.fixture
    def trainer(self, mock_pipeline, mock_config):
        """Create a DualCNNTrainer instance for testing."""
        return DualCNNTrainer(mock_pipeline, mock_config)
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        return [
            "This is a sample training text for testing the dual CNN trainer.",
            "Another sample text to ensure we have enough data for training.",
            "The third sample text provides more variety in the training dataset.",
            "Fourth text sample for comprehensive testing of the training loop.",
            "Final sample text to complete our small training dataset."
        ]
    
    def test_trainer_initialization(self, trainer, mock_pipeline, mock_config):
        """Test trainer initialization."""
        assert trainer.pipeline == mock_pipeline
        assert trainer.config == mock_config
        assert trainer.current_epoch == 0
        assert not trainer.is_training
        assert not trainer._stop_training
        assert trainer.first_cnn_optimizer is not None
        assert trainer.second_cnn_optimizer is not None
        assert trainer.loss_fn is not None
        assert len(trainer.metrics) > 0
        assert len(trainer.training_history) > 0
    
    def test_trainer_initialization_uninitialized_pipeline(self, mock_config):
        """Test trainer initialization with uninitialized pipeline."""
        mock_pipeline = Mock(spec=DualCNNPipeline)
        mock_pipeline.is_initialized.return_value = False
        
        with pytest.raises(Exception) as exc_info:
            DualCNNTrainer(mock_pipeline, mock_config)
        
        assert "Pipeline must be fully initialized" in str(exc_info.value)
    
    def test_add_progress_callback(self, trainer):
        """Test adding progress callbacks."""
        callback = Mock()
        trainer.add_progress_callback(callback)
        
        assert callback in trainer.progress_callbacks
    
    def test_create_dataset(self, trainer, sample_training_data):
        """Test dataset creation from text data."""
        dataset = trainer._create_dataset(sample_training_data, batch_size=2, shuffle=False)
        
        # Check that dataset is created
        assert dataset is not None
        
        # Check dataset structure
        for batch in dataset.take(1):
            inputs, targets = batch
            assert inputs.shape[0] <= 2  # Batch size
            assert len(inputs.shape) == 2  # (batch, sequence)
            assert len(targets.shape) == 1  # (batch,)
            assert inputs.dtype == tf.int32
            assert targets.dtype == tf.int32
    
    def test_extract_wave_features(self, trainer):
        """Test wave feature extraction."""
        reservoir_states = tf.random.normal((4, 16, 64))
        attention_weights = tf.random.normal((4, 4, 16, 16))
        
        wave_features = trainer._extract_wave_features(reservoir_states, attention_weights)
        
        assert wave_features.shape[0] == 4  # Batch size
        assert wave_features.shape[1] == 16  # Sequence length
        assert wave_features.shape[2] == trainer.config.wave_feature_dim
    
    def test_prepare_second_cnn_input(self, trainer):
        """Test preparation of second CNN input."""
        batch_size = tf.constant(4)
        wave_features = tf.random.normal((4, 20, 64))  # Longer than window size
        
        second_cnn_input = trainer._prepare_second_cnn_input(wave_features, batch_size)
        
        assert second_cnn_input.shape[0] == 4  # Batch size
        assert second_cnn_input.shape[1] == trainer.config.wave_window_size
        assert second_cnn_input.shape[2] == 64  # Feature dimension
    
    def test_prepare_second_cnn_input_short_sequence(self, trainer):
        """Test preparation of second CNN input with short sequence."""
        batch_size = tf.constant(4)
        wave_features = tf.random.normal((4, 5, 64))  # Shorter than window size
        
        second_cnn_input = trainer._prepare_second_cnn_input(wave_features, batch_size)
        
        assert second_cnn_input.shape[0] == 4  # Batch size
        assert second_cnn_input.shape[1] == trainer.config.wave_window_size
        assert second_cnn_input.shape[2] == 64  # Feature dimension
    
    def test_train_batch(self, trainer):
        """Test training a single batch."""
        input_batch = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0]] * 4, dtype=tf.int32)
        target_batch = tf.constant([6, 7, 8, 9], dtype=tf.int32)
        
        batch_losses = trainer._train_batch(input_batch, target_batch)
        
        assert 'first_cnn_loss' in batch_losses
        assert 'second_cnn_loss' in batch_losses
        assert 'combined_loss' in batch_losses
        
        # Check that losses are tensors with reasonable values
        for loss_name, loss_value in batch_losses.items():
            assert isinstance(loss_value, tf.Tensor)
            assert loss_value.shape == ()  # Scalar
            assert tf.math.is_finite(loss_value)
    
    def test_validate_epoch(self, trainer, sample_training_data):
        """Test epoch validation."""
        val_dataset = trainer._create_dataset(sample_training_data[:2], batch_size=2, shuffle=False)
        
        val_losses = trainer._validate_epoch(val_dataset)
        
        assert 'val_first_cnn_loss' in val_losses
        assert 'val_second_cnn_loss' in val_losses
        assert 'val_combined_loss' in val_losses
        
        # Check that validation losses are reasonable
        for loss_name, loss_value in val_losses.items():
            assert isinstance(loss_value, float)
            assert loss_value >= 0.0
            assert np.isfinite(loss_value)
    
    def test_update_progress(self, trainer):
        """Test progress update functionality."""
        callback = Mock()
        trainer.add_progress_callback(callback)
        
        batch_losses = {
            'first_cnn_loss': tf.constant(1.5),
            'second_cnn_loss': tf.constant(1.2),
            'combined_loss': tf.constant(1.3)
        }
        
        trainer._update_progress(0, 2, 5, 10, batch_losses)
        
        # Check that progress was updated
        assert trainer.last_progress is not None
        assert trainer.last_progress.current_epoch == 1
        assert trainer.last_progress.total_epochs == 2
        assert trainer.last_progress.batch_processed == 6
        assert trainer.last_progress.total_batches == 10
        
        # Check that callback was called
        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert isinstance(call_args, TrainingProgress)
    
    def test_update_training_history(self, trainer):
        """Test training history update."""
        epoch_results = {
            'first_cnn_loss': 1.5,
            'second_cnn_loss': 1.2,
            'combined_loss': 1.3
        }
        
        trainer._update_training_history(epoch_results)
        
        assert len(trainer.training_history['first_cnn_loss']) == 1
        assert len(trainer.training_history['second_cnn_loss']) == 1
        assert len(trainer.training_history['combined_loss']) == 1
        assert trainer.training_history['first_cnn_loss'][0] == 1.5
        assert trainer.training_history['second_cnn_loss'][0] == 1.2
        assert trainer.training_history['combined_loss'][0] == 1.3
    
    def test_should_stop_early(self, trainer):
        """Test early stopping logic."""
        # Test with no validation loss (should not stop)
        epoch_results = {'combined_loss': 1.0}
        assert not trainer._should_stop_early(epoch_results)
        
        # Test with validation loss but insufficient history
        epoch_results = {'val_combined_loss': 1.0}
        assert not trainer._should_stop_early(epoch_results)
        
        # Test with sufficient history but improving validation loss
        trainer.training_history['combined_loss'] = [2.0, 1.8, 1.6, 1.4, 1.2]
        epoch_results = {'val_combined_loss': 1.0}
        assert not trainer._should_stop_early(epoch_results)
        
        # Test with worsening validation loss (should stop)
        trainer.training_history['combined_loss'] = [1.0, 1.0, 1.0, 1.0, 1.0]
        epoch_results = {'val_combined_loss': 1.5}
        assert trainer._should_stop_early(epoch_results)
    
    def test_calculate_final_results(self, trainer):
        """Test final results calculation."""
        # Set up some training history
        trainer.training_history['combined_loss'] = [2.0, 1.5, 1.0]
        trainer.training_history['attention_entropy'] = [2.5, 2.3, 2.1]
        trainer.training_history['epoch_times'] = [10.0, 9.5, 9.0]
        
        final_results = trainer._calculate_final_results(total_time=30.0)
        
        assert 'training_history' in final_results
        assert 'final_metrics' in final_results
        assert 'pipeline_status' in final_results
        
        final_metrics = final_results['final_metrics']
        assert 'initial_loss' in final_metrics
        assert 'final_loss' in final_metrics
        assert 'loss_improvement' in final_metrics
        assert 'total_training_time' in final_metrics
        assert 'epochs_completed' in final_metrics
        assert 'avg_epoch_time' in final_metrics
        
        assert final_metrics['initial_loss'] == 2.0
        assert final_metrics['final_loss'] == 1.0
        assert final_metrics['loss_improvement'] == 1.0
        assert final_metrics['total_training_time'] == 30.0
        assert final_metrics['epochs_completed'] == 3
    
    def test_stop_training(self, trainer):
        """Test training stop functionality."""
        assert not trainer._stop_training
        
        trainer.stop_training()
        
        assert trainer._stop_training
    
    def test_get_current_progress(self, trainer):
        """Test getting current progress."""
        # Initially no progress
        assert trainer.get_current_progress() is None
        
        # Set some progress
        progress = TrainingProgress(
            current_epoch=1,
            total_epochs=2,
            first_cnn_loss=1.5,
            second_cnn_loss=1.2,
            combined_loss=1.3,
            wave_storage_utilization=50.0,
            attention_entropy=2.5,
            estimated_time_remaining=60.0,
            learning_rate=0.001,
            batch_processed=5,
            total_batches=10
        )
        trainer.last_progress = progress
        
        current_progress = trainer.get_current_progress()
        assert current_progress == progress
    
    def test_save_and_load_training_state(self, trainer):
        """Test saving and loading training state."""
        # Set up some training state
        trainer.current_epoch = 5
        trainer.training_history['combined_loss'] = [2.0, 1.5, 1.0]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save state
            trainer.save_training_state(filepath)
            assert os.path.exists(filepath)
            
            # Verify saved content
            with open(filepath, 'r') as f:
                saved_state = json.load(f)
            
            assert saved_state['current_epoch'] == 5
            assert saved_state['training_history']['combined_loss'] == [2.0, 1.5, 1.0]
            assert 'config' in saved_state
            assert 'optimizer_states' in saved_state
            
            # Create new trainer and load state
            new_trainer = DualCNNTrainer(trainer.pipeline, trainer.config)
            new_trainer.load_training_state(filepath)
            
            assert new_trainer.current_epoch == 5
            assert new_trainer.training_history['combined_loss'] == [2.0, 1.5, 1.0]
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_train_dual_cnn_basic(self, trainer, sample_training_data):
        """Test basic dual CNN training functionality."""
        # Mock the training methods to avoid complex TensorFlow operations
        trainer._prepare_training_data = Mock(return_value=(Mock(), None))
        trainer._train_epoch = Mock(return_value={
            'first_cnn_loss': 1.5,
            'second_cnn_loss': 1.2,
            'combined_loss': 1.3
        })
        trainer._calculate_final_results = Mock(return_value={
            'training_history': trainer.training_history,
            'final_metrics': {'final_loss': 1.0},
            'pipeline_status': {'fully_initialized': True}
        })
        
        # Mock dataset length calculation
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([1, 2, 3, 4, 5]))  # 5 batches
        trainer._prepare_training_data.return_value = (mock_dataset, None)
        
        results = trainer.train_dual_cnn(
            training_data=sample_training_data,
            epochs=2,
            batch_size=4
        )
        
        assert 'training_history' in results
        assert 'final_metrics' in results
        assert 'pipeline_status' in results
        
        # Verify training methods were called
        trainer._prepare_training_data.assert_called_once()
        assert trainer._train_epoch.call_count == 2  # 2 epochs
        trainer._calculate_final_results.assert_called_once()
    
    def test_train_dual_cnn_with_validation(self, trainer, sample_training_data):
        """Test dual CNN training with validation data."""
        validation_data = sample_training_data[:2]
        
        # Mock the training methods
        trainer._prepare_training_data = Mock(return_value=(Mock(), Mock()))
        trainer._train_epoch = Mock(return_value={
            'first_cnn_loss': 1.5,
            'second_cnn_loss': 1.2,
            'combined_loss': 1.3,
            'val_combined_loss': 1.4
        })
        trainer._calculate_final_results = Mock(return_value={
            'training_history': trainer.training_history,
            'final_metrics': {'final_loss': 1.0},
            'pipeline_status': {'fully_initialized': True}
        })
        
        # Mock dataset length calculation
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([1, 2, 3]))  # 3 batches
        trainer._prepare_training_data.return_value = (mock_dataset, Mock())
        
        results = trainer.train_dual_cnn(
            training_data=sample_training_data,
            validation_data=validation_data,
            epochs=1,
            batch_size=4
        )
        
        assert 'training_history' in results
        assert 'final_metrics' in results
        assert 'pipeline_status' in results
        
        # Verify validation data was passed
        call_args = trainer._prepare_training_data.call_args[0]
        assert call_args[1] == validation_data  # validation_data argument
    
    def test_train_dual_cnn_early_stopping(self, trainer, sample_training_data):
        """Test dual CNN training with early stopping."""
        # Mock early stopping to trigger after first epoch
        trainer._should_stop_early = Mock(side_effect=[False, True])
        trainer._prepare_training_data = Mock(return_value=(Mock(), None))
        trainer._train_epoch = Mock(return_value={
            'first_cnn_loss': 1.5,
            'second_cnn_loss': 1.2,
            'combined_loss': 1.3
        })
        trainer._calculate_final_results = Mock(return_value={
            'training_history': trainer.training_history,
            'final_metrics': {'final_loss': 1.0},
            'pipeline_status': {'fully_initialized': True}
        })
        
        # Mock dataset length calculation
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([1, 2, 3]))
        trainer._prepare_training_data.return_value = (mock_dataset, None)
        
        results = trainer.train_dual_cnn(
            training_data=sample_training_data,
            epochs=5  # Should stop early after 2 epochs
        )
        
        # Should have stopped after 2 epochs (1 completed + early stop check)
        assert trainer._train_epoch.call_count == 2
        assert trainer._should_stop_early.call_count == 2
    
    def test_repr(self, trainer):
        """Test string representation of trainer."""
        repr_str = repr(trainer)
        assert "DualCNNTrainer" in repr_str
        assert "status=idle" in repr_str
        assert "epoch=0" in repr_str
        
        # Test during training
        trainer.is_training = True
        trainer.current_epoch = 3
        repr_str = repr(trainer)
        assert "status=training" in repr_str
        assert "epoch=3" in repr_str


class TestTrainingProgress:
    """Test cases for TrainingProgress dataclass."""
    
    def test_training_progress_creation(self):
        """Test TrainingProgress creation."""
        progress = TrainingProgress(
            current_epoch=1,
            total_epochs=10,
            first_cnn_loss=1.5,
            second_cnn_loss=1.2,
            combined_loss=1.3,
            wave_storage_utilization=75.0,
            attention_entropy=2.5,
            estimated_time_remaining=300.0,
            learning_rate=0.001,
            batch_processed=50,
            total_batches=100
        )
        
        assert progress.current_epoch == 1
        assert progress.total_epochs == 10
        assert progress.first_cnn_loss == 1.5
        assert progress.second_cnn_loss == 1.2
        assert progress.combined_loss == 1.3
        assert progress.wave_storage_utilization == 75.0
        assert progress.attention_entropy == 2.5
        assert progress.estimated_time_remaining == 300.0
        assert progress.learning_rate == 0.001
        assert progress.batch_processed == 50
        assert progress.total_batches == 100


class TestWaveOutput:
    """Test cases for WaveOutput dataclass."""
    
    def test_wave_output_creation(self):
        """Test WaveOutput creation."""
        wave_features = tf.random.normal((64,))
        attention_weights = tf.random.normal((4, 32, 32))
        
        wave_output = WaveOutput(
            sequence_position=10,
            wave_features=wave_features,
            attention_weights=attention_weights,
            timestamp=1234567890.0,
            confidence_score=0.85
        )
        
        assert wave_output.sequence_position == 10
        assert tf.equal(wave_output.wave_features, wave_features).numpy().all()
        assert tf.equal(wave_output.attention_weights, attention_weights).numpy().all()
        assert wave_output.timestamp == 1234567890.0
        assert wave_output.confidence_score == 0.85


if __name__ == '__main__':
    pytest.main([__file__])