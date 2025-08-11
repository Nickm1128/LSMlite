"""
Integration tests for DualCNNPipeline orchestrator.

This module tests the complete dual CNN pipeline setup and component coordination,
including initialization, error handling, and integration between components.
"""

import pytest
import tensorflow as tf
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock

# Import from the working test file for now
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the classes from our working test file
exec(open('test_pipeline_creation.py').read())

# Now we have DualCNNPipeline, ComponentInitializationError, DualCNNTrainingError available
from lsm_lite.utils.config import DualCNNConfig, LSMConfig


class TestDualCNNPipelineInitialization:
    """Test pipeline initialization and component setup."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DualCNNConfig(
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
            first_cnn_filters=[16, 32],
            second_cnn_filters=[32, 64],
            dual_training_epochs=2,
            training_batch_size=8
        )
        
        self.sample_training_data = [
            "This is a test sentence for training.",
            "Another example text for the pipeline.",
            "More training data to test the system.",
            "Final sample text for initialization."
        ]
    
    def test_pipeline_creation_with_dual_cnn_config(self):
        """Test pipeline creation with DualCNNConfig."""
        pipeline = DualCNNPipeline(self.config)
        
        assert pipeline.config == self.config
        assert not pipeline.is_initialized()
        assert pipeline.tokenizer is None
        assert pipeline.embedder is None
        assert pipeline.reservoir is None
        assert pipeline.first_cnn is None
        assert pipeline.second_cnn is None
        assert pipeline.wave_storage is None
    
    def test_pipeline_creation_with_lsm_config(self):
        """Test pipeline creation with LSMConfig (backward compatibility)."""
        lsm_config = LSMConfig(
            max_length=32,
            embedding_dim=64,
            reservoir_size=64,
            cnn_filters=[16, 32],
            epochs=2,
            batch_size=8
        )
        
        pipeline = DualCNNPipeline(lsm_config)
        
        assert isinstance(pipeline.config, DualCNNConfig)
        assert pipeline.config.embedder_max_length == 32
        assert pipeline.config.wave_feature_dim == 64
        assert pipeline.config.reservoir_size == 64
    
    def test_pipeline_creation_with_invalid_config(self):
        """Test pipeline creation with invalid configuration."""
        invalid_config = DualCNNConfig(
            reservoir_size=-1,  # Invalid negative size
            attention_heads=0,  # Invalid zero heads
        )
        
        with pytest.raises(ComponentInitializationError) as exc_info:
            DualCNNPipeline(invalid_config)
        
        assert "Configuration validation failed" in str(exc_info.value)
    
    @patch('lsm_lite.core.dual_cnn_pipeline.UnifiedTokenizer')
    @patch('lsm_lite.core.dual_cnn_pipeline.SinusoidalEmbedder')
    @patch('lsm_lite.core.dual_cnn_pipeline.AttentiveReservoir')
    @patch('lsm_lite.core.dual_cnn_pipeline.RollingWaveStorage')
    @patch('lsm_lite.core.dual_cnn_pipeline.CNNProcessor')
    def test_successful_fit_and_initialize(self, mock_cnn, mock_storage, mock_reservoir, 
                                         mock_embedder, mock_tokenizer):
        """Test successful pipeline initialization."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 1000
        mock_tokenizer_instance.tokenize.return_value = {
            'input_ids': np.array([[1, 2, 3, 4, 5]]),
            'attention_mask': np.array([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_embedder.return_value = Mock()
        mock_reservoir.return_value = Mock()
        mock_storage.return_value = Mock()
        mock_cnn.return_value = Mock()
        
        # Test initialization
        pipeline = DualCNNPipeline(self.config)
        pipeline.fit_and_initialize(self.sample_training_data)
        
        # Verify initialization
        assert pipeline.is_initialized()
        assert pipeline.tokenizer is not None
        assert pipeline.embedder is not None
        assert pipeline.reservoir is not None
        assert pipeline.wave_storage is not None
        assert pipeline.first_cnn is not None
        assert pipeline.second_cnn is not None
        
        # Verify component creation calls
        mock_tokenizer.assert_called_once()
        mock_embedder.assert_called_once()
        mock_reservoir.assert_called_once()
        mock_storage.assert_called_once()
        assert mock_cnn.call_count == 2  # First and second CNN
    
    @patch('lsm_lite.core.dual_cnn_pipeline.UnifiedTokenizer')
    def test_tokenizer_initialization_failure(self, mock_tokenizer):
        """Test handling of tokenizer initialization failure."""
        mock_tokenizer.side_effect = Exception("Tokenizer creation failed")
        
        pipeline = DualCNNPipeline(self.config)
        
        with pytest.raises(ComponentInitializationError) as exc_info:
            pipeline.fit_and_initialize(self.sample_training_data)
        
        assert "tokenizer" in str(exc_info.value)
        assert "Tokenizer creation failed" in str(exc_info.value)
        assert not pipeline.is_initialized()
    
    @patch('lsm_lite.core.dual_cnn_pipeline.UnifiedTokenizer')
    @patch('lsm_lite.core.dual_cnn_pipeline.SinusoidalEmbedder')
    def test_embedder_initialization_failure(self, mock_embedder, mock_tokenizer):
        """Test handling of embedder initialization failure."""
        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 1000
        mock_tokenizer_instance.tokenize.return_value = {
            'input_ids': np.array([[1, 2, 3, 4, 5]]),
            'attention_mask': np.array([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Setup embedder to fail
        mock_embedder.side_effect = Exception("Embedder creation failed")
        
        pipeline = DualCNNPipeline(self.config)
        
        with pytest.raises(ComponentInitializationError) as exc_info:
            pipeline.fit_and_initialize(self.sample_training_data)
        
        assert "embedder" in str(exc_info.value)
        assert "Embedder creation failed" in str(exc_info.value)
        assert not pipeline.is_initialized()
    
    def test_progress_callback_functionality(self):
        """Test progress callback during initialization."""
        progress_updates = []
        
        def progress_callback(component, status, full_progress):
            progress_updates.append((component, status))
        
        with patch.multiple(
            'lsm_lite.core.dual_cnn_pipeline',
            UnifiedTokenizer=Mock(return_value=Mock(vocab_size=1000, tokenize=Mock(return_value={
                'input_ids': np.array([[1, 2, 3]]), 'attention_mask': np.array([[1, 1, 1]])
            }))),
            SinusoidalEmbedder=Mock(return_value=Mock()),
            AttentiveReservoir=Mock(return_value=Mock()),
            RollingWaveStorage=Mock(return_value=Mock()),
            CNNProcessor=Mock(return_value=Mock())
        ):
            pipeline = DualCNNPipeline(self.config)
            pipeline.fit_and_initialize(self.sample_training_data, progress_callback=progress_callback)
        
        # Verify progress updates
        expected_components = ['tokenizer', 'embedder', 'reservoir', 'wave_storage', 'first_cnn', 'second_cnn']
        for component in expected_components:
            assert (component, 'initializing') in progress_updates
            assert (component, 'completed') in progress_updates
    
    def test_component_status_tracking(self):
        """Test component status tracking during initialization."""
        with patch.multiple(
            'lsm_lite.core.dual_cnn_pipeline',
            UnifiedTokenizer=Mock(return_value=Mock(vocab_size=1000, tokenize=Mock(return_value={
                'input_ids': np.array([[1, 2, 3]]), 'attention_mask': np.array([[1, 1, 1]])
            }))),
            SinusoidalEmbedder=Mock(return_value=Mock()),
            AttentiveReservoir=Mock(return_value=Mock()),
            RollingWaveStorage=Mock(return_value=Mock()),
            CNNProcessor=Mock(return_value=Mock())
        ):
            pipeline = DualCNNPipeline(self.config)
            
            # Check initial status
            initial_status = pipeline.get_component_status()
            assert all(not status for component, status in initial_status.items() if component != 'fully_initialized')
            
            # Initialize pipeline
            pipeline.fit_and_initialize(self.sample_training_data)
            
            # Check final status
            final_status = pipeline.get_component_status()
            assert all(status for status in final_status.values())


class TestDualCNNPipelineProcessing:
    """Test pipeline processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DualCNNConfig(
            embedder_max_length=16,
            reservoir_size=32,
            wave_feature_dim=32,
            wave_window_size=8,
            first_cnn_filters=[8, 16],
            second_cnn_filters=[16, 32]
        )
        
        # Create mock components
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.vocab_size = 100
        self.mock_tokenizer.tokenize.return_value = {
            'input_ids': np.array([[1, 2, 3, 4, 5]]),
            'attention_mask': np.array([[1, 1, 1, 1, 1]])
        }
        self.mock_tokenizer.decode.return_value = "decoded text"
        self.mock_tokenizer.get_special_tokens.return_value = {'eos_token_id': 2}
        
        self.mock_embedder = Mock()
        self.mock_embedder.return_value = tf.random.normal((1, 5, 32))
        
        self.mock_reservoir = Mock()
        self.mock_reservoir.return_value = (
            tf.random.normal((1, 5, 32)),  # reservoir states
            tf.random.normal((1, 4, 5, 5))  # attention weights
        )
        
        self.mock_first_cnn = Mock()
        self.mock_first_cnn.return_value = tf.random.normal((1, 5, 100))
        
        self.mock_second_cnn = Mock()
        self.mock_second_cnn.return_value = tf.random.normal((1, 8, 100))
        
        self.mock_wave_storage = Mock()
        self.mock_wave_storage.__len__.return_value = 5
        self.mock_wave_storage.get_wave_sequence.return_value = tf.random.normal((8, 32))
    
    def test_process_sequence_with_initialized_pipeline(self):
        """Test sequence processing with fully initialized pipeline."""
        pipeline = DualCNNPipeline(self.config)
        
        # Manually set components (simulating successful initialization)
        pipeline.tokenizer = self.mock_tokenizer
        pipeline.embedder = self.mock_embedder
        pipeline.reservoir = self.mock_reservoir
        pipeline.first_cnn = self.mock_first_cnn
        pipeline.second_cnn = self.mock_second_cnn
        pipeline.wave_storage = self.mock_wave_storage
        pipeline._is_initialized = True
        
        # Process sequence
        first_output, second_output, attention_weights = pipeline.process_sequence(
            "test input text", store_waves=True
        )
        
        # Verify outputs
        assert first_output is not None
        assert second_output is not None
        assert attention_weights is not None
        
        # Verify component calls
        self.mock_tokenizer.tokenize.assert_called_once()
        self.mock_embedder.assert_called_once()
        self.mock_reservoir.assert_called_once()
        self.mock_first_cnn.assert_called_once()
        self.mock_second_cnn.assert_called_once()
        self.mock_wave_storage.store_wave.assert_called()
    
    def test_process_sequence_without_initialization(self):
        """Test sequence processing without initialization raises error."""
        pipeline = DualCNNPipeline(self.config)
        
        with pytest.raises(ComponentInitializationError) as exc_info:
            pipeline.process_sequence("test input")
        
        assert "Pipeline must be initialized" in str(exc_info.value)
    
    def test_generate_tokens_basic_functionality(self):
        """Test basic token generation functionality."""
        pipeline = DualCNNPipeline(self.config)
        
        # Setup initialized pipeline
        pipeline.tokenizer = self.mock_tokenizer
        pipeline.embedder = self.mock_embedder
        pipeline.reservoir = self.mock_reservoir
        pipeline.first_cnn = self.mock_first_cnn
        pipeline.second_cnn = self.mock_second_cnn
        pipeline.wave_storage = self.mock_wave_storage
        pipeline._is_initialized = True
        
        # Mock the process_sequence method to avoid complex setup
        pipeline.process_sequence = Mock(return_value=(
            tf.constant([[[0.1, 0.8, 0.1] + [0.0] * 97]]),  # first_output (logits for 3 tokens)
            tf.constant([[[0.2, 0.6, 0.2] + [0.0] * 97]]),  # second_output
            None  # attention_weights
        ))
        
        # Generate tokens
        generated = pipeline.generate_tokens("test prompt", max_length=2)
        
        # Verify generation
        assert isinstance(generated, list)
        assert len(generated) <= 2
        assert all(isinstance(token, int) for token in generated)
    
    def test_generate_tokens_without_initialization(self):
        """Test token generation without initialization raises error."""
        pipeline = DualCNNPipeline(self.config)
        
        with pytest.raises(ComponentInitializationError) as exc_info:
            pipeline.generate_tokens("test prompt")
        
        assert "Pipeline must be initialized" in str(exc_info.value)


class TestDualCNNPipelineUtilities:
    """Test pipeline utility functions and statistics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DualCNNConfig()
        self.pipeline = DualCNNPipeline(self.config)
    
    def test_get_pipeline_stats_uninitialized(self):
        """Test getting pipeline statistics before initialization."""
        stats = self.pipeline.get_pipeline_stats()
        
        assert 'config' in stats
        assert 'initialized' in stats
        assert 'components' in stats
        assert 'initialization_progress' in stats
        
        assert not stats['initialized']
        assert not any(stats['components'].values())
    
    def test_get_pipeline_stats_initialized(self):
        """Test getting pipeline statistics after initialization."""
        # Mock initialized state
        self.pipeline._is_initialized = True
        self.pipeline.tokenizer = Mock()
        self.pipeline.embedder = Mock()
        self.pipeline.reservoir = Mock()
        self.pipeline.first_cnn = Mock()
        self.pipeline.second_cnn = Mock()
        
        mock_wave_storage = Mock()
        mock_wave_storage.get_storage_stats.return_value = {
            'stored_count': 10,
            'utilization_percent': 50.0
        }
        self.pipeline.wave_storage = mock_wave_storage
        
        stats = self.pipeline.get_pipeline_stats()
        
        assert stats['initialized']
        assert all(stats['components'].values())
        assert 'wave_storage' in stats
    
    def test_cleanup_functionality(self):
        """Test pipeline cleanup functionality."""
        # Setup mock components
        mock_wave_storage = Mock()
        self.pipeline.wave_storage = mock_wave_storage
        self.pipeline.tokenizer = Mock()
        self.pipeline.embedder = Mock()
        self.pipeline.reservoir = Mock()
        self.pipeline.first_cnn = Mock()
        self.pipeline.second_cnn = Mock()
        self.pipeline._is_initialized = True
        
        # Perform cleanup
        self.pipeline.cleanup()
        
        # Verify cleanup
        assert not self.pipeline.is_initialized()
        assert self.pipeline.tokenizer is None
        assert self.pipeline.embedder is None
        assert self.pipeline.reservoir is None
        assert self.pipeline.first_cnn is None
        assert self.pipeline.second_cnn is None
        assert self.pipeline.wave_storage is None
        
        # Verify wave storage was cleared
        mock_wave_storage.clear_storage.assert_called_once()
    
    def test_pipeline_string_representation(self):
        """Test pipeline string representation."""
        # Test uninitialized pipeline
        repr_str = repr(self.pipeline)
        assert "not initialized" in repr_str
        assert "DualCNNConfig" in repr_str
        
        # Test initialized pipeline
        self.pipeline._is_initialized = True
        repr_str = repr(self.pipeline)
        assert "initialized" in repr_str
        assert "not initialized" not in repr_str


class TestDualCNNPipelineErrorHandling:
    """Test error handling and recovery scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DualCNNConfig(
            embedder_fit_samples=10,
            embedder_max_length=8,
            reservoir_size=16
        )
    
    def test_initialization_error_recovery(self):
        """Test that initialization errors don't leave pipeline in inconsistent state."""
        with patch('lsm_lite.core.dual_cnn_pipeline.UnifiedTokenizer') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Tokenizer failed")
            
            pipeline = DualCNNPipeline(self.config)
            
            with pytest.raises(ComponentInitializationError):
                pipeline.fit_and_initialize(["test data"])
            
            # Verify pipeline is not marked as initialized
            assert not pipeline.is_initialized()
            
            # Verify components are not set
            status = pipeline.get_component_status()
            assert not any(status.values())
    
    def test_partial_initialization_tracking(self):
        """Test tracking of partial initialization progress."""
        def failing_embedder(*args, **kwargs):
            raise Exception("Embedder initialization failed")
        
        with patch.multiple(
            'lsm_lite.core.dual_cnn_pipeline',
            UnifiedTokenizer=Mock(return_value=Mock(vocab_size=100, tokenize=Mock(return_value={
                'input_ids': np.array([[1, 2, 3]]), 'attention_mask': np.array([[1, 1, 1]])
            }))),
            SinusoidalEmbedder=failing_embedder
        ):
            pipeline = DualCNNPipeline(self.config)
            
            with pytest.raises(ComponentInitializationError):
                pipeline.fit_and_initialize(["test data"])
            
            # Check that tokenizer was completed but embedder failed
            progress = pipeline.get_initialization_progress()
            assert progress.get('tokenizer') == 'completed'
            assert 'embedder' in progress  # Should be present but not completed
    
    def test_configuration_validation_errors(self):
        """Test handling of configuration validation errors."""
        invalid_configs = [
            DualCNNConfig(reservoir_size=0),  # Invalid reservoir size
            DualCNNConfig(attention_heads=-1),  # Invalid attention heads
            DualCNNConfig(wave_window_size=0),  # Invalid window size
        ]
        
        for config in invalid_configs:
            with pytest.raises(ComponentInitializationError) as exc_info:
                DualCNNPipeline(config)
            
            assert "Configuration validation failed" in str(exc_info.value)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])