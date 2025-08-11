"""
Simple integration tests for DualCNNPipeline orchestrator.

This module tests the basic functionality of the DualCNNPipeline without
complex mocking, focusing on the core orchestration logic.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the working classes from our test file
exec(open('test_pipeline_creation.py').read())


class TestDualCNNPipelineBasic:
    """Test basic DualCNNPipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DualCNNConfig(
            embedder_fit_samples=100,
            embedder_batch_size=16,
            embedder_max_length=32,
            reservoir_size=64,
            attention_heads=4,
            attention_dim=16,
            wave_window_size=20,
            wave_overlap=5,
            max_wave_storage=50,
            wave_feature_dim=64,
            first_cnn_filters=[16, 32],
            second_cnn_filters=[32, 64],
            dual_training_epochs=2,
            training_batch_size=8
        )
    
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
    
    def test_component_status_tracking(self):
        """Test component status tracking."""
        pipeline = DualCNNPipeline(self.config)
        
        # Check initial status
        initial_status = pipeline.get_component_status()
        assert not initial_status['tokenizer']
        assert not initial_status['embedder']
        assert not initial_status['reservoir']
        assert not initial_status['wave_storage']
        assert not initial_status['first_cnn']
        assert not initial_status['second_cnn']
        assert not initial_status['fully_initialized']
    
    def test_initialization_progress_tracking(self):
        """Test initialization progress tracking."""
        pipeline = DualCNNPipeline(self.config)
        
        # Check initial progress
        initial_progress = pipeline.get_initialization_progress()
        assert len(initial_progress) == 0
        
        # Manually update progress to test the mechanism
        pipeline._update_progress("test_component", "initializing", None)
        progress = pipeline.get_initialization_progress()
        assert progress["test_component"] == "initializing"
        
        pipeline._update_progress("test_component", "completed", None)
        progress = pipeline.get_initialization_progress()
        assert progress["test_component"] == "completed"
    
    def test_cleanup_functionality(self):
        """Test pipeline cleanup functionality."""
        pipeline = DualCNNPipeline(self.config)
        
        # Simulate some initialized state
        pipeline._is_initialized = True
        pipeline._initialization_progress = {"test": "completed"}
        
        # Perform cleanup
        pipeline.cleanup()
        
        # Verify cleanup
        assert not pipeline.is_initialized()
        assert len(pipeline.get_initialization_progress()) == 0
        assert pipeline.tokenizer is None
        assert pipeline.embedder is None
        assert pipeline.reservoir is None
        assert pipeline.first_cnn is None
        assert pipeline.second_cnn is None
        assert pipeline.wave_storage is None
    
    def test_pipeline_string_representation(self):
        """Test pipeline string representation."""
        pipeline = DualCNNPipeline(self.config)
        
        # Test uninitialized pipeline
        repr_str = repr(pipeline)
        assert "not initialized" in repr_str
        assert "DualCNNConfig" in repr_str
        
        # Test initialized pipeline
        pipeline._is_initialized = True
        repr_str = repr(pipeline)
        assert "initialized" in repr_str
        assert "not initialized" not in repr_str
    
    def test_lsm_config_conversion(self):
        """Test LSMConfig to DualCNNConfig conversion."""
        lsm_config = LSMConfig(
            max_length=64,
            embedding_dim=128,
            reservoir_size=256,
            cnn_filters=[32, 64, 128],
            epochs=5,
            batch_size=16,
            learning_rate=0.002,
            generation_max_length=100,
            generation_temperature=0.8
        )
        
        pipeline = DualCNNPipeline(lsm_config)
        dual_config = pipeline.config
        
        # Verify conversion
        assert dual_config.embedder_max_length == 64
        assert dual_config.wave_feature_dim == 128
        assert dual_config.reservoir_size == 256
        assert dual_config.first_cnn_filters == [32, 64, 128]
        assert dual_config.dual_training_epochs == 5
        assert dual_config.training_batch_size == 16
        assert dual_config.learning_rate == 0.002
        assert dual_config.generation_max_length == 100
        assert dual_config.generation_temperature == 0.8
    
    def test_progress_callback_mechanism(self):
        """Test progress callback mechanism."""
        pipeline = DualCNNPipeline(self.config)
        
        progress_updates = []
        
        def progress_callback(component, status, full_progress):
            progress_updates.append((component, status))
        
        # Test progress updates
        pipeline._update_progress("tokenizer", "initializing", progress_callback)
        pipeline._update_progress("tokenizer", "completed", progress_callback)
        pipeline._update_progress("embedder", "initializing", progress_callback)
        
        # Verify callback was called
        assert len(progress_updates) == 3
        assert progress_updates[0] == ("tokenizer", "initializing")
        assert progress_updates[1] == ("tokenizer", "completed")
        assert progress_updates[2] == ("embedder", "initializing")
    
    def test_current_step_tracking(self):
        """Test current step tracking for error reporting."""
        pipeline = DualCNNPipeline(self.config)
        
        # Test with no progress
        assert pipeline._get_current_step() == "unknown"
        
        # Test with some progress
        pipeline._initialization_progress = {
            "tokenizer": "completed",
            "embedder": "initializing",
            "reservoir": "not_started"
        }
        
        # Should return the first non-completed step
        assert pipeline._get_current_step() == "embedder"
        
        # Test with all completed
        pipeline._initialization_progress = {
            "tokenizer": "completed",
            "embedder": "completed",
            "reservoir": "completed"
        }
        
        assert pipeline._get_current_step() == "unknown"


class TestDualCNNPipelineConfiguration:
    """Test configuration handling and validation."""
    
    def test_valid_configuration_acceptance(self):
        """Test that valid configurations are accepted."""
        valid_configs = [
            DualCNNConfig(),  # Default config
            DualCNNConfig(
                reservoir_size=128,
                attention_heads=8,
                wave_window_size=30,
                wave_overlap=5
            ),
            DualCNNConfig(
                first_cnn_filters=[32, 64],
                second_cnn_filters=[64, 128],
                wave_feature_dim=128
            )
        ]
        
        for config in valid_configs:
            pipeline = DualCNNPipeline(config)
            assert pipeline.config == config
    
    def test_invalid_configuration_rejection(self):
        """Test that invalid configurations are rejected."""
        invalid_configs = [
            DualCNNConfig(reservoir_size=0),  # Invalid reservoir size
            DualCNNConfig(attention_heads=-1),  # Invalid attention heads
            DualCNNConfig(wave_window_size=0),  # Invalid window size
            DualCNNConfig(wave_window_size=10, wave_overlap=15),  # Overlap > window
        ]
        
        for config in invalid_configs:
            with pytest.raises(ComponentInitializationError):
                DualCNNPipeline(config)
    
    def test_configuration_parameter_access(self):
        """Test access to configuration parameters."""
        config = DualCNNConfig(
            embedder_fit_samples=500,
            reservoir_size=256,
            attention_heads=8,
            wave_window_size=40,
            first_cnn_filters=[64, 128, 256]
        )
        
        pipeline = DualCNNPipeline(config)
        
        assert pipeline.config.embedder_fit_samples == 500
        assert pipeline.config.reservoir_size == 256
        assert pipeline.config.attention_heads == 8
        assert pipeline.config.wave_window_size == 40
        assert pipeline.config.first_cnn_filters == [64, 128, 256]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])