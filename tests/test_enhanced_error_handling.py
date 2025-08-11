"""
Tests for enhanced error handling and user experience improvements.

This module tests the error handling, validation, graceful degradation,
and logging improvements implemented in task 12.
"""

import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import logging
import tempfile
import os

# Import the modules we're testing
from lsm_lite.utils.error_handling import (
    ErrorHandler, LSMError, ConfigurationError, DataValidationError,
    MemoryError, ComputationError, ValidationUtils, handle_lsm_error,
    global_error_handler, ErrorCategory, ErrorSeverity
)
from lsm_lite.utils.logging_config import (
    setup_logging, LSMFormatter, PerformanceLogger, ErrorTracker,
    TimedOperation
)
from lsm_lite.utils.config import DualCNNConfig
from lsm_lite.core.dual_cnn_pipeline import DualCNNPipeline, ComponentInitializationError
from lsm_lite.training.dual_cnn_trainer import DualCNNTrainer


class TestErrorHandling:
    """Test enhanced error handling functionality."""
    
    def test_lsm_error_creation(self):
        """Test LSMError creation with context and solutions."""
        from lsm_lite.utils.error_handling import ErrorContext, ErrorSolution
        
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            input_shapes={"input": [32, 128]},
            memory_usage=100.5
        )
        
        solution = ErrorSolution(
            description="Test solution",
            action_steps=["Step 1", "Step 2"],
            code_example="test_code()"
        )
        
        error = LSMError(
            message="Test error message",
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            solutions=[solution]
        )
        
        error_str = str(error)
        assert "Test error message" in error_str
        assert "test_component" in error_str
        assert "test_operation" in error_str
        assert "Test solution" in error_str
        assert "Step 1" in error_str
    
    def test_error_handler_pattern_matching(self):
        """Test error handler pattern matching and solution generation."""
        handler = ErrorHandler()
        
        # Test shape mismatch error
        shape_error = ValueError("Tensor shape mismatch: expected (32, 128), got (32, 64)")
        enhanced_error = handler.handle_error(shape_error)
        
        assert isinstance(enhanced_error, LSMError)
        assert enhanced_error.category == ErrorCategory.COMPUTATION
        assert len(enhanced_error.solutions) > 0
        assert "shape" in enhanced_error.solutions[0].description.lower()
    
    def test_validation_utils_tensor_validation(self):
        """Test tensor shape validation."""
        # Valid tensor
        valid_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
        ValidationUtils.validate_tensor_shape(valid_tensor, (2, 3), "test_tensor")
        
        # Invalid tensor shape
        invalid_tensor = tf.constant([[1, 2], [3, 4]])
        with pytest.raises(DataValidationError) as exc_info:
            ValidationUtils.validate_tensor_shape(invalid_tensor, (2, 3), "test_tensor")
        
        assert "dimension mismatch" in str(exc_info.value).lower()
    
    def test_validation_utils_config_validation(self):
        """Test configuration parameter validation."""
        # Create a mock config with some issues
        class MockConfig:
            training_batch_size = -1  # Invalid
            learning_rate = 2.0  # Invalid (> 1)
            attention_heads = 0  # Invalid
            reservoir_size = 100  # Valid
        
        config = MockConfig()
        issues = ValidationUtils.validate_config_parameters(config)
        
        assert len(issues) >= 3  # Should find at least 3 issues
        assert any("training_batch_size must be positive" in issue for issue in issues)
        assert any("learning_rate must be between 0 and 1" in issue for issue in issues)
        assert any("attention_heads must be positive" in issue for issue in issues)
    
    def test_validation_utils_training_data_validation(self):
        """Test training data validation."""
        # Valid data
        valid_data = ["This is a good sentence.", "Another valid sentence here."]
        issues = ValidationUtils.validate_training_data(valid_data)
        assert len(issues) == 0
        
        # Invalid data
        invalid_data = ["", "  ", "short", None, 123]  # Mix of issues
        issues = ValidationUtils.validate_training_data(invalid_data)
        assert len(issues) > 0
    
    def test_handle_lsm_error_decorator(self):
        """Test the error handling decorator."""
        @handle_lsm_error
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(LSMError) as exc_info:
            failing_function()
        
        enhanced_error = exc_info.value
        assert isinstance(enhanced_error, LSMError)
        assert enhanced_error.original_error is not None
        assert isinstance(enhanced_error.original_error, ValueError)
    
    def test_error_recovery_strategies(self):
        """Test error recovery strategies."""
        handler = ErrorHandler()
        
        # Mock config and component
        mock_config = Mock()
        mock_config.training_batch_size = 32
        mock_config.reservoir_size = 512
        
        mock_component = Mock()
        
        # Test memory error recovery
        memory_error = LSMError(
            "Out of memory",
            ErrorCategory.MEMORY,
            ErrorSeverity.HIGH
        )
        
        recovery_success = handler.attempt_recovery(memory_error, mock_component, mock_config)
        assert recovery_success
        assert mock_config.training_batch_size == 16  # Should be reduced


class TestLoggingConfiguration:
    """Test enhanced logging configuration."""
    
    def test_lsm_formatter(self):
        """Test custom LSM formatter."""
        formatter = LSMFormatter(include_context=True)
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function"
        )
        
        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "test_function" in formatted
        assert "42" in formatted
    
    def test_performance_logger(self):
        """Test performance logging functionality."""
        perf_logger = PerformanceLogger("test_perf")
        
        # Test timer functionality
        perf_logger.start_timer("test_operation")
        import time
        time.sleep(0.01)  # Small delay
        duration = perf_logger.end_timer("test_operation")
        
        assert duration is not None
        assert duration > 0
    
    def test_error_tracker(self):
        """Test error tracking functionality."""
        tracker = ErrorTracker("test_tracker")
        
        # Log some errors
        error1 = ValueError("Test error 1")
        error2 = ValueError("Test error 2")
        error3 = TypeError("Different error type")
        
        tracker.log_error(error1, {"context": "test1"})
        tracker.log_error(error2, {"context": "test2"})
        tracker.log_error(error3, {"context": "test3"})
        
        summary = tracker.get_error_summary()
        assert summary['total_errors'] == 3
        assert summary['error_counts']['ValueError'] == 2
        assert summary['error_counts']['TypeError'] == 1
        assert summary['most_common_error'] == 'ValueError'
    
    def test_timed_operation_context_manager(self):
        """Test timed operation context manager."""
        with TimedOperation("test_operation") as timer:
            import time
            time.sleep(0.01)  # Small delay
        
        # Should complete without errors
        assert timer is not None
    
    def test_logging_setup(self):
        """Test logging setup with various configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            config = setup_logging(
                level="DEBUG",
                log_file=log_file,
                include_context=True,
                enable_performance_logging=True,
                enable_error_tracking=True
            )
            
            assert 'loggers' in config
            assert 'root' in config['loggers']
            assert 'performance' in config['loggers']
            assert 'error_tracker' in config['loggers']
            
            # Test that log file was created
            logger = logging.getLogger('lsm_lite.test')
            logger.info("Test log message")
            
            assert os.path.exists(log_file)


class TestDualCNNPipelineErrorHandling:
    """Test error handling in dual CNN pipeline."""
    
    def test_pipeline_initialization_with_invalid_config(self):
        """Test pipeline initialization with invalid configuration."""
        # Create invalid config
        config = DualCNNConfig()
        config.training_batch_size = -1  # Invalid
        config.attention_heads = 0  # Invalid
        
        with pytest.raises(ConfigurationError) as exc_info:
            pipeline = DualCNNPipeline(config)
        
        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        assert "validation failed" in str(error).lower()
    
    @patch('lsm_lite.core.dual_cnn_pipeline.UnifiedTokenizer')
    @patch('lsm_lite.core.dual_cnn_pipeline.SinusoidalEmbedder')
    def test_pipeline_fallback_modes(self, mock_embedder, mock_tokenizer):
        """Test pipeline fallback modes when components fail."""
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.vocab_size = 10000
        mock_tokenizer.return_value.tokenize.return_value = {'input_ids': [tf.constant([1, 2, 3])]}
        
        mock_embedder.return_value = Mock()
        
        # Create valid config
        config = DualCNNConfig()
        pipeline = DualCNNPipeline(config)
        
        # Test fallback status
        fallback_status = pipeline.get_fallback_status()
        assert 'fallback_mode_enabled' in fallback_status
        assert 'single_cnn_fallback' in fallback_status
    
    def test_pipeline_component_recovery(self):
        """Test component recovery mechanisms."""
        config = DualCNNConfig()
        pipeline = DualCNNPipeline(config)
        
        # Test recovery attempt
        recovery_success = pipeline._attempt_component_recovery("reservoir", ValueError("Test error"))
        assert isinstance(recovery_success, bool)
        
        # Check that config was modified for recovery
        if recovery_success:
            assert config.attention_heads <= 4  # Should be reduced
            assert config.reservoir_size <= 256  # Should be reduced


class TestDualCNNTrainerErrorHandling:
    """Test error handling in dual CNN trainer."""
    
    @patch('lsm_lite.training.dual_cnn_trainer.DualCNNPipeline')
    def test_trainer_initialization_validation(self, mock_pipeline_class):
        """Test trainer initialization with validation."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = False
        mock_pipeline.get_fallback_status.return_value = {
            "fallback_mode_enabled": False,
            "single_cnn_fallback": False
        }
        
        config = DualCNNConfig()
        
        with pytest.raises(ComputationError) as exc_info:
            trainer = DualCNNTrainer(mock_pipeline, config)
        
        error = exc_info.value
        assert "must be fully initialized" in str(error).lower()
    
    @patch('lsm_lite.training.dual_cnn_trainer.DualCNNPipeline')
    def test_trainer_memory_monitoring(self, mock_pipeline_class):
        """Test memory monitoring in trainer."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = True
        mock_pipeline.get_fallback_status.return_value = {
            "fallback_mode_enabled": False,
            "single_cnn_fallback": False
        }
        mock_pipeline.tokenizer = Mock()
        mock_pipeline.first_cnn = Mock()
        mock_pipeline.first_cnn.trainable_variables = []
        mock_pipeline.reservoir = Mock()
        mock_pipeline.reservoir.trainable_variables = []
        
        config = DualCNNConfig()
        
        # This should not raise an error now
        trainer = DualCNNTrainer(mock_pipeline, config)
        
        # Test memory monitor
        assert hasattr(trainer, '_memory_monitor')
        memory_stats = trainer._memory_monitor.get_stats()
        assert 'current_mb' in memory_stats
        assert 'peak_mb' in memory_stats
    
    def test_trainer_recovery_strategies(self):
        """Test trainer recovery strategies."""
        # Create mock trainer with minimal setup
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = True
        mock_pipeline.get_fallback_status.return_value = {
            "fallback_mode_enabled": False,
            "single_cnn_fallback": False
        }
        
        config = DualCNNConfig()
        
        # Mock the training components setup to avoid actual initialization
        with patch.object(DualCNNTrainer, '_setup_training_components'):
            trainer = DualCNNTrainer(mock_pipeline, config)
        
        # Test memory error recovery
        recovery_success = trainer._recover_from_memory_error()
        assert isinstance(recovery_success, bool)
        
        # Test generic recovery
        recovery_success = trainer._generic_recovery()
        assert isinstance(recovery_success, bool)


class TestIntegrationErrorHandling:
    """Test integration of error handling across components."""
    
    def test_end_to_end_error_handling(self):
        """Test end-to-end error handling flow."""
        # This test verifies that errors propagate correctly through the system
        # and are enhanced with proper context and solutions
        
        # Create invalid training data
        invalid_data = []  # Empty data
        
        config = DualCNNConfig()
        
        # This should raise a DataValidationError
        with pytest.raises((DataValidationError, ConfigurationError)) as exc_info:
            pipeline = DualCNNPipeline(config)
            pipeline.fit_and_initialize(invalid_data)
        
        error = exc_info.value
        assert isinstance(error, (DataValidationError, ConfigurationError))
        assert hasattr(error, 'context')
        assert hasattr(error, 'solutions')
    
    def test_graceful_degradation_flow(self):
        """Test graceful degradation when components fail."""
        config = DualCNNConfig()
        
        # Create pipeline (this should work with valid config)
        pipeline = DualCNNPipeline(config)
        
        # Check that fallback mechanisms are available
        assert hasattr(pipeline, '_use_single_cnn_fallback')
        assert hasattr(pipeline, '_fallback_mode')
        
        # Test fallback status reporting
        fallback_status = pipeline.get_fallback_status()
        assert isinstance(fallback_status, dict)
        assert 'fallback_mode_enabled' in fallback_status
    
    @patch('lsm_lite.utils.error_handling.global_error_handler')
    def test_global_error_handler_integration(self, mock_global_handler):
        """Test integration with global error handler."""
        mock_global_handler.handle_error.return_value = Mock()
        mock_global_handler.attempt_recovery.return_value = True
        
        # Test that global error handler is called
        @handle_lsm_error
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(LSMError):
            test_function()
        
        # Verify global handler was used
        assert mock_global_handler.handle_error.called


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])