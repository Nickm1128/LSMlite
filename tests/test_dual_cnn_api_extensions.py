"""
Integration tests for dual CNN API extensions.

This module tests the convenience methods added to the LSMLite API:
- setup_dual_cnn_pipeline
- quick_dual_cnn_train  
- dual_cnn_generate
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf
import numpy as np

# Import the main API class
from lsm_lite.api import LSMLite
from lsm_lite.utils.config import DualCNNConfig, LSMConfig
from lsm_lite.core.dual_cnn_pipeline import DualCNNPipeline, ComponentInitializationError
from lsm_lite.training.dual_cnn_trainer import DualCNNTrainer, DualCNNTrainingError


class TestDualCNNAPIExtensions:
    """Test suite for dual CNN API convenience methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api = LSMLite()
        self.sample_training_data = [
            "This is a sample text for training the dual CNN model.",
            "Another example sentence to provide more training data.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require diverse training examples.",
            "Natural language processing is a fascinating field of study."
        ]
        
        # Create a minimal dual CNN config for testing
        self.test_config = DualCNNConfig(
            embedder_fit_samples=100,
            embedder_batch_size=16,
            embedder_max_length=32,
            reservoir_size=64,
            attention_heads=2,
            attention_dim=32,
            first_cnn_filters=[16, 32],
            second_cnn_filters=[32, 64],
            wave_window_size=10,
            wave_overlap=2,
            max_wave_storage=50,
            wave_feature_dim=64,
            dual_training_epochs=2,
            training_batch_size=8
        )
    
    def test_setup_dual_cnn_pipeline_basic(self):
        """Test basic dual CNN pipeline setup."""
        with patch('lsm_lite.api.DualCNNPipeline') as mock_pipeline_class:
            # Mock pipeline instance
            mock_pipeline = Mock()
            mock_pipeline.is_initialized.return_value = True
            mock_pipeline_class.return_value = mock_pipeline
            
            # Test pipeline setup
            result = self.api.setup_dual_cnn_pipeline(
                training_data=self.sample_training_data,
                dual_cnn_config=self.test_config
            )
            
            # Verify pipeline was created and initialized
            mock_pipeline_class.assert_called_once_with(self.test_config)
            mock_pipeline.fit_and_initialize.assert_called_once()
            assert result == mock_pipeline
            assert self.api._dual_cnn_pipeline == mock_pipeline
    
    def test_setup_dual_cnn_pipeline_with_intelligent_defaults(self):
        """Test pipeline setup with intelligent defaults."""
        with patch('lsm_lite.api.DualCNNPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.is_initialized.return_value = True
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock intelligent defaults
            with patch.object(DualCNNConfig, 'get_intelligent_defaults') as mock_defaults:
                mock_defaults.return_value = self.test_config
                
                # Test without providing config
                result = self.api.setup_dual_cnn_pipeline(
                    training_data=self.sample_training_data
                )
                
                # Verify intelligent defaults were used
                mock_defaults.assert_called_once()
                mock_pipeline_class.assert_called_once_with(self.test_config)
    
    def test_setup_dual_cnn_pipeline_empty_data(self):
        """Test pipeline setup with empty training data."""
        with pytest.raises(ValueError, match="training_data cannot be empty"):
            self.api.setup_dual_cnn_pipeline(training_data=[])
    
    def test_setup_dual_cnn_pipeline_with_progress_callback(self):
        """Test pipeline setup with progress callback."""
        progress_updates = []
        
        def progress_callback(component, status, progress):
            progress_updates.append((component, status))
        
        with patch('lsm_lite.api.DualCNNPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.is_initialized.return_value = True
            mock_pipeline_class.return_value = mock_pipeline
            
            # Test with progress callback
            self.api.setup_dual_cnn_pipeline(
                training_data=self.sample_training_data,
                dual_cnn_config=self.test_config,
                progress_callback=progress_callback
            )
            
            # Verify callback was passed to fit_and_initialize
            mock_pipeline.fit_and_initialize.assert_called_once()
            call_args = mock_pipeline.fit_and_initialize.call_args
            assert call_args[1]['progress_callback'] == progress_callback
    
    def test_quick_dual_cnn_train_basic(self):
        """Test basic quick dual CNN training."""
        with patch('lsm_lite.api.DataLoader') as mock_loader_class:
            # Mock data loader
            mock_loader = Mock()
            mock_loader.load_conversations.return_value = self.sample_training_data
            mock_loader_class.return_value = mock_loader
            
            with patch('lsm_lite.api.DualCNNPipeline') as mock_pipeline_class:
                # Mock pipeline
                mock_pipeline = Mock()
                mock_pipeline.is_initialized.return_value = True
                mock_pipeline.config = self.test_config
                mock_pipeline_class.return_value = mock_pipeline
                
                with patch('lsm_lite.api.DualCNNTrainer') as mock_trainer_class:
                    # Mock trainer
                    mock_trainer = Mock()
                    mock_training_results = {
                        'training_history': {'combined_loss': [1.0, 0.8, 0.6]},
                        'final_metrics': {'combined_accuracy': 0.85},
                        'pipeline_status': {'fully_initialized': True}
                    }
                    mock_trainer.train_dual_cnn.return_value = mock_training_results
                    mock_trainer_class.return_value = mock_trainer
                    
                    # Test quick training
                    result = self.api.quick_dual_cnn_train(
                        dataset_name='test_dataset',
                        max_samples=100,
                        dual_cnn_config=self.test_config,
                        epochs=2,
                        batch_size=8
                    )
                    
                    # Verify components were created and training was executed
                    mock_loader_class.assert_called_once()
                    mock_pipeline_class.assert_called_once()
                    mock_trainer_class.assert_called_once_with(mock_pipeline, self.test_config)
                    mock_trainer.train_dual_cnn.assert_called_once()
                    
                    assert result == mock_training_results
                    assert self.api._dual_cnn_pipeline == mock_pipeline
                    assert self.api._dual_cnn_trainer == mock_trainer
    
    def test_quick_dual_cnn_train_no_data(self):
        """Test quick training with no data loaded."""
        with patch('lsm_lite.api.DataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_conversations.return_value = []  # Empty data
            mock_loader_class.return_value = mock_loader
            
            with pytest.raises(ValueError, match="No conversations loaded"):
                self.api.quick_dual_cnn_train(dataset_name='empty_dataset')
    
    def test_quick_dual_cnn_train_with_existing_pipeline(self):
        """Test quick training with existing pipeline."""
        # Set up existing pipeline
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = True
        mock_pipeline.config = self.test_config
        self.api._dual_cnn_pipeline = mock_pipeline
        
        with patch('lsm_lite.api.DataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_conversations.return_value = self.sample_training_data
            mock_loader_class.return_value = mock_loader
            
            with patch('lsm_lite.api.DualCNNTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer.train_dual_cnn.return_value = {'status': 'completed'}
                mock_trainer_class.return_value = mock_trainer
                
                # Test training with existing pipeline
                result = self.api.quick_dual_cnn_train(dataset_name='test_dataset')
                
                # Verify existing pipeline was used (no new setup call)
                mock_trainer_class.assert_called_once_with(mock_pipeline, self.test_config)
    
    def test_dual_cnn_generate_basic(self):
        """Test basic dual CNN text generation."""
        # Set up mock pipeline and trainer
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = True
        mock_pipeline.config = self.test_config
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = {'input_ids': [tf.constant([1, 2, 3, 4])]}
        mock_tokenizer.decode.return_value = "Generated text output"
        mock_pipeline.tokenizer = mock_tokenizer
        
        mock_trainer = Mock()
        
        self.api._dual_cnn_pipeline = mock_pipeline
        self.api._dual_cnn_trainer = mock_trainer
        
        # Mock the generation method
        with patch.object(self.api, '_generate_with_dual_cnn') as mock_generate:
            mock_generate.return_value = [5, 6, 7, 8]  # Mock generated tokens
            
            # Test generation
            result = self.api.dual_cnn_generate(
                prompt="Test prompt",
                max_length=20,
                temperature=0.8,
                use_wave_coordination=True
            )
            
            # Verify generation process
            mock_tokenizer.tokenize.assert_called_once()
            mock_generate.assert_called_once()
            mock_tokenizer.decode.assert_called_once_with([5, 6, 7, 8])
            
            assert result == "Generated text output"
    
    def test_dual_cnn_generate_no_pipeline(self):
        """Test generation without initialized pipeline."""
        with pytest.raises(ValueError, match="Dual CNN pipeline not initialized"):
            self.api.dual_cnn_generate("Test prompt")
    
    def test_dual_cnn_generate_no_trainer(self):
        """Test generation without trainer."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = True
        self.api._dual_cnn_pipeline = mock_pipeline
        
        with pytest.raises(ValueError, match="Dual CNN trainer not available"):
            self.api.dual_cnn_generate("Test prompt")
    
    def test_dual_cnn_generate_pipeline_not_initialized(self):
        """Test generation with uninitialized pipeline."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = False
        mock_trainer = Mock()
        
        self.api._dual_cnn_pipeline = mock_pipeline
        self.api._dual_cnn_trainer = mock_trainer
        
        with pytest.raises(ValueError, match="Dual CNN pipeline not fully initialized"):
            self.api.dual_cnn_generate("Test prompt")
    
    def test_analyze_training_data(self):
        """Test training data analysis for intelligent defaults."""
        characteristics = self.api._analyze_training_data(self.sample_training_data)
        
        # Verify analysis results
        assert 'dataset_size' in characteristics
        assert 'avg_length' in characteristics
        assert 'max_length' in characteristics
        assert 'min_length' in characteristics
        assert 'vocab_size' in characteristics
        assert 'sample_analyzed' in characteristics
        
        assert characteristics['dataset_size'] == len(self.sample_training_data)
        assert characteristics['avg_length'] > 0
        assert characteristics['vocab_size'] > 0
    
    def test_analyze_training_data_empty(self):
        """Test training data analysis with empty data."""
        characteristics = self.api._analyze_training_data([])
        assert characteristics == {}
    
    def test_analyze_training_data_large_dataset(self):
        """Test training data analysis with large dataset."""
        large_dataset = self.sample_training_data * 500  # 2500 samples
        characteristics = self.api._analyze_training_data(large_dataset)
        
        # Should only analyze first 1000 samples
        assert characteristics['sample_analyzed'] == 1000
        assert characteristics['dataset_size'] == 2500
    
    def test_sample_token_basic(self):
        """Test token sampling functionality."""
        # Create probability distribution
        probs = tf.constant([0.1, 0.3, 0.4, 0.2])
        
        # Test basic sampling
        token = self.api._sample_token(probs, temperature=1.0, top_k=None, top_p=None)
        assert isinstance(token, (int, np.integer))
        assert 0 <= token < len(probs)
    
    def test_sample_token_with_top_k(self):
        """Test token sampling with top-k filtering."""
        probs = tf.constant([0.1, 0.3, 0.4, 0.2])
        
        # Test top-k sampling
        token = self.api._sample_token(probs, temperature=1.0, top_k=2, top_p=None)
        assert isinstance(token, (int, np.integer))
        assert 0 <= token < len(probs)
    
    def test_sample_token_with_top_p(self):
        """Test token sampling with top-p filtering."""
        probs = tf.constant([0.1, 0.3, 0.4, 0.2])
        
        # Test top-p sampling
        token = self.api._sample_token(probs, temperature=1.0, top_k=None, top_p=0.8)
        assert isinstance(token, (int, np.integer))
        assert 0 <= token < len(probs)
    
    def test_properties(self):
        """Test dual CNN API properties."""
        # Test initial state
        assert self.api.dual_cnn_pipeline is None
        assert self.api.dual_cnn_trainer is None
        assert not self.api.is_dual_cnn_ready
        
        # Set up mock components
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = True
        mock_trainer = Mock()
        
        self.api._dual_cnn_pipeline = mock_pipeline
        self.api._dual_cnn_trainer = mock_trainer
        
        # Test with components
        assert self.api.dual_cnn_pipeline == mock_pipeline
        assert self.api.dual_cnn_trainer == mock_trainer
        assert self.api.is_dual_cnn_ready
    
    def test_integration_workflow(self):
        """Test complete integration workflow."""
        with patch('lsm_lite.api.DataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_conversations.return_value = self.sample_training_data
            mock_loader_class.return_value = mock_loader
            
            with patch('lsm_lite.api.DualCNNPipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.is_initialized.return_value = True
                mock_pipeline.config = self.test_config
                mock_pipeline.tokenizer = Mock()
                mock_pipeline.tokenizer.tokenize.return_value = {'input_ids': [tf.constant([1, 2, 3])]}
                mock_pipeline.tokenizer.decode.return_value = "Generated text"
                mock_pipeline_class.return_value = mock_pipeline
                
                with patch('lsm_lite.api.DualCNNTrainer') as mock_trainer_class:
                    mock_trainer = Mock()
                    mock_trainer.train_dual_cnn.return_value = {'status': 'completed'}
                    mock_trainer_class.return_value = mock_trainer
                    
                    with patch.object(self.api, '_generate_with_dual_cnn') as mock_generate:
                        mock_generate.return_value = [4, 5, 6]
                        
                        # Complete workflow: setup -> train -> generate
                        
                        # 1. Setup pipeline
                        pipeline = self.api.setup_dual_cnn_pipeline(
                            training_data=self.sample_training_data,
                            dual_cnn_config=self.test_config
                        )
                        assert pipeline == mock_pipeline
                        
                        # 2. Train model
                        training_results = self.api.quick_dual_cnn_train(
                            dataset_name='test_dataset',
                            epochs=2
                        )
                        assert training_results['status'] == 'completed'
                        
                        # 3. Generate text
                        generated_text = self.api.dual_cnn_generate(
                            prompt="Test prompt",
                            max_length=10
                        )
                        assert generated_text == "Generated text"
                        
                        # Verify all components were called
                        mock_pipeline_class.assert_called()
                        mock_trainer_class.assert_called()
                        mock_generate.assert_called()


class TestDualCNNAPIErrorHandling:
    """Test error handling in dual CNN API methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api = LSMLite()
        self.sample_data = ["Test data for error handling"]
    
    def test_setup_pipeline_initialization_error(self):
        """Test handling of component initialization errors."""
        with patch('lsm_lite.api.DualCNNPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.fit_and_initialize.side_effect = ComponentInitializationError(
                "test_component", "Test initialization error"
            )
            mock_pipeline_class.return_value = mock_pipeline
            
            with pytest.raises(ComponentInitializationError):
                self.api.setup_dual_cnn_pipeline(training_data=self.sample_data)
    
    def test_quick_train_training_error(self):
        """Test handling of training errors."""
        with patch('lsm_lite.api.DataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_conversations.return_value = self.sample_data
            mock_loader_class.return_value = mock_loader
            
            with patch('lsm_lite.api.DualCNNPipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline.is_initialized.return_value = True
                mock_pipeline.config = DualCNNConfig()
                mock_pipeline_class.return_value = mock_pipeline
                
                with patch('lsm_lite.api.DualCNNTrainer') as mock_trainer_class:
                    mock_trainer = Mock()
                    mock_trainer.train_dual_cnn.side_effect = DualCNNTrainingError(
                        "training", "dual_cnn", "Test training error"
                    )
                    mock_trainer_class.return_value = mock_trainer
                    
                    with pytest.raises(DualCNNTrainingError):
                        self.api.quick_dual_cnn_train(dataset_name='test_dataset')
    
    def test_generate_runtime_error(self):
        """Test handling of generation runtime errors."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized.return_value = True
        mock_trainer = Mock()
        
        self.api._dual_cnn_pipeline = mock_pipeline
        self.api._dual_cnn_trainer = mock_trainer
        
        # Mock tokenizer to raise an exception
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.side_effect = Exception("Tokenization failed")
        mock_pipeline.tokenizer = mock_tokenizer
        
        with pytest.raises(RuntimeError, match="Dual CNN generation failed"):
            self.api.dual_cnn_generate("Test prompt")


if __name__ == '__main__':
    pytest.main([__file__])