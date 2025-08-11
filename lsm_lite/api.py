"""
Main convenience API for LSM Lite.

This module provides the primary interface for users to interact with the LSM system.
It handles model creation, training, and inference through a simple API.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import tensorflow as tf
import numpy as np

from .utils.config import LSMConfig, DualCNNConfig
from .utils.persistence import ModelPersistence
from .core.tokenizer import UnifiedTokenizer
from .core.reservoir import SparseReservoir
from .core.cnn import CNNProcessor
from .core.dual_cnn_pipeline import DualCNNPipeline
from .data.loader import DataLoader
from .data.embeddings import SinusoidalEmbedder
from .training.trainer import LSMTrainer
from .training.dual_cnn_trainer import DualCNNTrainer
from .inference.generator import TextGenerator

# Enhanced error handling
from .utils.error_handling import (
    handle_lsm_error, ErrorContext, ConfigurationError, 
    DataValidationError, ComputationError, global_error_handler,
    ValidationUtils
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSMLite:
    """Main convenience API for LSM Lite."""
    
    def __init__(self, config: Optional[LSMConfig] = None):
        """
        Initialize LSM Lite with configuration.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or LSMConfig()
        self._model = None
        self._tokenizer = None
        self._embedder = None
        self._reservoir = None
        self._cnn = None
        self._trainer = None
        self._generator = None
        
        # Dual CNN components
        self._dual_cnn_pipeline = None
        self._dual_cnn_trainer = None
        
        logger.info("LSM Lite initialized with config: %s", self.config)
    
    def build_model(self) -> None:
        """Build all model components based on configuration."""
        logger.info("Building model components...")
        
        # Initialize tokenizer
        self._tokenizer = UnifiedTokenizer(
            backend=self.config.tokenizer_backend,
            max_length=self.config.max_length
        )
        logger.info("Tokenizer built with backend: %s", self.config.tokenizer_backend)
        
        # Initialize embedder
        vocab_size = self._tokenizer.vocab_size or 10000  # Default vocab size if None
        self._embedder = SinusoidalEmbedder(
            vocab_size=vocab_size,
            embedding_dim=self.config.embedding_dim,
            max_length=self.config.max_length
        )
        logger.info("Embedder built with vocab size: %d", self._tokenizer.vocab_size)
        
        # Initialize reservoir
        self._reservoir = SparseReservoir(
            input_dim=self.config.embedding_dim,
            reservoir_size=self.config.reservoir_size,
            sparsity=self.config.sparsity,
            spectral_radius=self.config.spectral_radius
        )
        logger.info("Reservoir built with size: %d", self.config.reservoir_size)
        
        # Initialize CNN processor
        # Calculate input shape for CNN based on reservoir output
        reservoir_output_shape = (self.config.max_length, self.config.reservoir_size)
        self._cnn = CNNProcessor(
            input_shape=reservoir_output_shape,
            architecture=self.config.cnn_architecture,
            filters=self.config.cnn_filters,
            vocab_size=vocab_size
        )
        logger.info("CNN processor built with architecture: %s", self.config.cnn_architecture)
        
        # Initialize trainer
        self._trainer = LSMTrainer(
            tokenizer=self._tokenizer,
            embedder=self._embedder,
            reservoir=self._reservoir,
            cnn=self._cnn,
            config=self.config
        )
        
        # Initialize generator
        self._generator = TextGenerator(
            model=self._trainer.model,
            tokenizer=self._tokenizer,
            embedder=self._embedder
        )
        
        logger.info("Model build complete!")
    
    def train(self, dataset_name: str = 'cosmopedia-v2', 
              max_samples: Optional[int] = None,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the LSM model on conversational data.
        
        Args:
            dataset_name: Name of dataset to use for training
            max_samples: Maximum number of samples to use (None for all)
            epochs: Number of training epochs (None to use config default)
            batch_size: Batch size for training (None to use config default)
            
        Returns:
            Training history dictionary
        """
        if self._trainer is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info("Starting training on dataset: %s", dataset_name)
        
        # Load training data
        data_loader = DataLoader(
            dataset_name=dataset_name,
            max_samples=max_samples or self.config.max_samples
        )
        conversations = data_loader.load_conversations()
        logger.info("Loaded %d conversations for training", len(conversations))
        
        # Train the model
        training_epochs = epochs or self.config.epochs
        training_batch_size = batch_size or self.config.batch_size
        
        history = self._trainer.train(
            conversations=conversations,
            epochs=training_epochs,
            batch_size=training_batch_size
        )
        
        logger.info("Training completed!")
        return history
    
    def generate(self, prompt: str, max_length: int = 50, 
                 temperature: float = 1.0) -> str:
        """
        Generate text continuation for a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text continuation
        """
        if self._generator is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self._generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
    
    def evaluate(self, test_conversations: List[str]) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_conversations: List of test conversation strings
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self._trainer is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self._trainer.evaluate(test_conversations)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Directory path to save the model
        """
        if self._trainer is None or self._trainer.model is None:
            raise ValueError("No trained model to save. Train the model first.")
        
        if self._tokenizer is None or self._embedder is None:
            raise ValueError("Tokenizer or embedder not initialized.")
        
        ModelPersistence.save_model(
            model=self._trainer.model,
            tokenizer=self._tokenizer,
            embedder=self._embedder,
            config=self.config,
            path=path
        )
        logger.info("Model saved to: %s", path)
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Directory path to load the model from
        """
        components = ModelPersistence.load_model(path)
        
        self._model = components['model']
        self._tokenizer = components['tokenizer']
        self._embedder = components['embedder']
        self.config = components['config']
        
        # Rebuild other components
        self._generator = TextGenerator(
            model=self._model,
            tokenizer=self._tokenizer,
            embedder=self._embedder
        )
        
        logger.info("Model loaded from: %s", path)
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._trainer is not None and self._trainer.model is not None
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call build_model() first.")
        vocab_size = self._tokenizer.vocab_size
        if vocab_size is None:
            raise ValueError("Tokenizer vocab size not available.")
        return vocab_size
    
    # Dual CNN Convenience Methods
    
    @handle_lsm_error
    def setup_dual_cnn_pipeline(self, 
                               training_data: List[str],
                               dual_cnn_config: Optional[DualCNNConfig] = None,
                               embedder_params: Optional[Dict[str, Any]] = None,
                               reservoir_params: Optional[Dict[str, Any]] = None,
                               cnn_params: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[callable] = None,
                               enable_fallback: bool = True) -> DualCNNPipeline:
        """
        Create and initialize a dual CNN pipeline with enhanced error handling.
        
        This convenience method handles the complete pipeline setup:
        - Validates training data format and content
        - Fits embedder to training data with fallback options
        - Initializes attentive reservoir with graceful degradation
        - Sets up first CNN for next-token prediction
        - Enables rolling wave output storage with memory management
        - Initializes second CNN with fallback to single CNN if needed
        
        Args:
            training_data: List of training text strings for embedder fitting
            dual_cnn_config: Optional dual CNN configuration (uses intelligent defaults if None)
            embedder_params: Optional parameters for embedder fitting
            reservoir_params: Optional parameters for reservoir initialization
            cnn_params: Optional parameters for CNN setup
            progress_callback: Optional callback for initialization progress updates
            enable_fallback: Whether to enable fallback modes on component failures
            
        Returns:
            Initialized DualCNNPipeline instance
            
        Raises:
            DataValidationError: If training_data is empty or invalid
            ConfigurationError: If configuration validation fails
            ComponentInitializationError: If critical component initialization fails
        """
        # Enhanced data validation
        if not training_data:
            raise DataValidationError("training_data cannot be empty")
        
        data_issues = ValidationUtils.validate_training_data(training_data)
        if data_issues:
            context = ErrorContext(
                component="LSMLite",
                operation="setup_dual_cnn_pipeline",
                config_values={"data_size": len(training_data)}
            )
            raise DataValidationError(
                f"Training data validation failed: {'; '.join(data_issues)}",
                context=context
            )
        
        logger.info("Setting up dual CNN pipeline with %d training samples", len(training_data))
        
        # Create dual CNN configuration with enhanced defaults
        if dual_cnn_config is None:
            try:
                # Analyze training data characteristics for intelligent defaults
                data_characteristics = self._analyze_training_data_enhanced(training_data)
                dual_cnn_config = DualCNNConfig().get_intelligent_defaults(data_characteristics)
                logger.info("Using intelligent defaults for dual CNN configuration")
            except Exception as e:
                logger.warning(f"Data analysis failed, using basic defaults: {e}")
                dual_cnn_config = DualCNNConfig()  # Use basic defaults
        
        # Validate configuration
        config_issues = ValidationUtils.validate_config_parameters(dual_cnn_config)
        if config_issues:
            context = ErrorContext(
                component="LSMLite",
                operation="config_validation",
                config_values=self._safe_config_dict(dual_cnn_config)
            )
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(config_issues)}",
                context=context
            )
        
        # Create and initialize pipeline with error handling
        try:
            self._dual_cnn_pipeline = DualCNNPipeline(dual_cnn_config)
            
            # Fit and initialize all components
            self._dual_cnn_pipeline.fit_and_initialize(
                training_data=training_data,
                embedder_params=embedder_params,
                reservoir_params=reservoir_params,
                cnn_params=cnn_params,
                progress_callback=progress_callback,
                enable_fallback=enable_fallback
            )
            
            # Check final status and warn about fallbacks
            fallback_status = self._dual_cnn_pipeline.get_fallback_status()
            if fallback_status["fallback_mode_enabled"]:
                logger.warning("Pipeline initialized with fallback modes:")
                if fallback_status["single_cnn_fallback"]:
                    logger.warning("  - Single CNN fallback enabled (second CNN unavailable)")
                if not fallback_status["attentive_reservoir_available"]:
                    logger.warning("  - Standard reservoir fallback (attention unavailable)")
            
            logger.info("Dual CNN pipeline setup completed successfully")
            return self._dual_cnn_pipeline
            
        except Exception as e:
            # Attempt recovery if enabled
            if enable_fallback:
                logger.warning(f"Pipeline setup failed, attempting recovery: {e}")
                
                try:
                    # Try with reduced configuration
                    reduced_config = self._get_reduced_config(dual_cnn_config)
                    self._dual_cnn_pipeline = DualCNNPipeline(reduced_config)
                    
                    self._dual_cnn_pipeline.fit_and_initialize(
                        training_data=training_data[:1000],  # Use smaller sample
                        embedder_params=embedder_params,
                        reservoir_params=reservoir_params,
                        cnn_params=cnn_params,
                        progress_callback=progress_callback,
                        enable_fallback=True
                    )
                    
                    logger.info("Pipeline setup completed with reduced configuration")
                    return self._dual_cnn_pipeline
                    
                except Exception as recovery_error:
                    logger.error(f"Pipeline recovery failed: {recovery_error}")
            
            # Final failure
            context = ErrorContext(
                component="LSMLite",
                operation="setup_dual_cnn_pipeline",
                config_values=self._safe_config_dict(dual_cnn_config)
            )
            enhanced_error = global_error_handler.handle_error(e, context)
            raise enhanced_error
    
    def quick_dual_cnn_train(self, 
                           dataset_name: str = 'cosmopedia-v2',
                           max_samples: Optional[int] = None,
                           dual_cnn_config: Optional[DualCNNConfig] = None,
                           epochs: Optional[int] = None,
                           batch_size: Optional[int] = None,
                           validation_split: float = 0.1,
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        One-line dual CNN training setup and execution.
        
        This convenience method provides a complete dual CNN training workflow:
        - Loads training data from specified dataset
        - Sets up dual CNN pipeline with intelligent defaults
        - Trains both CNNs with rolling wave coordination
        - Returns comprehensive training results
        
        Args:
            dataset_name: Name of dataset to use for training
            max_samples: Maximum number of samples to use (None for intelligent default)
            dual_cnn_config: Optional dual CNN configuration
            epochs: Number of training epochs (None to use config default)
            batch_size: Training batch size (None to use config default)
            validation_split: Validation split ratio
            progress_callback: Optional callback for training progress updates
            
        Returns:
            Dictionary with training results, metrics, and pipeline status
            
        Raises:
            ValueError: If dataset cannot be loaded or is invalid
            DualCNNTrainingError: If training fails
        """
        logger.info("Starting quick dual CNN training on dataset: %s", dataset_name)
        
        # Load training data
        data_loader = DataLoader(
            dataset_name=dataset_name,
            max_samples=max_samples or 10000  # Default sample size
        )
        conversations = data_loader.load_conversations()
        
        if not conversations:
            raise ValueError(f"No conversations loaded from dataset: {dataset_name}")
        
        logger.info("Loaded %d conversations for dual CNN training", len(conversations))
        
        # Setup pipeline if not already done
        if self._dual_cnn_pipeline is None:
            self._dual_cnn_pipeline = self.setup_dual_cnn_pipeline(
                training_data=conversations,
                dual_cnn_config=dual_cnn_config,
                progress_callback=progress_callback
            )
        
        # Create dual CNN trainer
        config = dual_cnn_config or self._dual_cnn_pipeline.config
        self._dual_cnn_trainer = DualCNNTrainer(self._dual_cnn_pipeline, config)
        
        # Add progress callback if provided
        if progress_callback:
            self._dual_cnn_trainer.add_progress_callback(progress_callback)
        
        # Train dual CNN
        training_results = self._dual_cnn_trainer.train_dual_cnn(
            training_data=conversations,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        logger.info("Quick dual CNN training completed successfully")
        return training_results
    
    def dual_cnn_generate(self, 
                         prompt: str,
                         max_length: int = 50,
                         temperature: float = 1.0,
                         top_k: Optional[int] = None,
                         top_p: Optional[float] = None,
                         use_wave_coordination: bool = True) -> str:
        """
        Generate text using the dual CNN approach.
        
        This method uses both CNNs in coordination:
        - First CNN processes input through attentive reservoir
        - Rolling wave outputs are captured and stored
        - Second CNN uses wave features for final prediction
        - Results are combined using configured weights
        
        Args:
            prompt: Input text prompt for generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter (None to disable)
            top_p: Top-p (nucleus) sampling parameter (None to disable)
            use_wave_coordination: Whether to use dual CNN coordination
            
        Returns:
            Generated text continuation
            
        Raises:
            ValueError: If dual CNN pipeline is not initialized or trained
            RuntimeError: If generation fails
        """
        if self._dual_cnn_pipeline is None:
            raise ValueError("Dual CNN pipeline not initialized. Call setup_dual_cnn_pipeline() first.")
        
        if not self._dual_cnn_pipeline.is_initialized():
            raise ValueError("Dual CNN pipeline not fully initialized.")
        
        if self._dual_cnn_trainer is None:
            raise ValueError("Dual CNN trainer not available. Call quick_dual_cnn_train() first.")
        
        logger.info("Generating text with dual CNN approach: '%s'", prompt[:50])
        
        try:
            # Tokenize input prompt
            tokenized = self._dual_cnn_pipeline.tokenizer.tokenize([prompt], padding=True, truncation=True)
            input_ids = tokenized['input_ids'][0]
            
            # Generate tokens using dual CNN coordination
            generated_tokens = self._generate_with_dual_cnn(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_wave_coordination=use_wave_coordination
            )
            
            # Decode generated tokens to text
            generated_text = self._dual_cnn_pipeline.tokenizer.decode(generated_tokens)
            
            logger.info("Generated %d tokens successfully", len(generated_tokens))
            return generated_text
            
        except Exception as e:
            logger.error("Text generation failed: %s", str(e))
            raise RuntimeError(f"Dual CNN generation failed: {str(e)}")
    
    # Helper methods for dual CNN functionality
    
    def _analyze_training_data(self, training_data: List[str]) -> Dict[str, Any]:
        """Analyze training data characteristics for intelligent configuration."""
        return self._analyze_training_data_enhanced(training_data)
    
    def _analyze_training_data_enhanced(self, training_data: List[str]) -> Dict[str, Any]:
        """Enhanced training data analysis with error handling and validation."""
        if not training_data:
            return {}
        
        try:
            # Sample data for analysis (avoid processing all data)
            sample_size = min(1000, len(training_data))
            sample_data = training_data[:sample_size]
            
            # Filter out empty or invalid texts
            valid_texts = [text for text in sample_data if text and text.strip()]
            
            if not valid_texts:
                logger.warning("No valid texts found in training data sample")
                return {'dataset_size': len(training_data), 'valid_texts': 0}
            
            # Calculate text length statistics
            lengths = []
            char_counts = []
            
            for text in valid_texts:
                try:
                    words = text.split()
                    lengths.append(len(words))
                    char_counts.append(len(text))
                except Exception as e:
                    logger.debug(f"Failed to analyze text: {e}")
                    continue
            
            if not lengths:
                return {'dataset_size': len(training_data), 'valid_texts': len(valid_texts)}
            
            # Basic statistics
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            median_length = sorted(lengths)[len(lengths) // 2]
            
            avg_chars = sum(char_counts) / len(char_counts)
            
            # Estimate vocabulary size with error handling
            estimated_vocab_size = 10000  # Default fallback
            try:
                all_words = set()
                for text in valid_texts[:100]:  # Sample smaller for vocab estimation
                    words = text.lower().split()
                    all_words.update(words)
                
                if all_words:
                    estimated_vocab_size = len(all_words) * 10  # Rough scaling factor
            except Exception as e:
                logger.debug(f"Vocabulary estimation failed: {e}")
            
            # Detect language complexity
            complexity_score = self._estimate_text_complexity(valid_texts[:50])
            
            # Memory estimation
            estimated_memory_gb = self._estimate_memory_requirements(
                len(training_data), avg_length, estimated_vocab_size
            )
            
            characteristics = {
                'dataset_size': len(training_data),
                'valid_texts': len(valid_texts),
                'avg_length': avg_length,
                'median_length': median_length,
                'max_length': max_length,
                'min_length': min_length,
                'avg_chars': avg_chars,
                'vocab_size': estimated_vocab_size,
                'complexity_score': complexity_score,
                'estimated_memory_gb': estimated_memory_gb,
                'sample_analyzed': sample_size
            }
            
            logger.info("Enhanced training data analysis: %s", characteristics)
            return characteristics
            
        except Exception as e:
            logger.error(f"Training data analysis failed: {e}")
            return {
                'dataset_size': len(training_data),
                'analysis_failed': True,
                'error': str(e)
            }
    
    def _estimate_text_complexity(self, texts: List[str]) -> float:
        """Estimate text complexity based on various factors."""
        try:
            if not texts:
                return 0.5  # Medium complexity default
            
            complexity_factors = []
            
            for text in texts:
                try:
                    words = text.split()
                    if not words:
                        continue
                    
                    # Average word length
                    avg_word_len = sum(len(word) for word in words) / len(words)
                    
                    # Sentence complexity (rough estimate)
                    sentences = text.split('.')
                    avg_sentence_len = len(words) / max(len(sentences), 1)
                    
                    # Vocabulary diversity
                    unique_words = len(set(word.lower() for word in words))
                    vocab_diversity = unique_words / len(words) if words else 0
                    
                    # Combine factors
                    complexity = (
                        min(avg_word_len / 10, 1.0) * 0.3 +
                        min(avg_sentence_len / 20, 1.0) * 0.4 +
                        vocab_diversity * 0.3
                    )
                    
                    complexity_factors.append(complexity)
                    
                except Exception as e:
                    logger.debug(f"Failed to analyze text complexity: {e}")
                    continue
            
            if complexity_factors:
                return sum(complexity_factors) / len(complexity_factors)
            else:
                return 0.5  # Default medium complexity
                
        except Exception as e:
            logger.debug(f"Complexity estimation failed: {e}")
            return 0.5
    
    def _estimate_memory_requirements(self, dataset_size: int, avg_length: float, vocab_size: int) -> float:
        """Estimate memory requirements for training."""
        try:
            # Rough estimation based on model parameters and data size
            
            # Model parameters (rough estimate)
            embedding_params = vocab_size * 512  # Embedding dimension
            reservoir_params = 512 * 512  # Reservoir size
            cnn_params = 256 * 256 * 3  # CNN filters
            
            total_params = embedding_params + reservoir_params + cnn_params
            
            # Memory for parameters (4 bytes per float32 parameter)
            param_memory_gb = (total_params * 4) / (1024**3)
            
            # Memory for activations and gradients (rough estimate)
            batch_memory_gb = (avg_length * 512 * 32 * 4) / (1024**3)  # Batch size 32
            
            # Wave storage memory
            wave_memory_gb = 0.1  # Conservative estimate
            
            # Total with safety margin
            total_memory_gb = (param_memory_gb + batch_memory_gb + wave_memory_gb) * 2
            
            return max(total_memory_gb, 1.0)  # Minimum 1GB
            
        except Exception as e:
            logger.debug(f"Memory estimation failed: {e}")
            return 2.0  # Default 2GB estimate
    
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
    
    def _get_reduced_config(self, original_config: DualCNNConfig) -> DualCNNConfig:
        """Get a reduced configuration for recovery attempts."""
        try:
            # Create a copy of the original config
            reduced_config = DualCNNConfig()
            
            # Copy basic settings
            for attr in ['embedder_max_length', 'wave_feature_dim', 'training_batch_size']:
                if hasattr(original_config, attr):
                    setattr(reduced_config, attr, getattr(original_config, attr))
            
            # Reduce complex settings
            reduced_config.reservoir_size = min(256, getattr(original_config, 'reservoir_size', 512))
            reduced_config.attention_heads = min(4, getattr(original_config, 'attention_heads', 8))
            reduced_config.attention_dim = min(32, getattr(original_config, 'attention_dim', 64))
            
            # Reduce CNN complexity
            original_filters = getattr(original_config, 'first_cnn_filters', [64, 128, 256])
            reduced_config.first_cnn_filters = [f // 2 for f in original_filters]
            reduced_config.second_cnn_filters = [f // 2 for f in original_filters]
            
            # Reduce memory usage
            reduced_config.max_memory_usage_gb = min(1.0, getattr(original_config, 'max_memory_usage_gb', 2.0))
            reduced_config.wave_window_size = min(25, getattr(original_config, 'wave_window_size', 50))
            
            # Reduce training parameters
            reduced_config.training_batch_size = min(16, getattr(original_config, 'training_batch_size', 32))
            
            logger.info("Created reduced configuration for recovery")
            return reduced_config
            
        except Exception as e:
            logger.error(f"Failed to create reduced config: {e}")
            return DualCNNConfig()  # Return basic default config
    
    def _generate_with_dual_cnn(self, 
                               input_ids: List[int],
                               max_length: int,
                               temperature: float,
                               top_k: Optional[int],
                               top_p: Optional[float],
                               use_wave_coordination: bool) -> List[int]:
        """Generate tokens using dual CNN coordination."""
        import tensorflow as tf
        import numpy as np
        
        # Convert input to tensor
        input_tensor = tf.constant([input_ids], dtype=tf.int32)
        initial_length = len(input_ids)
        generated_tokens = input_ids.copy().tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
        
        # Clear wave storage for new generation
        self._dual_cnn_pipeline.wave_storage.clear_storage()
        
        for step in range(max_length):
            # Prepare current input (last N tokens based on max_length)
            current_input = generated_tokens[-self._dual_cnn_pipeline.config.embedder_max_length:]
            if len(current_input) < self._dual_cnn_pipeline.config.embedder_max_length:
                # Pad with zeros if needed
                padding = [0] * (self._dual_cnn_pipeline.config.embedder_max_length - len(current_input))
                current_input = padding + current_input
            
            current_tensor = tf.constant([current_input], dtype=tf.int32)
            
            # Forward pass through pipeline
            embedded = self._dual_cnn_pipeline.embedder(current_tensor)
            reservoir_states, attention_weights = self._dual_cnn_pipeline.reservoir(embedded, training=False)
            
            # First CNN prediction
            first_cnn_output = self._dual_cnn_pipeline.first_cnn(reservoir_states, training=False)
            first_cnn_probs = tf.nn.softmax(first_cnn_output / temperature, axis=-1)
            
            if use_wave_coordination:
                # Extract and store wave features
                wave_features = self._extract_wave_features_for_generation(reservoir_states, attention_weights)
                
                # Store wave output for potential second CNN use
                try:
                    self._dual_cnn_pipeline.wave_storage.store_wave(
                        wave_output=wave_features[0],  # Remove batch dimension
                        sequence_position=step
                    )
                except Exception as e:
                    logger.warning("Wave storage failed, using first CNN only: %s", e)
                    use_wave_coordination = False
                
                # Second CNN prediction if we have enough wave data
                if step >= self._dual_cnn_pipeline.config.wave_window_size // 2:
                    try:
                        # Prepare second CNN input from wave storage
                        wave_sequence = self._dual_cnn_pipeline.wave_storage.get_wave_sequence(
                            start_pos=max(0, step - self._dual_cnn_pipeline.config.wave_window_size + 1),
                            length=self._dual_cnn_pipeline.config.wave_window_size
                        )
                        
                        if wave_sequence is not None:
                            wave_input = tf.expand_dims(wave_sequence, axis=0)  # Add batch dimension
                            second_cnn_output = self._dual_cnn_pipeline.second_cnn(wave_input, training=False)
                            second_cnn_probs = tf.nn.softmax(second_cnn_output / temperature, axis=-1)
                            
                            # Combine predictions using configured weights
                            combined_probs = (
                                self._dual_cnn_pipeline.config.wave_coordination_weight * first_cnn_probs +
                                self._dual_cnn_pipeline.config.final_prediction_weight * second_cnn_probs
                            )
                        else:
                            combined_probs = first_cnn_probs
                    except Exception as e:
                        logger.warning("Second CNN prediction failed, using first CNN only: %s", e)
                        combined_probs = first_cnn_probs
                else:
                    combined_probs = first_cnn_probs
            else:
                combined_probs = first_cnn_probs
            
            # Sample next token
            next_token = self._sample_token(
                probs=combined_probs[0],  # Remove batch dimension
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            generated_tokens.append(int(next_token))
            
            # Stop if we hit an end token (if defined)
            if next_token == 0:  # Assuming 0 is padding/end token
                break
        
        return generated_tokens[initial_length:]  # Return only newly generated tokens
    
    def _extract_wave_features_for_generation(self, reservoir_states: tf.Tensor, attention_weights: tf.Tensor) -> tf.Tensor:
        """Extract wave features during generation (similar to training but for single step)."""
        # Apply attention weighting to reservoir states
        attended_weights = tf.reduce_mean(attention_weights, axis=[1, -1])  # (batch, sequence)
        attended_weights = tf.expand_dims(attended_weights, axis=-1)  # (batch, sequence, 1)
        
        # Weight reservoir states by attention
        weighted_states = reservoir_states * attended_weights
        
        # Project to wave feature dimension if needed
        if weighted_states.shape[-1] != self._dual_cnn_pipeline.config.wave_feature_dim:
            # Use a simple linear projection (in practice, this should be learned)
            projection_matrix = tf.random.normal(
                (weighted_states.shape[-1], self._dual_cnn_pipeline.config.wave_feature_dim),
                stddev=0.1
            )
            wave_features = tf.matmul(weighted_states, projection_matrix)
        else:
            wave_features = weighted_states
        
        return wave_features
    
    def _sample_token(self, 
                     probs: tf.Tensor,
                     temperature: float,
                     top_k: Optional[int],
                     top_p: Optional[float]) -> int:
        """Sample next token from probability distribution."""
        import numpy as np
        
        probs_np = probs.numpy()
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k_indices = np.argpartition(probs_np, -top_k)[-top_k:]
            filtered_probs = np.zeros_like(probs_np)
            filtered_probs[top_k_indices] = probs_np[top_k_indices]
            probs_np = filtered_probs
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1:
            sorted_indices = np.argsort(probs_np)[::-1]
            sorted_probs = probs_np[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            
            # Find cutoff index
            cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1
            
            # Keep only top-p tokens
            filtered_probs = np.zeros_like(probs_np)
            filtered_probs[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
            probs_np = filtered_probs
        
        # Renormalize probabilities
        if probs_np.sum() > 0:
            probs_np = probs_np / probs_np.sum()
        else:
            # Fallback to uniform distribution if all probabilities are zero
            probs_np = np.ones_like(probs_np) / len(probs_np)
        
        # Sample token
        token = np.random.choice(len(probs_np), p=probs_np)
        return token
    
    # Properties for dual CNN components
    
    @property
    def dual_cnn_pipeline(self) -> Optional[DualCNNPipeline]:
        """Get the dual CNN pipeline instance."""
        return self._dual_cnn_pipeline
    
    @property
    def dual_cnn_trainer(self) -> Optional[DualCNNTrainer]:
        """Get the dual CNN trainer instance."""
        return self._dual_cnn_trainer
    
    @property
    def is_dual_cnn_ready(self) -> bool:
        """Check if dual CNN pipeline is ready for training/generation."""
        return (self._dual_cnn_pipeline is not None and 
                self._dual_cnn_pipeline.is_initialized())
