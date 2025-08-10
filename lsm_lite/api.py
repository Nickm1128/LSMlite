"""
Main convenience API for LSM Lite.

This module provides the primary interface for users to interact with the LSM system.
It handles model creation, training, and inference through a simple API.
"""

import os
import logging
from typing import List, Optional, Dict, Any

from .utils.config import LSMConfig
from .utils.persistence import ModelPersistence
from .core.tokenizer import UnifiedTokenizer
from .core.reservoir import SparseReservoir
from .core.cnn import CNNProcessor
from .data.loader import DataLoader
from .data.embeddings import SinusoidalEmbedder
from .training.trainer import LSMTrainer
from .inference.generator import TextGenerator

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
