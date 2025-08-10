"""
Sinusoidal positional embeddings for LSM.

This module implements sinusoidal positional encodings combined with token
embeddings for use in the LSM architecture.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SinusoidalEmbedder(keras.layers.Layer):
    """Sinusoidal positional embeddings for LSM."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int = 512,
                 temperature: float = 10000.0, trainable_embeddings: bool = True,
                 name: str = "sinusoidal_embedder", **kwargs):
        """
        Initialize sinusoidal embedder.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            max_length: Maximum sequence length
            temperature: Temperature parameter for positional encoding
            trainable_embeddings: Whether token embeddings are trainable
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.temperature = temperature
        self.trainable_embeddings = trainable_embeddings
        
        # Build embedding components
        self._build_embeddings()
        
        logger.info("Sinusoidal embedder initialized: vocab_size=%d, embedding_dim=%d, max_length=%d",
                   vocab_size, embedding_dim, max_length)
    
    def _build_embeddings(self):
        """Build token and positional embedding matrices."""
        # Token embeddings (trainable)
        self.token_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            trainable=self.trainable_embeddings,
            name="token_embeddings"
        )
        
        # Create sinusoidal positional encodings (fixed)
        self.positional_encoding = self._create_positional_encoding()
        
        # Scale factor for embeddings
        self.embedding_scale = tf.Variable(
            tf.sqrt(tf.cast(self.embedding_dim, tf.float32)),
            trainable=False,
            name="embedding_scale"
        )
    
    def _create_positional_encoding(self) -> tf.Variable:
        """Create sinusoidal positional encoding matrix."""
        # Create position and dimension indices
        position = np.arange(self.max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * 
                         -(math.log(self.temperature) / self.embedding_dim))
        
        # Initialize positional encoding matrix
        pos_encoding = np.zeros((self.max_length, self.embedding_dim))
        
        # Apply sine to even indices
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        
        # Apply cosine to odd indices
        if self.embedding_dim > 1:
            pos_encoding[:, 1::2] = np.cos(position * div_term[:self.embedding_dim//2])
        
        # Convert to TensorFlow variable (non-trainable)
        return tf.Variable(
            pos_encoding.astype(np.float32),
            trainable=False,
            name="positional_encoding"
        )
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Apply token and positional embeddings.
        
        Args:
            inputs: Token IDs tensor of shape (batch_size, sequence_length)
            training: Whether in training mode
            
        Returns:
            Embedded sequences of shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
        
        # Ensure sequence length doesn't exceed max_length
        # Use tf.cond for symbolic tensor comparison
        def truncate_fn():
            return inputs[:, :self.max_length]
        
        def keep_fn():
            return inputs
            
        inputs = tf.cond(
            sequence_length > self.max_length,
            truncate_fn,
            keep_fn
        )
        sequence_length = tf.minimum(sequence_length, self.max_length)
        
        # Get token embeddings
        token_embeds = self.token_embedding(inputs)
        token_embeds = token_embeds * self.embedding_scale
        
        # Get positional encodings for the sequence
        positions = tf.range(sequence_length)
        pos_embeds = tf.gather(self.positional_encoding, positions)
        
        # Expand positional embeddings to match batch size
        pos_embeds = tf.expand_dims(pos_embeds, 0)
        pos_embeds = tf.tile(pos_embeds, [batch_size, 1, 1])
        
        # Combine token and positional embeddings
        embeddings = token_embeds + pos_embeds
        
        return embeddings
    
    def get_positional_encoding(self, sequence_length: int) -> tf.Tensor:
        """
        Get positional encoding for a specific sequence length.
        
        Args:
            sequence_length: Length of sequence
            
        Returns:
            Positional encoding tensor
        """
        positions = tf.range(min(sequence_length, self.max_length))
        return tf.gather(self.positional_encoding, positions)
    
    def compute_output_shape(self, input_shape):
        """Compute output shape for the layer."""
        batch_size, sequence_length = input_shape
        return (batch_size, sequence_length, self.embedding_dim)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'trainable_embeddings': self.trainable_embeddings,
        })
        return config


class PositionalEncodingAnalyzer:
    """Utility class for analyzing positional encodings."""
    
    def __init__(self, embedder: SinusoidalEmbedder):
        """
        Initialize analyzer.
        
        Args:
            embedder: SinusoidalEmbedder instance to analyze
        """
        self.embedder = embedder
    
    def visualize_positional_patterns(self, max_positions: int = 100) -> dict:
        """
        Analyze patterns in positional encodings.
        
        Args:
            max_positions: Maximum number of positions to analyze
            
        Returns:
            Dictionary with analysis results
        """
        pos_length = min(max_positions, self.embedder.max_length)
        pos_encoding = self.embedder.positional_encoding[:pos_length].numpy()
        
        # Compute various statistics
        analysis = {
            'shape': pos_encoding.shape,
            'mean_per_position': np.mean(pos_encoding, axis=1).tolist(),
            'std_per_position': np.std(pos_encoding, axis=1).tolist(),
            'mean_per_dimension': np.mean(pos_encoding, axis=0).tolist(),
            'std_per_dimension': np.std(pos_encoding, axis=0).tolist(),
            'max_value': float(np.max(pos_encoding)),
            'min_value': float(np.min(pos_encoding)),
        }
        
        return analysis
    
    def compute_position_similarities(self, positions: list) -> np.ndarray:
        """
        Compute cosine similarities between positional encodings.
        
        Args:
            positions: List of positions to compare
            
        Returns:
            Similarity matrix
        """
        pos_encodings = tf.gather(self.embedder.positional_encoding, positions)
        
        # Normalize encodings
        pos_encodings_norm = tf.nn.l2_normalize(pos_encodings, axis=-1)
        
        # Compute cosine similarity matrix
        similarity_matrix = tf.matmul(pos_encodings_norm, pos_encodings_norm, transpose_b=True)
        
        return similarity_matrix.numpy()


class LearnablePositionalEmbedding(keras.layers.Layer):
    """Alternative learnable positional embedding layer."""
    
    def __init__(self, max_length: int, embedding_dim: int, 
                 name: str = "learnable_positional", **kwargs):
        """
        Initialize learnable positional embedding.
        
        Args:
            max_length: Maximum sequence length
            embedding_dim: Embedding dimension
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Learnable positional embeddings
        self.position_embedding = keras.layers.Embedding(
            input_dim=max_length,
            output_dim=embedding_dim,
            name="position_embeddings"
        )
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply learnable positional embeddings.
        
        Args:
            inputs: Input tensor (typically from token embeddings)
            
        Returns:
            Positional embeddings to add to token embeddings
        """
        batch_size, sequence_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
        
        # Create position indices
        positions = tf.range(sequence_length)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        
        # Get positional embeddings
        pos_embeddings = self.position_embedding(positions)
        
        return pos_embeddings
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
        })
        return config


def create_hybrid_embedder(vocab_size: int, embedding_dim: int, max_length: int,
                          use_learnable: bool = False) -> keras.layers.Layer:
    """
    Create a hybrid embedder with both token and positional embeddings.
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        max_length: Maximum sequence length
        use_learnable: Whether to use learnable vs. sinusoidal positional embeddings
        
    Returns:
        Combined embedding layer
    """
    if use_learnable:
        return HybridEmbedder(vocab_size, embedding_dim, max_length, learnable_pos=True)
    else:
        return SinusoidalEmbedder(vocab_size, embedding_dim, max_length)


class HybridEmbedder(keras.layers.Layer):
    """Hybrid embedder combining token embeddings with optional learnable positions."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int,
                 learnable_pos: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.learnable_pos = learnable_pos
        
        # Token embeddings
        self.token_embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        
        # Positional embeddings
        if learnable_pos:
            self.pos_embedding = LearnablePositionalEmbedding(max_length, embedding_dim)
        else:
            self.pos_embedding = SinusoidalEmbedder(vocab_size, embedding_dim, max_length)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        token_embeds = self.token_embedding(inputs)
        
        if self.learnable_pos:
            pos_embeds = self.pos_embedding(token_embeds)
            return token_embeds + pos_embeds
        else:
            # For sinusoidal, pass the token IDs directly
            return self.pos_embedding(inputs)
