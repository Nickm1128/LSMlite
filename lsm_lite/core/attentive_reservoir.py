"""
Attentive reservoir implementation extending SparseReservoir with attention mechanisms.

This module implements an attention-enhanced reservoir that adds multi-head attention
to the standard sparse reservoir processing for improved context awareness.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import logging

from .reservoir import SparseReservoir

logger = logging.getLogger(__name__)


class AttentiveReservoir(SparseReservoir):
    """Sparse reservoir with multi-head attention mechanism."""
    
    def __init__(self, input_dim: int, reservoir_size: int = 512,
                 attention_heads: int = 8, attention_dim: int = 64,
                 sparsity: float = 0.1, spectral_radius: float = 0.9,
                 sine_amplitude: float = 1.0, sine_frequency: float = 1.0,
                 sine_decay: float = 0.1, leak_rate: float = 0.3,
                 name: str = "attentive_reservoir", **kwargs):
        """
        Initialize attentive reservoir with multi-head attention.
        
        Args:
            input_dim: Dimension of input features
            reservoir_size: Number of reservoir neurons
            attention_heads: Number of attention heads
            attention_dim: Dimension of each attention head
            sparsity: Connection sparsity (fraction of connections to keep)
            spectral_radius: Spectral radius for stability
            sine_amplitude: Amplitude parameter for sine activation
            sine_frequency: Frequency parameter for sine activation  
            sine_decay: Decay parameter for sine activation
            leak_rate: Leaking rate for reservoir dynamics
            name: Layer name
        """
        # Initialize parent class
        super().__init__(
            input_dim=input_dim,
            reservoir_size=reservoir_size,
            sparsity=sparsity,
            spectral_radius=spectral_radius,
            sine_amplitude=sine_amplitude,
            sine_frequency=sine_frequency,
            sine_decay=sine_decay,
            leak_rate=leak_rate,
            name=name,
            **kwargs
        )
        
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.total_attention_dim = attention_heads * attention_dim
        
        # Build attention components
        self._build_attention_layers()
        
        logger.info("Attentive reservoir initialized: size=%d, attention_heads=%d, attention_dim=%d",
                   self.reservoir_size, self.attention_heads, self.attention_dim)
    
    def _build_attention_layers(self):
        """Build multi-head attention layers."""
        # Query, Key, Value projection layers
        self.query_projection = self.add_weight(
            name="query_projection",
            shape=(self.reservoir_size, self.total_attention_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        self.key_projection = self.add_weight(
            name="key_projection", 
            shape=(self.reservoir_size, self.total_attention_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        self.value_projection = self.add_weight(
            name="value_projection",
            shape=(self.reservoir_size, self.total_attention_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        # Output projection layer
        self.output_projection = self.add_weight(
            name="output_projection",
            shape=(self.total_attention_dim, self.reservoir_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        # Layer normalization for attention output
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="attention_layer_norm"
        )
        
        # Storage for attention weights (for analysis/visualization)
        self.last_attention_weights = None
    
    def _compute_multi_head_attention(self, reservoir_states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute multi-head attention over reservoir states.
        
        Args:
            reservoir_states: Tensor of shape (batch_size, sequence_length, reservoir_size)
            
        Returns:
            Tuple of (attended_states, attention_weights)
        """
        batch_size, sequence_length, _ = tf.shape(reservoir_states)[0], tf.shape(reservoir_states)[1], tf.shape(reservoir_states)[2]
        
        # Project to queries, keys, values
        queries = tf.matmul(reservoir_states, self.query_projection)  # (batch, seq, total_att_dim)
        keys = tf.matmul(reservoir_states, self.key_projection)       # (batch, seq, total_att_dim)
        values = tf.matmul(reservoir_states, self.value_projection)   # (batch, seq, total_att_dim)
        
        # Reshape for multi-head attention
        # Shape: (batch, seq, heads, att_dim) -> (batch, heads, seq, att_dim)
        queries = self._reshape_for_attention(queries, batch_size, sequence_length)
        keys = self._reshape_for_attention(keys, batch_size, sequence_length)
        values = self._reshape_for_attention(values, batch_size, sequence_length)
        
        # Compute attention scores
        # Shape: (batch, heads, seq, seq)
        attention_scores = tf.matmul(queries, keys, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.attention_dim, tf.float32))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Store attention weights for analysis
        self.last_attention_weights = attention_weights
        
        # Apply attention to values
        # Shape: (batch, heads, seq, att_dim)
        attended_values = tf.matmul(attention_weights, values)
        
        # Reshape back to (batch, seq, total_att_dim)
        attended_values = self._reshape_from_attention(attended_values, batch_size, sequence_length)
        
        # Apply output projection
        attended_output = tf.matmul(attended_values, self.output_projection)
        
        return attended_output, attention_weights
    
    def _reshape_for_attention(self, tensor: tf.Tensor, batch_size: tf.Tensor, sequence_length: tf.Tensor) -> tf.Tensor:
        """Reshape tensor for multi-head attention computation."""
        # (batch, seq, total_att_dim) -> (batch, seq, heads, att_dim) -> (batch, heads, seq, att_dim)
        tensor = tf.reshape(tensor, [batch_size, sequence_length, self.attention_heads, self.attention_dim])
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        return tensor
    
    def _reshape_from_attention(self, tensor: tf.Tensor, batch_size: tf.Tensor, sequence_length: tf.Tensor) -> tf.Tensor:
        """Reshape tensor back from multi-head attention computation."""
        # (batch, heads, seq, att_dim) -> (batch, seq, heads, att_dim) -> (batch, seq, total_att_dim)
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        tensor = tf.reshape(tensor, [batch_size, sequence_length, self.total_attention_dim])
        return tensor
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Process inputs through the attentive reservoir.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, input_dim)
            training: Whether in training mode
            
        Returns:
            Tuple of (reservoir_states, attention_weights)
            - reservoir_states: Enhanced states of shape (batch_size, sequence_length, reservoir_size)
            - attention_weights: Attention weights of shape (batch_size, heads, seq_len, seq_len)
        """
        # First, get standard reservoir states from parent class
        reservoir_states = super().call(inputs, training=training)
        
        # Apply multi-head attention to reservoir states
        attended_states, attention_weights = self._compute_multi_head_attention(reservoir_states)
        
        # Residual connection and layer normalization
        enhanced_states = self.attention_layer_norm(reservoir_states + attended_states)
        
        return enhanced_states, attention_weights
    
    def get_attention_weights(self) -> Optional[tf.Tensor]:
        """
        Get the last computed attention weights.
        
        Returns:
            Attention weights tensor of shape (batch_size, heads, seq_len, seq_len) or None
        """
        return self.last_attention_weights
    
    def compute_attention_entropy(self) -> Optional[tf.Tensor]:
        """
        Compute entropy of attention weights as a measure of attention diversity.
        
        Returns:
            Attention entropy tensor or None if no attention weights available
        """
        if self.last_attention_weights is None:
            return None
        
        # Compute entropy across the last dimension (attended positions)
        # Shape: (batch, heads, seq, seq) -> (batch, heads, seq)
        attention_entropy = -tf.reduce_sum(
            self.last_attention_weights * tf.math.log(self.last_attention_weights + 1e-10),
            axis=-1
        )
        
        # Average across sequence and heads
        # Shape: (batch,)
        mean_entropy = tf.reduce_mean(attention_entropy, axis=[1, 2])
        
        return mean_entropy
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'attention_heads': self.attention_heads,
            'attention_dim': self.attention_dim,
        })
        return config
    
    def compute_output_shape(self, input_shape):
        """Compute output shape for the layer."""
        batch_size, sequence_length, _ = input_shape
        # Returns tuple of (reservoir_states_shape, attention_weights_shape)
        reservoir_shape = (batch_size, sequence_length, self.reservoir_size)
        attention_shape = (batch_size, self.attention_heads, sequence_length, sequence_length)
        return reservoir_shape, attention_shape


class AttentiveReservoirAnalyzer:
    """Utility class for analyzing attentive reservoir properties."""
    
    @staticmethod
    def analyze_attention_patterns(reservoir: AttentiveReservoir, 
                                 test_inputs: tf.Tensor) -> dict:
        """
        Analyze attention patterns in the reservoir.
        
        Args:
            reservoir: The attentive reservoir to analyze
            test_inputs: Test input tensor
            
        Returns:
            Dictionary with attention analysis results
        """
        # Process inputs to get attention weights
        _, attention_weights = reservoir(test_inputs)
        
        if attention_weights is None:
            return {'error': 'No attention weights available'}
        
        # Compute attention statistics
        attention_entropy = reservoir.compute_attention_entropy()
        
        # Compute attention concentration (how focused the attention is)
        attention_max = tf.reduce_max(attention_weights, axis=-1)  # Max attention per position
        attention_concentration = tf.reduce_mean(attention_max)
        
        # Compute attention diversity (how spread out attention is)
        attention_std = tf.math.reduce_std(attention_weights, axis=-1)
        attention_diversity = tf.reduce_mean(attention_std)
        
        return {
            'attention_entropy': float(tf.reduce_mean(attention_entropy)) if attention_entropy is not None else None,
            'attention_concentration': float(attention_concentration),
            'attention_diversity': float(attention_diversity),
            'attention_heads': reservoir.attention_heads,
            'attention_dim': reservoir.attention_dim,
            'attention_weights_shape': attention_weights.shape.as_list(),
        }
    
    @staticmethod
    def visualize_attention_heads(reservoir: AttentiveReservoir,
                                test_inputs: tf.Tensor,
                                head_indices: Optional[list] = None) -> dict:
        """
        Extract attention patterns for visualization.
        
        Args:
            reservoir: The attentive reservoir
            test_inputs: Test input tensor
            head_indices: List of head indices to visualize (default: all heads)
            
        Returns:
            Dictionary with attention patterns for visualization
        """
        # Process inputs
        _, attention_weights = reservoir(test_inputs)
        
        if attention_weights is None:
            return {'error': 'No attention weights available'}
        
        # Select heads to visualize
        if head_indices is None:
            head_indices = list(range(reservoir.attention_heads))
        
        # Extract attention patterns for selected heads
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        visualization_data = {}
        for head_idx in head_indices:
            if head_idx < num_heads:
                # Get attention weights for this head (first batch item)
                head_attention = attention_weights[0, head_idx, :, :].numpy()
                visualization_data[f'head_{head_idx}'] = head_attention
        
        return {
            'attention_patterns': visualization_data,
            'sequence_length': seq_len,
            'selected_heads': head_indices,
        }