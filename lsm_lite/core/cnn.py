"""
2D/3D CNN processors for spatial-temporal processing.

This module provides CNN architectures for processing reservoir outputs,
supporting both 2D spatial and 3D spatial-temporal convolutions.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class CNNProcessor(keras.Model):
    """2D/3D CNN for processing reservoir outputs."""
    
    def __init__(self, input_shape: Tuple[int, ...], architecture: str = '2d',
                 filters: List[int] = None, vocab_size: int = 50257,
                 dropout_rate: float = 0.1, name: str = "cnn_processor", **kwargs):
        """
        Initialize CNN processor.
        
        Args:
            input_shape: Shape of input data (excluding batch dimension)
            architecture: CNN architecture ('2d' or '3d')
            filters: List of filter sizes for conv layers
            vocab_size: Vocabulary size for final output layer
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.input_shape_val = input_shape
        self.architecture = architecture.lower()
        self.filters = filters or [64, 128, 256]
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        
        # Build the appropriate model
        self._build_model()
        
        logger.info("CNN processor initialized: architecture=%s, filters=%s, vocab_size=%d",
                   self.architecture, self.filters, self.vocab_size)
    
    def _build_model(self):
        """Build CNN model based on architecture."""
        if self.architecture == '2d':
            self._build_2d_cnn()
        elif self.architecture == '3d':
            self._build_3d_cnn()
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def _build_2d_cnn(self):
        """Build 2D CNN for spatial processing."""
        # Input layer
        # Reshape input to add channel dimension if needed
        if len(self.input_shape_val) == 2:
            # Add channel dimension for 2D input
            self.reshape_layer = layers.Reshape(self.input_shape_val + (1,))
        else:
            self.reshape_layer = None
        
        # Convolutional layers
        self.conv_layers = []
        self.pool_layers = []
        self.dropout_layers = []
        
        for i, num_filters in enumerate(self.filters):
            # Convolutional layer
            conv = layers.Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                name=f'conv2d_{i}'
            )
            self.conv_layers.append(conv)
            
            # Max pooling layer
            pool = layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'maxpool2d_{i}'
            )
            self.pool_layers.append(pool)
            
            # Dropout layer
            dropout = layers.Dropout(
                rate=self.dropout_rate,
                name=f'dropout_{i}'
            )
            self.dropout_layers.append(dropout)
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')
        
        # Dense layers
        self.dense1 = layers.Dense(512, activation='relu', name='dense_1')
        self.dense1_dropout = layers.Dropout(self.dropout_rate, name='dense_1_dropout')
        
        # Output layer
        self.output_layer = layers.Dense(
            self.vocab_size,
            activation='softmax',
            name='output'
        )
    
    def _build_3d_cnn(self):
        """Build 3D CNN for spatial-temporal processing."""
        # For 3D CNN, we need to ensure proper input shape
        # Reshape input to add time and channel dimensions if needed
        if len(self.input_shape_val) == 2:
            # Treat sequence dimension as time, spatial dim as space
            seq_len, feature_dim = self.input_shape_val
            # Reshape to (time_steps, height, width, channels)
            # We'll treat this as a 1D spatial dimension replicated
            spatial_size = int(feature_dim ** 0.5)
            if spatial_size * spatial_size != feature_dim:
                spatial_size = feature_dim  # Keep as 1D if not perfect square
                self.reshape_layer = layers.Reshape((seq_len, spatial_size, 1, 1))
            else:
                self.reshape_layer = layers.Reshape((seq_len, spatial_size, spatial_size, 1))
        else:
            self.reshape_layer = None
        
        # 3D Convolutional layers
        self.conv3d_layers = []
        self.pool3d_layers = []
        self.dropout_layers = []
        
        for i, num_filters in enumerate(self.filters):
            # 3D Convolutional layer
            conv3d = layers.Conv3D(
                filters=num_filters,
                kernel_size=(3, 3, 3),
                activation='relu',
                padding='same',
                name=f'conv3d_{i}'
            )
            self.conv3d_layers.append(conv3d)
            
            # 3D Max pooling layer
            pool3d = layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                name=f'maxpool3d_{i}'
            )
            self.pool3d_layers.append(pool3d)
            
            # Dropout layer
            dropout = layers.Dropout(
                rate=self.dropout_rate,
                name=f'dropout3d_{i}'
            )
            self.dropout_layers.append(dropout)
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling3D(name='global_avg_pool_3d')
        
        # Dense layers
        self.dense1 = layers.Dense(512, activation='relu', name='dense_1')
        self.dense1_dropout = layers.Dropout(self.dropout_rate, name='dense_1_dropout')
        
        # Output layer
        self.output_layer = layers.Dense(
            self.vocab_size,
            activation='softmax',
            name='output'
        )
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through CNN processor.
        
        Args:
            inputs: Input tensor from reservoir
            training: Whether in training mode
            
        Returns:
            Output tensor with vocabulary probabilities
        """
        x = inputs
        
        # Reshape if necessary
        if self.reshape_layer is not None:
            x = self.reshape_layer(x)
        
        # Apply convolutional layers
        if self.architecture == '2d':
            for conv, pool, dropout in zip(self.conv_layers, self.pool_layers, self.dropout_layers):
                x = conv(x)
                x = pool(x)
                x = dropout(x, training=training)
        
        elif self.architecture == '3d':
            for conv3d, pool3d, dropout in zip(self.conv3d_layers, self.pool3d_layers, self.dropout_layers):
                x = conv3d(x)
                x = pool3d(x)
                x = dropout(x, training=training)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Dense layers
        x = self.dense1(x)
        x = self.dense1_dropout(x, training=training)
        
        # Output layer
        outputs = self.output_layer(x)
        
        return outputs
    
    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'input_shape': self.input_shape_val,
            'architecture': self.architecture,
            'filters': self.filters,
            'vocab_size': self.vocab_size,
            'dropout_rate': self.dropout_rate,
        }
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)


class CNNVisualization:
    """Utility class for visualizing CNN feature maps and filters."""
    
    def __init__(self, cnn_model: CNNProcessor):
        """
        Initialize visualization helper.
        
        Args:
            cnn_model: Trained CNN processor model
        """
        self.model = cnn_model
    
    def get_feature_maps(self, inputs: tf.Tensor, 
                        layer_names: Optional[List[str]] = None) -> dict:
        """
        Extract feature maps from specified layers.
        
        Args:
            inputs: Input tensor to process
            layer_names: List of layer names to extract features from
            
        Returns:
            Dictionary mapping layer names to feature maps
        """
        if layer_names is None:
            # Default to all convolutional layers
            if self.model.architecture == '2d':
                layer_names = [f'conv2d_{i}' for i in range(len(self.model.filters))]
            else:
                layer_names = [f'conv3d_{i}' for i in range(len(self.model.filters))]
        
        # Create models to extract intermediate outputs
        feature_maps = {}
        
        for layer_name in layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                extractor = keras.Model(inputs=self.model.input, outputs=layer.output)
                feature_map = extractor(inputs)
                feature_maps[layer_name] = feature_map
            except ValueError:
                logger.warning("Layer '%s' not found in model", layer_name)
        
        return feature_maps
    
    def analyze_filters(self) -> dict:
        """
        Analyze the learned filters in convolutional layers.
        
        Returns:
            Dictionary with filter analysis results
        """
        analysis = {}
        
        # Get filter weights from each convolutional layer
        if self.model.architecture == '2d':
            conv_layers = self.model.conv_layers
        else:
            conv_layers = self.model.conv3d_layers
        
        for i, layer in enumerate(conv_layers):
            weights = layer.get_weights()[0]  # Filter weights
            biases = layer.get_weights()[1]   # Biases
            
            analysis[f'layer_{i}'] = {
                'filter_shape': weights.shape,
                'num_filters': weights.shape[-1],
                'weight_mean': float(tf.reduce_mean(weights)),
                'weight_std': float(tf.math.reduce_std(weights)),
                'bias_mean': float(tf.reduce_mean(biases)),
                'bias_std': float(tf.math.reduce_std(biases)),
            }
        
        return analysis


def create_adaptive_cnn(reservoir_output_shape: Tuple[int, ...], 
                       vocab_size: int,
                       complexity: str = 'medium') -> CNNProcessor:
    """
    Create an adaptive CNN architecture based on input characteristics.
    
    Args:
        reservoir_output_shape: Shape of reservoir output
        vocab_size: Vocabulary size for output
        complexity: Complexity level ('simple', 'medium', 'complex')
        
    Returns:
        Configured CNN processor
    """
    # Choose architecture based on input dimensionality
    if len(reservoir_output_shape) == 2:
        # 2D input - use 2D CNN
        architecture = '2d'
    else:
        # Higher dimensional input - use 3D CNN
        architecture = '3d'
    
    # Choose filters based on complexity
    filter_configs = {
        'simple': [32, 64],
        'medium': [64, 128, 256],
        'complex': [64, 128, 256, 512]
    }
    
    filters = filter_configs.get(complexity, filter_configs['medium'])
    
    return CNNProcessor(
        input_shape=reservoir_output_shape,
        architecture=architecture,
        filters=filters,
        vocab_size=vocab_size
    )
