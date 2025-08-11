"""
Sparse sine-activated liquid state machine reservoir.

This module implements a sparse reservoir computing system with parameterized
sine activation functions for processing sequential data.
"""

import numpy as np
import tensorflow as tf
from scipy import sparse
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SparseReservoir(tf.keras.layers.Layer):
    """Sparse sine-activated liquid state machine."""
    
    def __init__(self, input_dim: int, reservoir_size: int = 512, 
                 sparsity: float = 0.1, spectral_radius: float = 0.9,
                 sine_amplitude: float = 1.0, sine_frequency: float = 1.0,
                 sine_decay: float = 0.1, leak_rate: float = 0.3,
                 name: str = "sparse_reservoir", **kwargs):
        """
        Initialize sparse sine-activated reservoir.
        
        Args:
            input_dim: Dimension of input features
            reservoir_size: Number of reservoir neurons
            sparsity: Connection sparsity (fraction of connections to keep)
            spectral_radius: Spectral radius for stability
            sine_amplitude: Amplitude parameter for sine activation
            sine_frequency: Frequency parameter for sine activation  
            sine_decay: Decay parameter for sine activation
            leak_rate: Leaking rate for reservoir dynamics
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.sine_amplitude = sine_amplitude
        self.sine_frequency = sine_frequency
        self.sine_decay = sine_decay
        self.leak_rate = leak_rate
        
        # Build reservoir matrices
        self._build_reservoir()
        
        logger.info("Sparse reservoir initialized: size=%d, sparsity=%.3f, spectral_radius=%.3f",
                   self.reservoir_size, self.sparsity, self.spectral_radius)
    
    def _build_reservoir(self):
        """Build sparse reservoir weight matrices."""
        # Input-to-reservoir weights (dense)
        self.W_in = self.add_weight(
            name="input_weights",
            shape=(self.input_dim, self.reservoir_size),
            initializer=tf.keras.initializers.RandomUniform(-0.5, 0.5),
            trainable=False
        )
        
        # Reservoir-to-reservoir weights (sparse)
        self._build_sparse_reservoir_weights()
        
        # Reservoir state
        self.reservoir_state = tf.Variable(
            tf.zeros((1, self.reservoir_size)),
            trainable=False,
            name="reservoir_state"
        )
    
    def _build_sparse_reservoir_weights(self):
        """Create sparse reservoir-to-reservoir weight matrix."""
        # Generate random sparse matrix
        reservoir_weights = self._create_sparse_matrix(
            self.reservoir_size, self.reservoir_size, self.sparsity
        )
        
        # Scale to desired spectral radius
        reservoir_weights = self._scale_spectral_radius(reservoir_weights)
        
        # Convert to TensorFlow SparseTensor
        indices = np.array(np.nonzero(reservoir_weights)).T
        values = reservoir_weights.data
        dense_shape = reservoir_weights.shape
        
        self.W_res_sparse = tf.SparseTensor(
            indices=indices.astype(np.int64),
            values=values.astype(np.float32),
            dense_shape=dense_shape
        )
        
        # Also keep dense version for gradient computation
        self.W_res = self.add_weight(
            name="reservoir_weights",
            shape=(self.reservoir_size, self.reservoir_size),
            initializer='zeros',
            trainable=False
        )
        self.W_res.assign(tf.sparse.to_dense(self.W_res_sparse))
    
    def _create_sparse_matrix(self, rows: int, cols: int, sparsity: float) -> sparse.csr_matrix:
        """Create random sparse matrix with given sparsity."""
        # Calculate number of non-zero entries
        nnz = int(rows * cols * sparsity)
        
        # Generate random indices
        row_indices = np.random.randint(0, rows, size=nnz)
        col_indices = np.random.randint(0, cols, size=nnz)
        
        # Generate random values
        values = np.random.normal(0, 1, size=nnz)
        
        # Create sparse matrix
        matrix = sparse.csr_matrix((values, (row_indices, col_indices)), shape=(rows, cols))
        
        return matrix
    
    def _scale_spectral_radius(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """Scale matrix to have desired spectral radius."""
        try:
            # Compute largest eigenvalue magnitude
            eigenvalues = sparse.linalg.eigs(matrix, k=1, which='LM', return_eigenvectors=False, maxiter=1000)
            current_radius = np.abs(eigenvalues[0]).real
        except sparse.linalg.ArpackNoConvergence:
            # If eigenvalue computation fails, use power iteration as fallback
            logger.warning("Eigenvalue computation failed, using power iteration fallback")
            current_radius = self._power_iteration_spectral_radius(matrix)
        
        # Scale to desired spectral radius
        if current_radius > 1e-10:  # Avoid division by zero
            scaling_factor = self.spectral_radius / current_radius
            matrix = matrix * scaling_factor
        
        return matrix
    
    def _power_iteration_spectral_radius(self, matrix: sparse.csr_matrix, max_iter: int = 100) -> float:
        """Compute spectral radius using power iteration method."""
        n = matrix.shape[0]
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        
        for _ in range(max_iter):
            x_new = matrix.dot(x)
            eigenvalue = np.dot(x, x_new)
            x_new_norm = np.linalg.norm(x_new)
            
            if x_new_norm < 1e-10:
                break
                
            x = x_new / x_new_norm
        
        return abs(eigenvalue)
    
    def sine_activation(self, x: tf.Tensor) -> tf.Tensor:
        """
        Parameterized sine activation function.
        
        Formula: A * exp(-α * |x|) * sin(ω * x)
        where A = amplitude, α = decay, ω = frequency
        """
        abs_x = tf.abs(x)
        exp_decay = tf.exp(-self.sine_decay * abs_x)
        sine_term = tf.sin(self.sine_frequency * x)
        
        return self.sine_amplitude * exp_decay * sine_term
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Process inputs through the reservoir.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, input_dim)
            training: Whether in training mode
            
        Returns:
            Reservoir states of shape (batch_size, sequence_length, reservoir_size)
        """
        batch_size, sequence_length, _ = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Initialize reservoir states for the batch
        states = []
        current_state = tf.tile(self.reservoir_state, [batch_size, 1])
        
        # Use tf.while_loop for dynamic sequence processing
        def reservoir_step(t, states_ta, current_state):
            # Get input at time t
            input_t = inputs[:, t, :]  # Shape: (batch_size, input_dim)
            
            # Compute input contribution
            input_contrib = tf.matmul(input_t, self.W_in)
            
            # Compute reservoir contribution
            reservoir_contrib = tf.matmul(current_state, self.W_res)
            
            # Combine contributions
            pre_activation = input_contrib + reservoir_contrib
            
            # Apply sine activation
            activated = self.sine_activation(pre_activation)
            
            # Apply leaking (reservoir dynamics)
            new_state = (1 - self.leak_rate) * current_state + self.leak_rate * activated
            
            # Store state
            states_ta = states_ta.write(t, new_state)
            
            return t + 1, states_ta, new_state
        
        # Initialize TensorArray to store states
        states_ta = tf.TensorArray(
            dtype=tf.float32,
            size=sequence_length,
            dynamic_size=False
        )
        
        # Run the reservoir dynamics
        _, final_states_ta, _ = tf.while_loop(
            cond=lambda t, _, __: t < sequence_length,
            body=reservoir_step,
            loop_vars=(0, states_ta, current_state)
        )
        
        # Stack all states into output tensor
        reservoir_output = final_states_ta.stack()  # Shape: (seq_len, batch_size, reservoir_size)
        reservoir_output = tf.transpose(reservoir_output, [1, 0, 2])  # Shape: (batch_size, seq_len, reservoir_size)
        
        return reservoir_output
    
    def reset_state(self, batch_size: int = 1):
        """Reset reservoir state to zeros."""
        self.reservoir_state.assign(tf.zeros((1, self.reservoir_size)))
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'reservoir_size': self.reservoir_size,
            'sparsity': self.sparsity,
            'spectral_radius': self.spectral_radius,
            'sine_amplitude': self.sine_amplitude,
            'sine_frequency': self.sine_frequency,
            'sine_decay': self.sine_decay,
            'leak_rate': self.leak_rate,
        })
        return config
    
    def compute_output_shape(self, input_shape):
        """Compute output shape for the layer."""
        batch_size, sequence_length, _ = input_shape
        return (batch_size, sequence_length, self.reservoir_size)


class ReservoirAnalyzer:
    """Utility class for analyzing reservoir properties."""
    
    @staticmethod
    def compute_memory_capacity(reservoir: SparseReservoir, 
                              test_length: int = 1000) -> float:
        """
        Compute linear memory capacity of the reservoir.
        
        Args:
            reservoir: The reservoir to analyze
            test_length: Length of test sequence
            
        Returns:
            Memory capacity score
        """
        # Generate random input sequence
        test_input = tf.random.normal((1, test_length, reservoir.input_dim))
        
        # Get reservoir states
        reservoir_states = reservoir(test_input)
        
        # Compute linear memory capacity (simplified)
        # This is a basic implementation - full analysis would require more sophisticated metrics
        correlation_sum = 0.0
        
        for delay in range(1, min(50, test_length)):
            if test_length - delay > 0:
                # Compute correlation between input and reservoir state with delay
                input_delayed = test_input[:, :-delay, :]
                state_current = reservoir_states[:, delay:, :]
                
                # Simplified correlation computation
                corr = tf.reduce_mean(tf.abs(tf.reduce_mean(input_delayed * state_current, axis=-1)))
                correlation_sum += corr.numpy()
        
        return correlation_sum
    
    @staticmethod
    def analyze_dynamics(reservoir: SparseReservoir) -> dict:
        """
        Analyze reservoir dynamics and properties.
        
        Returns:
            Dictionary with analysis results
        """
        # Get actual spectral radius of weight matrix
        W_dense = reservoir.W_res.numpy()
        eigenvalues = np.linalg.eigvals(W_dense)
        actual_spectral_radius = np.max(np.abs(eigenvalues))
        
        # Compute sparsity
        actual_sparsity = np.sum(np.abs(W_dense) > 1e-10) / W_dense.size
        
        return {
            'actual_spectral_radius': float(actual_spectral_radius),
            'target_spectral_radius': reservoir.spectral_radius,
            'actual_sparsity': float(actual_sparsity),
            'target_sparsity': reservoir.sparsity,
            'reservoir_size': reservoir.reservoir_size,
            'input_dim': reservoir.input_dim,
        }
