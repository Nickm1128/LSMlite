"""
Unit tests for AttentiveReservoir class.

Tests cover multi-head attention mechanisms, attention weight computation,
integration with base reservoir functionality, and analysis utilities.
"""

import unittest
import numpy as np
import tensorflow as tf

# Import the classes to test
from lsm_lite.core.attentive_reservoir import AttentiveReservoir, AttentiveReservoirAnalyzer
from lsm_lite.core.reservoir import SparseReservoir


class TestAttentiveReservoir(unittest.TestCase):
    """Test cases for AttentiveReservoir class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test configuration."""
        # Configure TensorFlow for testing
        tf.config.experimental.enable_op_determinism()
        tf.random.set_seed(42)
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 64
        self.reservoir_size = 128
        self.attention_heads = 4
        self.attention_dim = 16
        self.batch_size = 2
        self.sequence_length = 10
        
        # Create test reservoir
        self.reservoir = AttentiveReservoir(
            input_dim=self.input_dim,
            reservoir_size=self.reservoir_size,
            attention_heads=self.attention_heads,
            attention_dim=self.attention_dim,
            sparsity=0.1,
            spectral_radius=0.9
        )
        
        # Create test input
        self.test_input = tf.random.normal((self.batch_size, self.sequence_length, self.input_dim))
    
    def test_initialization(self):
        """Test proper initialization of AttentiveReservoir."""
        # Check that all required attributes are set
        self.assertEqual(self.reservoir.input_dim, self.input_dim)
        self.assertEqual(self.reservoir.reservoir_size, self.reservoir_size)
        self.assertEqual(self.reservoir.attention_heads, self.attention_heads)
        self.assertEqual(self.reservoir.attention_dim, self.attention_dim)
        self.assertEqual(self.reservoir.total_attention_dim, self.attention_heads * self.attention_dim)
        
        # Check that attention layers are built
        self.assertIsNotNone(self.reservoir.query_projection)
        self.assertIsNotNone(self.reservoir.key_projection)
        self.assertIsNotNone(self.reservoir.value_projection)
        self.assertIsNotNone(self.reservoir.output_projection)
        self.assertIsNotNone(self.reservoir.attention_layer_norm)
        
        # Check weight shapes
        expected_qkv_shape = (self.reservoir_size, self.attention_heads * self.attention_dim)
        expected_output_shape = (self.attention_heads * self.attention_dim, self.reservoir_size)
        
        self.assertEqual(self.reservoir.query_projection.shape, expected_qkv_shape)
        self.assertEqual(self.reservoir.key_projection.shape, expected_qkv_shape)
        self.assertEqual(self.reservoir.value_projection.shape, expected_qkv_shape)
        self.assertEqual(self.reservoir.output_projection.shape, expected_output_shape)
    
    def test_inheritance_from_sparse_reservoir(self):
        """Test that AttentiveReservoir properly inherits from SparseReservoir."""
        # Check inheritance
        self.assertIsInstance(self.reservoir, SparseReservoir)
        
        # Check that parent class attributes are accessible
        self.assertTrue(hasattr(self.reservoir, 'W_in'))
        self.assertTrue(hasattr(self.reservoir, 'W_res'))
        self.assertTrue(hasattr(self.reservoir, 'reservoir_state'))
        self.assertTrue(hasattr(self.reservoir, 'sine_activation'))
    
    def test_call_method_returns_tuple(self):
        """Test that call method returns tuple of (states, attention_weights)."""
        result = self.reservoir(self.test_input)
        
        # Should return tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        states, attention_weights = result
        
        # Check output shapes
        expected_states_shape = (self.batch_size, self.sequence_length, self.reservoir_size)
        expected_attention_shape = (self.batch_size, self.attention_heads, self.sequence_length, self.sequence_length)
        
        self.assertEqual(states.shape, expected_states_shape)
        self.assertEqual(attention_weights.shape, expected_attention_shape)
    
    def test_multi_head_attention_computation(self):
        """Test multi-head attention mechanism computation."""
        # Process input to get attention weights
        states, attention_weights = self.reservoir(self.test_input)
        
        # Check attention weights properties
        self.assertIsInstance(attention_weights, tf.Tensor)
        
        # Attention weights should sum to 1 across last dimension (softmax property)
        attention_sums = tf.reduce_sum(attention_weights, axis=-1)
        expected_sums = tf.ones_like(attention_sums)
        tf.debugging.assert_near(attention_sums, expected_sums, atol=1e-6)
        
        # Attention weights should be non-negative
        self.assertTrue(tf.reduce_all(attention_weights >= 0))
    
    def test_attention_weight_storage(self):
        """Test that attention weights are properly stored for analysis."""
        # Initially no attention weights
        self.assertIsNone(self.reservoir.last_attention_weights)
        
        # Process input
        _, attention_weights = self.reservoir(self.test_input)
        
        # Check that weights are stored
        stored_weights = self.reservoir.get_attention_weights()
        self.assertIsNotNone(stored_weights)
        tf.debugging.assert_equal(stored_weights, attention_weights)
    
    def test_attention_entropy_computation(self):
        """Test attention entropy computation."""
        # Process input to generate attention weights
        self.reservoir(self.test_input)
        
        # Compute attention entropy
        entropy = self.reservoir.compute_attention_entropy()
        
        # Check entropy properties
        self.assertIsNotNone(entropy)
        self.assertEqual(entropy.shape, (self.batch_size,))
        
        # Entropy should be non-negative
        self.assertTrue(tf.reduce_all(entropy >= 0))
    
    def test_get_config(self):
        """Test configuration serialization."""
        config = self.reservoir.get_config()
        
        # Check that attention-specific config is included
        self.assertIn('attention_heads', config)
        self.assertIn('attention_dim', config)
        self.assertEqual(config['attention_heads'], self.attention_heads)
        self.assertEqual(config['attention_dim'], self.attention_dim)
        
        # Check that parent config is also included
        self.assertIn('input_dim', config)
        self.assertIn('reservoir_size', config)
        self.assertIn('sparsity', config)


class TestAttentiveReservoirAnalyzer(unittest.TestCase):
    """Test cases for AttentiveReservoirAnalyzer utility class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test configuration."""
        # Configure TensorFlow for testing
        tf.config.experimental.enable_op_determinism()
        tf.random.set_seed(42)
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 32
        self.reservoir_size = 64
        self.attention_heads = 4
        self.attention_dim = 16
        self.batch_size = 2
        self.sequence_length = 8
        
        self.reservoir = AttentiveReservoir(
            input_dim=self.input_dim,
            reservoir_size=self.reservoir_size,
            attention_heads=self.attention_heads,
            attention_dim=self.attention_dim
        )
        
        self.test_input = tf.random.normal((self.batch_size, self.sequence_length, self.input_dim))
    
    def test_analyze_attention_patterns(self):
        """Test attention pattern analysis."""
        analysis = AttentiveReservoirAnalyzer.analyze_attention_patterns(
            self.reservoir, self.test_input
        )
        
        # Check that analysis contains expected keys
        expected_keys = [
            'attention_entropy', 'attention_concentration', 'attention_diversity',
            'attention_heads', 'attention_dim', 'attention_weights_shape'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Check value types and ranges
        self.assertIsInstance(analysis['attention_entropy'], (float, type(None)))
        self.assertIsInstance(analysis['attention_concentration'], float)
        self.assertIsInstance(analysis['attention_diversity'], float)
        self.assertEqual(analysis['attention_heads'], self.attention_heads)
        self.assertEqual(analysis['attention_dim'], self.attention_dim)
        
        # Concentration and diversity should be non-negative
        self.assertGreaterEqual(analysis['attention_concentration'], 0)
        self.assertGreaterEqual(analysis['attention_diversity'], 0)
    
    def test_visualize_attention_heads(self):
        """Test attention head visualization."""
        # Test with default head selection (all heads)
        viz_data = AttentiveReservoirAnalyzer.visualize_attention_heads(
            self.reservoir, self.test_input
        )
        
        # Check structure
        self.assertIn('attention_patterns', viz_data)
        self.assertIn('sequence_length', viz_data)
        self.assertIn('selected_heads', viz_data)
        
        # Check that all heads are included by default
        patterns = viz_data['attention_patterns']
        self.assertEqual(len(patterns), self.attention_heads)
        
        for head_idx in range(self.attention_heads):
            head_key = f'head_{head_idx}'
            self.assertIn(head_key, patterns)
            
            # Check attention pattern shape
            pattern = patterns[head_key]
            self.assertEqual(pattern.shape, (self.sequence_length, self.sequence_length))


if __name__ == '__main__':
    unittest.main(verbosity=2)