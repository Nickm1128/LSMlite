"""
Configuration management for LSM Lite.

This module provides configuration dataclasses and utilities for managing
LSM model parameters and training settings.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class LSMConfig:
    """Simple configuration for LSM models."""
    
    # Tokenizer settings
    tokenizer_backend: str = 'gpt2'
    max_length: int = 128
    vocab_size: Optional[int] = None  # Auto-detected if None
    
    # Embedding settings
    embedding_dim: int = 256
    max_samples: int = 10000
    
    # Reservoir settings
    reservoir_size: int = 512
    sparsity: float = 0.1
    spectral_radius: float = 0.9
    sine_amplitude: float = 1.0
    sine_frequency: float = 1.0
    sine_decay: float = 0.1
    leak_rate: float = 0.3
    
    # CNN settings
    cnn_architecture: str = '2d'
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout_rate: float = 0.1
    
    # Training settings
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.1
    
    # Generation settings
    generation_temperature: float = 1.0
    generation_max_length: int = 50
    generation_top_k: Optional[int] = None
    generation_top_p: Optional[float] = None
    
    # Data loading settings
    dataset_name: str = 'cosmopedia-v2'
    min_text_length: int = 10
    max_text_length: int = 1000
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration file
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("Configuration saved to: %s", filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LSMConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            LSMConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create instance with loaded config
        config = cls(**config_dict)
        logger.info("Configuration loaded from: %s", filepath)
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LSMConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            LSMConfig instance
        """
        # Filter out unknown keys
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    def update(self, **kwargs) -> 'LSMConfig':
        """
        Create a new configuration with updated values.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            New LSMConfig instance with updated values
        """
        config_dict = asdict(self)
        config_dict.update(kwargs)
        return LSMConfig.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Tokenizer validation
        if self.tokenizer_backend not in ['gpt2', 'bert', 'spacy']:
            errors.append(f"Invalid tokenizer backend: {self.tokenizer_backend}")
        
        if self.max_length <= 0:
            errors.append(f"max_length must be positive, got: {self.max_length}")
        
        # Embedding validation
        if self.embedding_dim <= 0:
            errors.append(f"embedding_dim must be positive, got: {self.embedding_dim}")
        
        # Reservoir validation
        if self.reservoir_size <= 0:
            errors.append(f"reservoir_size must be positive, got: {self.reservoir_size}")
        
        if not (0.0 <= self.sparsity <= 1.0):
            errors.append(f"sparsity must be between 0 and 1, got: {self.sparsity}")
        
        if self.spectral_radius <= 0:
            errors.append(f"spectral_radius must be positive, got: {self.spectral_radius}")
        
        if not (0.0 <= self.leak_rate <= 1.0):
            errors.append(f"leak_rate must be between 0 and 1, got: {self.leak_rate}")
        
        # CNN validation
        if self.cnn_architecture not in ['2d', '3d']:
            errors.append(f"Invalid CNN architecture: {self.cnn_architecture}")
        
        if not self.cnn_filters or not all(f > 0 for f in self.cnn_filters):
            errors.append("CNN filters must be non-empty list of positive integers")
        
        if not (0.0 <= self.dropout_rate < 1.0):
            errors.append(f"dropout_rate must be between 0 and 1, got: {self.dropout_rate}")
        
        # Training validation
        if self.epochs <= 0:
            errors.append(f"epochs must be positive, got: {self.epochs}")
        
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got: {self.batch_size}")
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got: {self.learning_rate}")
        
        if not (0.0 <= self.validation_split < 1.0):
            errors.append(f"validation_split must be between 0 and 1, got: {self.validation_split}")
        
        return errors
    
    def get_model_size_estimate(self) -> Dict[str, int]:
        """
        Estimate model size in parameters.
        
        Returns:
            Dictionary with parameter count estimates for each component
        """
        vocab_size = self.vocab_size or 50257  # Default GPT-2 vocab size
        
        # Token embeddings
        token_embedding_params = vocab_size * self.embedding_dim
        
        # Positional embeddings (sinusoidal - no parameters)
        pos_embedding_params = 0
        
        # Reservoir (sparse, so fewer actual parameters)
        reservoir_input_params = self.embedding_dim * self.reservoir_size
        reservoir_recurrent_params = int(self.reservoir_size ** 2 * self.sparsity)
        
        # CNN parameters (rough estimate)
        cnn_params = 0
        prev_filters = 1  # Input channels
        
        for filters in self.cnn_filters:
            if self.cnn_architecture == '2d':
                kernel_size = 3 * 3
            else:  # 3D
                kernel_size = 3 * 3 * 3
            
            cnn_params += prev_filters * filters * kernel_size
            prev_filters = filters
        
        # Output layer
        output_params = self.cnn_filters[-1] * vocab_size if self.cnn_filters else 0
        
        return {
            'token_embeddings': token_embedding_params,
            'positional_embeddings': pos_embedding_params,
            'reservoir_input': reservoir_input_params,
            'reservoir_recurrent': reservoir_recurrent_params,
            'cnn': cnn_params,
            'output_layer': output_params,
            'total': (token_embedding_params + pos_embedding_params + 
                     reservoir_input_params + reservoir_recurrent_params + 
                     cnn_params + output_params)
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["LSM Configuration:"]
        lines.append(f"  Tokenizer: {self.tokenizer_backend} (max_length={self.max_length})")
        lines.append(f"  Embedding: dim={self.embedding_dim}")
        lines.append(f"  Reservoir: size={self.reservoir_size}, sparsity={self.sparsity:.2f}")
        lines.append(f"  CNN: {self.cnn_architecture}, filters={self.cnn_filters}")
        lines.append(f"  Training: epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate}")
        
        return "\n".join(lines)


@dataclass
class DualCNNConfig:
    """
    Configuration for dual CNN convenience features.
    
    This configuration manages parameters for the complete dual CNN pipeline:
    - Embedder fitting and initialization
    - Attentive reservoir with attention mechanisms
    - First CNN for next-token prediction with rolling wave output
    - Second CNN for final token prediction generation
    """
    
    # Embedder configuration
    embedder_fit_samples: int = 10000
    embedder_batch_size: int = 256
    embedder_max_length: int = 128
    
    # Reservoir configuration
    reservoir_size: int = 512
    attention_heads: int = 8
    attention_dim: int = 64
    reservoir_sparsity: float = 0.1
    reservoir_spectral_radius: float = 0.9
    reservoir_leak_rate: float = 0.3
    
    # First CNN configuration (next-token prediction)
    first_cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    first_cnn_architecture: str = '2d'
    first_cnn_dropout_rate: float = 0.1
    first_cnn_kernel_size: int = 3
    
    # Rolling wave configuration
    wave_window_size: int = 50
    wave_overlap: int = 10
    max_wave_storage: int = 1000
    wave_feature_dim: int = 256
    
    # Second CNN configuration (final prediction)
    second_cnn_filters: List[int] = field(default_factory=lambda: [128, 256, 512])
    second_cnn_architecture: str = '2d'
    second_cnn_dropout_rate: float = 0.1
    second_cnn_kernel_size: int = 3
    
    # Training configuration
    dual_training_epochs: int = 10
    training_batch_size: int = 32
    learning_rate: float = 0.001
    wave_coordination_weight: float = 0.3
    final_prediction_weight: float = 0.7
    validation_split: float = 0.1
    
    # Generation configuration
    generation_max_length: int = 50
    generation_temperature: float = 1.0
    generation_top_k: Optional[int] = None
    generation_top_p: Optional[float] = None
    
    # Memory management
    max_memory_usage_gb: float = 4.0
    enable_gradient_checkpointing: bool = False
    mixed_precision: bool = True
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters and parameter combinations.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Embedder validation
        if self.embedder_fit_samples <= 0:
            errors.append(f"embedder_fit_samples must be positive, got: {self.embedder_fit_samples}")
        
        if self.embedder_batch_size <= 0:
            errors.append(f"embedder_batch_size must be positive, got: {self.embedder_batch_size}")
        
        if self.embedder_max_length <= 0:
            errors.append(f"embedder_max_length must be positive, got: {self.embedder_max_length}")
        
        # Reservoir validation
        if self.reservoir_size <= 0:
            errors.append(f"reservoir_size must be positive, got: {self.reservoir_size}")
        
        if self.attention_heads <= 0:
            errors.append(f"attention_heads must be positive, got: {self.attention_heads}")
        
        if self.attention_dim <= 0:
            errors.append(f"attention_dim must be positive, got: {self.attention_dim}")
        
        if not (0.0 <= self.reservoir_sparsity <= 1.0):
            errors.append(f"reservoir_sparsity must be between 0 and 1, got: {self.reservoir_sparsity}")
        
        if self.reservoir_spectral_radius <= 0:
            errors.append(f"reservoir_spectral_radius must be positive, got: {self.reservoir_spectral_radius}")
        
        if not (0.0 <= self.reservoir_leak_rate <= 1.0):
            errors.append(f"reservoir_leak_rate must be between 0 and 1, got: {self.reservoir_leak_rate}")
        
        # First CNN validation
        if not self.first_cnn_filters or not all(f > 0 for f in self.first_cnn_filters):
            errors.append("first_cnn_filters must be non-empty list of positive integers")
        
        if self.first_cnn_architecture not in ['2d', '3d']:
            errors.append(f"Invalid first_cnn_architecture: {self.first_cnn_architecture}")
        
        if not (0.0 <= self.first_cnn_dropout_rate < 1.0):
            errors.append(f"first_cnn_dropout_rate must be between 0 and 1, got: {self.first_cnn_dropout_rate}")
        
        if self.first_cnn_kernel_size <= 0 or self.first_cnn_kernel_size % 2 == 0:
            errors.append(f"first_cnn_kernel_size must be positive odd number, got: {self.first_cnn_kernel_size}")
        
        # Rolling wave validation
        if self.wave_window_size <= 0:
            errors.append(f"wave_window_size must be positive, got: {self.wave_window_size}")
        
        if self.wave_overlap < 0 or self.wave_overlap >= self.wave_window_size:
            errors.append(f"wave_overlap must be between 0 and wave_window_size, got: {self.wave_overlap}")
        
        if self.max_wave_storage <= 0:
            errors.append(f"max_wave_storage must be positive, got: {self.max_wave_storage}")
        
        if self.wave_feature_dim <= 0:
            errors.append(f"wave_feature_dim must be positive, got: {self.wave_feature_dim}")
        
        # Second CNN validation
        if not self.second_cnn_filters or not all(f > 0 for f in self.second_cnn_filters):
            errors.append("second_cnn_filters must be non-empty list of positive integers")
        
        if self.second_cnn_architecture not in ['2d', '3d']:
            errors.append(f"Invalid second_cnn_architecture: {self.second_cnn_architecture}")
        
        if not (0.0 <= self.second_cnn_dropout_rate < 1.0):
            errors.append(f"second_cnn_dropout_rate must be between 0 and 1, got: {self.second_cnn_dropout_rate}")
        
        if self.second_cnn_kernel_size <= 0 or self.second_cnn_kernel_size % 2 == 0:
            errors.append(f"second_cnn_kernel_size must be positive odd number, got: {self.second_cnn_kernel_size}")
        
        # Training validation
        if self.dual_training_epochs <= 0:
            errors.append(f"dual_training_epochs must be positive, got: {self.dual_training_epochs}")
        
        if self.training_batch_size <= 0:
            errors.append(f"training_batch_size must be positive, got: {self.training_batch_size}")
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got: {self.learning_rate}")
        
        if not (0.0 <= self.validation_split < 1.0):
            errors.append(f"validation_split must be between 0 and 1, got: {self.validation_split}")
        
        # Weight validation
        if not (0.0 <= self.wave_coordination_weight <= 1.0):
            errors.append(f"wave_coordination_weight must be between 0 and 1, got: {self.wave_coordination_weight}")
        
        if not (0.0 <= self.final_prediction_weight <= 1.0):
            errors.append(f"final_prediction_weight must be between 0 and 1, got: {self.final_prediction_weight}")
        
        # Check that weights sum to approximately 1.0
        weight_sum = self.wave_coordination_weight + self.final_prediction_weight
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"wave_coordination_weight and final_prediction_weight should sum to 1.0, got: {weight_sum}")
        
        # Generation validation
        if self.generation_max_length <= 0:
            errors.append(f"generation_max_length must be positive, got: {self.generation_max_length}")
        
        if self.generation_temperature <= 0:
            errors.append(f"generation_temperature must be positive, got: {self.generation_temperature}")
        
        if self.generation_top_k is not None and self.generation_top_k <= 0:
            errors.append(f"generation_top_k must be positive if specified, got: {self.generation_top_k}")
        
        if self.generation_top_p is not None and not (0.0 < self.generation_top_p <= 1.0):
            errors.append(f"generation_top_p must be between 0 and 1 if specified, got: {self.generation_top_p}")
        
        # Memory validation
        if self.max_memory_usage_gb <= 0:
            errors.append(f"max_memory_usage_gb must be positive, got: {self.max_memory_usage_gb}")
        
        # Parameter combination validation
        if self.attention_heads > 0 and self.attention_dim % self.attention_heads != 0:
            errors.append(f"attention_dim ({self.attention_dim}) must be divisible by attention_heads ({self.attention_heads})")
        
        # Check if wave storage can handle the expected sequence length
        if self.wave_window_size > 0:
            effective_window = self.wave_window_size - self.wave_overlap
            if effective_window <= 0:
                errors.append("wave_overlap must be less than wave_window_size for effective windowing")
        else:
            errors.append("wave_window_size must be positive for effective windowing")
        
        # Memory constraint validation
        estimated_memory = self._estimate_memory_usage()
        if estimated_memory > self.max_memory_usage_gb:
            errors.append(f"Estimated memory usage ({estimated_memory:.2f}GB) exceeds limit ({self.max_memory_usage_gb}GB)")
        
        return errors
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in GB for the dual CNN configuration.
        
        Returns:
            Estimated memory usage in GB
        """
        # Rough estimation based on model parameters and batch size
        
        # Reservoir memory
        reservoir_params = self.reservoir_size * self.reservoir_size * self.reservoir_sparsity
        attention_params = self.attention_heads * self.attention_dim * self.reservoir_size
        
        # First CNN memory
        first_cnn_params = sum(self.first_cnn_filters) * 1000  # Rough estimate
        
        # Second CNN memory
        second_cnn_params = sum(self.second_cnn_filters) * 1000  # Rough estimate
        
        # Wave storage memory
        wave_storage_params = self.max_wave_storage * self.wave_feature_dim
        
        # Total parameters
        total_params = reservoir_params + attention_params + first_cnn_params + second_cnn_params + wave_storage_params
        
        # Memory estimation (4 bytes per float32 parameter, plus overhead)
        memory_gb = (total_params * 4 * self.training_batch_size) / (1024**3)
        memory_gb *= 2  # Account for gradients and optimizer states
        memory_gb += 0.5  # Base overhead
        
        return memory_gb
    
    def get_intelligent_defaults(self, input_data_characteristics: Optional[Dict[str, Any]] = None) -> 'DualCNNConfig':
        """
        Generate intelligent defaults based on input data characteristics.
        
        Args:
            input_data_characteristics: Dictionary containing data info like vocab_size, avg_length, etc.
            
        Returns:
            New DualCNNConfig with adjusted defaults
        """
        config_dict = asdict(self)
        
        if input_data_characteristics:
            vocab_size = input_data_characteristics.get('vocab_size', 50000)
            avg_length = input_data_characteristics.get('avg_length', 100)
            dataset_size = input_data_characteristics.get('dataset_size', 10000)
            
            # Adjust embedder samples based on dataset size
            if dataset_size < 5000:
                config_dict['embedder_fit_samples'] = min(dataset_size // 2, 2000)
            elif dataset_size > 50000:
                config_dict['embedder_fit_samples'] = 20000
            
            # Adjust wave window based on average sequence length
            if avg_length < 50:
                config_dict['wave_window_size'] = max(10, avg_length // 2)
                config_dict['wave_overlap'] = max(2, config_dict['wave_window_size'] // 5)
            elif avg_length > 200:
                config_dict['wave_window_size'] = 100
                config_dict['wave_overlap'] = 20
            
            # Adjust reservoir size based on vocabulary size
            if vocab_size < 10000:
                config_dict['reservoir_size'] = 256
                config_dict['attention_heads'] = 4
            elif vocab_size > 100000:
                config_dict['reservoir_size'] = 1024
                config_dict['attention_heads'] = 16
        
        return DualCNNConfig(**config_dict)
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration file
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("DualCNN configuration saved to: %s", filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'DualCNNConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            DualCNNConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls(**config_dict)
        logger.info("DualCNN configuration loaded from: %s", filepath)
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DualCNNConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            DualCNNConfig instance
        """
        # Filter out unknown keys
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    def to_lsm_config(self) -> LSMConfig:
        """
        Convert to compatible LSMConfig for backward compatibility.
        
        Returns:
            LSMConfig instance with compatible parameters
        """
        return LSMConfig(
            max_length=self.embedder_max_length,
            embedding_dim=self.wave_feature_dim,
            max_samples=self.embedder_fit_samples,
            reservoir_size=self.reservoir_size,
            sparsity=self.reservoir_sparsity,
            spectral_radius=self.reservoir_spectral_radius,
            leak_rate=self.reservoir_leak_rate,
            cnn_architecture=self.first_cnn_architecture,
            cnn_filters=self.first_cnn_filters,
            dropout_rate=self.first_cnn_dropout_rate,
            epochs=self.dual_training_epochs,
            batch_size=self.training_batch_size,
            learning_rate=self.learning_rate,
            validation_split=self.validation_split,
            generation_temperature=self.generation_temperature,
            generation_max_length=self.generation_max_length,
            generation_top_k=self.generation_top_k,
            generation_top_p=self.generation_top_p
        )
    
    def __str__(self) -> str:
        """String representation of dual CNN configuration."""
        lines = ["Dual CNN Configuration:"]
        lines.append(f"  Embedder: samples={self.embedder_fit_samples}, batch_size={self.embedder_batch_size}")
        lines.append(f"  Reservoir: size={self.reservoir_size}, attention_heads={self.attention_heads}")
        lines.append(f"  First CNN: {self.first_cnn_architecture}, filters={self.first_cnn_filters}")
        lines.append(f"  Wave Storage: window={self.wave_window_size}, overlap={self.wave_overlap}, max={self.max_wave_storage}")
        lines.append(f"  Second CNN: {self.second_cnn_architecture}, filters={self.second_cnn_filters}")
        lines.append(f"  Training: epochs={self.dual_training_epochs}, batch_size={self.training_batch_size}")
        lines.append(f"  Weights: wave={self.wave_coordination_weight:.2f}, final={self.final_prediction_weight:.2f}")
        lines.append(f"  Memory: max={self.max_memory_usage_gb}GB, estimated={self._estimate_memory_usage():.2f}GB")
        
        return "\n".join(lines)


class ConfigManager:
    """Utility class for managing multiple configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config: LSMConfig, name: str) -> str:
        """
        Save configuration with a name.
        
        Args:
            config: Configuration to save
            name: Configuration name
            
        Returns:
            Path to saved configuration file
        """
        filepath = os.path.join(self.config_dir, f"{name}.json")
        config.save(filepath)
        return filepath
    
    def load_config(self, name: str) -> LSMConfig:
        """
        Load configuration by name.
        
        Args:
            name: Configuration name
            
        Returns:
            Loaded configuration
        """
        filepath = os.path.join(self.config_dir, f"{name}.json")
        return LSMConfig.load(filepath)
    
    def list_configs(self) -> List[str]:
        """
        List available configuration names.
        
        Returns:
            List of configuration names
        """
        config_files = [f for f in os.listdir(self.config_dir) if f.endswith('.json')]
        return [os.path.splitext(f)[0] for f in config_files]
    
    def delete_config(self, name: str) -> bool:
        """
        Delete a configuration.
        
        Args:
            name: Configuration name to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        filepath = os.path.join(self.config_dir, f"{name}.json")
        try:
            os.remove(filepath)
            logger.info("Deleted configuration: %s", name)
            return True
        except FileNotFoundError:
            logger.warning("Configuration not found: %s", name)
            return False
        except Exception as e:
            logger.error("Failed to delete configuration %s: %s", name, e)
            return False


def create_preset_configs() -> Dict[str, LSMConfig]:
    """
    Create preset configurations for common use cases.
    
    Returns:
        Dictionary mapping preset names to configurations
    """
    presets = {}
    
    # Small model for testing
    presets['small'] = LSMConfig(
        embedding_dim=128,
        reservoir_size=256,
        cnn_filters=[32, 64],
        epochs=5,
        batch_size=16,
        max_samples=1000
    )
    
    # Medium model for general use
    presets['medium'] = LSMConfig(
        embedding_dim=256,
        reservoir_size=512,
        cnn_filters=[64, 128, 256],
        epochs=10,
        batch_size=32,
        max_samples=10000
    )
    
    # Large model for better performance
    presets['large'] = LSMConfig(
        embedding_dim=512,
        reservoir_size=1024,
        cnn_filters=[128, 256, 512],
        epochs=20,
        batch_size=16,
        max_samples=50000
    )
    
    # Fast training configuration
    presets['fast'] = LSMConfig(
        embedding_dim=128,
        reservoir_size=256,
        cnn_filters=[32, 64],
        epochs=3,
        batch_size=64,
        max_samples=5000,
        learning_rate=0.002
    )
    
    # High quality generation
    presets['quality'] = LSMConfig(
        embedding_dim=384,
        reservoir_size=768,
        cnn_filters=[96, 192, 384],
        epochs=15,
        batch_size=24,
        sparsity=0.05,  # Less sparse for better performance
        generation_temperature=0.8,
        generation_top_p=0.9
    )
    
    return presets


def get_config_for_gpu_memory(available_memory_gb: float) -> LSMConfig:
    """
    Get recommended configuration based on available GPU memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        
    Returns:
        Recommended configuration
    """
    if available_memory_gb < 2:
        return create_preset_configs()['small']
    elif available_memory_gb < 6:
        return create_preset_configs()['medium']
    elif available_memory_gb < 12:
        return create_preset_configs()['large']
    else:
        # Very large configuration for high-memory systems
        return LSMConfig(
            embedding_dim=768,
            reservoir_size=1536,
            cnn_filters=[192, 384, 768],
            epochs=25,
            batch_size=8,
            max_samples=100000
        )
