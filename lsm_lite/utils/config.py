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
