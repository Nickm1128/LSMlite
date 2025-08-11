"""
Wave output data structures for dual CNN convenience features.

This module provides dataclasses for managing wave output data and training progress
in the dual CNN pipeline, with serialization support for persistence.
"""

import json
import pickle
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

logger = logging.getLogger(__name__)


@dataclass
class WaveOutput:
    """
    Structure for rolling wave output data from the first CNN.
    
    This dataclass encapsulates wave features, attention weights, and metadata
    for efficient storage and retrieval in the dual CNN pipeline.
    """
    
    sequence_position: int
    wave_features: Union[np.ndarray, 'tf.Tensor']
    attention_weights: Union[np.ndarray, 'tf.Tensor']
    timestamp: float
    confidence_score: float
    
    # Optional metadata
    sequence_id: Optional[str] = None
    batch_index: Optional[int] = None
    epoch: Optional[int] = None
    loss_value: Optional[float] = None
    gradient_norm: Optional[float] = None
    
    def __post_init__(self):
        """Validate and normalize data after initialization."""
        # Validate sequence position
        if self.sequence_position < 0:
            raise ValueError(f"sequence_position must be non-negative, got: {self.sequence_position}")
        
        # Validate confidence score
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"confidence_score must be between 0 and 1, got: {self.confidence_score}")
        
        # Validate timestamp
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got: {self.timestamp}")
        
        # Convert TensorFlow tensors to numpy for serialization compatibility
        if TF_AVAILABLE and tf is not None:
            if isinstance(self.wave_features, tf.Tensor):
                self.wave_features = self.wave_features.numpy()
            if isinstance(self.attention_weights, tf.Tensor):
                self.attention_weights = self.attention_weights.numpy()
        
        # Ensure arrays are numpy arrays
        if not isinstance(self.wave_features, np.ndarray):
            self.wave_features = np.array(self.wave_features)
        if not isinstance(self.attention_weights, np.ndarray):
            self.attention_weights = np.array(self.attention_weights)
        
        # Validate array shapes
        if self.wave_features.ndim == 0:
            raise ValueError("wave_features must be at least 1-dimensional")
        if self.attention_weights.ndim == 0:
            raise ValueError("attention_weights must be at least 1-dimensional")
    
    @property
    def feature_dim(self) -> int:
        """Get the feature dimension of wave features."""
        return self.wave_features.shape[-1] if self.wave_features.ndim > 0 else 0
    
    @property
    def attention_heads(self) -> int:
        """Get the number of attention heads."""
        return self.attention_weights.shape[0] if self.attention_weights.ndim > 1 else 1
    
    @property
    def memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        wave_bytes = self.wave_features.nbytes if hasattr(self.wave_features, 'nbytes') else 0
        attention_bytes = self.attention_weights.nbytes if hasattr(self.attention_weights, 'nbytes') else 0
        metadata_bytes = 200  # Rough estimate for other fields
        return wave_bytes + attention_bytes + metadata_bytes
    
    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_arrays: Whether to include numpy arrays in the dictionary
            
        Returns:
            Dictionary representation
        """
        data = {
            'sequence_position': self.sequence_position,
            'timestamp': self.timestamp,
            'confidence_score': self.confidence_score,
            'sequence_id': self.sequence_id,
            'batch_index': self.batch_index,
            'epoch': self.epoch,
            'loss_value': self.loss_value,
            'gradient_norm': self.gradient_norm,
            'feature_dim': self.feature_dim,
            'attention_heads': self.attention_heads,
            'memory_usage_bytes': self.memory_usage_bytes
        }
        
        if include_arrays:
            data['wave_features'] = self.wave_features.tolist()
            data['attention_weights'] = self.attention_weights.tolist()
            data['wave_features_shape'] = self.wave_features.shape
            data['attention_weights_shape'] = self.attention_weights.shape
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaveOutput':
        """
        Create WaveOutput from dictionary.
        
        Args:
            data: Dictionary containing wave output data
            
        Returns:
            WaveOutput instance
        """
        # Extract arrays if present
        wave_features = data.get('wave_features', [])
        attention_weights = data.get('attention_weights', [])
        
        # Reshape if shape information is available
        if 'wave_features_shape' in data:
            wave_features = np.array(wave_features).reshape(data['wave_features_shape'])
        else:
            wave_features = np.array(wave_features)
        
        if 'attention_weights_shape' in data:
            attention_weights = np.array(attention_weights).reshape(data['attention_weights_shape'])
        else:
            attention_weights = np.array(attention_weights)
        
        return cls(
            sequence_position=data['sequence_position'],
            wave_features=wave_features,
            attention_weights=attention_weights,
            timestamp=data['timestamp'],
            confidence_score=data['confidence_score'],
            sequence_id=data.get('sequence_id'),
            batch_index=data.get('batch_index'),
            epoch=data.get('epoch'),
            loss_value=data.get('loss_value'),
            gradient_norm=data.get('gradient_norm')
        )
    
    def save_json(self, filepath: str) -> None:
        """
        Save wave output to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("WaveOutput saved to JSON: %s", filepath)
        except Exception as e:
            logger.error("Failed to save WaveOutput to JSON %s: %s", filepath, e)
            raise
    
    @classmethod
    def load_json(cls, filepath: str) -> 'WaveOutput':
        """
        Load wave output from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            WaveOutput instance
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info("WaveOutput loaded from JSON: %s", filepath)
            return cls.from_dict(data)
        except Exception as e:
            logger.error("Failed to load WaveOutput from JSON %s: %s", filepath, e)
            raise
    
    def save_pickle(self, filepath: str) -> None:
        """
        Save wave output to pickle file (more efficient for large arrays).
        
        Args:
            filepath: Path to save pickle file
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info("WaveOutput saved to pickle: %s", filepath)
        except Exception as e:
            logger.error("Failed to save WaveOutput to pickle %s: %s", filepath, e)
            raise
    
    @classmethod
    def load_pickle(cls, filepath: str) -> 'WaveOutput':
        """
        Load wave output from pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            WaveOutput instance
        """
        try:
            with open(filepath, 'rb') as f:
                wave_output = pickle.load(f)
            logger.info("WaveOutput loaded from pickle: %s", filepath)
            return wave_output
        except Exception as e:
            logger.error("Failed to load WaveOutput from pickle %s: %s", filepath, e)
            raise
    
    def __str__(self) -> str:
        """String representation of wave output."""
        return (f"WaveOutput(pos={self.sequence_position}, "
                f"features={self.wave_features.shape}, "
                f"attention={self.attention_weights.shape}, "
                f"confidence={self.confidence_score:.3f}, "
                f"memory={self.memory_usage_bytes}B)")


@dataclass
class TrainingProgress:
    """
    Progress tracking for dual CNN training.
    
    This dataclass tracks training metrics, progress indicators, and performance
    statistics for both CNNs in the dual CNN pipeline.
    """
    
    current_epoch: int
    total_epochs: int
    current_batch: int
    total_batches: int
    first_cnn_loss: float
    second_cnn_loss: float
    combined_loss: float
    wave_storage_utilization: float
    attention_entropy: float
    
    # Time tracking
    epoch_start_time: float
    batch_start_time: float
    estimated_time_remaining: float
    
    # Performance metrics
    first_cnn_accuracy: Optional[float] = None
    second_cnn_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Validation metrics
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    best_validation_loss: Optional[float] = None
    epochs_without_improvement: Optional[int] = None
    
    # Additional metadata
    model_checkpoint_path: Optional[str] = None
    training_stage: str = "training"  # "training", "validation", "completed", "failed"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate progress data after initialization."""
        # Validate epoch and batch counts
        if self.current_epoch < 0 or self.current_epoch > self.total_epochs:
            raise ValueError(f"current_epoch must be between 0 and {self.total_epochs}, got: {self.current_epoch}")
        
        if self.current_batch < 0 or self.current_batch > self.total_batches:
            raise ValueError(f"current_batch must be between 0 and {self.total_batches}, got: {self.current_batch}")
        
        # Validate loss values
        if self.first_cnn_loss < 0:
            raise ValueError(f"first_cnn_loss must be non-negative, got: {self.first_cnn_loss}")
        if self.second_cnn_loss < 0:
            raise ValueError(f"second_cnn_loss must be non-negative, got: {self.second_cnn_loss}")
        if self.combined_loss < 0:
            raise ValueError(f"combined_loss must be non-negative, got: {self.combined_loss}")
        
        # Validate utilization and entropy
        if not (0.0 <= self.wave_storage_utilization <= 1.0):
            raise ValueError(f"wave_storage_utilization must be between 0 and 1, got: {self.wave_storage_utilization}")
        
        if self.attention_entropy < 0:
            raise ValueError(f"attention_entropy must be non-negative, got: {self.attention_entropy}")
        
        # Validate time values
        if self.epoch_start_time < 0:
            raise ValueError(f"epoch_start_time must be non-negative, got: {self.epoch_start_time}")
        if self.batch_start_time < 0:
            raise ValueError(f"batch_start_time must be non-negative, got: {self.batch_start_time}")
        if self.estimated_time_remaining < 0:
            raise ValueError(f"estimated_time_remaining must be non-negative, got: {self.estimated_time_remaining}")
        
        # Validate optional accuracy values
        if self.first_cnn_accuracy is not None and not (0.0 <= self.first_cnn_accuracy <= 1.0):
            raise ValueError(f"first_cnn_accuracy must be between 0 and 1, got: {self.first_cnn_accuracy}")
        if self.second_cnn_accuracy is not None and not (0.0 <= self.second_cnn_accuracy <= 1.0):
            raise ValueError(f"second_cnn_accuracy must be between 0 and 1, got: {self.second_cnn_accuracy}")
        if self.validation_accuracy is not None and not (0.0 <= self.validation_accuracy <= 1.0):
            raise ValueError(f"validation_accuracy must be between 0 and 1, got: {self.validation_accuracy}")
        
        # Validate training stage
        valid_stages = ["training", "validation", "completed", "failed"]
        if self.training_stage not in valid_stages:
            raise ValueError(f"training_stage must be one of {valid_stages}, got: {self.training_stage}")
    
    @property
    def epoch_progress(self) -> float:
        """Get epoch progress as a fraction (0.0 to 1.0)."""
        if self.total_epochs == 0:
            return 1.0
        return self.current_epoch / self.total_epochs
    
    @property
    def batch_progress(self) -> float:
        """Get batch progress within current epoch as a fraction (0.0 to 1.0)."""
        if self.total_batches == 0:
            return 1.0
        return self.current_batch / self.total_batches
    
    @property
    def overall_progress(self) -> float:
        """Get overall training progress as a fraction (0.0 to 1.0)."""
        if self.total_epochs == 0:
            return 1.0
        
        completed_epochs = self.current_epoch
        current_epoch_progress = self.batch_progress
        
        return (completed_epochs + current_epoch_progress) / self.total_epochs
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time for current epoch in seconds."""
        current_time = datetime.now().timestamp()
        return current_time - self.epoch_start_time
    
    @property
    def is_improving(self) -> bool:
        """Check if validation loss is improving."""
        if self.validation_loss is None or self.best_validation_loss is None:
            return True  # Assume improving if no validation data
        return self.validation_loss <= self.best_validation_loss
    
    def update_batch(self, batch_idx: int, first_loss: float, second_loss: float, 
                    combined_loss: float, wave_utilization: float, attention_entropy: float) -> None:
        """
        Update progress for a new batch.
        
        Args:
            batch_idx: Current batch index
            first_loss: Loss from first CNN
            second_loss: Loss from second CNN
            combined_loss: Combined loss value
            wave_utilization: Wave storage utilization fraction
            attention_entropy: Attention entropy value
        """
        self.current_batch = batch_idx
        self.first_cnn_loss = first_loss
        self.second_cnn_loss = second_loss
        self.combined_loss = combined_loss
        self.wave_storage_utilization = wave_utilization
        self.attention_entropy = attention_entropy
        self.batch_start_time = datetime.now().timestamp()
        
        # Update time estimation
        if self.current_batch > 0:
            elapsed = self.elapsed_time
            batches_remaining = (self.total_epochs - self.current_epoch) * self.total_batches - self.current_batch
            if batches_remaining > 0:
                avg_batch_time = elapsed / self.current_batch
                self.estimated_time_remaining = avg_batch_time * batches_remaining
    
    def update_epoch(self, epoch_idx: int) -> None:
        """
        Update progress for a new epoch.
        
        Args:
            epoch_idx: Current epoch index
        """
        self.current_epoch = epoch_idx
        self.current_batch = 0
        self.epoch_start_time = datetime.now().timestamp()
        self.batch_start_time = self.epoch_start_time
    
    def update_validation(self, val_loss: float, val_accuracy: Optional[float] = None) -> None:
        """
        Update validation metrics.
        
        Args:
            val_loss: Validation loss
            val_accuracy: Optional validation accuracy
        """
        self.validation_loss = val_loss
        self.validation_accuracy = val_accuracy
        
        # Update best validation loss
        if self.best_validation_loss is None or val_loss < self.best_validation_loss:
            self.best_validation_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement = (self.epochs_without_improvement or 0) + 1
    
    def mark_completed(self) -> None:
        """Mark training as completed."""
        self.training_stage = "completed"
        self.estimated_time_remaining = 0.0
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark training as failed.
        
        Args:
            error_message: Error description
        """
        self.training_stage = "failed"
        self.error_message = error_message
        self.estimated_time_remaining = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingProgress':
        """
        Create TrainingProgress from dictionary.
        
        Args:
            data: Dictionary containing progress data
            
        Returns:
            TrainingProgress instance
        """
        return cls(**data)
    
    def save_json(self, filepath: str) -> None:
        """
        Save training progress to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("TrainingProgress saved to JSON: %s", filepath)
        except Exception as e:
            logger.error("Failed to save TrainingProgress to JSON %s: %s", filepath, e)
            raise
    
    @classmethod
    def load_json(cls, filepath: str) -> 'TrainingProgress':
        """
        Load training progress from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            TrainingProgress instance
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info("TrainingProgress loaded from JSON: %s", filepath)
            return cls.from_dict(data)
        except Exception as e:
            logger.error("Failed to load TrainingProgress from JSON %s: %s", filepath, e)
            raise
    
    def get_progress_summary(self) -> str:
        """
        Get a human-readable progress summary.
        
        Returns:
            Progress summary string
        """
        lines = []
        lines.append(f"Training Progress: {self.overall_progress:.1%}")
        lines.append(f"Epoch: {self.current_epoch}/{self.total_epochs} ({self.epoch_progress:.1%})")
        lines.append(f"Batch: {self.current_batch}/{self.total_batches} ({self.batch_progress:.1%})")
        lines.append(f"Losses: First={self.first_cnn_loss:.4f}, Second={self.second_cnn_loss:.4f}, Combined={self.combined_loss:.4f}")
        lines.append(f"Wave Storage: {self.wave_storage_utilization:.1%} utilized")
        lines.append(f"Attention Entropy: {self.attention_entropy:.3f}")
        
        if self.estimated_time_remaining > 0:
            hours = int(self.estimated_time_remaining // 3600)
            minutes = int((self.estimated_time_remaining % 3600) // 60)
            seconds = int(self.estimated_time_remaining % 60)
            lines.append(f"ETA: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        if self.validation_loss is not None:
            lines.append(f"Validation Loss: {self.validation_loss:.4f}")
            if self.validation_accuracy is not None:
                lines.append(f"Validation Accuracy: {self.validation_accuracy:.3f}")
        
        lines.append(f"Status: {self.training_stage}")
        
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation of training progress."""
        return (f"TrainingProgress(epoch={self.current_epoch}/{self.total_epochs}, "
                f"batch={self.current_batch}/{self.total_batches}, "
                f"loss={self.combined_loss:.4f}, "
                f"progress={self.overall_progress:.1%}, "
                f"status={self.training_stage})")


class WaveOutputBatch:
    """
    Utility class for managing batches of WaveOutput objects.
    
    This class provides efficient storage and retrieval of multiple wave outputs,
    with batch serialization and memory management capabilities.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize wave output batch.
        
        Args:
            max_size: Maximum number of wave outputs to store
        """
        self.max_size = max_size
        self.wave_outputs: List[WaveOutput] = []
        self._creation_time = datetime.now().timestamp()
    
    def add(self, wave_output: WaveOutput) -> None:
        """
        Add a wave output to the batch.
        
        Args:
            wave_output: WaveOutput to add
        """
        if len(self.wave_outputs) >= self.max_size:
            # Remove oldest wave output to make space
            self.wave_outputs.pop(0)
            logger.warning("WaveOutputBatch at capacity, removed oldest wave output")
        
        self.wave_outputs.append(wave_output)
    
    def get_by_position(self, position: int) -> Optional[WaveOutput]:
        """
        Get wave output by sequence position.
        
        Args:
            position: Sequence position to find
            
        Returns:
            WaveOutput if found, None otherwise
        """
        for wave_output in self.wave_outputs:
            if wave_output.sequence_position == position:
                return wave_output
        return None
    
    def get_range(self, start_pos: int, end_pos: int) -> List[WaveOutput]:
        """
        Get wave outputs in a position range.
        
        Args:
            start_pos: Start position (inclusive)
            end_pos: End position (exclusive)
            
        Returns:
            List of WaveOutput objects in the range
        """
        return [wo for wo in self.wave_outputs 
                if start_pos <= wo.sequence_position < end_pos]
    
    def clear(self) -> None:
        """Clear all wave outputs from the batch."""
        self.wave_outputs.clear()
    
    def get_memory_usage(self) -> int:
        """
        Get total memory usage of all wave outputs in bytes.
        
        Returns:
            Total memory usage in bytes
        """
        return sum(wo.memory_usage_bytes for wo in self.wave_outputs)
    
    def save_batch(self, filepath: str, format: str = 'pickle') -> None:
        """
        Save entire batch to file.
        
        Args:
            filepath: Path to save file
            format: Format to use ('pickle' or 'json')
        """
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self.wave_outputs, f)
        elif format == 'json':
            batch_data = {
                'creation_time': self._creation_time,
                'max_size': self.max_size,
                'wave_outputs': [wo.to_dict() for wo in self.wave_outputs]
            }
            with open(filepath, 'w') as f:
                json.dump(batch_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info("WaveOutputBatch saved to %s (%s format)", filepath, format)
    
    @classmethod
    def load_batch(cls, filepath: str, format: str = 'pickle') -> 'WaveOutputBatch':
        """
        Load batch from file.
        
        Args:
            filepath: Path to load file
            format: Format to use ('pickle' or 'json')
            
        Returns:
            WaveOutputBatch instance
        """
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                wave_outputs = pickle.load(f)
            batch = cls()
            batch.wave_outputs = wave_outputs
        elif format == 'json':
            with open(filepath, 'r') as f:
                batch_data = json.load(f)
            
            batch = cls(max_size=batch_data.get('max_size', 1000))
            batch._creation_time = batch_data.get('creation_time', datetime.now().timestamp())
            batch.wave_outputs = [WaveOutput.from_dict(wo_data) 
                                 for wo_data in batch_data['wave_outputs']]
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info("WaveOutputBatch loaded from %s (%s format)", filepath, format)
        return batch
    
    def __len__(self) -> int:
        """Get number of wave outputs in batch."""
        return len(self.wave_outputs)
    
    def __iter__(self):
        """Iterate over wave outputs."""
        return iter(self.wave_outputs)
    
    def __str__(self) -> str:
        """String representation of wave output batch."""
        return (f"WaveOutputBatch(size={len(self.wave_outputs)}/{self.max_size}, "
                f"memory={self.get_memory_usage()}B)")