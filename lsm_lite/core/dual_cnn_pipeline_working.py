#!/usr/bin/env python3
"""Test script to create DualCNNPipeline step by step"""

import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import tensorflow as tf
import numpy as np

# Import each component individually
print("Importing config...")
from lsm_lite.utils.config import DualCNNConfig, LSMConfig

print("Importing tokenizer...")
from lsm_lite.core.tokenizer import UnifiedTokenizer

print("Importing embedder...")
from lsm_lite.data.embeddings import SinusoidalEmbedder

print("Importing reservoir...")
from lsm_lite.core.attentive_reservoir import AttentiveReservoir

print("Importing wave storage...")
from lsm_lite.core.rolling_wave_storage import RollingWaveStorage, WaveStorageError

print("Importing CNN...")
from lsm_lite.core.cnn import CNNProcessor

logger = logging.getLogger(__name__)

print("Creating exception classes...")

class ComponentInitializationError(Exception):
    """Exception raised when component initialization fails."""
    
    def __init__(self, component_name: str, error_details: str):
        self.component_name = component_name
        self.error_details = error_details
        super().__init__(f"Failed to initialize {component_name}: {error_details}")


class DualCNNTrainingError(Exception):
    """Exception raised during dual CNN training."""
    
    def __init__(self, stage: str, cnn_id: str, details: str):
        super().__init__(f"Dual CNN training failed at {stage} for {cnn_id}: {details}")

print("Creating DualCNNPipeline class...")

class DualCNNPipeline:
    """
    Main orchestrator for the dual CNN workflow.
    """
    
    def __init__(self, config: Union[DualCNNConfig, LSMConfig]):
        """Initialize the dual CNN pipeline."""
        # Convert LSMConfig to DualCNNConfig if needed
        if isinstance(config, LSMConfig):
            self.config = self._convert_lsm_config(config)
        else:
            self.config = config
        
        # Validate configuration
        validation_errors = self.config.validate()
        if validation_errors:
            raise ComponentInitializationError(
                "DualCNNPipeline", 
                f"Configuration validation failed: {'; '.join(validation_errors)}"
            )
        
        # Initialize component placeholders
        self.tokenizer = None
        self.embedder = None
        self.reservoir = None
        self.first_cnn = None
        self.second_cnn = None
        self.wave_storage = None
        
        # Pipeline state
        self._is_initialized = False
        self._initialization_progress = {}
        
        logger.info("DualCNNPipeline created with config: %s", self.config)
    
    def _convert_lsm_config(self, lsm_config: LSMConfig) -> DualCNNConfig:
        """Convert LSMConfig to DualCNNConfig for backward compatibility."""
        return DualCNNConfig(
            embedder_fit_samples=lsm_config.max_samples,
            embedder_batch_size=lsm_config.batch_size,
            embedder_max_length=lsm_config.max_length,
            reservoir_size=lsm_config.reservoir_size,
            reservoir_sparsity=lsm_config.sparsity,
            reservoir_spectral_radius=lsm_config.spectral_radius,
            reservoir_leak_rate=lsm_config.leak_rate,
            first_cnn_filters=lsm_config.cnn_filters,
            first_cnn_architecture=lsm_config.cnn_architecture,
            first_cnn_dropout_rate=lsm_config.dropout_rate,
            wave_feature_dim=lsm_config.embedding_dim,
            dual_training_epochs=lsm_config.epochs,
            training_batch_size=lsm_config.batch_size,
            learning_rate=lsm_config.learning_rate,
            validation_split=lsm_config.validation_split,
            generation_max_length=lsm_config.generation_max_length,
            generation_temperature=lsm_config.generation_temperature,
            generation_top_k=lsm_config.generation_top_k,
            generation_top_p=lsm_config.generation_top_p
        )
    
    def fit_and_initialize(self, 
                          training_data: List[str],
                          embedder_params: Optional[Dict[str, Any]] = None,
                          reservoir_params: Optional[Dict[str, Any]] = None,
                          cnn_params: Optional[Dict[str, Any]] = None,
                          progress_callback: Optional[callable] = None) -> None:
        """One-shot setup of the entire dual CNN pipeline."""
        try:
            logger.info("Starting dual CNN pipeline initialization...")
            self._initialization_progress = {}
            
            # Step 1: Initialize tokenizer
            self._update_progress("tokenizer", "initializing", progress_callback)
            self._initialize_tokenizer(training_data)
            self._update_progress("tokenizer", "completed", progress_callback)
            
            # Step 2: Fit embedder
            self._update_progress("embedder", "fitting", progress_callback)
            self._fit_embedder(training_data, embedder_params or {})
            self._update_progress("embedder", "completed", progress_callback)
            
            # Step 3: Initialize attentive reservoir
            self._update_progress("reservoir", "initializing", progress_callback)
            self._initialize_reservoir(reservoir_params or {})
            self._update_progress("reservoir", "completed", progress_callback)
            
            # Step 4: Set up rolling wave storage
            self._update_progress("wave_storage", "initializing", progress_callback)
            self._initialize_wave_storage()
            self._update_progress("wave_storage", "completed", progress_callback)
            
            # Step 5: Initialize first CNN
            self._update_progress("first_cnn", "initializing", progress_callback)
            self._initialize_first_cnn(cnn_params or {})
            self._update_progress("first_cnn", "completed", progress_callback)
            
            # Step 6: Initialize second CNN
            self._update_progress("second_cnn", "initializing", progress_callback)
            self._initialize_second_cnn(cnn_params or {})
            self._update_progress("second_cnn", "completed", progress_callback)
            
            self._is_initialized = True
            logger.info("Dual CNN pipeline initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Pipeline initialization failed at step {self._get_current_step()}: {str(e)}"
            logger.error(error_msg)
            raise ComponentInitializationError("DualCNNPipeline", error_msg)
    
    def _update_progress(self, component: str, status: str, callback: Optional[callable]):
        """Update initialization progress and call callback if provided."""
        self._initialization_progress[component] = status
        if callback:
            callback(component, status, self._initialization_progress)
    
    def _get_current_step(self) -> str:
        """Get the current initialization step for error reporting."""
        for component, status in self._initialization_progress.items():
            if status != "completed":
                return component
        return "unknown"
    
    def _initialize_tokenizer(self, training_data: List[str]) -> None:
        """Initialize the tokenizer."""
        try:
            self.tokenizer = UnifiedTokenizer(
                backend='gpt2',
                max_length=self.config.embedder_max_length
            )
            logger.info("Tokenizer initialized successfully")
        except Exception as e:
            raise ComponentInitializationError("tokenizer", str(e))
    
    def _fit_embedder(self, training_data: List[str], embedder_params: Dict[str, Any]) -> None:
        """Fit the sinusoidal embedder to training data."""
        try:
            # Sample data for embedder fitting
            sample_size = min(len(training_data), self.config.embedder_fit_samples)
            sample_data = training_data[:sample_size]
            
            # Tokenize sample data
            tokenized_data = []
            for text in sample_data:
                result = self.tokenizer.tokenize(text, padding=False, truncation=True)
                tokens = result['input_ids'][0].tolist()
                tokenized_data.extend(tokens)
            
            # Initialize embedder
            vocab_size = self.tokenizer.vocab_size
            self.embedder = SinusoidalEmbedder(
                vocab_size=vocab_size,
                embedding_dim=self.config.wave_feature_dim,
                max_length=self.config.embedder_max_length,
                **embedder_params
            )
            
            logger.info("Embedder fitted successfully on %d samples", sample_size)
            
        except Exception as e:
            raise ComponentInitializationError("embedder", str(e))
    
    def _initialize_reservoir(self, reservoir_params: Dict[str, Any]) -> None:
        """Initialize the attentive reservoir."""
        try:
            self.reservoir = AttentiveReservoir(
                input_dim=self.config.wave_feature_dim,
                reservoir_size=self.config.reservoir_size,
                attention_heads=self.config.attention_heads,
                attention_dim=self.config.attention_dim,
                sparsity=self.config.reservoir_sparsity,
                spectral_radius=self.config.reservoir_spectral_radius,
                leak_rate=self.config.reservoir_leak_rate,
                **reservoir_params
            )
            
            logger.info("Attentive reservoir initialized successfully")
            
        except Exception as e:
            raise ComponentInitializationError("reservoir", str(e))
    
    def _initialize_wave_storage(self) -> None:
        """Initialize the rolling wave storage."""
        try:
            max_memory_mb = (self.config.max_memory_usage_gb * 1024 * 0.1)
            
            self.wave_storage = RollingWaveStorage(
                max_sequence_length=self.config.max_wave_storage,
                feature_dim=self.config.wave_feature_dim,
                window_size=self.config.wave_window_size,
                overlap=self.config.wave_overlap,
                max_memory_mb=max_memory_mb
            )
            
            logger.info("Rolling wave storage initialized successfully")
            
        except Exception as e:
            raise ComponentInitializationError("wave_storage", str(e))
    
    def _initialize_first_cnn(self, cnn_params: Dict[str, Any]) -> None:
        """Initialize the first CNN for next-token prediction."""
        try:
            vocab_size = self.tokenizer.vocab_size
            input_shape = (self.config.embedder_max_length, self.config.reservoir_size)
            
            self.first_cnn = CNNProcessor(
                input_shape=input_shape,
                architecture=self.config.first_cnn_architecture,
                filters=self.config.first_cnn_filters,
                vocab_size=vocab_size,
                dropout_rate=self.config.first_cnn_dropout_rate,
                name="first_cnn",
                **cnn_params
            )
            
            logger.info("First CNN initialized successfully")
            
        except Exception as e:
            raise ComponentInitializationError("first_cnn", str(e))
    
    def _initialize_second_cnn(self, cnn_params: Dict[str, Any]) -> None:
        """Initialize the second CNN for final token prediction."""
        try:
            vocab_size = self.tokenizer.vocab_size
            input_shape = (self.config.wave_window_size, self.config.wave_feature_dim)
            
            self.second_cnn = CNNProcessor(
                input_shape=input_shape,
                architecture=self.config.second_cnn_architecture,
                filters=self.config.second_cnn_filters,
                vocab_size=vocab_size,
                dropout_rate=self.config.second_cnn_dropout_rate,
                name="second_cnn",
                **cnn_params
            )
            
            logger.info("Second CNN initialized successfully")
            
        except Exception as e:
            raise ComponentInitializationError("second_cnn", str(e))
    
    def is_initialized(self) -> bool:
        """Check if the pipeline is fully initialized."""
        return self._is_initialized
    
    def get_initialization_progress(self) -> Dict[str, str]:
        """Get current initialization progress."""
        return self._initialization_progress.copy()
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all pipeline components."""
        return {
            'tokenizer': self.tokenizer is not None,
            'embedder': self.embedder is not None,
            'reservoir': self.reservoir is not None,
            'wave_storage': self.wave_storage is not None,
            'first_cnn': self.first_cnn is not None,
            'second_cnn': self.second_cnn is not None,
            'fully_initialized': self._is_initialized
        }
    
    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            if self.wave_storage is not None:
                self.wave_storage.clear_storage()
            
            # Clear component references
            self.tokenizer = None
            self.embedder = None
            self.reservoir = None
            self.first_cnn = None
            self.second_cnn = None
            self.wave_storage = None
            
            self._is_initialized = False
            self._initialization_progress = {}
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error("Error during pipeline cleanup: %s", str(e))
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        status = "initialized" if self._is_initialized else "not initialized"
        return f"DualCNNPipeline(status={status}, config={type(self.config).__name__})"

print("Testing pipeline creation...")

# Test basic creation
config = DualCNNConfig(
    embedder_fit_samples=100,
    embedder_max_length=32,
    reservoir_size=64,
    attention_heads=4,
    attention_dim=16,
    wave_window_size=20,
    wave_overlap=5,  # Must be less than wave_window_size
    wave_feature_dim=64,
    first_cnn_filters=[16, 32],
    second_cnn_filters=[32, 64]
)

pipeline = DualCNNPipeline(config)
print(f"Pipeline created successfully: {pipeline}")
print("All tests passed!")