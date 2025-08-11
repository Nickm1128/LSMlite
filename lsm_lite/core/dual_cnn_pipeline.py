"""
Dual CNN Pipeline orchestrator for streamlined dual CNN training workflow.

This module implements the main DualCNNPipeline class that orchestrates the complete
workflow: embedder fitting, attentive reservoir initialization, first CNN for 
next-token prediction with rolling wave output storage, and a second CNN for 
final token prediction generation.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import tensorflow as tf
import numpy as np

# Import dependencies with fallback handling
try:
    from ..utils.config import DualCNNConfig, LSMConfig
except ImportError:
    from lsm_lite.utils.config import DualCNNConfig, LSMConfig

try:
    from ..utils.error_handling import (
        handle_lsm_error, ErrorContext, ConfigurationError, 
        ComputationError, MemoryError, global_error_handler,
        ValidationUtils
    )
except ImportError:
    from lsm_lite.utils.error_handling import (
        handle_lsm_error, ErrorContext, ConfigurationError,
        ComputationError, MemoryError, global_error_handler,
        ValidationUtils
    )

try:
    from .tokenizer import UnifiedTokenizer
except ImportError:
    from lsm_lite.core.tokenizer import UnifiedTokenizer

try:
    from ..data.embeddings import SinusoidalEmbedder
except ImportError:
    from lsm_lite.data.embeddings import SinusoidalEmbedder

try:
    from .attentive_reservoir import AttentiveReservoir
except ImportError:
    from lsm_lite.core.attentive_reservoir import AttentiveReservoir

try:
    from .rolling_wave_storage import RollingWaveStorage, WaveStorageError
except ImportError:
    from lsm_lite.core.rolling_wave_storage import RollingWaveStorage, WaveStorageError

try:
    from .cnn import CNNProcessor
except ImportError:
    from lsm_lite.core.cnn import CNNProcessor

logger = logging.getLogger(__name__)

# Debug print to verify module execution
print("DEBUG: dual_cnn_pipeline.py module loaded successfully")


# Legacy exception classes for backward compatibility
class ComponentInitializationError(ConfigurationError):
    """Exception raised when component initialization fails."""
    
    def __init__(self, component_name: str, error_details: str):
        context = ErrorContext(
            component=component_name,
            operation="initialization"
        )
        super().__init__(f"Failed to initialize {component_name}: {error_details}", context=context)


class DualCNNTrainingError(ComputationError):
    """Exception raised during dual CNN training."""
    
    def __init__(self, stage: str, cnn_id: str, details: str):
        context = ErrorContext(
            component=cnn_id,
            operation=stage
        )
        super().__init__(f"Dual CNN training failed at {stage} for {cnn_id}: {details}", context=context)


class DualCNNPipeline:
    """
    Main orchestrator for the dual CNN workflow.
    
    This class coordinates the complete pipeline: embedder fitting, attentive reservoir
    initialization, first CNN for next-token prediction with rolling wave output storage,
    and a second CNN for final token prediction generation.
    """
    
    @handle_lsm_error
    def __init__(self, config: Union[DualCNNConfig, LSMConfig]):
        """
        Initialize the dual CNN pipeline.
        
        Args:
            config: Configuration object (DualCNNConfig or LSMConfig)
        """
        # Convert LSMConfig to DualCNNConfig if needed
        if isinstance(config, LSMConfig):
            self.config = self._convert_lsm_config(config)
        else:
            self.config = config
        
        # Enhanced configuration validation
        validation_errors = ValidationUtils.validate_config_parameters(self.config)
        if validation_errors:
            context = ErrorContext(
                component="DualCNNPipeline",
                operation="initialization",
                config_values=self._safe_config_dict(self.config)
            )
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(validation_errors)}",
                context=context
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
        self._fallback_mode = False
        self._use_single_cnn_fallback = False
        
        # Error recovery state
        self._initialization_attempts = {}
        self._max_retry_attempts = 3
        
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
    
    @handle_lsm_error
    def fit_and_initialize(self, 
                          training_data: List[str],
                          embedder_params: Optional[Dict[str, Any]] = None,
                          reservoir_params: Optional[Dict[str, Any]] = None,
                          cnn_params: Optional[Dict[str, Any]] = None,
                          progress_callback: Optional[callable] = None,
                          enable_fallback: bool = True) -> None:
        """
        One-shot setup of the entire dual CNN pipeline with enhanced error handling.
        
        This method handles the complete initialization workflow:
        1. Validate training data
        2. Initialize tokenizer with fallback options
        3. Fit embedder to training data
        4. Initialize attentive reservoir with graceful degradation
        5. Set up rolling wave storage with memory management
        6. Initialize first CNN for next-token prediction
        7. Initialize second CNN with fallback to single CNN if needed
        
        Args:
            training_data: List of training text strings
            embedder_params: Optional parameters for embedder fitting
            reservoir_params: Optional parameters for reservoir initialization
            cnn_params: Optional parameters for CNN setup
            progress_callback: Optional callback for progress updates
            enable_fallback: Whether to enable fallback modes on component failures
            
        Raises:
            ConfigurationError: If configuration or data validation fails
            ComponentInitializationError: If critical component initialization fails
        """
        # Validate training data first
        data_issues = ValidationUtils.validate_training_data(training_data)
        if data_issues:
            context = ErrorContext(
                component="DualCNNPipeline",
                operation="data_validation",
                config_values={"data_size": len(training_data)}
            )
            raise ConfigurationError(
                f"Training data validation failed: {'; '.join(data_issues)}",
                context=context
            )
        
        logger.info("Starting dual CNN pipeline initialization with %d training samples...", len(training_data))
        self._initialization_progress = {}
        self._fallback_mode = False
        
        initialization_steps = [
            ("tokenizer", self._initialize_tokenizer_with_fallback, [training_data]),
            ("embedder", self._fit_embedder_with_validation, [training_data, embedder_params or {}]),
            ("reservoir", self._initialize_reservoir_with_fallback, [reservoir_params or {}]),
            ("wave_storage", self._initialize_wave_storage_with_monitoring, []),
            ("first_cnn", self._initialize_first_cnn_with_validation, [cnn_params or {}]),
            ("second_cnn", self._initialize_second_cnn_with_fallback, [cnn_params or {}, enable_fallback])
        ]
        
        for step_name, step_func, step_args in initialization_steps:
            try:
                self._update_progress(step_name, "initializing", progress_callback)
                step_func(*step_args)
                self._update_progress(step_name, "completed", progress_callback)
                logger.debug(f"Successfully initialized {step_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {step_name}: {str(e)}")
                
                # Attempt recovery if enabled
                if enable_fallback and self._attempt_component_recovery(step_name, e):
                    self._update_progress(step_name, "completed_with_fallback", progress_callback)
                    logger.warning(f"Initialized {step_name} with fallback mode")
                    continue
                
                # Critical failure - cannot continue
                context = ErrorContext(
                    component="DualCNNPipeline",
                    operation=f"initialize_{step_name}",
                    config_values=self._get_relevant_config_for_step(step_name)
                )
                
                enhanced_error = global_error_handler.handle_error(e, context)
                
                # Try automatic recovery
                if global_error_handler.attempt_recovery(enhanced_error, self, self.config):
                    logger.info(f"Automatic recovery successful for {step_name}, retrying...")
                    try:
                        step_func(*step_args)
                        self._update_progress(step_name, "completed_after_recovery", progress_callback)
                        continue
                    except Exception as retry_error:
                        logger.error(f"Recovery attempt failed for {step_name}: {retry_error}")
                
                # Final failure
                raise ComponentInitializationError(
                    step_name, 
                    f"Initialization failed after recovery attempts: {str(e)}"
                )
        
        self._is_initialized = True
        
        # Log final status
        if self._fallback_mode:
            logger.warning("Dual CNN pipeline initialized with fallback modes enabled")
        else:
            logger.info("Dual CNN pipeline initialization completed successfully")
        
        # Validate final pipeline state
        self._validate_pipeline_state()
    
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
                backend='basic',  # Use reliable basic tokenizer
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
                tokens = result['input_ids'][0].tolist()  # Get first (and only) sequence
                tokenized_data.extend(tokens)
            
            # Initialize embedder
            vocab_size = self.tokenizer.vocab_size
            self.embedder = SinusoidalEmbedder(
                vocab_size=vocab_size,
                embedding_dim=self.config.wave_feature_dim,
                max_length=self.config.embedder_max_length,
                **embedder_params
            )
            
            # Note: SinusoidalEmbedder doesn't need fitting - it's based on fixed sinusoidal patterns
            
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
            # Calculate memory limit for wave storage
            max_memory_mb = (self.config.max_memory_usage_gb * 1024 * 0.1)  # 10% of total memory
            
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
            
            # Calculate input shape for CNN based on reservoir output
            # Assuming reservoir outputs (batch, sequence, reservoir_size)
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
            
            # Calculate input shape for second CNN based on wave storage
            # Wave storage provides (window_size, feature_dim)
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
    
    def _initialize_tokenizer_with_fallback(self, training_data: List[str]) -> None:
        """Initialize tokenizer with fallback options."""
        backends_to_try = ['gpt2', 'bert-base-uncased', 'basic']
        
        for backend in backends_to_try:
            try:
                self.tokenizer = UnifiedTokenizer(
                    backend=backend,
                    max_length=self.config.embedder_max_length
                )
                logger.info(f"Tokenizer initialized successfully with backend: {backend}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer with backend {backend}: {e}")
                continue
        
        raise ComponentInitializationError("tokenizer", "All tokenizer backends failed")
    
    def _fit_embedder_with_validation(self, training_data: List[str], embedder_params: Dict[str, Any]) -> None:
        """Fit embedder with enhanced validation and error handling."""
        try:
            # Sample data for embedder fitting with validation
            sample_size = min(len(training_data), self.config.embedder_fit_samples)
            sample_data = training_data[:sample_size]
            
            # Validate sample data
            if not sample_data:
                raise ValueError("No valid training data for embedder fitting")
            
            # Tokenize sample data with error handling
            tokenized_data = []
            failed_tokenizations = 0
            
            for text in sample_data:
                try:
                    result = self.tokenizer.tokenize(text, padding=False, truncation=True)
                    tokens = result['input_ids'][0].tolist()
                    tokenized_data.extend(tokens)
                except Exception as e:
                    failed_tokenizations += 1
                    logger.debug(f"Failed to tokenize text: {e}")
                    continue
            
            if failed_tokenizations > len(sample_data) * 0.5:
                logger.warning(f"High tokenization failure rate: {failed_tokenizations}/{len(sample_data)}")
            
            if not tokenized_data:
                raise ValueError("No tokens generated from training data")
            
            # Initialize embedder with validation
            vocab_size = self.tokenizer.vocab_size
            if vocab_size is None or vocab_size <= 0:
                vocab_size = max(max(tokenized_data) + 1, 10000)  # Fallback vocab size
                logger.warning(f"Using fallback vocab size: {vocab_size}")
            
            self.embedder = SinusoidalEmbedder(
                vocab_size=vocab_size,
                embedding_dim=self.config.wave_feature_dim,
                max_length=self.config.embedder_max_length,
                **embedder_params
            )
            
            logger.info(f"Embedder fitted successfully on {sample_size} samples, vocab_size={vocab_size}")
            
        except Exception as e:
            context = ErrorContext(
                component="embedder",
                operation="fitting",
                config_values={
                    "sample_size": len(training_data),
                    "vocab_size": getattr(self.tokenizer, 'vocab_size', None),
                    "embedding_dim": self.config.wave_feature_dim
                }
            )
            raise ComponentInitializationError("embedder", str(e))
    
    def _initialize_reservoir_with_fallback(self, reservoir_params: Dict[str, Any]) -> None:
        """Initialize reservoir with fallback to standard reservoir if attention fails."""
        try:
            # Try attentive reservoir first
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
            logger.warning(f"Attentive reservoir failed, falling back to standard reservoir: {e}")
            
            try:
                # Fallback to standard reservoir
                from .reservoir import SparseReservoir
                self.reservoir = SparseReservoir(
                    input_dim=self.config.wave_feature_dim,
                    reservoir_size=self.config.reservoir_size,
                    sparsity=self.config.reservoir_sparsity,
                    spectral_radius=self.config.reservoir_spectral_radius,
                    leak_rate=self.config.reservoir_leak_rate,
                    **reservoir_params
                )
                self._fallback_mode = True
                logger.info("Standard reservoir initialized as fallback")
                
            except Exception as fallback_error:
                raise ComponentInitializationError("reservoir", str(fallback_error))
    
    def _initialize_wave_storage_with_monitoring(self) -> None:
        """Initialize wave storage with memory monitoring and adaptive sizing."""
        try:
            # Calculate conservative memory limit
            max_memory_mb = (self.config.max_memory_usage_gb * 1024 * 0.05)  # 5% of total memory
            
            # Adaptive sizing based on available memory
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_memory_gb < 2.0:  # Less than 2GB available
                max_memory_mb = min(max_memory_mb, 50.0)  # Limit to 50MB
                logger.warning("Low memory detected, reducing wave storage size")
            
            self.wave_storage = RollingWaveStorage(
                max_sequence_length=self.config.max_wave_storage,
                feature_dim=self.config.wave_feature_dim,
                window_size=self.config.wave_window_size,
                overlap=self.config.wave_overlap,
                max_memory_mb=max_memory_mb
            )
            
            # Test storage functionality
            test_wave = tf.zeros((self.config.wave_feature_dim,))
            self.wave_storage.store_wave(test_wave, 0)
            self.wave_storage.clear_storage()
            
            logger.info(f"Rolling wave storage initialized with {max_memory_mb:.1f}MB limit")
            
        except Exception as e:
            context = ErrorContext(
                component="wave_storage",
                operation="initialization",
                memory_usage=max_memory_mb,
                config_values={
                    "feature_dim": self.config.wave_feature_dim,
                    "window_size": self.config.wave_window_size
                }
            )
            raise ComponentInitializationError("wave_storage", str(e))
    
    def _initialize_first_cnn_with_validation(self, cnn_params: Dict[str, Any]) -> None:
        """Initialize first CNN with input validation."""
        try:
            vocab_size = self.tokenizer.vocab_size
            if vocab_size is None or vocab_size <= 0:
                raise ValueError("Invalid vocabulary size from tokenizer")
            
            # Validate input shape calculation
            input_shape = (self.config.embedder_max_length, self.config.reservoir_size)
            
            # Check if input shape is reasonable
            total_params = np.prod(input_shape) * len(self.config.first_cnn_filters)
            if total_params > 10**8:  # 100M parameters
                logger.warning(f"Large model detected ({total_params:,} parameters), consider reducing size")
            
            self.first_cnn = CNNProcessor(
                input_shape=input_shape,
                architecture=self.config.first_cnn_architecture,
                filters=self.config.first_cnn_filters,
                vocab_size=vocab_size,
                dropout_rate=self.config.first_cnn_dropout_rate,
                name="first_cnn",
                **cnn_params
            )
            
            logger.info(f"First CNN initialized successfully with input shape {input_shape}")
            
        except Exception as e:
            context = ErrorContext(
                component="first_cnn",
                operation="initialization",
                input_shapes={"cnn_input": input_shape},
                config_values={
                    "vocab_size": vocab_size,
                    "filters": self.config.first_cnn_filters
                }
            )
            raise ComponentInitializationError("first_cnn", str(e))
    
    def _initialize_second_cnn_with_fallback(self, cnn_params: Dict[str, Any], enable_fallback: bool) -> None:
        """Initialize second CNN with fallback to single CNN mode."""
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
            
            logger.info(f"Second CNN initialized successfully with input shape {input_shape}")
            
        except Exception as e:
            if enable_fallback:
                logger.warning(f"Second CNN initialization failed, enabling single CNN fallback: {e}")
                self.second_cnn = None
                self._use_single_cnn_fallback = True
                self._fallback_mode = True
            else:
                context = ErrorContext(
                    component="second_cnn",
                    operation="initialization",
                    input_shapes={"cnn_input": input_shape},
                    config_values={
                        "vocab_size": vocab_size,
                        "filters": self.config.second_cnn_filters
                    }
                )
                raise ComponentInitializationError("second_cnn", str(e))
    
    def _attempt_component_recovery(self, component_name: str, error: Exception) -> bool:
        """Attempt to recover from component initialization failure."""
        recovery_attempts = self._initialization_attempts.get(component_name, 0)
        
        if recovery_attempts >= self._max_retry_attempts:
            logger.error(f"Maximum recovery attempts reached for {component_name}")
            return False
        
        self._initialization_attempts[component_name] = recovery_attempts + 1
        
        # Component-specific recovery strategies
        if component_name == "reservoir":
            # Reduce complexity for reservoir
            self.config.attention_heads = max(1, self.config.attention_heads // 2)
            self.config.reservoir_size = max(64, self.config.reservoir_size // 2)
            logger.info(f"Reduced reservoir complexity for recovery attempt {recovery_attempts + 1}")
            return True
        
        elif component_name == "wave_storage":
            # Reduce memory usage for wave storage
            self.config.max_memory_usage_gb = max(0.5, self.config.max_memory_usage_gb / 2)
            self.config.wave_window_size = max(10, self.config.wave_window_size // 2)
            logger.info(f"Reduced wave storage size for recovery attempt {recovery_attempts + 1}")
            return True
        
        elif component_name in ["first_cnn", "second_cnn"]:
            # Reduce CNN complexity
            self.config.first_cnn_filters = [f // 2 for f in self.config.first_cnn_filters]
            self.config.second_cnn_filters = [f // 2 for f in self.config.second_cnn_filters]
            logger.info(f"Reduced CNN complexity for recovery attempt {recovery_attempts + 1}")
            return True
        
        return False
    
    def _get_relevant_config_for_step(self, step_name: str) -> Dict[str, Any]:
        """Get relevant configuration values for a specific initialization step."""
        config_dict = self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        
        relevant_configs = {
            "tokenizer": ["embedder_max_length"],
            "embedder": ["wave_feature_dim", "embedder_max_length", "embedder_fit_samples"],
            "reservoir": ["reservoir_size", "attention_heads", "attention_dim", "reservoir_sparsity"],
            "wave_storage": ["wave_window_size", "wave_overlap", "max_wave_storage", "max_memory_usage_gb"],
            "first_cnn": ["first_cnn_filters", "first_cnn_architecture", "first_cnn_dropout_rate"],
            "second_cnn": ["second_cnn_filters", "second_cnn_architecture", "second_cnn_dropout_rate"]
        }
        
        relevant_keys = relevant_configs.get(step_name, [])
        return {key: config_dict.get(key) for key in relevant_keys if key in config_dict}
    
    def _validate_pipeline_state(self) -> None:
        """Validate the final state of the initialized pipeline."""
        issues = []
        
        if self.tokenizer is None:
            issues.append("Tokenizer not initialized")
        
        if self.embedder is None:
            issues.append("Embedder not initialized")
        
        if self.reservoir is None:
            issues.append("Reservoir not initialized")
        
        if self.first_cnn is None:
            issues.append("First CNN not initialized")
        
        if self.wave_storage is None:
            issues.append("Wave storage not initialized")
        
        # Second CNN is optional in fallback mode
        if self.second_cnn is None and not self._use_single_cnn_fallback:
            issues.append("Second CNN not initialized and fallback not enabled")
        
        if issues:
            raise ComponentInitializationError("pipeline_validation", f"Pipeline validation failed: {'; '.join(issues)}")
        
        logger.info("Pipeline state validation completed successfully")
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get information about fallback modes and degraded functionality."""
        return {
            "fallback_mode_enabled": self._fallback_mode,
            "single_cnn_fallback": self._use_single_cnn_fallback,
            "attentive_reservoir_available": isinstance(self.reservoir, AttentiveReservoir),
            "second_cnn_available": self.second_cnn is not None,
            "initialization_attempts": self._initialization_attempts.copy()
        }
    
    def cleanup(self) -> None:
        """Clean up pipeline resources with enhanced error handling."""
        cleanup_errors = []
        
        try:
            if self.wave_storage is not None:
                self.wave_storage.clear_storage()
        except Exception as e:
            cleanup_errors.append(f"Wave storage cleanup failed: {e}")
        
        try:
            # Clear TensorFlow memory
            tf.keras.backend.clear_session()
        except Exception as e:
            cleanup_errors.append(f"TensorFlow cleanup failed: {e}")
        
        # Clear component references
        self.tokenizer = None
        self.embedder = None
        self.reservoir = None
        self.first_cnn = None
        self.second_cnn = None
        self.wave_storage = None
        
        # Reset state
        self._is_initialized = False
        self._initialization_progress = {}
        self._fallback_mode = False
        self._use_single_cnn_fallback = False
        self._initialization_attempts = {}
        
        if cleanup_errors:
            logger.warning("Cleanup completed with errors: %s", "; ".join(cleanup_errors))
        else:
            logger.info("Pipeline cleanup completed successfully")
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        status = "initialized" if self._is_initialized else "not initialized"
        return f"DualCNNPipeline(status={status}, config={type(self.config).__name__})"  
  
    def _safe_config_dict(self, config) -> Dict[str, Any]:
        """Safely extract configuration values for error reporting."""
        safe_dict = {}
        if not hasattr(config, '__dict__'):
            return safe_dict
        
        for key, value in config.__dict__.items():
            try:
                # Only include simple types that can be safely formatted
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_dict[key] = value
                elif isinstance(value, (list, tuple)):
                    # Include lists/tuples of simple types
                    if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                        safe_dict[key] = value
                    else:
                        safe_dict[key] = f"<{type(value).__name__} with {len(value)} items>"
                elif isinstance(value, dict):
                    safe_dict[key] = f"<dict with {len(value)} keys>"
                else:
                    safe_dict[key] = f"<{type(value).__name__} object>"
            except Exception:
                safe_dict[key] = "<formatting error>"
        
        return safe_dict