"""
Enhanced error handling and user experience utilities for LSM Lite.

This module provides comprehensive error handling, validation, and user guidance
for the dual CNN pipeline and related components.
"""

import logging
import traceback
import sys
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorizing issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better organization."""
    CONFIGURATION = "configuration"
    DATA = "data"
    MEMORY = "memory"
    COMPUTATION = "computation"
    INITIALIZATION = "initialization"
    TRAINING = "training"
    INFERENCE = "inference"
    STORAGE = "storage"


@dataclass
class ErrorContext:
    """Context information for enhanced error reporting."""
    component: str
    operation: str
    input_shapes: Optional[Dict[str, Any]] = None
    memory_usage: Optional[float] = None
    config_values: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


@dataclass
class ErrorSolution:
    """Suggested solution for an error."""
    description: str
    action_steps: List[str]
    code_example: Optional[str] = None
    documentation_link: Optional[str] = None


class LSMError(Exception):
    """Base exception class for LSM Lite with enhanced error information."""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory,
                 severity: ErrorSeverity,
                 context: Optional[ErrorContext] = None,
                 solutions: Optional[List[ErrorSolution]] = None,
                 original_error: Optional[Exception] = None):
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.solutions = solutions or []
        self.original_error = original_error
        
        # Create enhanced error message
        enhanced_message = self._create_enhanced_message()
        super().__init__(enhanced_message)
    
    def _create_enhanced_message(self) -> str:
        """Create an enhanced error message with context and solutions."""
        lines = [f"[{self.severity.value.upper()}] {self.category.value.title()} Error: {self.message}"]
        
        # Add context information
        if self.context:
            lines.append("\nContext:")
            lines.append(f"  Component: {self.context.component}")
            lines.append(f"  Operation: {self.context.operation}")
            
            if self.context.input_shapes:
                lines.append("  Input Shapes:")
                for name, shape in self.context.input_shapes.items():
                    lines.append(f"    {name}: {shape}")
            
            if self.context.memory_usage:
                lines.append(f"  Memory Usage: {self.context.memory_usage:.2f} MB")
            
            if self.context.config_values:
                lines.append("  Configuration:")
                for key, value in self.context.config_values.items():
                    try:
                        # Safely format the value
                        if isinstance(value, (dict, list)):
                            value_str = str(value)
                        elif hasattr(value, '__dict__'):
                            value_str = f"{type(value).__name__} object"
                        else:
                            value_str = str(value)
                        lines.append(f"    {key}: {value_str}")
                    except Exception:
                        # Fallback if formatting fails
                        lines.append(f"    {key}: <formatting error>")
        
        # Add solutions
        if self.solutions:
            lines.append("\nSuggested Solutions:")
            for i, solution in enumerate(self.solutions, 1):
                lines.append(f"\n{i}. {solution.description}")
                for step in solution.action_steps:
                    lines.append(f"   - {step}")
                
                if solution.code_example:
                    lines.append("   Example:")
                    for line in solution.code_example.split('\n'):
                        lines.append(f"     {line}")
                
                if solution.documentation_link:
                    lines.append(f"   Documentation: {solution.documentation_link}")
        
        # Add original error information
        if self.original_error:
            lines.append(f"\nOriginal Error: {type(self.original_error).__name__}: {str(self.original_error)}")
        
        return '\n'.join(lines)


class ConfigurationError(LSMError):
    """Error related to configuration issues."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, 
                 solutions: Optional[List[ErrorSolution]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, context, solutions, original_error)


class DataValidationError(LSMError):
    """Error related to data validation issues."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 solutions: Optional[List[ErrorSolution]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.DATA, ErrorSeverity.MEDIUM, context, solutions, original_error)


class MemoryError(LSMError):
    """Error related to memory issues."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 solutions: Optional[List[ErrorSolution]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.MEMORY, ErrorSeverity.HIGH, context, solutions, original_error)


class ComputationError(LSMError):
    """Error related to computation issues."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 solutions: Optional[List[ErrorSolution]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM, context, solutions, original_error)


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.error_history = []
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common error patterns and their solutions."""
        return {
            "shape_mismatch": {
                "pattern": ["shape", "dimension", "mismatch", "incompatible"],
                "category": ErrorCategory.COMPUTATION,
                "severity": ErrorSeverity.MEDIUM,
                "solutions": [
                    ErrorSolution(
                        description="Check input tensor shapes and ensure they match expected dimensions",
                        action_steps=[
                            "Verify input data preprocessing",
                            "Check model configuration parameters",
                            "Ensure batch dimensions are consistent",
                            "Validate sequence lengths"
                        ],
                        code_example="""
# Check tensor shapes before processing
print(f"Input shape: {input_tensor.shape}")
print(f"Expected shape: {expected_shape}")

# Reshape if necessary
if input_tensor.shape != expected_shape:
    input_tensor = tf.reshape(input_tensor, expected_shape)
                        """
                    )
                ]
            },
            "memory_exhausted": {
                "pattern": ["memory", "oom", "out of memory", "allocation"],
                "category": ErrorCategory.MEMORY,
                "severity": ErrorSeverity.HIGH,
                "solutions": [
                    ErrorSolution(
                        description="Reduce memory usage by adjusting batch size or model parameters",
                        action_steps=[
                            "Reduce batch size in configuration",
                            "Decrease reservoir size or CNN filters",
                            "Enable gradient checkpointing",
                            "Clear unused variables and call gc.collect()",
                            "Use mixed precision training"
                        ],
                        code_example="""
# Reduce batch size
config.training_batch_size = config.training_batch_size // 2

# Reduce model size
config.reservoir_size = min(config.reservoir_size, 256)
config.first_cnn_filters = [f // 2 for f in config.first_cnn_filters]

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
                        """
                    )
                ]
            },
            "tokenizer_error": {
                "pattern": ["tokenizer", "vocab", "encoding", "decode"],
                "category": ErrorCategory.DATA,
                "severity": ErrorSeverity.MEDIUM,
                "solutions": [
                    ErrorSolution(
                        description="Fix tokenizer configuration or input data format",
                        action_steps=[
                            "Verify tokenizer backend is properly installed",
                            "Check input text encoding (UTF-8)",
                            "Ensure vocabulary file is accessible",
                            "Validate max_length parameter"
                        ],
                        code_example="""
# Reinitialize tokenizer with explicit parameters
tokenizer = UnifiedTokenizer(
    backend='gpt2',
    max_length=512,
    padding=True,
    truncation=True
)

# Test tokenization
test_result = tokenizer.tokenize(["test text"])
print(f"Tokenization successful: {test_result}")
                        """
                    )
                ]
            },
            "wave_storage_error": {
                "pattern": ["wave", "storage", "buffer", "circular"],
                "category": ErrorCategory.STORAGE,
                "severity": ErrorSeverity.MEDIUM,
                "solutions": [
                    ErrorSolution(
                        description="Adjust wave storage parameters or clear storage",
                        action_steps=[
                            "Increase max_memory_mb for wave storage",
                            "Reduce wave_window_size if too large",
                            "Clear storage before new sequences",
                            "Check feature dimension compatibility"
                        ],
                        code_example="""
# Adjust wave storage configuration
config.max_memory_usage_gb = 2.0  # Increase memory limit
config.wave_window_size = 25      # Reduce window size

# Clear storage manually
pipeline.wave_storage.clear_storage()

# Check storage stats
stats = pipeline.wave_storage.get_storage_stats()
print(f"Storage utilization: {stats['utilization_percent']:.1f}%")
                        """
                    )
                ]
            },
            "attention_computation": {
                "pattern": ["attention", "heads", "query", "key", "value"],
                "category": ErrorCategory.COMPUTATION,
                "severity": ErrorSeverity.MEDIUM,
                "solutions": [
                    ErrorSolution(
                        description="Adjust attention mechanism parameters",
                        action_steps=[
                            "Reduce number of attention heads",
                            "Decrease attention dimension",
                            "Check input sequence length",
                            "Verify attention mask if used"
                        ],
                        code_example="""
# Reduce attention complexity
config.attention_heads = min(config.attention_heads, 4)
config.attention_dim = min(config.attention_dim, 32)

# Check attention computation
reservoir = AttentiveReservoir(
    input_dim=config.wave_feature_dim,
    reservoir_size=config.reservoir_size,
    attention_heads=config.attention_heads,
    attention_dim=config.attention_dim
)
                        """
                    )
                ]
            },
            "training_divergence": {
                "pattern": ["loss", "nan", "inf", "diverge", "unstable"],
                "category": ErrorCategory.TRAINING,
                "severity": ErrorSeverity.HIGH,
                "solutions": [
                    ErrorSolution(
                        description="Stabilize training with better parameters and regularization",
                        action_steps=[
                            "Reduce learning rate",
                            "Add gradient clipping",
                            "Increase dropout rate",
                            "Check for data preprocessing issues",
                            "Use learning rate scheduling"
                        ],
                        code_example="""
# Stabilize training parameters
config.learning_rate = 0.0001  # Reduce learning rate
config.first_cnn_dropout_rate = 0.3  # Increase dropout

# Add gradient clipping in optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=config.learning_rate,
    clipnorm=1.0  # Clip gradients
)

# Monitor loss values
if tf.math.is_nan(loss) or tf.math.is_inf(loss):
    print("Warning: Loss is NaN or Inf, stopping training")
                        """
                    )
                ]
            }
        }
    
    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            "reduce_batch_size": self._reduce_batch_size,
            "clear_memory": self._clear_memory,
            "fallback_to_single_cnn": self._fallback_to_single_cnn,
            "reinitialize_component": self._reinitialize_component,
            "adjust_parameters": self._adjust_parameters
        }
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> LSMError:
        """
        Handle an error by analyzing it and providing enhanced information.
        
        Args:
            error: The original exception
            context: Optional context information
            
        Returns:
            Enhanced LSMError with solutions and context
        """
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Find matching error pattern
        matched_pattern = None
        for pattern_name, pattern_info in self.error_patterns.items():
            if any(keyword in error_message for keyword in pattern_info["pattern"]):
                matched_pattern = pattern_info
                break
        
        # Create enhanced error
        if matched_pattern:
            enhanced_error = LSMError(
                message=str(error),
                category=matched_pattern["category"],
                severity=matched_pattern["severity"],
                context=context,
                solutions=matched_pattern["solutions"],
                original_error=error
            )
        else:
            # Generic error handling
            enhanced_error = LSMError(
                message=str(error),
                category=ErrorCategory.COMPUTATION,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                solutions=[
                    ErrorSolution(
                        description="General troubleshooting steps",
                        action_steps=[
                            "Check input data format and shapes",
                            "Verify configuration parameters",
                            "Reduce model complexity if needed",
                            "Check system resources (memory, GPU)"
                        ]
                    )
                ],
                original_error=error
            )
        
        # Log the error
        self._log_error(enhanced_error)
        
        # Store in error history
        self.error_history.append({
            "timestamp": tf.timestamp(),
            "error": enhanced_error,
            "context": context
        })
        
        return enhanced_error
    
    def _log_error(self, error: LSMError):
        """Log error with appropriate level based on severity."""
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(str(error))
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(str(error))
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(str(error))
        else:
            logger.info(str(error))
    
    def attempt_recovery(self, error: LSMError, component: Any, config: Any) -> bool:
        """
        Attempt to recover from an error using available strategies.
        
        Args:
            error: The enhanced error
            component: The component that failed
            config: Configuration object
            
        Returns:
            True if recovery was successful, False otherwise
        """
        recovery_attempted = False
        
        # Try category-specific recovery strategies
        if error.category == ErrorCategory.MEMORY:
            recovery_attempted = self._recover_from_memory_error(component, config)
        elif error.category == ErrorCategory.COMPUTATION:
            recovery_attempted = self._recover_from_computation_error(component, config)
        elif error.category == ErrorCategory.CONFIGURATION:
            recovery_attempted = self._recover_from_config_error(component, config)
        
        if recovery_attempted:
            logger.info(f"Recovery attempted for {error.category.value} error")
        
        return recovery_attempted
    
    def _recover_from_memory_error(self, component: Any, config: Any) -> bool:
        """Attempt recovery from memory errors."""
        try:
            # Reduce batch size
            if hasattr(config, 'training_batch_size'):
                config.training_batch_size = max(1, config.training_batch_size // 2)
                logger.info(f"Reduced batch size to {config.training_batch_size}")
            
            # Clear memory
            self._clear_memory()
            
            return True
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _recover_from_computation_error(self, component: Any, config: Any) -> bool:
        """Attempt recovery from computation errors."""
        try:
            # Reduce model complexity
            if hasattr(config, 'reservoir_size'):
                config.reservoir_size = max(64, config.reservoir_size // 2)
                logger.info(f"Reduced reservoir size to {config.reservoir_size}")
            
            if hasattr(config, 'attention_heads'):
                config.attention_heads = max(1, config.attention_heads // 2)
                logger.info(f"Reduced attention heads to {config.attention_heads}")
            
            return True
        except Exception as e:
            logger.error(f"Computation recovery failed: {e}")
            return False
    
    def _recover_from_config_error(self, component: Any, config: Any) -> bool:
        """Attempt recovery from configuration errors."""
        try:
            # Reset to safe defaults
            if hasattr(config, 'get_safe_defaults'):
                safe_config = config.get_safe_defaults()
                for key, value in safe_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                logger.info("Reset configuration to safe defaults")
            
            return True
        except Exception as e:
            logger.error(f"Configuration recovery failed: {e}")
            return False
    
    # Recovery strategy implementations
    def _reduce_batch_size(self, config: Any) -> bool:
        """Reduce batch size to save memory."""
        if hasattr(config, 'training_batch_size'):
            config.training_batch_size = max(1, config.training_batch_size // 2)
            return True
        return False
    
    def _clear_memory(self) -> bool:
        """Clear GPU and system memory."""
        try:
            import gc
            gc.collect()
            
            # Clear TensorFlow memory
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
            
            return True
        except Exception:
            return False
    
    def _fallback_to_single_cnn(self, component: Any) -> bool:
        """Fallback to single CNN if dual CNN fails."""
        try:
            if hasattr(component, '_use_single_cnn_fallback'):
                component._use_single_cnn_fallback = True
                logger.info("Enabled single CNN fallback mode")
                return True
        except Exception:
            pass
        return False
    
    def _reinitialize_component(self, component: Any, config: Any) -> bool:
        """Reinitialize a component with safer parameters."""
        try:
            if hasattr(component, 'reinitialize'):
                component.reinitialize(config)
                return True
        except Exception:
            pass
        return False
    
    def _adjust_parameters(self, config: Any) -> bool:
        """Adjust parameters to safer values."""
        try:
            # Reduce learning rate
            if hasattr(config, 'learning_rate'):
                config.learning_rate *= 0.5
            
            # Increase regularization
            if hasattr(config, 'first_cnn_dropout_rate'):
                config.first_cnn_dropout_rate = min(0.5, config.first_cnn_dropout_rate + 0.1)
            
            return True
        except Exception:
            return False


class ValidationUtils:
    """Utilities for validating inputs and configurations."""
    
    @staticmethod
    def validate_tensor_shape(tensor: tf.Tensor, expected_shape: tuple, name: str) -> None:
        """
        Validate tensor shape matches expected dimensions.
        
        Args:
            tensor: Input tensor to validate
            expected_shape: Expected shape (use None for variable dimensions)
            name: Name of the tensor for error reporting
            
        Raises:
            DataValidationError: If shape validation fails
        """
        actual_shape = tensor.shape.as_list()
        
        # Check rank
        if len(actual_shape) != len(expected_shape):
            raise DataValidationError(
                f"Tensor '{name}' has wrong rank: expected {len(expected_shape)}, got {len(actual_shape)}",
                context=ErrorContext(
                    component="ValidationUtils",
                    operation="validate_tensor_shape",
                    input_shapes={name: actual_shape}
                ),
                solutions=[
                    ErrorSolution(
                        description="Fix tensor dimensions",
                        action_steps=[
                            "Check data preprocessing pipeline",
                            "Verify model input requirements",
                            "Add or remove dimensions as needed"
                        ]
                    )
                ]
            )
        
        # Check each dimension
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise DataValidationError(
                    f"Tensor '{name}' dimension {i} mismatch: expected {expected}, got {actual}",
                    context=ErrorContext(
                        component="ValidationUtils",
                        operation="validate_tensor_shape",
                        input_shapes={name: actual_shape}
                    ),
                    solutions=[
                        ErrorSolution(
                            description="Fix tensor dimension",
                            action_steps=[
                                f"Reshape tensor to match expected dimension {i}",
                                "Check data loading and preprocessing",
                                "Verify model configuration"
                            ]
                        )
                    ]
                )
    
    @staticmethod
    def validate_config_parameters(config: Any) -> List[str]:
        """
        Validate configuration parameters and return list of issues.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check required attributes
        required_attrs = [
            'embedder_max_length', 'reservoir_size', 'wave_feature_dim',
            'attention_heads', 'attention_dim', 'training_batch_size'
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                issues.append(f"Missing required configuration parameter: {attr}")
            elif getattr(config, attr) is None:
                issues.append(f"Configuration parameter {attr} is None")
        
        # Check parameter ranges
        if hasattr(config, 'training_batch_size') and config.training_batch_size <= 0:
            issues.append("training_batch_size must be positive")
        
        if hasattr(config, 'learning_rate') and (config.learning_rate <= 0 or config.learning_rate > 1):
            issues.append("learning_rate must be between 0 and 1")
        
        if hasattr(config, 'attention_heads') and config.attention_heads <= 0:
            issues.append("attention_heads must be positive")
        
        if hasattr(config, 'reservoir_size') and config.reservoir_size <= 0:
            issues.append("reservoir_size must be positive")
        
        # Check parameter compatibility
        if (hasattr(config, 'attention_heads') and hasattr(config, 'attention_dim') and
            hasattr(config, 'reservoir_size')):
            total_attention_dim = config.attention_heads * config.attention_dim
            if total_attention_dim > config.reservoir_size:
                issues.append(f"Total attention dimension ({total_attention_dim}) "
                            f"exceeds reservoir size ({config.reservoir_size})")
        
        return issues
    
    @staticmethod
    def validate_training_data(data: List[str]) -> List[str]:
        """
        Validate training data format and content.
        
        Args:
            data: List of training text strings
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not data:
            issues.append("Training data is empty")
            return issues
        
        if not isinstance(data, list):
            issues.append("Training data must be a list of strings")
            return issues
        
        # Check data content
        empty_count = 0
        too_short_count = 0
        too_long_count = 0
        
        for i, text in enumerate(data[:100]):  # Sample first 100 items
            if not isinstance(text, str):
                issues.append(f"Item {i} is not a string: {type(text)}")
                continue
            
            if not text.strip():
                empty_count += 1
            elif len(text.split()) < 3:
                too_short_count += 1
            elif len(text.split()) > 1000:
                too_long_count += 1
        
        if empty_count > len(data) * 0.1:
            issues.append(f"Too many empty texts: {empty_count}/{len(data)}")
        
        if too_short_count > len(data) * 0.5:
            issues.append(f"Too many short texts (< 3 words): {too_short_count}")
        
        if too_long_count > len(data) * 0.1:
            issues.append(f"Too many long texts (> 1000 words): {too_long_count}")
        
        return issues


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_lsm_error(func: Callable) -> Callable:
    """
    Decorator for handling LSM errors with enhanced reporting.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Create error context
            context = ErrorContext(
                component=func.__module__ or "unknown",
                operation=func.__name__,
                stack_trace=traceback.format_exc()
            )
            
            # Handle the error
            enhanced_error = global_error_handler.handle_error(e, context)
            
            # Re-raise the enhanced error
            raise enhanced_error
    
    return wrapper