"""
Demonstration of enhanced error handling and user experience improvements.

This example shows how the enhanced error handling provides better error messages,
graceful degradation, and improved debugging information.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsm_lite.utils.error_handling import (
    LSMError, ConfigurationError, DataValidationError,
    ValidationUtils, global_error_handler
)
from lsm_lite.utils.logging_config import setup_logging, TimedOperation
from lsm_lite.utils.config import DualCNNConfig
from lsm_lite.core.dual_cnn_pipeline import DualCNNPipeline
from lsm_lite.api import LSMLite
import logging


def demonstrate_enhanced_error_messages():
    """Demonstrate enhanced error messages with context and solutions."""
    print("=== Enhanced Error Messages Demo ===")
    
    try:
        # Create invalid configuration
        config = DualCNNConfig()
        config.training_batch_size = -1  # Invalid
        config.attention_heads = 0  # Invalid
        config.learning_rate = 2.0  # Invalid
        
        # This will trigger enhanced error handling
        pipeline = DualCNNPipeline(config)
        
    except ConfigurationError as e:
        print("Caught enhanced configuration error:")
        print(str(e))
        print("\nNotice the detailed context and suggested solutions!")
    
    print("\n" + "="*50 + "\n")


def demonstrate_data_validation():
    """Demonstrate data validation with helpful error messages."""
    print("=== Data Validation Demo ===")
    
    # Test with invalid training data
    invalid_data = ["", "  ", "x", None, 123]  # Various issues
    
    try:
        issues = ValidationUtils.validate_training_data(invalid_data)
        if issues:
            print("Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Try to use this data with LSMLite
        lsm = LSMLite()
        lsm.setup_dual_cnn_pipeline(invalid_data)
        
    except DataValidationError as e:
        print("\nCaught data validation error:")
        print(str(e))
    
    print("\n" + "="*50 + "\n")


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation when components fail."""
    print("=== Graceful Degradation Demo ===")
    
    try:
        # Create a configuration that might cause some components to fail
        config = DualCNNConfig()
        config.max_memory_usage_gb = 0.1  # Very low memory limit
        
        # Use minimal valid training data
        training_data = [
            "This is a simple sentence for testing.",
            "Another sentence to provide some variety.",
            "A third sentence to complete the minimal dataset."
        ]
        
        lsm = LSMLite()
        
        # This should work with fallback modes enabled
        pipeline = lsm.setup_dual_cnn_pipeline(
            training_data=training_data,
            dual_cnn_config=config,
            enable_fallback=True
        )
        
        # Check fallback status
        fallback_status = pipeline.get_fallback_status()
        print("Pipeline fallback status:")
        for key, value in fallback_status.items():
            print(f"  {key}: {value}")
        
        if fallback_status["fallback_mode_enabled"]:
            print("\nPipeline is running in fallback mode - some features may be limited")
            print("but the system continues to function!")
        
    except Exception as e:
        print(f"Error during graceful degradation demo: {e}")
    
    print("\n" + "="*50 + "\n")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and logging."""
    print("=== Performance Monitoring Demo ===")
    
    from lsm_lite.utils.logging_config import PerformanceLogger, TimedOperation
    
    # Setup performance logger
    perf_logger = PerformanceLogger("demo")
    
    # Demonstrate timing operations
    with TimedOperation("data_processing", perf_logger):
        # Simulate some work
        import time
        time.sleep(0.1)
        
        # Log memory usage
        memory_stats = perf_logger.log_memory_usage("during_processing")
        if memory_stats:
            print(f"Memory usage: {memory_stats['memory_mb']:.1f} MB")
    
    # Manual timing
    perf_logger.start_timer("manual_operation")
    time.sleep(0.05)
    duration = perf_logger.end_timer("manual_operation")
    print(f"Manual operation took: {duration:.3f} seconds")
    
    print("\n" + "="*50 + "\n")


def demonstrate_error_recovery():
    """Demonstrate error recovery mechanisms."""
    print("=== Error Recovery Demo ===")
    
    try:
        # Create a scenario that might trigger recovery
        config = DualCNNConfig()
        config.reservoir_size = 2048  # Large size that might cause issues
        config.attention_heads = 16   # Many attention heads
        
        training_data = ["Short text for testing recovery mechanisms."]
        
        lsm = LSMLite()
        
        # This might trigger recovery mechanisms
        pipeline = lsm.setup_dual_cnn_pipeline(
            training_data=training_data,
            dual_cnn_config=config,
            enable_fallback=True
        )
        
        print("Pipeline created successfully (possibly with recovery)")
        
        # Check if any recovery was needed
        fallback_status = pipeline.get_fallback_status()
        if fallback_status.get("initialization_attempts"):
            print("Recovery attempts were made:")
            for component, attempts in fallback_status["initialization_attempts"].items():
                print(f"  {component}: {attempts} attempts")
        
    except Exception as e:
        print(f"Error recovery demo failed: {e}")
        
        # Show that we can still get useful information from the error
        if hasattr(e, 'solutions'):
            print("Available solutions:")
            for i, solution in enumerate(e.solutions, 1):
                print(f"  {i}. {solution.description}")
    
    print("\n" + "="*50 + "\n")


def demonstrate_logging_configuration():
    """Demonstrate enhanced logging configuration."""
    print("=== Enhanced Logging Demo ===")
    
    from lsm_lite.utils.logging_config import setup_logging, get_debug_logger
    
    # Setup enhanced logging
    logging_config = setup_logging(
        level="DEBUG",
        include_context=True,
        enable_performance_logging=True,
        enable_error_tracking=True
    )
    
    print("Enhanced logging configured with:")
    print(f"  - Level: {logging_config['level']}")
    print(f"  - Performance logging: enabled")
    print(f"  - Error tracking: enabled")
    print(f"  - Context information: enabled")
    
    # Get a component-specific logger
    component_logger = get_debug_logger("demo_component")
    
    # Log some messages to show the enhanced formatting
    component_logger.info("This is an info message with component context")
    component_logger.warning("This is a warning message")
    component_logger.debug("This is a debug message with detailed context")
    
    print("\nCheck the console output above to see enhanced log formatting!")
    
    print("\n" + "="*50 + "\n")


def main():
    """Run all demonstrations."""
    print("Enhanced Error Handling and User Experience Demo")
    print("=" * 60)
    print()
    
    # Setup basic logging for the demo
    from lsm_lite.utils.logging_config import setup_logging
    setup_logging(level="INFO", include_context=False)
    
    try:
        demonstrate_enhanced_error_messages()
        demonstrate_data_validation()
        demonstrate_graceful_degradation()
        demonstrate_performance_monitoring()
        demonstrate_error_recovery()
        demonstrate_logging_configuration()
        
        print("Demo completed successfully!")
        print("\nKey improvements demonstrated:")
        print("✓ Enhanced error messages with context and solutions")
        print("✓ Data validation with specific guidance")
        print("✓ Graceful degradation when components fail")
        print("✓ Performance monitoring and timing")
        print("✓ Error recovery mechanisms")
        print("✓ Enhanced logging with component context")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        
        # Even if the demo fails, show that we get enhanced error information
        if hasattr(e, 'context'):
            print(f"Error context: {e.context}")
        if hasattr(e, 'solutions'):
            print(f"Available solutions: {len(e.solutions)}")


if __name__ == "__main__":
    main()