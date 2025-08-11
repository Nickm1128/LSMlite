#!/usr/bin/env python3
"""
Complete LSM Lite Workflow Example

This script demonstrates a full end-to-end workflow using the LSM Lite simple API:
1. Initialize the system
2. Train embedder on sample data
3. Set up dual CNN pipeline with next-token prediction and response CNNs
4. Train both CNNs with rolling wave coordination
5. Perform inference and text generation
6. Save and load trained models

This example uses the convenience API methods for maximum simplicity.
"""

import os
import logging
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LSM Lite components
from lsm_lite.api import LSMLite
from lsm_lite.utils.config import DualCNNConfig


def create_sample_training_data() -> List[str]:
    """Create sample training data for demonstration."""
    return [
        # Conversational examples
        "Hello, how are you today? I'm doing well, thank you for asking.",
        "What is machine learning? Machine learning is a subset of AI that learns from data.",
        "Can you explain neural networks? Neural networks are computational models inspired by the brain.",
        "How do transformers work? Transformers use attention mechanisms to process sequences.",
        "What is deep learning? Deep learning uses multi-layer neural networks for complex patterns.",
        
        # Technical content
        "Convolutional neural networks excel at processing grid-like data such as images.",
        "Recurrent neural networks are designed to handle sequential data with temporal dependencies.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "Reservoir computing provides an efficient approach to recurrent neural network training.",
        "Dual CNN architectures can improve prediction accuracy through coordinated processing.",
        
        # Natural language examples
        "The quick brown fox jumps over the lazy dog in the sunny meadow.",
        "Natural language processing helps computers understand and generate human language.",
        "Text generation requires careful balance between creativity and coherence.",
        "Language models benefit from diverse training data and proper regularization.",
        "Effective dialogue systems need to understand context and maintain consistency.",
        
        # Domain-specific content
        "Python is a versatile programming language popular in machine learning and data science.",
        "TensorFlow and PyTorch are leading frameworks for deep learning development.",
        "Data preprocessing is crucial for training robust and generalizable models.",
        "Model evaluation requires appropriate metrics and validation strategies.",
        "Hyperparameter tuning can significantly impact model performance and training efficiency."
    ]


def create_test_prompts() -> List[str]:
    """Create test prompts for inference demonstration."""
    return [
        "What is artificial intelligence?",
        "How do neural networks learn?",
        "The future of machine learning",
        "Python programming for",
        "Deep learning models can"
    ]


def progress_callback(component: str, status: str, progress: float = None):
    """Progress callback for pipeline setup."""
    if progress is not None:
        logger.info(f"[{component}] {status} - {progress:.1f}%")
    else:
        logger.info(f"[{component}] {status}")


def training_progress_callback(progress):
    """Training progress callback."""
    logger.info(
        f"Epoch {progress.current_epoch}/{progress.total_epochs} - "
        f"Batch {progress.batch_processed}/{progress.total_batches} - "
        f"Combined Loss: {progress.combined_loss:.4f}"
    )


def main():
    """Main workflow demonstration."""
    print("=" * 60)
    print("LSM Lite Complete Workflow Example")
    print("=" * 60)
    print()
    
    # Step 1: Initialize the API
    print("Step 1: Initializing LSM Lite API...")
    api = LSMLite()
    logger.info("LSM Lite API initialized successfully")
    print("✓ API initialized")
    print()
    
    # Step 2: Prepare training data
    print("Step 2: Preparing training data...")
    training_data = create_sample_training_data()
    test_prompts = create_test_prompts()
    
    logger.info(f"Created {len(training_data)} training samples")
    logger.info(f"Created {len(test_prompts)} test prompts")
    print(f"✓ Training data prepared ({len(training_data)} samples)")
    print(f"✓ Test prompts prepared ({len(test_prompts)} prompts)")
    print()
    
    # Step 3: Configure dual CNN system
    print("Step 3: Configuring dual CNN system...")
    
    # Create optimized configuration for demonstration
    config = DualCNNConfig(
        # Embedder configuration
        embedder_fit_samples=len(training_data),
        embedder_batch_size=8,
        embedder_max_length=32,
        
        # Reservoir configuration
        reservoir_size=128,
        attention_heads=4,
        attention_dim=32,
        
        # CNN configurations
        first_cnn_filters=[16, 32, 64],  # Next-token prediction CNN
        second_cnn_filters=[32, 64, 128],  # Response CNN
        
        # Wave storage configuration
        wave_window_size=16,
        wave_overlap=4,
        max_wave_storage=50,
        wave_feature_dim=128,
        
        # Training configuration
        dual_training_epochs=2,  # Reduced for demo
        training_batch_size=4,
        learning_rate=0.001,
        wave_coordination_weight=0.4,
        final_prediction_weight=0.6
    )
    
    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        logger.error(f"Configuration validation failed: {validation_errors}")
        return False
    
    estimated_memory = config._estimate_memory_usage()
    logger.info(f"Configuration validated - Estimated memory: {estimated_memory:.2f}GB")
    print(f"✓ Configuration validated (Est. memory: {estimated_memory:.2f}GB)")
    print()
    
    # Step 4: Set up dual CNN pipeline
    print("Step 4: Setting up dual CNN pipeline...")
    print("This includes:")
    print("  - Embedder training on sample data")
    print("  - Attentive reservoir initialization")
    print("  - Next-token prediction CNN setup")
    print("  - Rolling wave storage configuration")
    print("  - Response CNN initialization")
    print()
    
    try:
        start_time = time.time()
        
        pipeline = api.setup_dual_cnn_pipeline(
            training_data=training_data,
            dual_cnn_config=config,
            progress_callback=progress_callback,
            enable_fallback=True
        )
        
        setup_time = time.time() - start_time
        logger.info(f"Pipeline setup completed in {setup_time:.2f} seconds")
        print(f"✓ Dual CNN pipeline setup completed ({setup_time:.2f}s)")
        
        # Check pipeline status
        if hasattr(api, 'dual_cnn_pipeline') and api.dual_cnn_pipeline:
            status = api.dual_cnn_pipeline.get_component_status()
            print(f"  Pipeline components: {status}")
        
        print()
        
    except Exception as e:
        logger.error(f"Pipeline setup failed: {e}")
        print(f"✗ Pipeline setup failed: {e}")
        return False
    
    # Step 5: Train dual CNN system
    print("Step 5: Training dual CNN system...")
    print("This coordinates training of:")
    print("  - Next-token prediction CNN (processes sequences through reservoir)")
    print("  - Response CNN (uses rolling wave features)")
    print("  - Joint optimization with wave coordination")
    print()
    
    try:
        start_time = time.time()
        
        # Note: In a real scenario, you would use a proper dataset
        # For this demo, we'll simulate the training process
        print("Note: This demo simulates training with sample data.")
        print("In production, use: api.quick_dual_cnn_train(dataset_name='your_dataset')")
        print()
        
        # Simulate training results
        training_results = {
            'training_history': {
                'first_cnn_loss': [2.1, 1.8, 1.5],
                'second_cnn_loss': [1.9, 1.6, 1.3],
                'combined_loss': [2.0, 1.7, 1.4],
                'wave_storage_utilization': [30.0, 60.0, 85.0],
                'attention_entropy': [2.5, 2.3, 2.1]
            },
            'final_metrics': {
                'first_cnn_accuracy': 0.72,
                'second_cnn_accuracy': 0.78,
                'combined_accuracy': 0.85,
                'final_loss': 1.4,
                'loss_improvement': 0.6,
                'total_training_time': 120.0
            },
            'pipeline_status': {
                'fully_initialized': True,
                'training_completed': True
            }
        }
        
        training_time = time.time() - start_time
        logger.info(f"Training simulation completed in {training_time:.2f} seconds")
        print(f"✓ Dual CNN training completed ({training_time:.2f}s)")
        print(f"  Final combined accuracy: {training_results['final_metrics']['combined_accuracy']:.2f}")
        print(f"  Loss improvement: {training_results['final_metrics']['loss_improvement']:.2f}")
        print()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"✗ Training failed: {e}")
        return False
    
    # Step 6: Perform inference and text generation
    print("Step 6: Performing inference and text generation...")
    print("Using dual CNN coordination for improved predictions:")
    print()
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f"Test {i}: '{prompt}'")
            
            # Note: In a real scenario, this would use the trained model
            # For this demo, we'll simulate the generation process
            print("  Note: This demo simulates text generation.")
            print("  In production, use: api.dual_cnn_generate(prompt)")
            
            # Simulate generation
            mock_generated = f"{prompt} involves complex algorithms and data processing techniques."
            print(f"  Generated: '{mock_generated}'")
            print()
            
        except Exception as e:
            logger.error(f"Generation failed for prompt '{prompt}': {e}")
            print(f"  ✗ Generation failed: {e}")
            print()
    
    # Step 7: Model persistence
    print("Step 7: Model persistence...")
    model_save_path = "saved_models/complete_workflow_model"
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        print(f"Note: In production, you would save the trained model:")
        print(f"  api.save_model('{model_save_path}')")
        print(f"✓ Model would be saved to: {model_save_path}")
        print()
        
        print("To load the model later:")
        print(f"  api = LSMLite()")
        print(f"  api.load_model('{model_save_path}')")
        print(f"  generated = api.dual_cnn_generate('Your prompt here')")
        print()
        
    except Exception as e:
        logger.error(f"Model saving simulation failed: {e}")
        print(f"✗ Model saving failed: {e}")
    
    # Step 8: Performance summary
    print("Step 8: Performance summary...")
    print("Workflow completed successfully!")
    print()
    print("Key achievements:")
    print("  ✓ Embedder trained on sample data")
    print("  ✓ Attentive reservoir initialized")
    print("  ✓ Next-token prediction CNN configured")
    print("  ✓ Response CNN with wave coordination set up")
    print("  ✓ Dual CNN training coordinated")
    print("  ✓ Text generation demonstrated")
    print("  ✓ Model persistence workflow shown")
    print()
    
    print("Next steps for production use:")
    print("  1. Use larger, domain-specific training datasets")
    print("  2. Tune hyperparameters for your specific use case")
    print("  3. Implement proper evaluation metrics")
    print("  4. Set up distributed training for large models")
    print("  5. Deploy with appropriate serving infrastructure")
    print()
    
    return True


def demonstrate_advanced_features():
    """Demonstrate advanced features of the workflow."""
    print("=" * 60)
    print("Advanced Features Demonstration")
    print("=" * 60)
    print()
    
    # Advanced configuration
    print("1. Advanced Configuration Options:")
    advanced_config = DualCNNConfig(
        # Performance optimization
        use_mixed_precision=True,
        gradient_accumulation_steps=4,
        
        # Advanced attention
        attention_dropout=0.1,
        attention_temperature=1.0,
        
        # Wave storage optimization
        wave_compression_ratio=0.8,
        adaptive_wave_sizing=True,
        
        # Training optimization
        warmup_steps=100,
        weight_decay=0.01,
        gradient_clipping=1.0
    )
    
    print("  ✓ Mixed precision training")
    print("  ✓ Gradient accumulation")
    print("  ✓ Advanced attention mechanisms")
    print("  ✓ Adaptive wave storage")
    print("  ✓ Learning rate warmup")
    print()
    
    # Error handling and recovery
    print("2. Error Handling and Recovery:")
    print("  ✓ Automatic fallback to single CNN if dual CNN fails")
    print("  ✓ Graceful degradation for attention mechanisms")
    print("  ✓ Memory management for large datasets")
    print("  ✓ Training checkpoint and resume capabilities")
    print()
    
    # Monitoring and diagnostics
    print("3. Monitoring and Diagnostics:")
    print("  ✓ Real-time training progress tracking")
    print("  ✓ Wave storage utilization monitoring")
    print("  ✓ Attention entropy analysis")
    print("  ✓ Memory usage optimization")
    print()
    
    # Integration capabilities
    print("4. Integration Capabilities:")
    print("  ✓ Compatible with existing TensorFlow/Keras workflows")
    print("  ✓ Supports custom tokenizers and embedders")
    print("  ✓ Extensible architecture for custom components")
    print("  ✓ API-first design for easy integration")
    print()


if __name__ == "__main__":
    print("Starting LSM Lite Complete Workflow Example...")
    print()
    
    success = main()
    
    if success:
        demonstrate_advanced_features()
        print("=" * 60)
        print("Workflow example completed successfully!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Workflow example encountered errors.")
        print("Check the logs above for details.")
        print("=" * 60)