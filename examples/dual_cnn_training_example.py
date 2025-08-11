"""
Example usage of DualCNNTrainer for coordinated dual CNN training.

This example demonstrates how to use the DualCNNTrainer class to train
both CNNs with rolling wave coordination, including progress tracking
and metrics collection.
"""

import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required components
from lsm_lite.utils.config import DualCNNConfig
from lsm_lite.core.dual_cnn_pipeline import DualCNNPipeline
from lsm_lite.training.dual_cnn_trainer import DualCNNTrainer, TrainingProgress


def progress_callback(progress: TrainingProgress):
    """Example progress callback function."""
    print(f"Epoch {progress.current_epoch}/{progress.total_epochs} - "
          f"Batch {progress.batch_processed}/{progress.total_batches}")
    print(f"  Combined Loss: {progress.combined_loss:.4f}")
    print(f"  First CNN Loss: {progress.first_cnn_loss:.4f}")
    print(f"  Second CNN Loss: {progress.second_cnn_loss:.4f}")
    print(f"  Wave Storage Utilization: {progress.wave_storage_utilization:.1f}%")
    print(f"  Attention Entropy: {progress.attention_entropy:.3f}")
    print(f"  Learning Rate: {progress.learning_rate:.6f}")
    if progress.estimated_time_remaining > 0:
        print(f"  Estimated Time Remaining: {progress.estimated_time_remaining:.1f}s")
    print()


def main():
    """Main example function."""
    print("=== Dual CNN Training Example ===\n")
    
    # 1. Create configuration
    print("1. Creating dual CNN configuration...")
    config = DualCNNConfig(
        # Embedder settings
        embedder_fit_samples=1000,
        embedder_batch_size=32,
        embedder_max_length=64,
        
        # Reservoir settings
        reservoir_size=256,
        attention_heads=8,
        attention_dim=32,
        
        # Wave storage settings
        wave_window_size=20,
        wave_overlap=5,
        max_wave_storage=100,
        wave_feature_dim=256,
        
        # Training settings
        dual_training_epochs=3,
        training_batch_size=16,
        learning_rate=0.001,
        wave_coordination_weight=0.3,
        final_prediction_weight=0.7
    )
    
    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        print(f"Configuration validation errors: {validation_errors}")
        return
    
    print(f"✓ Configuration created and validated")
    print(f"  Estimated memory usage: {config._estimate_memory_usage():.2f}GB")
    print()
    
    # 2. Create sample training data
    print("2. Creating sample training data...")
    training_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Transformers have revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant parts.",
        "Convolutional neural networks are great for image processing.",
        "Recurrent neural networks can handle sequential data.",
        "Reservoir computing is an efficient approach to neural networks.",
        "Dual CNN architectures can improve prediction accuracy."
    ]
    
    validation_data = [
        "Artificial intelligence is transforming many industries.",
        "Neural networks learn patterns from training data."
    ]
    
    print(f"✓ Created {len(training_data)} training samples")
    print(f"✓ Created {len(validation_data)} validation samples")
    print()
    
    # 3. Initialize pipeline (this would normally be done with real components)
    print("3. Initializing dual CNN pipeline...")
    print("   Note: This example shows the training interface.")
    print("   In practice, you would initialize a real DualCNNPipeline with:")
    print("   - pipeline = DualCNNPipeline(config)")
    print("   - pipeline.fit_and_initialize(training_data)")
    print()
    
    # 4. Create and configure trainer (mock example)
    print("4. Creating dual CNN trainer...")
    print("   Note: This would normally be:")
    print("   - trainer = DualCNNTrainer(pipeline, config)")
    print("   - trainer.add_progress_callback(progress_callback)")
    print()
    
    # 5. Show training interface
    print("5. Training interface example:")
    print("   The trainer would be used like this:")
    print()
    print("   # Start training with progress tracking")
    print("   results = trainer.train_dual_cnn(")
    print("       training_data=training_data,")
    print("       validation_data=validation_data,")
    print("       epochs=config.dual_training_epochs,")
    print("       batch_size=config.training_batch_size")
    print("   )")
    print()
    print("   # Access training results")
    print("   training_history = results['training_history']")
    print("   final_metrics = results['final_metrics']")
    print("   pipeline_status = results['pipeline_status']")
    print()
    
    # 6. Show expected results structure
    print("6. Expected training results structure:")
    expected_results = {
        'training_history': {
            'first_cnn_loss': [2.5, 2.1, 1.8],
            'second_cnn_loss': [2.3, 1.9, 1.6],
            'combined_loss': [2.4, 2.0, 1.7],
            'wave_storage_utilization': [25.0, 50.0, 75.0],
            'attention_entropy': [2.8, 2.6, 2.4],
            'learning_rate': [0.001, 0.001, 0.001],
            'epoch_times': [45.2, 43.8, 42.1]
        },
        'final_metrics': {
            'first_cnn_accuracy': 0.75,
            'second_cnn_accuracy': 0.78,
            'combined_accuracy': 0.82,
            'initial_loss': 2.4,
            'final_loss': 1.7,
            'loss_improvement': 0.7,
            'final_wave_utilization': 75.0,
            'avg_attention_entropy': 2.6,
            'total_training_time': 131.1,
            'epochs_completed': 3,
            'avg_epoch_time': 43.7
        },
        'pipeline_status': {
            'tokenizer': True,
            'embedder': True,
            'reservoir': True,
            'wave_storage': True,
            'first_cnn': True,
            'second_cnn': True,
            'fully_initialized': True
        }
    }
    
    print("   Training History:")
    for key, values in expected_results['training_history'].items():
        print(f"     {key}: {values}")
    
    print("\n   Final Metrics:")
    for key, value in expected_results['final_metrics'].items():
        print(f"     {key}: {value}")
    
    print("\n   Pipeline Status:")
    for key, value in expected_results['pipeline_status'].items():
        print(f"     {key}: {value}")
    print()
    
    # 7. Show progress tracking
    print("7. Progress tracking example:")
    print("   During training, the progress callback would receive updates like:")
    print()
    
    # Simulate progress updates
    example_progress = TrainingProgress(
        current_epoch=2,
        total_epochs=3,
        first_cnn_loss=1.9,
        second_cnn_loss=1.6,
        combined_loss=1.7,
        wave_storage_utilization=65.0,
        attention_entropy=2.5,
        estimated_time_remaining=45.0,
        learning_rate=0.001,
        batch_processed=8,
        total_batches=12
    )
    
    print("   Example progress update:")
    progress_callback(example_progress)
    
    # 8. Show additional features
    print("8. Additional trainer features:")
    print("   - trainer.stop_training()  # Graceful early stopping")
    print("   - trainer.get_current_progress()  # Get current progress")
    print("   - trainer.save_training_state('checkpoint.json')  # Save state")
    print("   - trainer.load_training_state('checkpoint.json')  # Resume training")
    print()
    
    print("=== Example Complete ===")
    print("\nThis example demonstrates the DualCNNTrainer interface.")
    print("The actual implementation coordinates training of both CNNs")
    print("with rolling wave storage and comprehensive progress tracking.")


if __name__ == "__main__":
    main()