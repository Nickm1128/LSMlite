#!/usr/bin/env python3
"""
Example usage of LSM Lite - Lightweight Liquid State Machine for Conversational AI

This script demonstrates various features and use cases of the LSM Lite library,
including training, text generation, and model analysis.
"""

import os
import sys
import logging
import time
from typing import List

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsm_lite import LSMLite, LSMConfig
from lsm_lite.utils.config import create_preset_configs
from lsm_lite.data.loader import DataLoader
from lsm_lite.core.reservoir import ReservoirAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_example():
    """Basic example showing minimal usage of LSM Lite."""
    print("\n" + "=" * 60)
    print("BASIC EXAMPLE - Minimal LSM Lite Usage")
    print("=" * 60)

    # Create LSM with default configuration
    lsm = LSMLite()

    # Build the model
    print("Building model...")
    lsm.build_model()

    # Train on a small dataset using the specific parquet file
    print("Training model...")
    history = lsm.train('https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/train-00000-of-00104.parquet', max_samples=100, epochs=2)

    # Generate some text
    print("Generating text...")
    generated = lsm.generate("Hello, how are you?", max_length=30)
    print(f"Generated: {generated}")

    print("Basic example completed!")


def configuration_example():
    """Example showing different configuration options."""
    print("\n" + "=" * 60)
    print("CONFIGURATION EXAMPLE - Custom Settings")
    print("=" * 60)

    # Show available presets
    presets = create_preset_configs()
    print("Available presets:")
    for name, config in presets.items():
        print(
            f"  {name}: {config.reservoir_size} reservoir, {config.embedding_dim} embedding dim"
        )

    # Use a small configuration for quick testing
    config = presets['small']
    print(f"\nUsing 'small' configuration: {config}")

    # Customize configuration
    custom_config = config.update(epochs=3,
                                  learning_rate=0.002,
                                  generation_temperature=0.8)

    # Create LSM with custom config
    lsm = LSMLite(custom_config)
    lsm.build_model()

    # Train briefly
    print("Training with custom configuration...")
    lsm.train(max_samples=50, epochs=2)

    # Test different generation parameters
    prompts = [
        "What is machine learning?", "Tell me about neural networks.",
        "How does AI work?"
    ]

    print("\nTesting different generation temperatures:")
    for temp in [0.5, 1.0, 1.5]:
        print(f"\nTemperature {temp}:")
        for prompt in prompts[:1]:  # Just use first prompt
            generated = lsm.generate(prompt, temperature=temp, max_length=20)
            print(f"  '{prompt}' -> '{generated}'")


def data_loading_example():
    """Example showing data loading capabilities."""
    print("\n" + "=" * 60)
    print("DATA LOADING EXAMPLE - Working with Different Data Sources")
    print("=" * 60)

    # Create sample data file for demonstration
    sample_data = [
        "Hello, how can I help you today?",
        "I'm interested in learning about artificial intelligence.",
        "AI is a fascinating field that involves creating intelligent machines.",
        "Can you tell me more about machine learning?",
        "Machine learning is a subset of AI that focuses on algorithms that learn from data.",
        "What are some practical applications of AI?",
        "AI is used in many areas including healthcare, finance, transportation, and entertainment.",
    ]

    # Save to a file
    with open("sample_conversations.txt", "w") as f:
        f.write("\n\n".join(sample_data))

    print("Created sample data file: sample_conversations.txt")

    # Load data using DataLoader
    data_loader = DataLoader(dataset_name="sample_conversations.txt",
                             max_samples=10)
    conversations = data_loader.load_conversations()

    print(f"Loaded {len(conversations)} conversations")

    # Get dataset statistics
    stats = data_loader.get_dataset_statistics(conversations)
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Create conversation pairs for training
    pairs = data_loader.create_conversation_pairs(conversations)
    print(f"Created {len(pairs)} training pairs")

    # Show some examples
    print("\nSample training pairs:")
    for i, pair in enumerate(pairs[:3]):
        print(f"  {i+1}. Input: '{pair['input']}'")
        print(f"     Target: '{pair['target']}'")

    # Clean up
    os.remove("sample_conversations.txt")


def model_analysis_example():
    """Example showing model analysis and diagnostics."""
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS EXAMPLE - Understanding Your Model")
    print("=" * 60)

    # Create a model with specific configuration
    config = LSMConfig(reservoir_size=256,
                       embedding_dim=128,
                       sparsity=0.1,
                       spectral_radius=0.9)

    lsm = LSMLite(config)
    lsm.build_model()

    # Analyze reservoir properties
    print("Analyzing reservoir dynamics...")
    analyzer = ReservoirAnalyzer()
    reservoir_analysis = analyzer.analyze_dynamics(lsm._reservoir)

    print("Reservoir analysis:")
    for key, value in reservoir_analysis.items():
        print(f"  {key}: {value}")

    # Check model size estimate
    size_estimate = config.get_model_size_estimate()
    print("\nModel size estimate (parameters):")
    for component, params in size_estimate.items():
        if component != 'total':
            print(f"  {component}: {params:,}")
    print(f"  TOTAL: {size_estimate['total']:,} parameters")

    # Train briefly and get model summary
    print("\nTraining briefly to initialize weights...")
    lsm.train(max_samples=20, epochs=1)

    # Get model summary
    print("\nModel architecture summary:")
    print(lsm._trainer.get_model_summary())


def text_generation_example():
    """Example showing advanced text generation features."""
    print("\n" + "=" * 60)
    print("TEXT GENERATION EXAMPLE - Advanced Generation Techniques")
    print("=" * 60)

    # Create and train a model
    config = create_preset_configs()['fast']  # Quick training for demo
    lsm = LSMLite(config)
    lsm.build_model()

    print("Training model for text generation...")
    lsm.train(max_samples=100, epochs=3)

    # Test different generation strategies
    prompt = "The future of artificial intelligence"

    print(f"\nPrompt: '{prompt}'\n")

    # Greedy decoding (temperature = 0)
    print("1. Greedy decoding (temperature=0.0):")
    generated = lsm.generate(prompt, temperature=0.0, max_length=30)
    print(f"   {generated}")

    # Low temperature (more focused)
    print("\n2. Low temperature (temperature=0.5):")
    generated = lsm.generate(prompt, temperature=0.5, max_length=30)
    print(f"   {generated}")

    # Medium temperature (balanced)
    print("\n3. Medium temperature (temperature=1.0):")
    generated = lsm.generate(prompt, temperature=1.0, max_length=30)
    print(f"   {generated}")

    # High temperature (more creative)
    print("\n4. High temperature (temperature=1.5):")
    generated = lsm.generate(prompt, temperature=1.5, max_length=30)
    print(f"   {generated}")

    # Batch generation
    prompts = [
        "Machine learning is", "Neural networks can", "Deep learning helps"
    ]

    print("\n5. Batch generation:")
    generated_texts = lsm._generator.generate_batch(prompts,
                                                    max_length=20,
                                                    temperature=0.8)
    for prompt, generated in zip(prompts, generated_texts):
        print(f"   '{prompt}' -> '{generated}'")


def persistence_example():
    """Example showing model saving and loading."""
    print("\n" + "=" * 60)
    print("PERSISTENCE EXAMPLE - Saving and Loading Models")
    print("=" * 60)

    # Create and train a model
    config = create_preset_configs()['small']
    lsm = LSMLite(config)
    lsm.build_model()

    print("Training model...")
    lsm.train(max_samples=50, epochs=2)

    # Test generation before saving
    test_prompt = "Hello world"
    original_output = lsm.generate(test_prompt, temperature=0.0, max_length=10)
    print(f"Original output: '{original_output}'")

    # Save the model
    model_path = "example_model"
    print(f"Saving model to: {model_path}")
    lsm.save_model(model_path)

    # Load the model in a new instance
    print("Loading model...")
    lsm_loaded = LSMLite()
    lsm_loaded.load_model(model_path)

    # Test that generation is consistent
    loaded_output = lsm_loaded.generate(test_prompt,
                                        temperature=0.0,
                                        max_length=10)
    print(f"Loaded output: '{loaded_output}'")

    if original_output == loaded_output:
        print("✓ Model persistence working correctly!")
    else:
        print(
            "⚠ Model outputs differ - this may be due to randomness in generation"
        )

    # Show model information
    from lsm_lite.utils.persistence import ModelPersistence
    model_info = ModelPersistence.get_model_info(model_path)
    print(f"\nModel info:")
    print(f"  Size: {model_info['size_mb']:.1f} MB")
    print(
        f"  All components present: {model_info['integrity']['all_components_present']}"
    )

    # Clean up
    import shutil
    shutil.rmtree(model_path)
    print("Cleaned up example model files")


def performance_comparison_example():
    """Example comparing different model configurations."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON - Different Model Sizes")
    print("=" * 60)

    presets = ['small', 'medium']  # Skip large for quick demo
    results = {}

    for preset_name in presets:
        print(f"\nTesting {preset_name} configuration...")

        config = create_preset_configs()[preset_name]
        config = config.update(epochs=2, max_samples=100)  # Quick training

        lsm = LSMLite(config)
        lsm.build_model()

        # Time the training
        start_time = time.time()
        history = lsm.train(max_samples=config.max_samples,
                            epochs=config.epochs)
        training_time = time.time() - start_time

        # Test generation speed
        prompt = "Test prompt for generation"
        start_time = time.time()
        for _ in range(5):  # Generate 5 times
            lsm.generate(prompt, max_length=20)
        generation_time = (time.time() - start_time) / 5

        # Get final metrics
        final_loss = history.get('loss', [0])[-1] if history.get('loss') else 0
        final_accuracy = history.get('accuracy',
                                     [0])[-1] if history.get('accuracy') else 0

        results[preset_name] = {
            'training_time': training_time,
            'generation_time': generation_time,
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'parameters': config.get_model_size_estimate()['total']
        }

    # Display comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)

    print(
        f"{'Config':<10} {'Params':<12} {'Train Time':<12} {'Gen Time':<12} {'Loss':<10} {'Accuracy':<10}"
    )
    print("-" * 70)

    for name, metrics in results.items():
        print(
            f"{name:<10} {metrics['parameters']:<12,} {metrics['training_time']:<12.2f} "
            f"{metrics['generation_time']:<12.4f} {metrics['final_loss']:<10.4f} {metrics['final_accuracy']:<10.4f}"
        )


def interactive_demo():
    """Interactive demonstration of LSM capabilities."""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO - Try LSM Live!")
    print("=" * 60)

    # Create a quick model for interaction
    config = create_preset_configs()['fast']
    lsm = LSMLite(config)
    lsm.build_model()

    print("Training a quick model for interaction...")
    lsm.train(max_samples=50, epochs=2)

    print("\nModel ready! Try some prompts:")
    print("(Press Enter with empty input to exit)")

    while True:
        try:
            prompt = input("\nYour prompt: ").strip()
            if not prompt:
                break

            generated = lsm.generate(prompt, max_length=40, temperature=0.8)
            print(f"LSM: {generated}")

        except KeyboardInterrupt:
            print("\nExiting demo...")
            break


def run_all_examples():
    """Run all examples in sequence."""
    print("LSM Lite Example Suite")
    print("Running comprehensive examples...")

    try:
        basic_example()
        configuration_example()
        data_loading_example()
        model_analysis_example()
        text_generation_example()
        persistence_example()
        performance_comparison_example()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Ask if user wants to try interactive demo
        try:
            response = input(
                "\nWould you like to try the interactive demo? (y/n): ").strip(
                ).lower()
            if response in ['y', 'yes']:
                interactive_demo()
        except KeyboardInterrupt:
            pass

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        logger.exception("Example execution failed")


def main():
    """Main function with menu selection."""
    examples = {
        '1': ('Basic Usage', basic_example),
        '2': ('Configuration Options', configuration_example),
        '3': ('Data Loading', data_loading_example),
        '4': ('Model Analysis', model_analysis_example),
        '5': ('Text Generation', text_generation_example),
        '6': ('Model Persistence', persistence_example),
        '7': ('Performance Comparison', performance_comparison_example),
        '8': ('Interactive Demo', interactive_demo),
        '9': ('Run All Examples', run_all_examples)
    }

    print("LSM Lite Examples")
    print("=" * 40)
    print("Choose an example to run:")

    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("  0. Exit")

    try:
        choice = input("\nEnter your choice (0-9): ").strip()

        if choice == '0':
            print("Goodbye!")
            return

        if choice in examples:
            name, func = examples[choice]
            print(f"\nRunning: {name}")
            func()
        else:
            print("Invalid choice. Please try again.")
            main()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Example failed")


if __name__ == "__main__":
    main()
