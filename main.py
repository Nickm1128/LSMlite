#!/usr/bin/env python3
"""
Main entry point for LSM Lite - Lightweight Liquid State Machine for Conversational AI

This script provides a command-line interface for training, evaluating, and using
the LSM model for conversational AI tasks.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Optional

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsm_lite import LSMLite, LSMConfig
from lsm_lite.utils.config import create_preset_configs, ConfigManager
from lsm_lite.utils.persistence import ModelPersistence
from lsm_lite.data.loader import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('lsm_lite.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="LSM Lite - Lightweight Liquid State Machine for Conversational AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model with default settings
  python main.py train --dataset cosmopedia-v2 --max-samples 10000

  # Train with custom configuration
  python main.py train --config configs/large.json --epochs 20

  # Generate text with basic settings
  python main.py generate --model-path saved_models/my_model --prompt "Hello, how are you?"

  # Generate with beam search and advanced sampling
  python main.py generate --model-path saved_models/my_model --prompt "Once upon a time" \
    --num-beams 5 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.1

  # Batch generate for multiple prompts
  python main.py generate --model-path saved_models/my_model \
    --batch-generate "Hello there" "The weather is" "AI technology"

  # Evaluate a model on test data
  python main.py evaluate --model-path saved_models/my_model --dataset test_data.txt

  # Interactive text generation
  python main.py interactive --model-path saved_models/my_model

  # Analyze model performance and generate reports
  python main.py analyze --model-path saved_models/my_model --include-plots

  # Compare two trained models
  python main.py compare --model1-path saved_models/model_v1 --model2-path saved_models/model_v2

  # Benchmark model inference speed
  python main.py benchmark --model-path saved_models/my_model --num-runs 20

  # List available configurations and models
  python main.py list-configs
  python main.py list-models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new LSM model')
    train_parser.add_argument('--config', type=str, help='Configuration file path')
    train_parser.add_argument('--preset', type=str, choices=['small', 'medium', 'large', 'fast', 'quality'],
                             help='Use preset configuration')
    train_parser.add_argument('--dataset', type=str, default='cosmopedia-v2',
                             help='Dataset name or file path')
    train_parser.add_argument('--max-samples', type=int, help='Maximum number of training samples')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--output-dir', type=str, default='saved_models',
                             help='Directory to save trained model')
    train_parser.add_argument('--model-name', type=str, help='Name for the saved model')
    
    # Generate command (enhanced)
    generate_parser = subparsers.add_parser('generate', help='Generate text using trained model')
    generate_parser.add_argument('--model-path', type=str, required=True,
                                help='Path to trained model')
    generate_parser.add_argument('--prompt', type=str, required=True,
                                help='Input prompt for generation')
    generate_parser.add_argument('--max-length', type=int, default=50,
                                help='Maximum generation length')
    generate_parser.add_argument('--temperature', type=float, default=1.0,
                                help='Generation temperature')
    generate_parser.add_argument('--top-k', type=int, help='Top-k sampling')
    generate_parser.add_argument('--top-p', type=float, help='Nucleus sampling threshold')
    generate_parser.add_argument('--num-beams', type=int, default=1,
                               help='Number of beams for beam search')
    generate_parser.add_argument('--repetition-penalty', type=float, default=1.0,
                               help='Repetition penalty')
    generate_parser.add_argument('--no-sampling', action='store_true',
                               help='Disable sampling (use with beam search)')
    generate_parser.add_argument('--batch-generate', nargs='+',
                               help='Generate for multiple prompts')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    evaluate_parser.add_argument('--model-path', type=str, required=True,
                                help='Path to trained model')
    evaluate_parser.add_argument('--dataset', type=str, required=True,
                                help='Test dataset name or file path')
    evaluate_parser.add_argument('--max-samples', type=int, help='Maximum test samples')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive text generation')
    interactive_parser.add_argument('--model-path', type=str, required=True,
                                   help='Path to trained model')
    interactive_parser.add_argument('--temperature', type=float, default=0.8,
                                   help='Generation temperature')
    
    # Utility commands
    subparsers.add_parser('list-configs', help='List available configurations')
    subparsers.add_parser('list-models', help='List saved models')
    
    # Config commands
    config_parser = subparsers.add_parser('create-config', help='Create a new configuration')
    config_parser.add_argument('--name', type=str, required=True, help='Configuration name')
    config_parser.add_argument('--preset', type=str, choices=['small', 'medium', 'large', 'fast', 'quality'],
                              help='Base preset to customize')
    
    # New: Advanced analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze trained model performance')
    analyze_parser.add_argument('--model-path', type=str, required=True,
                               help='Path to trained model')
    analyze_parser.add_argument('--test-texts', nargs='+',
                               default=['Hello, how are you?', 'The weather is nice today.'],
                               help='Test texts for analysis')
    analyze_parser.add_argument('--output-dir', type=str, default='analysis_output',
                               help='Directory to save analysis results')
    analyze_parser.add_argument('--include-plots', action='store_true',
                               help='Generate visualization plots')
    
    # New: Model comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare two trained models')
    compare_parser.add_argument('--model1-path', type=str, required=True,
                               help='Path to first model')
    compare_parser.add_argument('--model2-path', type=str, required=True,
                               help='Path to second model')
    compare_parser.add_argument('--test-texts', nargs='+',
                               default=['Hello', 'The quick brown fox'],
                               help='Test texts for comparison')
    
    # New: Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    benchmark_parser.add_argument('--model-path', type=str, required=True,
                                 help='Path to trained model')
    benchmark_parser.add_argument('--num-runs', type=int, default=10,
                                 help='Number of benchmark runs')
    
    return parser


def train_model(args) -> None:
    """Train a new LSM model."""
    logger.info("Starting LSM model training...")
    
    # Load or create configuration
    if args.config:
        config = LSMConfig.load(args.config)
        logger.info("Loaded configuration from: %s", args.config)
    elif args.preset:
        presets = create_preset_configs()
        config = presets[args.preset]
        logger.info("Using preset configuration: %s", args.preset)
    else:
        config = LSMConfig()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    config_updates = {}
    if args.max_samples:
        config_updates['max_samples'] = args.max_samples
    if args.epochs:
        config_updates['epochs'] = args.epochs
    if args.batch_size:
        config_updates['batch_size'] = args.batch_size
    if args.learning_rate:
        config_updates['learning_rate'] = args.learning_rate
    if args.dataset:
        config_updates['dataset_name'] = args.dataset
    
    if config_updates:
        config = config.update(**config_updates)
        logger.info("Updated configuration with CLI arguments")
    
    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        logger.error("Configuration validation errors:")
        for error in validation_errors:
            logger.error("  - %s", error)
        sys.exit(1)
    
    # Create LSM instance and build model
    lsm = LSMLite(config)
    lsm.build_model()
    
    # Train the model
    start_time = time.time()
    try:
        history = lsm.train(
            dataset_name=config.dataset_name,
            max_samples=config.max_samples,
            epochs=config.epochs,
            batch_size=config.batch_size
        )
        
        training_time = time.time() - start_time
        logger.info("Training completed in %.2f seconds", training_time)
        
        # Save the trained model
        model_name = args.model_name or f"lsm_model_{int(time.time())}"
        output_path = os.path.join(args.output_dir, model_name)
        
        os.makedirs(args.output_dir, exist_ok=True)
        lsm.save_model(output_path)
        
        # Save training history
        history_path = os.path.join(output_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info("Model saved to: %s", output_path)
        
        # Print training summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model saved to: {output_path}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final loss: {history.get('loss', [])[-1] if history.get('loss') else 'N/A'}")
        print(f"Final accuracy: {history.get('accuracy', [])[-1] if history.get('accuracy') else 'N/A'}")
        
    except Exception as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


def generate_text(args) -> None:
    """Generate text using a trained model with advanced features."""
    logger.info("Loading model from: %s", args.model_path)
    
    try:
        # Load the model
        lsm = LSMLite.load(args.model_path)
        
        # Handle batch generation
        if hasattr(args, 'batch_generate') and args.batch_generate:
            prompts = args.batch_generate
            print("\n" + "="*50)
            print("BATCH TEXT GENERATION")
            print("="*50)
            
            for i, prompt in enumerate(prompts):
                generated = lsm.generate(
                    prompt=prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=not args.no_sampling if hasattr(args, 'no_sampling') else True
                )
                print(f"\n{i+1}. Prompt: {prompt}")
                print(f"   Generated: {generated}")
                
        else:
            # Single generation with advanced parameters
            generation_params = {
                'prompt': args.prompt,
                'max_length': args.max_length,
                'temperature': args.temperature
            }
            
            # Add optional parameters
            if hasattr(args, 'top_k') and args.top_k:
                generation_params['top_k'] = args.top_k
            if hasattr(args, 'top_p') and args.top_p:
                generation_params['top_p'] = args.top_p
            if hasattr(args, 'num_beams') and args.num_beams > 1:
                generation_params['num_beams'] = args.num_beams
            if hasattr(args, 'repetition_penalty'):
                generation_params['repetition_penalty'] = args.repetition_penalty
            if hasattr(args, 'no_sampling') and args.no_sampling:
                generation_params['do_sample'] = False
                
            generated = lsm.generate(**generation_params)
            
            print("\n" + "="*50)
            print("ADVANCED TEXT GENERATION")
            print("="*50)
            print(f"Prompt: {args.prompt}")
            print(f"Generated: {generated}")
            
            # Show generation parameters
            print(f"\nGeneration Settings:")
            print(f"  Max Length: {args.max_length}")
            print(f"  Temperature: {args.temperature}")
            if hasattr(args, 'num_beams') and args.num_beams > 1:
                print(f"  Beam Search: {args.num_beams} beams")
            if hasattr(args, 'top_k') and args.top_k:
                print(f"  Top-k: {args.top_k}")
            if hasattr(args, 'top_p') and args.top_p:
                print(f"  Top-p: {args.top_p}")
            if hasattr(args, 'repetition_penalty') and args.repetition_penalty != 1.0:
                print(f"  Repetition Penalty: {args.repetition_penalty}")
            
        print("="*50)
        
        # Save to file if requested
        if hasattr(args, 'output_file') and args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"Prompt: {args.prompt}\n")
                f.write(f"Generated: {generated}\n")
            print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error("Text generation failed: %s", e)
        sys.exit(1)


def evaluate_model(args) -> None:
    """Evaluate a trained model."""
    logger.info("Evaluating model from: %s", args.model_path)
    
    try:
        # Load the model
        lsm = LSMLite()
        lsm.load_model(args.model_path)
        
        # Load test data
        data_loader = DataLoader(
            dataset_name=args.dataset,
            max_samples=args.max_samples
        )
        test_conversations = data_loader.load_conversations()
        
        # Evaluate
        results = lsm.evaluate(test_conversations)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        print(f"Test dataset: {args.dataset}")
        print(f"Test samples: {len(test_conversations)}")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        sys.exit(1)


def interactive_mode(args) -> None:
    """Run interactive text generation."""
    logger.info("Starting interactive mode with model: %s", args.model_path)
    
    try:
        # Load the model
        lsm = LSMLite()
        lsm.load_model(args.model_path)
        
        print("\n" + "="*50)
        print("INTERACTIVE TEXT GENERATION")
        print("="*50)
        print("Type your prompts and press Enter to generate text.")
        print("Type 'quit' to exit, 'help' for commands.")
        print("="*50)
        
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
                
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'help':
                    print("Commands:")
                    print("  quit - Exit interactive mode")
                    print("  help - Show this help message")
                    print("  Just type any text to generate a continuation")
                    continue
                elif not prompt:
                    continue
                
                # Generate text
                generated = lsm.generate(
                    prompt=prompt,
                    max_length=50,
                    temperature=args.temperature
                )
                
                print(f"Generated: {generated}")
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Generation error: {e}")
        
    except Exception as e:
        logger.error("Interactive mode failed: %s", e)
        sys.exit(1)


def list_configurations() -> None:
    """List available configurations."""
    config_manager = ConfigManager()
    configs = config_manager.list_configs()
    presets = list(create_preset_configs().keys())
    
    print("\n" + "="*50)
    print("AVAILABLE CONFIGURATIONS")
    print("="*50)
    
    print("Preset configurations:")
    for preset in presets:
        print(f"  - {preset}")
    
    if configs:
        print("\nSaved configurations:")
        for config_name in configs:
            print(f"  - {config_name}")
    else:
        print("\nNo saved configurations found.")
    
    print("="*50)


def list_models() -> None:
    """List saved models."""
    models = ModelPersistence.list_saved_models()
    
    print("\n" + "="*50)
    print("SAVED MODELS")
    print("="*50)
    
    if not models:
        print("No saved models found.")
    else:
        for model in models:
            print(f"Name: {model['name']}")
            print(f"  Path: {model['path']}")
            print(f"  Size: {model['size_mb']:.1f} MB")
            print(f"  Integrity: {'✓' if model['integrity']['all_components_present'] else '✗'}")
            if model['metadata']:
                print(f"  Version: {model['metadata'].get('lsm_lite_version', 'unknown')}")
                print(f"  Tokenizer: {model['metadata'].get('tokenizer_backend', 'unknown')}")
            print()
    
    print("="*50)


def create_configuration(args) -> None:
    """Create a new configuration."""
    config_manager = ConfigManager()
    
    # Start with preset if specified
    if args.preset:
        presets = create_preset_configs()
        config = presets[args.preset]
        logger.info("Starting with preset: %s", args.preset)
    else:
        config = LSMConfig()
        logger.info("Starting with default configuration")
    
    # Save the configuration
    config_manager.save_config(config, args.name)
    print(f"Configuration '{args.name}' created successfully!")
    print(f"Edit the file at: configs/{args.name}.json to customize settings.")


def analyze_model(args) -> None:
    """Analyze trained model performance."""
    from lsm_lite.utils.model_analysis import LSMAnalyzer, create_performance_plots
    
    logger.info("Loading model for analysis: %s", args.model_path)
    
    # Load model
    lsm = LSMLite.load(args.model_path)
    
    # Create analyzer
    analyzer = LSMAnalyzer(lsm._model, lsm._tokenizer, lsm._embedder)
    
    # Generate comprehensive analysis
    report_path = analyzer.export_analysis_report(args.output_dir, args.test_texts)
    
    # Create plots if requested
    if args.include_plots:
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        create_performance_plots(report, args.output_dir)
        print(f"Analysis complete with plots! Results saved to: {args.output_dir}")
    else:
        print(f"Analysis complete! Results saved to: {args.output_dir}")
    
    # Display key metrics
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    if 'complexity_analysis' in report and 'error' not in report['complexity_analysis']:
        complexity = report['complexity_analysis']
        print(f"\nModel Overview:")
        print(f"  Parameters: {complexity['total_parameters']:,}")
        print(f"  Memory: {complexity['estimated_memory_mb']:.1f} MB")
    
    if 'speed_profile' in report and 'error' not in report['speed_profile']:
        speed = report['speed_profile']
        print(f"  Speed: {speed['tokens_per_second']:.0f} tokens/sec")
        print(f"  Inference: {speed['avg_per_sample_ms']:.1f} ms/sample")


def compare_models(args) -> None:
    """Compare two trained models."""
    from lsm_lite.utils.model_analysis import LSMAnalyzer
    import json
    
    logger.info("Loading models for comparison...")
    
    # Load both models
    lsm1 = LSMLite.load(args.model1_path)
    lsm2 = LSMLite.load(args.model2_path)
    
    # Create analyzer for first model
    analyzer = LSMAnalyzer(lsm1._model, lsm1._tokenizer, lsm1._embedder)
    
    # Compare models
    comparison_results = analyzer.compare_model_versions(lsm2._model, args.test_texts)
    
    # Save results
    output_file = 'model_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Display summary
    print(f"\nModel Comparison Results:")
    print(f"  Agreement Rate: {comparison_results['agreement_rate']:.1%}")
    print(f"  Model 1 Avg Confidence: {comparison_results['confidence_comparison']['model1_avg_confidence']:.3f}")
    print(f"  Model 2 Avg Confidence: {comparison_results['confidence_comparison']['model2_avg_confidence']:.3f}")
    print(f"\nDetailed results saved to: {output_file}")


def benchmark_model(args) -> None:
    """Benchmark model performance."""
    from lsm_lite.utils.model_analysis import LSMAnalyzer
    
    logger.info("Benchmarking model: %s", args.model_path)
    
    # Load model
    lsm = LSMLite.load(args.model_path)
    
    # Create analyzer
    analyzer = LSMAnalyzer(lsm._model, lsm._tokenizer, lsm._embedder)
    
    # Benchmark with different sample texts
    test_texts = [
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Once upon a time in a distant land...",
        "The weather forecast shows sunny skies ahead."
    ]
    
    # Run performance profiling
    profile_results = analyzer.profile_inference_speed(test_texts, num_runs=args.num_runs)
    
    # Display results
    print(f"\nBenchmark Results ({args.num_runs} runs):")
    print(f"  Average Batch Time: {profile_results['avg_batch_time_ms']:.2f} ms")
    print(f"  Standard Deviation: {profile_results['std_batch_time_ms']:.2f} ms")
    print(f"  Tokens per Second: {profile_results['tokens_per_second']:.0f}")
    print(f"  Per Sample Time: {profile_results['avg_per_sample_ms']:.2f} ms")
    print(f"  Batch Size: {profile_results['batch_size']}")
    print(f"  Sequence Length: {profile_results['sequence_length']}")
    
    logger.info("Benchmark complete")


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_model(args)
        elif args.command == 'generate':
            generate_text(args)
        elif args.command == 'evaluate':
            evaluate_model(args)
        elif args.command == 'interactive':
            interactive_mode(args)
        elif args.command == 'list-configs':
            list_configurations()
        elif args.command == 'list-models':
            list_models()
        elif args.command == 'create-config':
            create_configuration(args)
        elif args.command == 'analyze':
            analyze_model(args)
        elif args.command == 'compare':
            compare_models(args)
        elif args.command == 'benchmark':
            benchmark_model(args)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
