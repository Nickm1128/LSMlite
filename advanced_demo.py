#!/usr/bin/env python3
"""
Advanced LSM Lite Demo - Showcase all enhanced features
This script demonstrates the new advanced capabilities including:
- Beam search generation with nucleus sampling
- Advanced training with validation and early stopping
- Model analysis and performance benchmarking
- Model comparison between different configurations
"""

import os
import json
import time
import logging
from pathlib import Path
from lsm_lite.api import LSMLite
from lsm_lite.utils.config import LSMConfig
from lsm_lite.training.advanced_trainer import AdvancedLSMTrainer
from lsm_lite.utils.model_analysis import LSMAnalyzer, create_performance_plots


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_advanced_generation():
    """Demonstrate advanced text generation features."""
    print("\n" + "="*60)
    print("🚀 ADVANCED TEXT GENERATION DEMO")
    print("="*60)
    
    # Check if we have a saved model
    saved_models = Path("saved_models")
    if not saved_models.exists() or not list(saved_models.glob("*")):
        print("❌ No saved models found. Training a quick demo model...")
        train_demo_model()
    
    # Find the first available model
    model_path = next(saved_models.glob("*"))
    print(f"📁 Using model: {model_path}")
    
    try:
        # Load model
        lsm = LSMLite.load(str(model_path))
        print("✅ Model loaded successfully!")
        
        # Demo 1: Basic generation
        print("\n1️⃣ Basic Generation:")
        basic_text = lsm.generate(
            prompt="Hello, how are you?",
            max_length=30,
            temperature=0.8
        )
        print(f"   Prompt: Hello, how are you?")
        print(f"   Output: {basic_text}")
        
        # Demo 2: Beam search generation
        print("\n2️⃣ Beam Search Generation:")
        beam_text = lsm.generate(
            prompt="Once upon a time in a distant land",
            max_length=40,
            num_beams=3,
            temperature=0.7,
            do_sample=False
        )
        print(f"   Prompt: Once upon a time in a distant land")
        print(f"   Output: {beam_text}")
        
        # Demo 3: Nucleus sampling (top-p)
        print("\n3️⃣ Nucleus Sampling (Top-p):")
        nucleus_text = lsm.generate(
            prompt="The future of artificial intelligence",
            max_length=35,
            temperature=0.9,
            top_p=0.8,
            repetition_penalty=1.1
        )
        print(f"   Prompt: The future of artificial intelligence")
        print(f"   Output: {nucleus_text}")
        
        # Demo 4: Top-k sampling
        print("\n4️⃣ Top-k Sampling:")
        topk_text = lsm.generate(
            prompt="Machine learning has revolutionized",
            max_length=30,
            temperature=0.8,
            top_k=40
        )
        print(f"   Prompt: Machine learning has revolutionized")
        print(f"   Output: {topk_text}")
        
        print("\n✨ Advanced generation demo complete!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_model_analysis():
    """Demonstrate model analysis and benchmarking."""
    print("\n" + "="*60)
    print("📊 MODEL ANALYSIS & BENCHMARKING DEMO")
    print("="*60)
    
    # Find saved models
    saved_models = Path("saved_models")
    if not saved_models.exists() or not list(saved_models.glob("*")):
        print("❌ No saved models found. Please train a model first.")
        return
        
    model_path = next(saved_models.glob("*"))
    print(f"📁 Analyzing model: {model_path}")
    
    try:
        # Load model
        lsm = LSMLite.load(str(model_path))
        print("✅ Model loaded for analysis!")
        
        # Create analyzer
        analyzer = LSMAnalyzer(lsm._model, lsm._tokenizer, lsm._embedder)
        
        # Demo 1: Model complexity analysis
        print("\n1️⃣ Model Complexity Analysis:")
        complexity = analyzer.analyze_model_complexity()
        print(f"   📈 Total Parameters: {complexity['total_parameters']:,}")
        print(f"   💾 Estimated Memory: {complexity['estimated_memory_mb']:.1f} MB")
        print(f"   📊 Trainable Parameters: {complexity['trainable_parameters']:,}")
        
        # Demo 2: Inference speed profiling
        print("\n2️⃣ Speed Benchmarking:")
        test_texts = [
            "Hello there, how are you today?",
            "The weather is beautiful outside.",
            "Machine learning is fascinating.",
        ]
        
        speed_profile = analyzer.profile_inference_speed(test_texts, num_runs=5)
        print(f"   ⚡ Tokens per Second: {speed_profile['tokens_per_second']:.0f}")
        print(f"   ⏱️  Average Time per Sample: {speed_profile['avg_per_sample_ms']:.2f} ms")
        print(f"   📦 Batch Processing: {speed_profile['avg_batch_time_ms']:.2f} ms")
        
        # Demo 3: Activation analysis
        print("\n3️⃣ Reservoir Activation Analysis:")
        activation_stats = analyzer.analyze_reservoir_activations(test_texts[:2])
        print(f"   🎯 Average Activation: {activation_stats['mean_activation']:.4f}")
        print(f"   📊 Activation Std: {activation_stats['std_activation']:.4f}")
        print(f"   🔥 Sparsity Level: {activation_stats['sparsity']:.2%}")
        
        # Demo 4: Generate comprehensive report
        print("\n4️⃣ Generating Comprehensive Analysis Report...")
        output_dir = "analysis_demo_output"
        report_path = analyzer.export_analysis_report(output_dir, test_texts)
        print(f"   📄 Full analysis report saved: {report_path}")
        
        # Create performance plots
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        try:
            create_performance_plots(report_data, output_dir)
            print(f"   📈 Performance plots saved to: {output_dir}")
        except Exception as e:
            print(f"   ⚠️  Plot generation skipped (matplotlib not available): {e}")
        
        print("\n✨ Model analysis demo complete!")
        
    except Exception as e:
        print(f"❌ Analysis demo failed: {e}")


def demo_advanced_training():
    """Demonstrate advanced training features."""
    print("\n" + "="*60)
    print("🎓 ADVANCED TRAINING DEMO")
    print("="*60)
    
    try:
        # Create a custom config for training
        config = LSMConfig(
            max_length=64,
            embedding_dim=128,
            reservoir_size=256,
            cnn_filters=[32, 64],
            epochs=3,
            batch_size=16,
            learning_rate=0.002,
            dataset_name="cosmopedia-v2"
        )
        
        print("📋 Training Configuration:")
        print(f"   🧠 Reservoir Size: {config.reservoir_size}")
        print(f"   📏 Max Length: {config.max_length}")
        print(f"   🔄 Epochs: {config.epochs}")
        print(f"   📦 Batch Size: {config.batch_size}")
        print(f"   📊 Learning Rate: {config.learning_rate}")
        
        # Create LSM and build model
        lsm = LSMLite(config)
        lsm.build_model()
        print("✅ Model architecture built!")
        
        # Demo advanced training with validation
        print("\n🚀 Starting Advanced Training...")
        
        # Create advanced trainer
        trainer = AdvancedLSMTrainer(
            model=lsm._model,
            tokenizer=lsm._tokenizer,
            embedder=lsm._embedder,
            config=config
        )
        
        # Get some training data (using fallback for demo)
        sample_conversations = [
            "Hello, how can I help you today? I'm doing well, thank you for asking.",
            "What's the weather like? It's sunny and warm outside today.",
            "Can you explain machine learning? It's a field of AI that learns from data.",
            "How does a computer work? It processes information using electrical circuits.",
            "What is the meaning of life? That's a profound philosophical question."
        ] * 10  # Repeat for more training data
        
        # Train with advanced features
        history = trainer.train_with_validation(
            conversations=sample_conversations,
            epochs=2,  # Quick demo
            validation_split=0.2,
            early_stopping_patience=5,
            reduce_lr_patience=3,
            min_lr=1e-6
        )
        
        print("✅ Advanced training completed!")
        print(f"📊 Final Training Loss: {history['loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"📈 Final Validation Loss: {history['val_loss'][-1]:.4f}")
        
        # Save the demo model
        demo_model_path = "saved_models/advanced_demo_model"
        os.makedirs("saved_models", exist_ok=True)
        lsm.save_model(demo_model_path)
        print(f"💾 Demo model saved: {demo_model_path}")
        
        print("\n✨ Advanced training demo complete!")
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")


def train_demo_model():
    """Train a quick model for demo purposes."""
    print("🔧 Training a quick demo model...")
    
    config = LSMConfig(
        max_length=32,
        embedding_dim=64,
        reservoir_size=128,
        epochs=1,
        batch_size=8,
        learning_rate=0.01
    )
    
    lsm = LSMLite(config)
    lsm.build_model()
    
    # Quick training with minimal data
    history = lsm.train(max_samples=20, epochs=1)
    
    # Save demo model
    demo_path = "saved_models/quick_demo_model"
    os.makedirs("saved_models", exist_ok=True)
    lsm.save_model(demo_path)
    print(f"✅ Quick demo model trained and saved: {demo_path}")


def demo_model_comparison():
    """Demonstrate model comparison features."""
    print("\n" + "="*60)
    print("⚖️  MODEL COMPARISON DEMO")
    print("="*60)
    
    # Check if we have multiple models
    saved_models = list(Path("saved_models").glob("*")) if Path("saved_models").exists() else []
    
    if len(saved_models) < 2:
        print("📝 Need at least 2 models for comparison. Training additional models...")
        
        # Train two different models for comparison
        train_comparison_models()
        saved_models = list(Path("saved_models").glob("*"))
    
    if len(saved_models) >= 2:
        model1_path = str(saved_models[0])
        model2_path = str(saved_models[1])
        
        print(f"📊 Comparing models:")
        print(f"   🥇 Model 1: {model1_path}")
        print(f"   🥈 Model 2: {model2_path}")
        
        try:
            # Load both models
            lsm1 = LSMLite.load(model1_path)
            lsm2 = LSMLite.load(model2_path)
            
            # Create analyzer for comparison
            analyzer = LSMAnalyzer(lsm1._model, lsm1._tokenizer, lsm1._embedder)
            
            # Compare models
            test_prompts = [
                "Hello, how are you?",
                "The weather is nice today.",
                "Machine learning is exciting."
            ]
            
            comparison_results = analyzer.compare_model_versions(lsm2._model, test_prompts)
            
            print("\n📋 Comparison Results:")
            print(f"   🎯 Agreement Rate: {comparison_results['agreement_rate']:.1%}")
            print(f"   🥇 Model 1 Avg Confidence: {comparison_results['confidence_comparison']['model1_avg_confidence']:.3f}")
            print(f"   🥈 Model 2 Avg Confidence: {comparison_results['confidence_comparison']['model2_avg_confidence']:.3f}")
            
            # Save detailed comparison
            with open("model_comparison_demo.json", 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"   📄 Detailed comparison saved: model_comparison_demo.json")
            
            print("\n✨ Model comparison demo complete!")
            
        except Exception as e:
            print(f"❌ Comparison demo failed: {e}")
    else:
        print("❌ Unable to create comparison models for demo.")


def train_comparison_models():
    """Train two different models for comparison demo."""
    print("🔧 Training comparison models...")
    
    # Model 1: Smaller, faster model
    config1 = LSMConfig(
        max_length=32,
        embedding_dim=64,
        reservoir_size=128,
        epochs=1,
        batch_size=8,
        learning_rate=0.01
    )
    
    lsm1 = LSMLite(config1)
    lsm1.build_model()
    lsm1.train(max_samples=20, epochs=1)
    lsm1.save_model("saved_models/comparison_model_small")
    
    # Model 2: Larger, more complex model
    config2 = LSMConfig(
        max_length=64,
        embedding_dim=128,
        reservoir_size=256,
        epochs=1,
        batch_size=8,
        learning_rate=0.005
    )
    
    lsm2 = LSMLite(config2)
    lsm2.build_model()
    lsm2.train(max_samples=20, epochs=1)
    lsm2.save_model("saved_models/comparison_model_large")
    
    print("✅ Comparison models trained successfully!")


def main():
    """Run all advanced feature demos."""
    print("🚀 LSM Lite Advanced Features Demo")
    print("=" * 60)
    print("This demo showcases all the enhanced capabilities:")
    print("• Advanced text generation with beam search")
    print("• Model analysis and performance benchmarking")
    print("• Advanced training with validation")
    print("• Model comparison tools")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Demo 1: Advanced text generation
        demo_advanced_generation()
        
        # Demo 2: Model analysis and benchmarking
        demo_model_analysis()
        
        # Demo 3: Advanced training features
        demo_advanced_training()
        
        # Demo 4: Model comparison
        demo_model_comparison()
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total Demo Time: {total_time:.2f} seconds")
        print("📁 Generated Files:")
        print("   • analysis_demo_output/ - Model analysis reports and plots")
        print("   • model_comparison_demo.json - Model comparison results")
        print("   • saved_models/ - Trained demo models")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()