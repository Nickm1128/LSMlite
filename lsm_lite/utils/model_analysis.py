"""
Advanced model analysis and visualization tools.

This module provides comprehensive analysis capabilities for trained LSM models
including parameter analysis, performance profiling, and model interpretability.
"""

import logging
import numpy as np
import tensorflow as tf
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LSMAnalyzer:
    """Comprehensive LSM model analyzer."""
    
    def __init__(self, model, tokenizer, embedder=None):
        """
        Initialize LSM analyzer.
        
        Args:
            model: Trained LSM model
            tokenizer: Model tokenizer
            embedder: Model embedder (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.analysis_cache = {}
        
        logger.info("LSM Analyzer initialized")
    
    def analyze_model_complexity(self) -> Dict[str, Any]:
        """Analyze model complexity and parameter distribution."""
        logger.info("Analyzing model complexity...")
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        # Layer-wise parameter analysis
        layer_analysis = []
        for layer in self.model.layers:
            layer_info = {
                'name': layer.name,
                'type': type(layer).__name__,
                'parameters': layer.count_params() if hasattr(layer, 'count_params') else 0,
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights]) if hasattr(layer, 'trainable_weights') else 0
            }
            
            # Add shape information
            if hasattr(layer, 'output_shape'):
                layer_info['output_shape'] = str(layer.output_shape)
            
            layer_analysis.append(layer_info)
        
        complexity_metrics = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(total_params - trainable_params),
            'parameter_efficiency': float(trainable_params / total_params) if total_params > 0 else 0.0,
            'layer_count': len(self.model.layers),
            'layer_analysis': layer_analysis
        }
        
        # Memory estimation (rough)
        memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        complexity_metrics['estimated_memory_mb'] = float(memory_mb)
        
        logger.info("Model complexity analysis complete: %d total params, %.1f MB", 
                   total_params, memory_mb)
        
        return complexity_metrics
    
    def profile_inference_speed(self, sample_texts: List[str], num_runs: int = 10) -> Dict[str, float]:
        """Profile model inference speed."""
        logger.info("Profiling inference speed with %d samples, %d runs", len(sample_texts), num_runs)
        
        # Prepare test data
        tokenized_samples = []
        for text in sample_texts[:10]:  # Limit samples for speed
            tokenized = self.tokenizer.tokenize([text], padding=True, truncation=True)
            tokenized_samples.append(tokenized['input_ids'][0])
        
        if not tokenized_samples:
            return {'error': 'No valid samples for profiling'}
        
        input_tensor = tf.constant(tokenized_samples, dtype=tf.int32)
        
        # Warmup runs
        for _ in range(3):
            _ = self.model(input_tensor, training=False)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start_time = tf.timestamp()
            _ = self.model(input_tensor, training=False)
            end_time = tf.timestamp()
            times.append(float(end_time - start_time))
        
        batch_size = len(tokenized_samples)
        sequence_length = len(tokenized_samples[0])
        
        profile_results = {
            'avg_batch_time_ms': float(np.mean(times) * 1000),
            'std_batch_time_ms': float(np.std(times) * 1000),
            'min_batch_time_ms': float(np.min(times) * 1000),
            'max_batch_time_ms': float(np.max(times) * 1000),
            'avg_per_sample_ms': float(np.mean(times) * 1000 / batch_size),
            'tokens_per_second': float(batch_size * sequence_length / np.mean(times)),
            'batch_size': batch_size,
            'sequence_length': sequence_length
        }
        
        logger.info("Inference profiling complete: %.2f ms/batch, %.0f tokens/sec", 
                   profile_results['avg_batch_time_ms'], profile_results['tokens_per_second'])
        
        return profile_results
    
    def analyze_token_predictions(self, sample_texts: List[str], top_k: int = 10) -> Dict[str, Any]:
        """Analyze model prediction patterns."""
        logger.info("Analyzing token prediction patterns...")
        
        prediction_stats = {
            'confidence_distribution': [],
            'top_predicted_tokens': {},
            'entropy_scores': [],
            'sample_predictions': []
        }
        
        for text in sample_texts[:20]:  # Limit for performance
            tokenized = self.tokenizer.tokenize([text], padding=True, truncation=True)
            input_tensor = tf.constant(tokenized['input_ids'], dtype=tf.int32)
            
            predictions = self.model(input_tensor, training=False)
            probs = tf.nn.softmax(predictions[0, -1, :])  # Last position probabilities
            
            # Calculate entropy
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))
            prediction_stats['entropy_scores'].append(float(entropy))
            
            # Get top predictions
            top_probs, top_indices = tf.nn.top_k(probs, k=top_k)
            
            sample_pred = {
                'input_text': text[:50] + '...' if len(text) > 50 else text,
                'entropy': float(entropy),
                'top_predictions': []
            }
            
            for i in range(top_k):
                token_id = int(top_indices[i])
                prob = float(top_probs[i])
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                
                sample_pred['top_predictions'].append({
                    'token': token_text,
                    'probability': prob,
                    'token_id': token_id
                })
                
                # Update global token frequency
                if token_text not in prediction_stats['top_predicted_tokens']:
                    prediction_stats['top_predicted_tokens'][token_text] = 0
                prediction_stats['top_predicted_tokens'][token_text] += prob
            
            prediction_stats['sample_predictions'].append(sample_pred)
            
            # Store max probability (confidence)
            max_prob = float(tf.reduce_max(probs))
            prediction_stats['confidence_distribution'].append(max_prob)
        
        # Calculate summary statistics
        prediction_stats['avg_confidence'] = float(np.mean(prediction_stats['confidence_distribution']))
        prediction_stats['avg_entropy'] = float(np.mean(prediction_stats['entropy_scores']))
        
        # Sort top predicted tokens
        sorted_tokens = sorted(
            prediction_stats['top_predicted_tokens'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        prediction_stats['most_predicted_tokens'] = sorted_tokens
        
        logger.info("Token prediction analysis complete: avg_confidence=%.3f, avg_entropy=%.3f", 
                   prediction_stats['avg_confidence'], prediction_stats['avg_entropy'])
        
        return prediction_stats
    
    def generate_layer_activation_analysis(self, sample_text: str) -> Dict[str, Any]:
        """Analyze layer activations for a sample text."""
        logger.info("Generating layer activation analysis...")
        
        tokenized = self.tokenizer.tokenize([sample_text], padding=True, truncation=True)
        input_tensor = tf.constant(tokenized['input_ids'], dtype=tf.int32)
        
        # Create model that outputs intermediate activations
        layer_outputs = []
        x = input_tensor
        
        for i, layer in enumerate(self.model.layers):
            try:
                x = layer(x)
                if hasattr(x, 'numpy'):
                    activation = x.numpy()
                    layer_outputs.append({
                        'layer_index': i,
                        'layer_name': layer.name,
                        'layer_type': type(layer).__name__,
                        'activation_shape': activation.shape,
                        'activation_mean': float(np.mean(activation)),
                        'activation_std': float(np.std(activation)),
                        'activation_min': float(np.min(activation)),
                        'activation_max': float(np.max(activation)),
                        'sparsity': float(np.mean(activation == 0)) if activation.dtype != np.bool_ else 0.0
                    })
            except Exception as e:
                logger.warning("Could not analyze layer %d (%s): %s", i, layer.name, e)
                layer_outputs.append({
                    'layer_index': i,
                    'layer_name': layer.name,
                    'layer_type': type(layer).__name__,
                    'error': str(e)
                })
        
        analysis = {
            'input_text': sample_text[:100] + '...' if len(sample_text) > 100 else sample_text,
            'layer_activations': layer_outputs,
            'total_layers': len(self.model.layers)
        }
        
        logger.info("Layer activation analysis complete for %d layers", len(layer_outputs))
        return analysis
    
    def compare_model_versions(self, other_model, comparison_texts: List[str]) -> Dict[str, Any]:
        """Compare this model with another model version."""
        logger.info("Comparing models on %d test texts", len(comparison_texts))
        
        comparison_results = {
            'model1_predictions': [],
            'model2_predictions': [],
            'agreement_rate': 0.0,
            'confidence_comparison': {
                'model1_avg_confidence': 0.0,
                'model2_avg_confidence': 0.0
            },
            'sample_comparisons': []
        }
        
        agreements = 0
        total_comparisons = 0
        
        model1_confidences = []
        model2_confidences = []
        
        for text in comparison_texts[:10]:  # Limit for performance
            tokenized = self.tokenizer.tokenize([text], padding=True, truncation=True)
            input_tensor = tf.constant(tokenized['input_ids'], dtype=tf.int32)
            
            # Model 1 predictions
            pred1 = self.model(input_tensor, training=False)
            probs1 = tf.nn.softmax(pred1[0, -1, :])
            top1_id = int(tf.argmax(probs1))
            top1_prob = float(tf.reduce_max(probs1))
            top1_token = self.tokenizer.decode([top1_id], skip_special_tokens=True)
            
            # Model 2 predictions
            pred2 = other_model(input_tensor, training=False)
            probs2 = tf.nn.softmax(pred2[0, -1, :])
            top2_id = int(tf.argmax(probs2))
            top2_prob = float(tf.reduce_max(probs2))
            top2_token = self.tokenizer.decode([top2_id], skip_special_tokens=True)
            
            # Compare
            if top1_id == top2_id:
                agreements += 1
            total_comparisons += 1
            
            model1_confidences.append(top1_prob)
            model2_confidences.append(top2_prob)
            
            comparison_results['sample_comparisons'].append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'model1_prediction': {
                    'token': top1_token,
                    'confidence': top1_prob,
                    'token_id': top1_id
                },
                'model2_prediction': {
                    'token': top2_token,
                    'confidence': top2_prob,
                    'token_id': top2_id
                },
                'agreement': top1_id == top2_id
            })
        
        comparison_results['agreement_rate'] = float(agreements / total_comparisons) if total_comparisons > 0 else 0.0
        comparison_results['confidence_comparison']['model1_avg_confidence'] = float(np.mean(model1_confidences))
        comparison_results['confidence_comparison']['model2_avg_confidence'] = float(np.mean(model2_confidences))
        
        logger.info("Model comparison complete: %.1f%% agreement, avg confidences: %.3f vs %.3f", 
                   comparison_results['agreement_rate'] * 100,
                   comparison_results['confidence_comparison']['model1_avg_confidence'],
                   comparison_results['confidence_comparison']['model2_avg_confidence'])
        
        return comparison_results
    
    def export_analysis_report(self, output_dir: str, sample_texts: List[str]) -> str:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive analysis report...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report = {
            'timestamp': tf.timestamp().numpy().item(),
            'model_info': {
                'layers': len(self.model.layers),
                'total_params': self.model.count_params()
            }
        }
        
        # Run all analyses
        try:
            report['complexity_analysis'] = self.analyze_model_complexity()
        except Exception as e:
            logger.error("Complexity analysis failed: %s", e)
            report['complexity_analysis'] = {'error': str(e)}
        
        try:
            report['speed_profile'] = self.profile_inference_speed(sample_texts)
        except Exception as e:
            logger.error("Speed profiling failed: %s", e)
            report['speed_profile'] = {'error': str(e)}
        
        try:
            report['prediction_analysis'] = self.analyze_token_predictions(sample_texts)
        except Exception as e:
            logger.error("Prediction analysis failed: %s", e)
            report['prediction_analysis'] = {'error': str(e)}
        
        if sample_texts:
            try:
                report['layer_analysis'] = self.generate_layer_activation_analysis(sample_texts[0])
            except Exception as e:
                logger.error("Layer analysis failed: %s", e)
                report['layer_analysis'] = {'error': str(e)}
        
        # Save report
        report_path = os.path.join(output_dir, 'lsm_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary
        summary_path = os.path.join(output_dir, 'analysis_summary.txt')
        self._generate_text_summary(report, summary_path)
        
        logger.info("Analysis report exported to: %s", output_dir)
        return report_path
    
    def _generate_text_summary(self, report: Dict[str, Any], summary_path: str):
        """Generate human-readable summary."""
        with open(summary_path, 'w') as f:
            f.write("LSM Model Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Model info
            if 'model_info' in report:
                f.write(f"Model Layers: {report['model_info']['layers']}\n")
                f.write(f"Total Parameters: {report['model_info']['total_params']:,}\n\n")
            
            # Complexity analysis
            if 'complexity_analysis' in report and 'error' not in report['complexity_analysis']:
                complexity = report['complexity_analysis']
                f.write("Model Complexity:\n")
                f.write(f"  - Trainable Parameters: {complexity['trainable_parameters']:,}\n")
                f.write(f"  - Parameter Efficiency: {complexity['parameter_efficiency']:.1%}\n")
                f.write(f"  - Estimated Memory: {complexity['estimated_memory_mb']:.1f} MB\n\n")
            
            # Speed profile
            if 'speed_profile' in report and 'error' not in report['speed_profile']:
                speed = report['speed_profile']
                f.write("Performance Profile:\n")
                f.write(f"  - Average Inference Time: {speed['avg_batch_time_ms']:.1f} ms/batch\n")
                f.write(f"  - Tokens per Second: {speed['tokens_per_second']:.0f}\n")
                f.write(f"  - Per Sample Time: {speed['avg_per_sample_ms']:.1f} ms\n\n")
            
            # Prediction analysis
            if 'prediction_analysis' in report and 'error' not in report['prediction_analysis']:
                pred = report['prediction_analysis']
                f.write("Prediction Analysis:\n")
                f.write(f"  - Average Confidence: {pred['avg_confidence']:.3f}\n")
                f.write(f"  - Average Entropy: {pred['avg_entropy']:.3f}\n")
                f.write(f"  - Samples Analyzed: {len(pred['sample_predictions'])}\n\n")
                
                if pred['most_predicted_tokens']:
                    f.write("  Most Predicted Tokens:\n")
                    for token, freq in pred['most_predicted_tokens'][:5]:
                        f.write(f"    '{token}': {freq:.3f}\n")
                    f.write("\n")


def create_performance_plots(analysis_report: Dict[str, Any], output_dir: str):
    """Create visualization plots for model analysis."""
    logger.info("Creating performance plots...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot 1: Parameter distribution by layer
    if 'complexity_analysis' in analysis_report and 'layer_analysis' in analysis_report['complexity_analysis']:
        layers = analysis_report['complexity_analysis']['layer_analysis']
        layer_names = [layer['name'][:15] for layer in layers]
        layer_params = [layer['parameters'] for layer in layers]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(layer_names)), layer_params)
        plt.title('Parameter Distribution by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Parameters')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Confidence distribution
    if 'prediction_analysis' in analysis_report and 'confidence_distribution' in analysis_report['prediction_analysis']:
        confidences = analysis_report['prediction_analysis']['confidence_distribution']
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Model Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info("Performance plots saved to: %s", output_dir)