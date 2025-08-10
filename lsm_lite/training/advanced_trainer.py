"""
Advanced training capabilities for LSM models.

This module provides enhanced training features including validation tracking,
learning rate scheduling, early stopping, and comprehensive metrics.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedLSMTrainer:
    """Enhanced training orchestrator with advanced features."""
    
    def __init__(self, tokenizer, embedder, reservoir, cnn, config):
        """
        Initialize advanced trainer.
        
        Args:
            tokenizer: Unified tokenizer
            embedder: Sinusoidal embedder
            reservoir: Sparse reservoir
            cnn: CNN processor
            config: LSM configuration
        """
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.reservoir = reservoir
        self.cnn = cnn
        self.config = config
        self._model = None
        self.training_history = {}
        self.best_model_path = None
        
        # Training callbacks
        self.callbacks = []
        
        logger.info("Advanced trainer initialized")
    
    def build_model_with_validation(self) -> tf.keras.Model:
        """Build complete LSM model pipeline with validation support."""
        # Input layer for token sequences
        input_layer = tf.keras.layers.Input(
            shape=(self.tokenizer.max_length,), 
            dtype=tf.int32, 
            name='token_input'
        )
        
        # Embedding layer
        embedded = self.embedder.embed(input_layer)
        
        # Reservoir processing
        reservoir_output = self.reservoir(embedded)
        
        # CNN processing
        cnn_output = self.cnn(reservoir_output)
        
        # Output layer for next token prediction
        output = tf.keras.layers.Dense(
            self.tokenizer.vocab_size,
            activation='softmax',
            name='token_output'
        )(cnn_output)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output, name='LSM_Advanced')
        
        # Compile with advanced optimizer
        optimizer = self._create_optimizer()
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                'sparse_top_k_categorical_accuracy',
                self._perplexity_metric
            ]
        )
        
        self._model = model
        logger.info("Advanced model built with %d parameters", model.count_params())
        return model
    
    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create optimizer with learning rate scheduling."""
        initial_learning_rate = getattr(self.config, 'learning_rate', 0.001)
        
        # Learning rate schedule
        if hasattr(self.config, 'use_lr_schedule') and self.config.use_lr_schedule:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=1000,
                alpha=0.1
            )
        else:
            lr_schedule = initial_learning_rate
        
        # Choose optimizer
        optimizer_name = getattr(self.config, 'optimizer', 'adam')
        
        if optimizer_name.lower() == 'adamw':
            return tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=getattr(self.config, 'weight_decay', 0.01)
            )
        elif optimizer_name.lower() == 'sgd':
            return tf.keras.optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=getattr(self.config, 'momentum', 0.9)
            )
        else:
            return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    def _perplexity_metric(self, y_true, y_pred):
        """Calculate perplexity as a metric."""
        # Calculate cross entropy
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # Return perplexity (exp of cross entropy)
        return tf.exp(tf.reduce_mean(cross_entropy))
    
    def setup_callbacks(self, validation_data=None, save_dir: str = "checkpoints") -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks for advanced features."""
        callbacks = []
        
        # Model checkpointing
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        checkpoint_path = os.path.join(save_dir, "best_model.keras")
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss' if validation_data else 'loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(model_checkpoint)
        self.best_model_path = checkpoint_path
        
        # Early stopping
        if hasattr(self.config, 'early_stopping') and self.config.early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=getattr(self.config, 'patience', 5),
                restore_best_weights=True,
                mode='min',
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        if hasattr(self.config, 'use_tensorboard') and self.config.use_tensorboard:
            log_dir = os.path.join(save_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
            callbacks.append(tensorboard)
        
        # Custom metrics logging
        metrics_callback = MetricsLoggingCallback()
        callbacks.append(metrics_callback)
        
        self.callbacks = callbacks
        return callbacks
    
    def train_with_validation(self, train_conversations: List[str], 
                             validation_conversations: Optional[List[str]] = None,
                             epochs: int = 10, batch_size: int = 32,
                             validation_split: float = 0.1) -> Dict[str, Any]:
        """
        Train model with validation and advanced features.
        
        Args:
            train_conversations: Training data
            validation_conversations: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation split ratio if validation_conversations is None
            
        Returns:
            Training history and metrics
        """
        logger.info("Starting advanced training with %d conversations", len(train_conversations))
        
        # Build model if not exists
        if self._model is None:
            self.build_model_with_validation()
        
        # Prepare data
        train_data = self._prepare_training_data(train_conversations, batch_size)
        
        # Handle validation data
        validation_data = None
        if validation_conversations:
            validation_data = self._prepare_training_data(validation_conversations, batch_size)
        elif validation_split > 0:
            # Split training data
            split_idx = int(len(train_conversations) * (1 - validation_split))
            val_conversations = train_conversations[split_idx:]
            train_conversations = train_conversations[:split_idx]
            
            train_data = self._prepare_training_data(train_conversations, batch_size)
            validation_data = self._prepare_training_data(val_conversations, batch_size)
        
        # Setup callbacks
        callbacks = self.setup_callbacks(validation_data)
        
        # Train model
        history = self._model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(validation_data or train_data)
        
        # Log training summary
        self._log_training_summary(final_metrics)
        
        return {
            'history': self.training_history,
            'final_metrics': final_metrics,
            'best_model_path': self.best_model_path
        }
    
    def _prepare_training_data(self, conversations: List[str], batch_size: int) -> tf.data.Dataset:
        """Prepare training data with advanced preprocessing."""
        # Tokenize all conversations
        all_sequences = []
        for conversation in conversations:
            tokenized = self.tokenizer.tokenize([conversation], padding=True, truncation=True)
            sequence = tokenized['input_ids'][0]
            
            # Create input-output pairs for language modeling
            for i in range(len(sequence) - 1):
                if sequence[i] != self.tokenizer.get_special_tokens()['pad_token_id']:
                    input_seq = sequence[:i+1]
                    target = sequence[i+1]
                    
                    # Pad input sequence
                    if len(input_seq) < self.tokenizer.max_length:
                        padding = [self.tokenizer.get_special_tokens()['pad_token_id']] * (self.tokenizer.max_length - len(input_seq))
                        input_seq = padding + input_seq
                    
                    all_sequences.append((input_seq[-self.tokenizer.max_length:], target))
        
        # Convert to tensorflow dataset
        if not all_sequences:
            raise ValueError("No valid training sequences found")
        
        inputs = np.array([seq[0] for seq in all_sequences])
        targets = np.array([seq[1] for seq in all_sequences])
        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.shuffle(buffer_size=min(10000, len(all_sequences)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info("Prepared dataset with %d sequences, batch_size=%d", len(all_sequences), batch_size)
        return dataset
    
    def _calculate_final_metrics(self, validation_data) -> Dict[str, float]:
        """Calculate comprehensive final metrics."""
        if self._model is None:
            return {}
        
        # Evaluate on validation data
        eval_results = self._model.evaluate(validation_data, verbose=0)
        metric_names = self._model.metrics_names
        
        metrics = {}
        for name, value in zip(metric_names, eval_results):
            metrics[name] = float(value)
        
        # Additional custom metrics
        try:
            # Calculate BLEU score approximation
            metrics['bleu_approx'] = self._calculate_bleu_approximation(validation_data)
        except Exception as e:
            logger.warning("Could not calculate BLEU approximation: %s", e)
            metrics['bleu_approx'] = 0.0
        
        return metrics
    
    def _calculate_bleu_approximation(self, validation_data) -> float:
        """Calculate a simple BLEU score approximation."""
        # This is a simplified version - in practice you'd use a proper BLEU implementation
        total_score = 0.0
        count = 0
        
        for batch in validation_data.take(5):  # Sample a few batches
            inputs, targets = batch
            predictions = self._model.predict(inputs, verbose=0)
            predicted_tokens = np.argmax(predictions, axis=-1)
            
            for pred, target in zip(predicted_tokens, targets):
                # Simple token overlap metric
                pred_tokens = set(pred.numpy().tolist())
                target_tokens = set([int(target.numpy())])
                
                if target_tokens:
                    overlap = len(pred_tokens.intersection(target_tokens))
                    score = overlap / len(target_tokens)
                    total_score += score
                    count += 1
        
        return total_score / max(count, 1)
    
    def _log_training_summary(self, final_metrics: Dict[str, float]):
        """Log comprehensive training summary."""
        logger.info("=== Training Summary ===")
        logger.info("Final Metrics:")
        for metric, value in final_metrics.items():
            logger.info("  %s: %.4f", metric, value)
        
        if 'loss' in self.training_history:
            logger.info("Training Loss: %.4f -> %.4f", 
                       self.training_history['loss'][0], 
                       self.training_history['loss'][-1])
        
        if 'val_loss' in self.training_history:
            logger.info("Validation Loss: %.4f -> %.4f", 
                       self.training_history['val_loss'][0], 
                       self.training_history['val_loss'][-1])
        
        logger.info("Best model saved to: %s", self.best_model_path)
        logger.info("========================")


class MetricsLoggingCallback(tf.keras.callbacks.Callback):
    """Custom callback for detailed metrics logging."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log detailed metrics at the end of each epoch."""
        if logs is None:
            logs = {}
        
        # Log key metrics
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        accuracy = logs.get('accuracy', 0)
        val_accuracy = logs.get('val_accuracy', 0)
        
        logger.info(
            "Epoch %d - Loss: %.4f, Val Loss: %.4f, Acc: %.4f, Val Acc: %.4f",
            epoch + 1, loss, val_loss, accuracy, val_accuracy
        )


class LSMModelAnalyzer:
    """Advanced model analysis and visualization tools."""
    
    def __init__(self, model, tokenizer, embedder):
        """Initialize model analyzer."""
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        
    def analyze_reservoir_dynamics(self, text_samples: List[str]) -> Dict[str, Any]:
        """Analyze reservoir state dynamics."""
        logger.info("Analyzing reservoir dynamics on %d samples", len(text_samples))
        
        reservoir_states = []
        for text in text_samples[:10]:  # Limit for performance
            tokenized = self.tokenizer.tokenize([text], padding=True, truncation=True)
            input_tensor = tf.constant(tokenized['input_ids'], dtype=tf.int32)
            
            # Get intermediate outputs
            embedded = self.embedder.embed(input_tensor)
            reservoir_output = self.model.layers[1](embedded)  # Assuming reservoir is second layer
            
            reservoir_states.append(reservoir_output.numpy())
        
        # Calculate statistics
        states_concat = np.concatenate(reservoir_states, axis=0)
        
        analysis = {
            'mean_activation': float(np.mean(states_concat)),
            'std_activation': float(np.std(states_concat)),
            'sparsity': float(np.mean(states_concat == 0)),
            'max_activation': float(np.max(states_concat)),
            'min_activation': float(np.min(states_concat)),
        }
        
        logger.info("Reservoir analysis complete: sparsity=%.3f, mean_act=%.3f", 
                   analysis['sparsity'], analysis['mean_activation'])
        
        return analysis
    
    def generate_attention_heatmap(self, text: str) -> np.ndarray:
        """Generate attention-like heatmap for model analysis."""
        tokenized = self.tokenizer.tokenize([text], padding=True, truncation=True)
        input_tensor = tf.constant(tokenized['input_ids'], dtype=tf.int32)
        
        # Get gradients with respect to input
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            embedded = self.embedder.embed(tf.cast(input_tensor, tf.float32))
            predictions = self.model(tf.cast(input_tensor, tf.int32))
            loss = tf.reduce_mean(predictions)
        
        gradients = tape.gradient(loss, embedded)
        
        # Convert gradients to attention-like scores
        attention_scores = tf.reduce_mean(tf.abs(gradients), axis=-1)
        
        return attention_scores.numpy()
    
    def export_model_summary(self, save_path: str):
        """Export comprehensive model summary."""
        summary_data = {
            'model_architecture': [],
            'total_parameters': self.model.count_params(),
            'tokenizer_info': {
                'backend': self.tokenizer.backend,
                'vocab_size': self.tokenizer.vocab_size,
                'max_length': self.tokenizer.max_length
            },
            'layer_details': []
        }
        
        # Get model architecture
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'unknown',
                'parameters': layer.count_params() if hasattr(layer, 'count_params') else 0
            }
            summary_data['layer_details'].append(layer_info)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info("Model summary exported to: %s", save_path)