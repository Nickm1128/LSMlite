"""
Unified training orchestrator for LSM models.

This module provides a single trainer class that coordinates all model components
and handles the complete training pipeline for the LSM architecture.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..core.tokenizer import UnifiedTokenizer
from ..core.reservoir import SparseReservoir
from ..core.cnn import CNNProcessor
from ..data.embeddings import SinusoidalEmbedder
from ..utils.config import LSMConfig

logger = logging.getLogger(__name__)


class LSMTrainer:
    """Unified training orchestrator for LSM models."""
    
    def __init__(self, tokenizer: UnifiedTokenizer, embedder: SinusoidalEmbedder,
                 reservoir: SparseReservoir, cnn: CNNProcessor, config: LSMConfig):
        """
        Initialize LSM trainer.
        
        Args:
            tokenizer: Unified tokenizer instance
            embedder: Sinusoidal embedder instance
            reservoir: Sparse reservoir instance
            cnn: CNN processor instance
            config: Training configuration
        """
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.reservoir = reservoir
        self.cnn = cnn
        self.config = config
        
        # Training components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        
        # Training state
        self.training_history = {}
        self.current_epoch = 0
        
        # Build the complete model
        self._build_model()
        
        logger.info("LSM trainer initialized with config: %s", config)
    
    def _build_model(self):
        """Build complete LSM model pipeline."""
        logger.info("Building LSM model pipeline...")
        
        # Create input layer
        input_layer = keras.layers.Input(
            shape=(self.config.max_length,),
            dtype=tf.int32,
            name='token_input'
        )
        
        # Embedding layer
        embedded = self.embedder(input_layer)
        
        # Reservoir layer
        reservoir_output = self.reservoir(embedded)
        
        # CNN processor
        cnn_output = self.cnn(reservoir_output)
        
        # Create model
        self.model = keras.Model(
            inputs=input_layer,
            outputs=cnn_output,
            name='lsm_model'
        )
        
        # Setup training components
        self._setup_training()
        
        logger.info("LSM model built successfully")
        self.model.summary(print_fn=logger.info)
    
    def _setup_training(self):
        """Setup optimizer, loss function, and metrics."""
        # Optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        # Loss function (sparse categorical crossentropy)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        # Metrics
        self.metrics = [
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
        
        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics
        )
    
    def prepare_training_data(self, conversations: List[str]) -> Tuple[tf.data.Dataset, int]:
        """
        Prepare training data from conversations.
        
        Args:
            conversations: List of conversation strings
            
        Returns:
            Tuple of (dataset, num_samples)
        """
        logger.info("Preparing training data from %d conversations...", len(conversations))
        
        if not conversations:
            raise ValueError("No conversations provided for training")
        
        # Debug: Show sample conversations
        logger.info("Sample conversation: %s", conversations[0][:100] + "..." if len(conversations[0]) > 100 else conversations[0])
        
        # Create input-target pairs
        input_sequences = []
        target_sequences = []
        
        for i, conv in enumerate(conversations):
            if not conv or not conv.strip():
                logger.warning("Empty conversation at index %d, skipping", i)
                continue
                
            # Tokenize conversation
            try:
                tokenized = self.tokenizer.tokenize([conv], padding=True, truncation=True)
                input_ids = tokenized['input_ids'][0]  # Get first (and only) sequence
                
                # Debug: Check tokenization result
                if i == 0:  # Log first conversation's tokenization
                    logger.info("First conversation tokenized to %d tokens", len(input_ids))
                    logger.info("First few tokens: %s", input_ids[:10])
                
                # Create shifted sequences for language modeling
                # Input: [BOS, token1, token2, ..., tokenN-1]
                # Target: [token1, token2, ..., tokenN-1, EOS]
                
                valid_tokens = [token for token in input_ids if token != 0]  # Remove padding
                
                if len(valid_tokens) < 2:
                    logger.warning("Conversation %d has too few valid tokens (%d), skipping", i, len(valid_tokens))
                    continue
                
                for j in range(len(valid_tokens) - 1):
                    # Create context window
                    start_idx = max(0, j - self.config.max_length + 1)
                    input_seq = valid_tokens[start_idx:j+1]
                    target_token = valid_tokens[j+1]
                    
                    # Pad input sequence if needed
                    if len(input_seq) < self.config.max_length:
                        padding = [0] * (self.config.max_length - len(input_seq))
                        input_seq = list(padding) + list(input_seq)
                    
                    input_sequences.append(input_seq)
                    target_sequences.append(target_token)
                    
            except Exception as e:
                logger.error("Error processing conversation %d: %s", i, e)
                continue
        
        if not input_sequences:
            raise ValueError("No valid training sequences created from conversations. Check that conversations contain meaningful text.")
        
        # Convert to tensors
        input_tensor = tf.constant(input_sequences, dtype=tf.int32)
        target_tensor = tf.constant(target_sequences, dtype=tf.int32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
        dataset = dataset.shuffle(buffer_size=min(10000, len(input_sequences)))
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info("Prepared %d training samples from %d conversations", len(input_sequences), len(conversations))
        return dataset, len(input_sequences)
    
    def train(self, conversations: List[str], epochs: int = None, 
              batch_size: int = None, validation_split: float = 0.1) -> Dict[str, Any]:
        """
        Train the complete LSM model.
        
        Args:
            conversations: List of conversation strings for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        
        logger.info("Starting LSM training for %d epochs...", epochs)
        start_time = time.time()
        
        # Prepare data
        train_dataset, num_samples = self.prepare_training_data(conversations)
        
        # Split for validation if requested
        if validation_split > 0:
            val_size = int(num_samples * validation_split)
            train_size = num_samples - val_size
            
            val_dataset = train_dataset.take(val_size // batch_size)
            train_dataset = train_dataset.skip(val_size // batch_size)
        else:
            val_dataset = None
        
        # Setup callbacks
        callbacks = self._create_callbacks()
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        self.current_epoch += epochs
        
        training_time = time.time() - start_time
        logger.info("Training completed in %.2f seconds", training_time)
        
        return self.training_history
    
    def _create_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if 'val_loss' in self.model.metrics_names else 'loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if 'val_loss' in self.model.metrics_names else 'loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Training progress logging
        progress_callback = TrainingProgressCallback()
        callbacks.append(progress_callback)
        
        return callbacks
    
    def evaluate(self, test_conversations: List[str]) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_conversations: List of test conversation strings
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on %d test conversations...", len(test_conversations))
        
        if self.model is None:
            raise ValueError("Model not built. Train the model first.")
        
        # Prepare test data
        test_dataset, num_samples = self.prepare_training_data(test_conversations)
        
        # Evaluate
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # Create results dictionary
        metric_names = self.model.metrics_names
        evaluation_results = dict(zip(metric_names, results))
        
        # Compute additional metrics
        additional_metrics = self._compute_additional_metrics(test_dataset)
        evaluation_results.update(additional_metrics)
        
        logger.info("Evaluation results: %s", evaluation_results)
        return evaluation_results
    
    def _compute_additional_metrics(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Compute additional evaluation metrics."""
        # Compute perplexity
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataset:
            predictions = self.model(batch_x, training=False)
            loss = self.loss_fn(batch_y, predictions)
            total_loss += loss.numpy()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': float(perplexity),
            'avg_loss': float(avg_loss)
        }
    
    def predict(self, input_texts: List[str]) -> np.ndarray:
        """
        Make predictions on input texts.
        
        Args:
            input_texts: List of input text strings
            
        Returns:
            Prediction probabilities array
        """
        if self.model is None:
            raise ValueError("Model not built. Train the model first.")
        
        # Tokenize inputs
        tokenized = self.tokenizer.tokenize(input_texts, padding=True, truncation=True)
        input_tensor = tf.constant(tokenized['input_ids'], dtype=tf.int32)
        
        # Make predictions
        predictions = self.model(input_tensor, training=False)
        
        return predictions.numpy()
    
    def save_training_state(self, filepath: str):
        """Save training state for resuming training later."""
        state = {
            'current_epoch': self.current_epoch,
            'training_history': self.training_history,
            'config': self.config,
        }
        
        # Save as JSON
        import json
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_state = self._make_json_serializable(state)
            json.dump(serializable_state, f, indent=2)
        
        logger.info("Training state saved to: %s", filepath)
    
    def load_training_state(self, filepath: str):
        """Load training state to resume training."""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_epoch = state['current_epoch']
        self.training_history = state['training_history']
        # Note: config is not loaded to avoid overriding current config
        
        logger.info("Training state loaded from: %s", filepath)
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def get_model_summary(self) -> str:
        """Get a string summary of the model architecture."""
        if self.model is None:
            return "Model not built yet."
        
        # Create a string buffer to capture the summary
        import io
        summary_buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        summary = summary_buffer.getvalue()
        summary_buffer.close()
        
        return summary


class TrainingProgressCallback(keras.callbacks.Callback):
    """Custom callback for logging training progress."""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info("Starting epoch %d...", epoch + 1)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            logger.info("Epoch %d completed in %.2f seconds - %s", 
                       epoch + 1, epoch_time, logs or {})
    
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0 and batch > 0:
            logger.info("Batch %d - %s", batch, logs or {})
