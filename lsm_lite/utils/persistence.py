"""
Model persistence functionality for LSM Lite.

This module provides utilities for saving and loading complete LSM models
including all components (model, tokenizer, embedder, configuration).
"""

import os
import json
import pickle
import logging
import shutil
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow import keras

from .config import LSMConfig
from ..core.tokenizer import UnifiedTokenizer
from ..data.embeddings import SinusoidalEmbedder

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Simple model save/load functionality."""
    
    MODEL_DIR = "model"
    TOKENIZER_DIR = "tokenizer"
    EMBEDDER_DIR = "embedder"
    CONFIG_FILE = "config.json"
    METADATA_FILE = "metadata.json"
    
    @staticmethod
    def save_model(model: keras.Model, tokenizer: UnifiedTokenizer, 
                   embedder: SinusoidalEmbedder, config: LSMConfig, 
                   path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save complete model to directory.
        
        Args:
            model: Trained TensorFlow/Keras model
            tokenizer: Tokenizer instance
            embedder: Embedder instance
            config: Model configuration
            path: Directory path to save to
            metadata: Optional metadata to save
        """
        logger.info("Saving complete LSM model to: %s", path)
        
        # Create directory structure
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, ModelPersistence.MODEL_DIR + '.keras')
        tokenizer_path = os.path.join(path, ModelPersistence.TOKENIZER_DIR)
        embedder_path = os.path.join(path, ModelPersistence.EMBEDDER_DIR)
        
        try:
            # Save TensorFlow model
            logger.info("Saving TensorFlow model...")
            model.save(model_path)  # Keras 3 with proper .keras extension
            
            # Save tokenizer
            logger.info("Saving tokenizer...")
            ModelPersistence._save_tokenizer(tokenizer, tokenizer_path)
            
            # Save embedder
            logger.info("Saving embedder...")
            ModelPersistence._save_embedder(embedder, embedder_path)
            
            # Save configuration
            logger.info("Saving configuration...")
            config_path = os.path.join(path, ModelPersistence.CONFIG_FILE)
            config.save(config_path)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'lsm_lite_version': '0.1.0',
                'tokenizer_backend': tokenizer.backend,
                'vocab_size': tokenizer.vocab_size,
                'embedding_dim': embedder.embedding_dim,
                'max_length': tokenizer.max_length,
                'model_components': {
                    'tokenizer': True,
                    'embedder': True,
                    'reservoir': True,
                    'cnn': True
                }
            })
            
            metadata_path = os.path.join(path, ModelPersistence.METADATA_FILE)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Model saved successfully to: %s", path)
            
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            # Clean up partial save
            if os.path.exists(path):
                shutil.rmtree(path)
            raise
    
    @staticmethod
    def _save_tokenizer(tokenizer: UnifiedTokenizer, path: str) -> None:
        """Save tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save tokenizer configuration and state
        tokenizer_config = {
            'backend': tokenizer.backend,
            'vocab_size': tokenizer.vocab_size,
            'max_length': tokenizer.max_length,
        }
        
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # Save backend-specific tokenizer if possible
        try:
            if hasattr(tokenizer._tokenizer, 'save_pretrained'):
                # HuggingFace tokenizers
                backend_path = os.path.join(path, 'backend')
                os.makedirs(backend_path, exist_ok=True)
                tokenizer._tokenizer.save_pretrained(backend_path)
            else:
                # For spaCy or other tokenizers, save with pickle
                backend_path = os.path.join(path, 'backend.pkl')
                with open(backend_path, 'wb') as f:
                    pickle.dump(tokenizer._tokenizer, f)
        except Exception as e:
            logger.warning("Failed to save tokenizer backend: %s", e)
            # Fallback to pickle save
            backend_path = os.path.join(path, 'backend.pkl')
            with open(backend_path, 'wb') as f:
                pickle.dump(tokenizer._tokenizer, f)
    
    @staticmethod
    def _save_embedder(embedder: SinusoidalEmbedder, path: str) -> None:
        """Save embedder to directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save embedder weights and configuration
        embedder_config = embedder.get_config()
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(embedder_config, f, indent=2)
        
        # Save weights
        weights_path = os.path.join(path, 'weights.h5')
        embedder.save_weights(weights_path)
    
    @staticmethod
    def load_model(path: str) -> Dict[str, Any]:
        """
        Load complete model from directory.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Dictionary containing loaded components
        """
        logger.info("Loading LSM model from: %s", path)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        try:
            # Load metadata first to check compatibility
            metadata_path = os.path.join(path, ModelPersistence.METADATA_FILE)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info("Loaded metadata: %s", metadata.get('lsm_lite_version', 'unknown'))
            else:
                metadata = {}
            
            # Load configuration
            config_path = os.path.join(path, ModelPersistence.CONFIG_FILE)
            config = LSMConfig.load(config_path)
            
            # Load tokenizer
            tokenizer_path = os.path.join(path, ModelPersistence.TOKENIZER_DIR)
            tokenizer = ModelPersistence._load_tokenizer(tokenizer_path, config)
            
            # Load embedder
            embedder_path = os.path.join(path, ModelPersistence.EMBEDDER_DIR)
            vocab_size = tokenizer.vocab_size or 10000  # Default if None
            embedder = ModelPersistence._load_embedder(embedder_path, config, vocab_size)
            
            # Load TensorFlow model
            model_path = os.path.join(path, ModelPersistence.MODEL_DIR + '.keras')
            model = keras.models.load_model(model_path, compile=False)
            
            logger.info("Model loaded successfully from: %s", path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'embedder': embedder,
                'config': config,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise
    
    @staticmethod
    def _load_tokenizer(path: str, config: LSMConfig) -> UnifiedTokenizer:
        """Load tokenizer from directory."""
        # Load tokenizer config
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            tokenizer_config = json.load(f)
        
        # Create tokenizer instance
        tokenizer = UnifiedTokenizer(
            backend=tokenizer_config['backend'],
            vocab_size=tokenizer_config['vocab_size'],
            max_length=tokenizer_config['max_length']
        )
        
        # Try to load backend-specific tokenizer
        backend_path = os.path.join(path, 'backend')
        backend_pkl_path = os.path.join(path, 'backend.pkl')
        
        if os.path.exists(backend_path) and hasattr(tokenizer._tokenizer, 'from_pretrained'):
            # HuggingFace tokenizers
            if tokenizer.backend == 'gpt2':
                from transformers import GPT2Tokenizer
                tokenizer._tokenizer = GPT2Tokenizer.from_pretrained(backend_path)
            elif tokenizer.backend == 'bert':
                from transformers import BertTokenizer
                tokenizer._tokenizer = BertTokenizer.from_pretrained(backend_path)
        elif os.path.exists(backend_pkl_path):
            # Pickled tokenizer
            with open(backend_pkl_path, 'rb') as f:
                tokenizer._tokenizer = pickle.load(f)
        
        return tokenizer
    
    @staticmethod
    def _load_embedder(path: str, config: LSMConfig, vocab_size: int) -> SinusoidalEmbedder:
        """Load embedder from directory."""
        # Load embedder config
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            embedder_config = json.load(f)
        
        # Create embedder instance
        embedder = SinusoidalEmbedder(
            vocab_size=vocab_size,
            embedding_dim=embedder_config['embedding_dim'],
            max_length=embedder_config['max_length'],
            temperature=embedder_config.get('temperature', 10000.0),
            trainable_embeddings=embedder_config.get('trainable_embeddings', True)
        )
        
        # Load weights
        weights_path = os.path.join(path, 'weights.h5')
        if os.path.exists(weights_path):
            # Build the layer first with dummy input
            dummy_input = tf.constant([[1, 2, 3]], dtype=tf.int32)
            embedder(dummy_input)  # This builds the layer
            embedder.load_weights(weights_path)
        
        return embedder
    
    @staticmethod
    def verify_model_integrity(path: str) -> Dict[str, bool]:
        """
        Verify integrity of saved model.
        
        Args:
            path: Path to model directory
            
        Returns:
            Dictionary with integrity check results
        """
        results = {
            'directory_exists': False,
            'model_exists': False,
            'tokenizer_exists': False,
            'embedder_exists': False,
            'config_exists': False,
            'metadata_exists': False,
            'all_components_present': False
        }
        
        if not os.path.exists(path):
            return results
        
        results['directory_exists'] = True
        
        # Check each component
        model_path = os.path.join(path, ModelPersistence.MODEL_DIR)
        tokenizer_path = os.path.join(path, ModelPersistence.TOKENIZER_DIR)
        embedder_path = os.path.join(path, ModelPersistence.EMBEDDER_DIR)
        config_path = os.path.join(path, ModelPersistence.CONFIG_FILE)
        metadata_path = os.path.join(path, ModelPersistence.METADATA_FILE)
        
        results['model_exists'] = os.path.exists(model_path)
        results['tokenizer_exists'] = os.path.exists(tokenizer_path)
        results['embedder_exists'] = os.path.exists(embedder_path)
        results['config_exists'] = os.path.exists(config_path)
        results['metadata_exists'] = os.path.exists(metadata_path)
        
        # Check if all essential components are present
        results['all_components_present'] = all([
            results['model_exists'],
            results['tokenizer_exists'],
            results['embedder_exists'],
            results['config_exists']
        ])
        
        return results
    
    @staticmethod
    def get_model_info(path: str) -> Dict[str, Any]:
        """
        Get information about saved model without loading it.
        
        Args:
            path: Path to model directory
            
        Returns:
            Dictionary with model information
        """
        info = {
            'path': path,
            'exists': False,
            'integrity': {},
            'metadata': {},
            'config': {},
            'size_mb': 0
        }
        
        if not os.path.exists(path):
            return info
        
        info['exists'] = True
        info['integrity'] = ModelPersistence.verify_model_integrity(path)
        
        # Get directory size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        info['size_mb'] = total_size / (1024 * 1024)
        
        # Load metadata if available
        metadata_path = os.path.join(path, ModelPersistence.METADATA_FILE)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    info['metadata'] = json.load(f)
            except Exception as e:
                logger.warning("Failed to load metadata: %s", e)
        
        # Load config if available
        config_path = os.path.join(path, ModelPersistence.CONFIG_FILE)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    info['config'] = json.load(f)
            except Exception as e:
                logger.warning("Failed to load config: %s", e)
        
        return info
    
    @staticmethod
    def list_saved_models(models_dir: str = "saved_models") -> List[Dict[str, Any]]:
        """
        List all saved models in a directory.
        
        Args:
            models_dir: Directory containing saved models
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        if not os.path.exists(models_dir):
            return models
        
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                model_info = ModelPersistence.get_model_info(item_path)
                model_info['name'] = item
                models.append(model_info)
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
        
        return models
    
    @staticmethod
    def cleanup_old_models(models_dir: str = "saved_models", 
                          keep_latest: int = 5) -> int:
        """
        Clean up old saved models, keeping only the latest ones.
        
        Args:
            models_dir: Directory containing saved models
            keep_latest: Number of latest models to keep
            
        Returns:
            Number of models deleted
        """
        models = ModelPersistence.list_saved_models(models_dir)
        
        if len(models) <= keep_latest:
            return 0
        
        models_to_delete = models[keep_latest:]
        deleted_count = 0
        
        for model_info in models_to_delete:
            try:
                shutil.rmtree(model_info['path'])
                logger.info("Deleted old model: %s", model_info['name'])
                deleted_count += 1
            except Exception as e:
                logger.error("Failed to delete model %s: %s", model_info['name'], e)
        
        return deleted_count


class ModelConverter:
    """Utility class for converting models between formats."""
    
    @staticmethod
    def export_to_onnx(model: keras.Model, output_path: str, 
                      input_shape: Optional[Tuple[int, ...]] = None) -> bool:
        """
        Export TensorFlow model to ONNX format.
        
        Args:
            model: TensorFlow model to export
            output_path: Path to save ONNX model
            input_shape: Input shape for the model
            
        Returns:
            True if export was successful
        """
        try:
            import tf2onnx
            
            # Convert model to ONNX
            spec = (tf.TensorSpec(input_shape, tf.int32, name="input"),)
            output_path = output_path if output_path.endswith('.onnx') else f"{output_path}.onnx"
            
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
            
            with open(output_path, "wb") as f:
                f.write(model_proto.SerializeToString())
            
            logger.info("Model exported to ONNX: %s", output_path)
            return True
            
        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
            return False
        except Exception as e:
            logger.error("Failed to export to ONNX: %s", e)
            return False
    
    @staticmethod
    def export_to_tflite(model: keras.Model, output_path: str,
                        quantize: bool = False) -> bool:
        """
        Export TensorFlow model to TensorFlow Lite format.
        
        Args:
            model: TensorFlow model to export
            output_path: Path to save TFLite model
            quantize: Whether to apply quantization
            
        Returns:
            True if export was successful
        """
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            output_path = output_path if output_path.endswith('.tflite') else f"{output_path}.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info("Model exported to TensorFlow Lite: %s", output_path)
            return True
            
        except Exception as e:
            logger.error("Failed to export to TensorFlow Lite: %s", e)
            return False
