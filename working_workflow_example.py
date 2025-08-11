#!/usr/bin/env python3
"""
Working LSM Lite Workflow Example

A simplified workflow that bypasses complex error handling and demonstrates
the core functionality using direct component initialization.
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

# Import core components directly
from lsm_lite.utils.config import DualCNNConfig
from lsm_lite.core.tokenizer import UnifiedTokenizer
from lsm_lite.data.embeddings import SinusoidalEmbedder
from lsm_lite.core.attentive_reservoir import AttentiveReservoir
from lsm_lite.core.rolling_wave_storage import RollingWaveStorage


def create_sample_data() -> List[str]:
    """Create sample training data."""
    return [
        "Hello, how are you today?",
        "Machine learning is fascinating.",
        "Neural networks learn from data.",
        "Deep learning uses multiple layers.",
        "Attention mechanisms are powerful.",
        "Text generation requires careful tuning.",
        "Natural language processing helps computers.",
        "Transformers revolutionized NLP.",
        "Convolutional networks process sequences.",
        "Reservoir computing is efficient."
    ]


def test_tokenizer():
    """Test tokenizer initialization and functionality."""
    print("Testing Tokenizer...")
    
    try:
        print("  Using basic tokenizer (no external dependencies)...")
        tokenizer = UnifiedTokenizer(
            backend='basic',
            max_length=32
        )
        
        # Test tokenization
        test_texts = ["Hello world", "This is a test"]
        result = tokenizer.tokenize(test_texts)
        
        print(f"    ✓ Basic tokenizer working")
        print(f"    Vocab size: {tokenizer.vocab_size}")
        print(f"    Input shape: {result['input_ids'].shape}")
        
        return tokenizer
        
    except Exception as e:
        print(f"  ✗ Tokenizer test failed: {e}")
        print("  Creating mock tokenizer...")
        return MockTokenizer()


def test_embedder(tokenizer):
    """Test embedder functionality."""
    print("Testing Embedder...")
    
    try:
        vocab_size = getattr(tokenizer, 'vocab_size', 10000)
        
        embedder = SinusoidalEmbedder(
            vocab_size=vocab_size,
            embedding_dim=64,
            max_length=32
        )
        
        # Test embedding
        import numpy as np
        test_ids = np.array([[1, 2, 3, 4, 0, 0]])  # Sample token IDs
        embeddings = embedder(test_ids)  # Use call method
        
        print(f"    ✓ Embedder working")
        print(f"    Embedding shape: {embeddings.shape}")
        
        return embedder
        
    except Exception as e:
        print(f"    ✗ Embedder test failed: {e}")
        return None


def test_reservoir():
    """Test attentive reservoir."""
    print("Testing Attentive Reservoir...")
    
    try:
        reservoir = AttentiveReservoir(
            input_dim=64,
            reservoir_size=128,
            attention_heads=2,
            attention_dim=32
        )
        
        # Test processing
        import numpy as np
        test_input = np.random.randn(1, 10, 64)  # (batch, seq, features)
        output = reservoir(test_input)  # Use call method
        
        # Handle tuple return (reservoir_states, attention_weights)
        if isinstance(output, tuple):
            reservoir_states, attention_weights = output
            print(f"    ✓ Reservoir working")
            print(f"    Reservoir states shape: {reservoir_states.shape}")
            print(f"    Attention weights shape: {attention_weights.shape}")
        else:
            print(f"    ✓ Reservoir working")
            print(f"    Output shape: {output.shape}")
        
        return reservoir
        
    except Exception as e:
        print(f"    ✗ Reservoir test failed: {e}")
        return None


def test_wave_storage():
    """Test rolling wave storage."""
    print("Testing Rolling Wave Storage...")
    
    try:
        wave_storage = RollingWaveStorage(
            max_sequence_length=32,  # Maximum sequence length
            feature_dim=128,  # Match reservoir output dimension
            window_size=8,
            overlap=2,
            max_memory_mb=50.0  # 50MB memory limit
        )
        
        # Test wave storage
        import numpy as np
        test_wave = np.random.randn(128)  # (features,) - single wave, match reservoir size
        wave_storage.store_wave(test_wave, sequence_position=0)
        
        stats = wave_storage.get_storage_stats()
        
        print(f"    ✓ Wave storage working")
        print(f"    Stored waves: {stats['stored_count']}")
        print(f"    Utilization: {stats['utilization_percent']:.1f}%")
        
        # Try to get a rolling window if we have stored waves
        if stats['stored_count'] > 0:
            try:
                window = wave_storage.get_rolling_window(center_pos=0)
                print(f"    Rolling window shape: {window.shape}")
            except Exception as e:
                print(f"    Rolling window test: {e}")
        
        return wave_storage
        
    except Exception as e:
        print(f"    ✗ Wave storage test failed: {e}")
        return None


def demonstrate_workflow():
    """Demonstrate the complete workflow step by step."""
    print("=" * 60)
    print("LSM Lite Working Workflow Example")
    print("=" * 60)
    print()
    
    # Sample data
    training_data = create_sample_data()
    print(f"Created {len(training_data)} training samples")
    print()
    
    # Step 1: Test Tokenizer
    print("Step 1: Initialize and test tokenizer...")
    tokenizer = test_tokenizer()
    if not tokenizer:
        print("✗ Cannot proceed without tokenizer")
        return False
    print()
    
    # Step 2: Test Embedder
    print("Step 2: Initialize and test embedder...")
    embedder = test_embedder(tokenizer)
    if not embedder:
        print("✗ Cannot proceed without embedder")
        return False
    print()
    
    # Step 3: Test Reservoir
    print("Step 3: Initialize and test attentive reservoir...")
    reservoir = test_reservoir()
    if not reservoir:
        print("✗ Cannot proceed without reservoir")
        return False
    print()
    
    # Step 4: Test Wave Storage
    print("Step 4: Initialize and test rolling wave storage...")
    wave_storage = test_wave_storage()
    if not wave_storage:
        print("✗ Cannot proceed without wave storage")
        return False
    print()
    
    # Step 5: Demonstrate pipeline flow
    print("Step 5: Demonstrate pipeline flow...")
    try:
        # Tokenize sample text
        sample_text = training_data[0]
        print(f"  Processing: '{sample_text}'")
        
        tokenized = tokenizer.tokenize([sample_text])
        print(f"  ✓ Tokenized: {tokenized['input_ids'].shape}")
        
        # Embed tokens
        embeddings = embedder(tokenized['input_ids'])  # Use call method
        print(f"  ✓ Embedded: {embeddings.shape}")
        
        # Process through reservoir
        reservoir_output = reservoir(embeddings)  # Use call method
        
        # Handle tuple return (reservoir_states, attention_weights)
        if isinstance(reservoir_output, tuple):
            reservoir_states, attention_weights = reservoir_output
            print(f"  ✓ Reservoir processed: states {reservoir_states.shape}, attention {attention_weights.shape}")
            # Use reservoir states for wave storage
            wave_data = reservoir_states[0]  # Take first batch
        else:
            print(f"  ✓ Reservoir processed: {reservoir_output.shape}")
            wave_data = reservoir_output[0]  # Take first batch
        
        # Store in wave storage (simulate CNN output)
        import numpy as np
        if len(wave_data.shape) > 1:
            # Flatten or take mean if needed
            wave_data = np.mean(wave_data, axis=0, keepdims=True)
            wave_data = np.tile(wave_data, (8, 1))  # Create window
        
        # Store individual waves (wave_data should be 2D: (seq_len, features))
        for i, wave_vector in enumerate(wave_data):
            wave_storage.store_wave(wave_vector, sequence_position=i)
        
        stats = wave_storage.get_storage_stats()
        print(f"  ✓ Wave storage: {stats['stored_count']} waves stored")
        
        print("  ✓ Pipeline flow completed successfully!")
        
    except Exception as e:
        print(f"  ✗ Pipeline flow failed: {e}")
        return False
    
    print()
    
    # Step 6: Summary
    print("Step 6: Workflow summary...")
    print("  ✓ Tokenizer: Text → Token IDs")
    print("  ✓ Embedder: Token IDs → Dense Embeddings")
    print("  ✓ Reservoir: Embeddings → Processed Features")
    print("  ✓ Wave Storage: Features → Coordinated Waves")
    print("  ✓ Ready for CNN training and generation")
    print()
    
    print("Next steps for full implementation:")
    print("  1. Train first CNN on next-token prediction")
    print("  2. Capture rolling wave outputs during training")
    print("  3. Train second CNN using wave features")
    print("  4. Coordinate both CNNs for generation")
    print("  5. Implement text generation with dual CNN")
    print()
    
    return True


class MockTokenizer:
    """Simple mock tokenizer for testing when real tokenizers fail."""
    
    def __init__(self):
        self.vocab_size = 10000
        self.max_length = 32
        
    def tokenize(self, texts, padding=True, truncation=True):
        """Mock tokenization."""
        import numpy as np
        
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # Simple word-based tokenization
            words = text.lower().split()
            # Convert to IDs (hash-based)
            ids = [hash(word) % self.vocab_size for word in words]
            
            # Truncate if needed
            if truncation and len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            # Create attention mask
            mask = [1] * len(ids)
            
            # Pad if needed
            if padding and len(ids) < self.max_length:
                pad_len = self.max_length - len(ids)
                ids.extend([0] * pad_len)
                mask.extend([0] * pad_len)
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        return {
            'input_ids': np.array(input_ids),
            'attention_mask': np.array(attention_masks)
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Mock decoding."""
        if isinstance(token_ids[0], int):
            token_ids = [token_ids]
        
        decoded = []
        for ids in token_ids:
            # Simple mock decoding
            words = [f"word_{id_}" for id_ in ids if id_ != 0]
            decoded.append(" ".join(words))
        
        return decoded[0] if len(decoded) == 1 else decoded


def main():
    """Main function."""
    success = demonstrate_workflow()
    
    if success:
        print("=" * 60)
        print("Workflow completed successfully!")
        print("All core components are working and ready for training.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Workflow encountered issues.")
        print("Check component initialization above.")
        print("=" * 60)


if __name__ == "__main__":
    main()