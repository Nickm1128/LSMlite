#!/usr/bin/env python3
"""
Simple test to verify the basic tokenizer works correctly.
"""

from lsm_lite.core.tokenizer import UnifiedTokenizer

def test_basic_tokenizer():
    """Test the basic tokenizer directly."""
    print("Testing Basic Tokenizer...")
    
    try:
        # Create basic tokenizer
        tokenizer = UnifiedTokenizer(backend='basic', max_length=32)
        print(f"✓ Tokenizer created successfully")
        print(f"  Backend: {tokenizer.backend}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Max length: {tokenizer.max_length}")
        
        # Test tokenization
        test_texts = [
            "Hello world",
            "This is a test of the basic tokenizer",
            "Machine learning is fascinating"
        ]
        
        for text in test_texts:
            result = tokenizer.tokenize([text])
            print(f"✓ Tokenized: '{text}'")
            print(f"  Input IDs shape: {result['input_ids'].shape}")
            print(f"  Attention mask shape: {result['attention_mask'].shape}")
            
            # Test decoding
            decoded = tokenizer.decode(result['input_ids'][0])
            print(f"  Decoded: '{decoded}'")
            print()
        
        print("✓ Basic tokenizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Basic tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_tokenizer()