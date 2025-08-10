"""
Unified tokenizer supporting multiple backends.

This module provides a single tokenizer class that can use different backend
implementations (GPT-2, BERT, spaCy) while maintaining a consistent interface.
"""

import logging
from typing import List, Union, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class UnifiedTokenizer:
    """Single tokenizer class supporting multiple backends."""
    
    def __init__(self, backend: str = 'gpt2', vocab_size: Optional[int] = None, 
                 max_length: int = 128):
        """
        Initialize unified tokenizer.
        
        Args:
            backend: Backend to use ('gpt2', 'bert', 'spacy')
            vocab_size: Override vocabulary size (None for auto-detection)
            max_length: Maximum sequence length for padding/truncation
        """
        self.backend = backend.lower()
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Initialize backend tokenizer
        self._tokenizer = self._create_backend_tokenizer()
        
        # Auto-detect vocab size if not provided
        if self.vocab_size is None:
            self.vocab_size = self._detect_vocab_size()
        
        logger.info("Tokenizer initialized with backend '%s', vocab_size=%d", 
                   self.backend, self.vocab_size)
    
    def _create_backend_tokenizer(self):
        """Factory method for backend tokenizers."""
        try:
            if self.backend == 'gpt2':
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                # Add padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
            
            elif self.backend == 'bert':
                from transformers import BertTokenizer
                return BertTokenizer.from_pretrained('bert-base-uncased')
            
            elif self.backend == 'spacy':
                return self._create_spacy_tokenizer()
            
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except ImportError as e:
            logger.error("Failed to import required tokenizer backend: %s", e)
            raise ImportError(f"Required library not found for backend '{self.backend}': {e}")
    
    def _create_spacy_tokenizer(self):
        """Create a simple spaCy-based tokenizer."""
        try:
            import spacy
            # Try to load English model, fallback to blank if not available
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, using blank tokenizer")
                nlp = spacy.blank("en")
            
            return SpacyTokenizerWrapper(nlp)
        except ImportError:
            raise ImportError("spaCy not installed. Install with: pip install spacy")
    
    def _detect_vocab_size(self) -> int:
        """Auto-detect vocabulary size from backend tokenizer."""
        if hasattr(self._tokenizer, 'vocab_size'):
            return self._tokenizer.vocab_size
        elif hasattr(self._tokenizer, 'vocab') and hasattr(self._tokenizer.vocab, '__len__'):
            return len(self._tokenizer.vocab)
        else:
            logger.warning("Could not auto-detect vocab size, using default 50257")
            return 50257  # GPT-2 default
    
    def tokenize(self, texts: Union[str, List[str]], 
                 padding: bool = True, truncation: bool = True) -> Dict[str, np.ndarray]:
        """
        Tokenize texts to integer sequences.
        
        Args:
            texts: Input text or list of texts
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences to max_length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' arrays
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.backend in ['gpt2', 'bert']:
            return self._tokenize_transformers(texts, padding, truncation)
        elif self.backend == 'spacy':
            return self._tokenize_spacy(texts, padding, truncation)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _tokenize_transformers(self, texts: List[str], padding: bool, 
                             truncation: bool) -> Dict[str, np.ndarray]:
        """Tokenize using transformers backend."""
        encoded = self._tokenizer(
            texts,
            padding='max_length' if padding else False,
            truncation=truncation,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='np'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def _tokenize_spacy(self, texts: List[str], padding: bool, 
                       truncation: bool) -> Dict[str, np.ndarray]:
        """Tokenize using spaCy backend."""
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # Tokenize text
            tokens = [token.text.lower() for token in self._tokenizer.nlp(text)]
            
            # Convert to IDs (simple hash-based approach for demo)
            token_ids = [hash(token) % self.vocab_size for token in tokens]
            
            # Handle truncation
            if truncation and len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Handle padding
            if padding and len(token_ids) < self.max_length:
                pad_length = self.max_length - len(token_ids)
                token_ids.extend([0] * pad_length)  # 0 is pad token
                attention_mask.extend([0] * pad_length)
            
            input_ids.append(token_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': np.array(input_ids),
            'attention_mask': np.array(attention_masks)
        }
    
    def decode(self, token_ids: Union[List[int], np.ndarray, List[List[int]]], 
               skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token sequences back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text or list of texts
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Handle single sequence
        if isinstance(token_ids[0], int):
            token_ids = [token_ids]
            single_sequence = True
        else:
            single_sequence = False
        
        if self.backend in ['gpt2', 'bert']:
            decoded = self._tokenizer.batch_decode(
                token_ids, 
                skip_special_tokens=skip_special_tokens
            )
        elif self.backend == 'spacy':
            decoded = [self._decode_spacy_sequence(seq) for seq in token_ids]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        return decoded[0] if single_sequence else decoded
    
    def _decode_spacy_sequence(self, token_ids: List[int]) -> str:
        """Decode a single sequence for spaCy backend."""
        # This is a simplified decoder - in practice you'd need a proper vocab mapping
        return " ".join([f"token_{tid}" for tid in token_ids if tid != 0])
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        if self.backend in ['gpt2', 'bert']:
            return {
                'pad_token_id': self._tokenizer.pad_token_id or 0,
                'eos_token_id': self._tokenizer.eos_token_id or 0,
                'bos_token_id': getattr(self._tokenizer, 'bos_token_id', 0) or 0,
                'unk_token_id': getattr(self._tokenizer, 'unk_token_id', 0) or 0,
            }
        else:
            # Default special tokens for spaCy
            return {
                'pad_token_id': 0,
                'eos_token_id': 1,
                'bos_token_id': 2,
                'unk_token_id': 3,
            }


class SpacyTokenizerWrapper:
    """Wrapper class for spaCy tokenizer to match transformers interface."""
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.vocab_size = 50000  # Default vocab size for spaCy wrapper
    
    def __call__(self, text):
        return self.nlp(text)
