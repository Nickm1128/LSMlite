"""
Data loading and preprocessing for conversational datasets.

This module handles loading data from various sources including HuggingFace
datasets and local files, with basic preprocessing for conversational AI.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional, Union
import random

logger = logging.getLogger(__name__)


class DataLoader:
    """Simple data loading for conversational datasets."""
    
    def __init__(self, dataset_name: str = 'cosmopedia-v2', 
                 max_samples: Optional[int] = 10000,
                 min_length: int = 10,
                 max_length: int = 5000):  # Increased max length for longer texts
        """
        Initialize data loader.
        
        Args:
            dataset_name: Name of dataset ('cosmopedia-v2', or local file path)
            max_samples: Maximum number of samples to load (None for all)
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
        """
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.min_length = min_length
        self.max_length = max_length
        
        logger.info("DataLoader initialized for dataset: %s", dataset_name)
    
    def load_conversations(self) -> List[str]:
        """
        Load conversation data from configured source.
        
        Returns:
            List of conversation strings
        """
        logger.info("Loading conversations from: %s", self.dataset_name)
        
        try:
            if self.dataset_name == 'cosmopedia-v2':
                conversations = self._load_huggingface_dataset()
            elif self.dataset_name.startswith('http'):
                # Handle direct URLs (parquet files)
                conversations = self._load_parquet_url()
            elif os.path.exists(self.dataset_name):
                conversations = self._load_local_dataset()
            else:
                # Try as HuggingFace dataset name
                conversations = self._load_huggingface_dataset(self.dataset_name)
            
            # Filter conversations by length
            filtered_conversations = self._filter_conversations(conversations)
            
            logger.info("Loaded %d conversations after filtering", len(filtered_conversations))
            return filtered_conversations
            
        except Exception as e:
            logger.error("Failed to load conversations: %s", e)
            # Return some fallback conversations for testing
            return self._get_fallback_conversations()
    
    def _load_huggingface_dataset(self, dataset_name: Optional[str] = None) -> List[str]:
        """Load from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets not installed. Install with: pip install datasets")
        
        dataset_name = dataset_name or 'HuggingFaceTB/cosmopedia-v2'
        
        try:
            # Load dataset
            logger.info("Loading HuggingFace dataset: %s", dataset_name)
            dataset = load_dataset(dataset_name, split='train', streaming=True)
            
            conversations = []
            count = 0
            
            for example in dataset:
                if self.max_samples and count >= self.max_samples:
                    break
                
                # Extract text content
                text = self._extract_text_from_example(example)
                if text:
                    conversations.append(text)
                    count += 1
                
                if count % 1000 == 0:
                    logger.info("Loaded %d conversations...", count)
            
            return conversations
            
        except Exception as e:
            logger.error("Failed to load HuggingFace dataset '%s': %s", dataset_name, e)
            raise
    
    def _load_parquet_url(self) -> List[str]:
        """Load data directly from a parquet URL."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas not installed. Install with: pip install pandas")
        
        try:
            logger.info("Loading parquet file from URL: %s", self.dataset_name)
            
            # Load the parquet file directly
            df = pd.read_parquet(self.dataset_name)
            
            conversations = []
            count = 0
            
            # Convert DataFrame rows to text conversations
            for _, row in df.iterrows():
                if self.max_samples and count >= self.max_samples:
                    break
                
                # Extract text content from the row
                text = self._extract_text_from_row(row)
                if text:
                    conversations.append(text)
                    count += 1
                
                if count % 1000 == 0:
                    logger.info("Loaded %d conversations from parquet...", count)
            
            logger.info("Successfully loaded %d conversations from parquet URL", len(conversations))
            return conversations
            
        except Exception as e:
            logger.error("Failed to load parquet from URL '%s': %s", self.dataset_name, e)
            raise
    
    def _extract_text_from_row(self, row) -> Optional[str]:
        """Extract text content from a pandas row."""
        # Common text fields in datasets
        text_fields = ['text', 'content', 'prompt', 'completion', 'conversation', 'dialogue']
        
        for field in text_fields:
            if field in row and row[field] and str(row[field]).strip():
                return str(row[field]).strip()
        
        # If no obvious text field, try to combine multiple fields
        combined_text = []
        for key, value in row.items():
            if isinstance(value, str) and len(value.strip()) > 10:
                combined_text.append(value.strip())
        
        if combined_text:
            return ' '.join(combined_text)
        
        return None
    
    def _extract_text_from_example(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a dataset example."""
        # Common text fields in datasets
        text_fields = ['text', 'content', 'prompt', 'completion', 'conversation', 'dialogue']
        
        for field in text_fields:
            if field in example and example[field]:
                text = str(example[field]).strip()
                if len(text) > 10:  # Ensure meaningful content
                    return text
        
        # If no obvious text field, try to combine multiple fields
        if 'prompt' in example and 'completion' in example:
            combined = f"{example['prompt']}\n{example['completion']}"
            if len(combined.strip()) > 10:
                return combined.strip()
        
        # Try combining all string fields that look like text
        text_parts = []
        for key, value in example.items():
            if isinstance(value, str) and len(value.strip()) > 20:  # Longer strings are more likely to be content
                text_parts.append(value.strip())
        
        if text_parts:
            combined = ' '.join(text_parts)
            if len(combined.strip()) > 10:
                return combined.strip()
        
        # Last resort - convert entire example to string, but only if it's not too short
        full_text = str(example).strip()
        if len(full_text) > 50:  # Avoid very short or empty content
            return full_text
            
        return None
    
    def _load_local_dataset(self) -> List[str]:
        """Load from local files (JSON, CSV, TXT)."""
        file_path = self.dataset_name
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        logger.info("Loading local file: %s", file_path)
        
        if ext == '.json':
            return self._load_json_file(file_path)
        elif ext == '.csv':
            return self._load_csv_file(file_path)
        elif ext in ['.txt', '.text']:
            return self._load_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _load_json_file(self, file_path: str) -> List[str]:
        """Load conversations from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        
        if isinstance(data, list):
            # List of conversations or objects
            for item in data:
                if isinstance(item, str):
                    conversations.append(item)
                elif isinstance(item, dict):
                    # Extract text from dict
                    text = self._extract_text_from_example(item)
                    if text:
                        conversations.append(text)
        elif isinstance(data, dict):
            # Single conversation or nested structure
            text = self._extract_text_from_example(data)
            if text:
                conversations.append(text)
        
        return conversations[:self.max_samples] if self.max_samples else conversations
    
    def _load_csv_file(self, file_path: str) -> List[str]:
        """Load conversations from CSV file."""
        conversations = []
        
        logger.info("Loading CSV file: %s", file_path)
        
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            # Log column names for debugging
            if reader.fieldnames:
                logger.info("CSV columns: %s", reader.fieldnames)
            
            row_count = 0
            for row in reader:
                row_count += 1
                
                if self.max_samples and len(conversations) >= self.max_samples:
                    break
                
                text = self._extract_text_from_example(row)
                if text:
                    conversations.append(text)
                    if len(conversations) == 1:  # Log first successful extraction
                        logger.info("First extracted text (first 200 chars): %s", text[:200])
                elif row_count <= 5:  # Log first few failed extractions for debugging
                    logger.warning("Could not extract text from row %d: %s", row_count, {k: str(v)[:50] for k, v in row.items()})
        
        logger.info("Loaded %d conversations from %d CSV rows", len(conversations), row_count)
        return conversations
    
    def _load_text_file(self, file_path: str) -> List[str]:
        """Load conversations from text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines (assume paragraph separation)
        conversations = [para.strip() for para in content.split('\n\n') if para.strip()]
        
        return conversations[:self.max_samples] if self.max_samples else conversations
    
    def _filter_conversations(self, conversations: List[str]) -> List[str]:
        """Filter conversations by length and quality."""
        filtered = []
        
        for conv in conversations:
            # Basic cleaning
            conv = conv.strip()
            
            # Length filtering
            if self.min_length <= len(conv) <= self.max_length:
                # Additional quality filters could be added here
                filtered.append(conv)
        
        return filtered
    
    def _get_fallback_conversations(self) -> List[str]:
        """Return fallback conversations for testing when loading fails."""
        logger.warning("Using fallback conversations for testing")
        
        fallback_conversations = [
            "Hello! How can I help you today?",
            "I'm looking for information about machine learning. Can you explain neural networks?",
            "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information.",
            "That's interesting! Can you tell me more about different types of neural networks?",
            "Sure! There are many types including feedforward networks, recurrent networks, and convolutional networks. Each has different strengths.",
            "What about reservoir computing? I've heard it's an interesting approach.",
            "Reservoir computing is a framework that uses a fixed, randomly connected recurrent network as a reservoir of dynamics.",
            "How does it differ from traditional RNNs?",
            "The key difference is that only the output weights are trained, while the reservoir weights remain fixed. This makes training much faster.",
            "That sounds very efficient! Are there any downsides?",
        ]
        
        # Repeat conversations if we need more samples
        if self.max_samples and self.max_samples > len(fallback_conversations):
            multiplier = (self.max_samples // len(fallback_conversations)) + 1
            fallback_conversations = fallback_conversations * multiplier
        
        return fallback_conversations[:self.max_samples] if self.max_samples else fallback_conversations
    
    def preprocess_conversations(self, conversations: List[str]) -> List[str]:
        """
        Basic conversation preprocessing.
        
        Args:
            conversations: Raw conversation strings
            
        Returns:
            Preprocessed conversation strings
        """
        preprocessed = []
        
        for conv in conversations:
            # Basic text cleaning
            processed = self._clean_text(conv)
            
            if processed:  # Only add non-empty strings
                preprocessed.append(processed)
        
        return preprocessed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove or replace special characters if needed
        # (Keep minimal for now to preserve conversation structure)
        
        return text.strip()
    
    def create_conversation_pairs(self, conversations: List[str]) -> List[Dict[str, str]]:
        """
        Create input-output pairs from conversations for training.
        
        Args:
            conversations: List of conversation strings
            
        Returns:
            List of dictionaries with 'input' and 'target' keys
        """
        pairs = []
        
        for conv in conversations:
            # Simple approach: split into sentences and create pairs
            sentences = self._split_into_sentences(conv)
            
            for i in range(len(sentences) - 1):
                pairs.append({
                    'input': sentences[i],
                    'target': sentences[i + 1]
                })
        
        return pairs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple approach)."""
        # Basic sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def get_dataset_statistics(self, conversations: List[str]) -> Dict[str, Any]:
        """
        Compute basic statistics about the dataset.
        
        Args:
            conversations: List of conversations
            
        Returns:
            Dictionary with dataset statistics
        """
        if not conversations:
            return {'num_conversations': 0}
        
        lengths = [len(conv) for conv in conversations]
        word_counts = [len(conv.split()) for conv in conversations]
        
        return {
            'num_conversations': len(conversations),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'total_characters': sum(lengths),
            'total_words': sum(word_counts),
        }
