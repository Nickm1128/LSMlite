"""
Text generation using trained LSM model.

This module provides text generation capabilities including autoregressive
generation with temperature sampling for the trained LSM model.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import List, Optional, Union, Dict, Any

from ..core.tokenizer import UnifiedTokenizer
from ..data.embeddings import SinusoidalEmbedder

logger = logging.getLogger(__name__)


class TextGenerator:
    """Text generation using trained LSM model."""
    
    def __init__(self, model: tf.keras.Model, tokenizer: UnifiedTokenizer,
                 embedder: SinusoidalEmbedder):
        """
        Initialize text generator.
        
        Args:
            model: Trained LSM model
            tokenizer: Tokenizer used for training
            embedder: Embedder used for training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.special_tokens = tokenizer.get_special_tokens()
        
        logger.info("Text generator initialized with vocab size: %d", tokenizer.vocab_size)
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 repetition_penalty: float = 1.0, stop_tokens: Optional[List[str]] = None,
                 num_beams: int = 1, do_sample: bool = True) -> str:
        """
        Generate text continuation for a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text (in tokens)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None to disable)
            top_p: Nucleus sampling threshold (None to disable)
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of tokens that stop generation
            
        Returns:
            Generated text continuation
        """
        logger.info("Generating text for prompt: '%.50s...'", prompt)
        
        # Tokenize the prompt
        tokenized_prompt = self.tokenizer.tokenize([prompt], padding=False, truncation=True)
        input_ids = tokenized_prompt['input_ids'][0].tolist()
        
        # Remove padding tokens
        input_ids = [token_id for token_id in input_ids if token_id != self.special_tokens['pad_token_id']]
        
        # Ensure prompt is not empty
        if not input_ids:
            input_ids = [self.special_tokens['bos_token_id']]
        
        # Choose generation strategy based on num_beams
        if num_beams > 1 and not do_sample:
            return self._generate_beam_search(input_ids, max_length, num_beams, stop_tokens)
        else:
            return self._generate_sampling(input_ids, max_length, temperature, top_k, top_p, 
                                         repetition_penalty, stop_tokens)
    
    def _generate_sampling(self, input_ids: List[int], max_length: int, temperature: float,
                          top_k: Optional[int], top_p: Optional[float], 
                          repetition_penalty: float, stop_tokens: Optional[List[str]]) -> str:
        """Generate text using sampling strategy."""
        # Generate tokens one by one
        generated_tokens = input_ids.copy()
        stop_token_ids = self._convert_stop_tokens_to_ids(stop_tokens) if stop_tokens else []
        
        for _ in range(max_length):
            # Prepare input sequence (last max_length tokens)
            context = generated_tokens[-self.tokenizer.max_length:]
            
            # Pad if necessary
            if len(context) < self.tokenizer.max_length:
                padding = [self.special_tokens['pad_token_id']] * (self.tokenizer.max_length - len(context))
                context = padding + context
            
            # Convert to tensor
            context_tensor = tf.constant([context], dtype=tf.int32)
            
            # Get model predictions
            predictions = self.model(context_tensor, training=False)
            
            # Handle both 2D and 3D output shapes
            if len(predictions.shape) == 3:
                logits = predictions[0, -1, :]  # Get logits for the last position (3D case)
            else:
                logits = predictions[0, :]  # Get logits for the batch (2D case)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated_tokens, repetition_penalty)
            
            # Sample next token
            next_token_id = self._sample_next_token(
                logits, temperature, top_k, top_p
            )
            
            # Check for stop conditions
            if (next_token_id == self.special_tokens['eos_token_id'] or
                next_token_id in stop_token_ids):
                break
            
            generated_tokens.append(next_token_id)
        
        # Extract only the generated part (excluding prompt)
        generated_part = generated_tokens[len(input_ids):]
        
        # Decode generated tokens
        if generated_part:
            generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
        else:
            generated_text = ""
        
        logger.info("Generated %d tokens", len(generated_part))
        return generated_text
    
    def _generate_beam_search(self, input_ids: List[int], max_length: int, 
                             num_beams: int, stop_tokens: Optional[List[str]]) -> str:
        """Generate text using beam search."""
        from heapq import heappush, heappop
        
        stop_token_ids = self._convert_stop_tokens_to_ids(stop_tokens) if stop_tokens else []
        
        # Initialize beams: (negative_log_prob, sequence)
        beams = [(0.0, input_ids.copy())]
        
        for _ in range(max_length):
            candidates = []
            
            # For each beam, generate next token candidates
            for neg_log_prob, sequence in beams:
                # Check for stop condition
                if (sequence and 
                    (sequence[-1] == self.special_tokens['eos_token_id'] or
                     sequence[-1] in stop_token_ids)):
                    candidates.append((neg_log_prob, sequence))
                    continue
                
                # Prepare context for prediction
                context = sequence[-self.tokenizer.max_length:]
                if len(context) < self.tokenizer.max_length:
                    padding = [self.special_tokens['pad_token_id']] * (self.tokenizer.max_length - len(context))
                    context = padding + context
                
                # Get predictions
                context_tensor = tf.constant([context], dtype=tf.int32)
                predictions = self.model(context_tensor, training=False)
                
                # Handle both 2D and 3D output shapes
                if len(predictions.shape) == 3:
                    logits = predictions[0, -1, :]  # Get logits for the last position (3D case)
                else:
                    logits = predictions[0, :]  # Get logits for the batch (2D case)
                    
                log_probs = tf.nn.log_softmax(logits)
                
                # Get top candidates
                top_k_log_probs, top_k_indices = tf.nn.top_k(log_probs, k=num_beams * 2)
                
                for i in range(len(top_k_log_probs)):
                    token_id = int(top_k_indices[i].numpy())
                    token_log_prob = float(top_k_log_probs[i].numpy())
                    
                    new_sequence = sequence + [token_id]
                    new_neg_log_prob = neg_log_prob - token_log_prob
                    
                    candidates.append((new_neg_log_prob, new_sequence))
            
            # Select top beams
            candidates.sort(key=lambda x: x[0])
            beams = candidates[:num_beams]
            
            # Check if all beams have ended
            all_ended = all(
                seq and (seq[-1] == self.special_tokens['eos_token_id'] or 
                        seq[-1] in stop_token_ids)
                for _, seq in beams
            )
            if all_ended:
                break
        
        # Return the best sequence
        best_sequence = min(beams, key=lambda x: x[0])[1]
        generated_part = best_sequence[len(input_ids):]
        
        if generated_part:
            generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
        else:
            generated_text = ""
        
        return generated_text
    
    def _convert_stop_tokens_to_ids(self, stop_tokens: List[str]) -> List[int]:
        """Convert stop token strings to token IDs."""
        stop_ids = []
        for token in stop_tokens:
            # Tokenize each stop token and get its ID
            tokenized = self.tokenizer.tokenize([token], padding=False)
            token_ids = tokenized['input_ids'][0].tolist()
            # Remove padding and add to stop IDs
            stop_ids.extend([tid for tid in token_ids if tid != self.special_tokens['pad_token_id']])
        return stop_ids
    
    def _apply_repetition_penalty(self, logits: tf.Tensor, generated_tokens: List[int],
                                 penalty: float) -> tf.Tensor:
        """Apply repetition penalty to logits."""
        # Count token frequencies in generated sequence
        token_counts = {}
        for token_id in generated_tokens:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
        
        # Apply penalty
        logits_array = logits.numpy()
        for token_id, count in token_counts.items():
            if token_id < len(logits_array):
                # Apply penalty based on frequency
                if logits_array[token_id] > 0:
                    logits_array[token_id] /= (penalty ** count)
                else:
                    logits_array[token_id] *= (penalty ** count)
        
        return tf.constant(logits_array)
    
    def _sample_next_token(self, logits: tf.Tensor, temperature: float = 1.0,
                          top_k: Optional[int] = None, top_p: Optional[float] = None) -> int:
        """
        Sample next token from logits with various sampling strategies.
        
        Args:
            logits: Model output logits
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Sampled token ID
        """
        # Apply temperature
        if temperature == 0.0:
            # Greedy decoding
            return int(tf.argmax(logits).numpy())
        
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            top_k_logits, top_k_indices = tf.nn.top_k(logits, k=top_k)
            # Set non-top-k logits to negative infinity
            logits_filtered = tf.fill(logits.shape, float('-inf'))
            logits_filtered = tf.tensor_scatter_nd_update(
                logits_filtered,
                tf.expand_dims(top_k_indices, axis=1),
                top_k_logits
            )
            logits = logits_filtered
        
        # Apply nucleus (top-p) sampling
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = tf.nn.top_k(logits, k=logits.shape[-1])
            cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
            
            # Find cutoff point
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove = tf.concat([
                [False],
                sorted_indices_to_remove[:-1]
            ], axis=0)
            
            # Create mask and apply
            indices_to_remove = tf.scatter_nd(
                tf.expand_dims(sorted_indices, axis=1),
                sorted_indices_to_remove,
                shape=logits.shape
            )
            logits = tf.where(indices_to_remove, float('-inf'), logits)
        
        # Sample from the filtered distribution
        probabilities = tf.nn.softmax(logits)
        token_id = tf.random.categorical(tf.expand_dims(logits, 0), 1)[0, 0]
        
        return int(token_id.numpy())
    
    def predict_next_token(self, context: str, return_probabilities: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Predict next token given context.
        
        Args:
            context: Input context string
            return_probabilities: Whether to return token probabilities
            
        Returns:
            Next token string or dictionary with token and probabilities
        """
        # Tokenize context
        tokenized = self.tokenizer.tokenize([context], padding=True, truncation=True)
        context_tensor = tf.constant(tokenized['input_ids'], dtype=tf.int32)
        
        # Get model predictions
        predictions = self.model(context_tensor, training=False)
        
        # Handle both 2D and 3D output shapes
        if len(predictions.shape) == 3:
            logits = predictions[0, -1, :]  # Get logits for the last position (3D case)
        else:
            logits = predictions[0, :]  # Get logits for the batch (2D case)
        
        # Get most likely token
        next_token_id = int(tf.argmax(logits).numpy())
        next_token = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
        
        if return_probabilities:
            # Get top-k tokens and probabilities
            probabilities = tf.nn.softmax(logits)
            top_k_probs, top_k_indices = tf.nn.top_k(probabilities, k=10)
            
            top_tokens = []
            for i in range(10):
                token_id = int(top_k_indices[i].numpy())
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                prob = float(top_k_probs[i].numpy())
                top_tokens.append({'token': token_text, 'probability': prob})
            
            return {
                'next_token': next_token,
                'top_tokens': top_tokens
            }
        
        return next_token
    
    def generate_batch(self, prompts: List[str], max_length: int = 50,
                      temperature: float = 1.0, **kwargs) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        logger.info("Generating text for %d prompts", len(prompts))
        
        generated_texts = []
        for prompt in prompts:
            generated_text = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                **kwargs
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def interactive_generation(self, initial_prompt: str = "", max_turns: int = 10):
        """
        Interactive text generation session.
        
        Args:
            initial_prompt: Initial prompt to start with
            max_turns: Maximum number of interaction turns
        """
        print("Interactive LSM Text Generation")
        print("Type 'quit' to exit, 'reset' to start over")
        print("-" * 50)
        
        context = initial_prompt
        
        for turn in range(max_turns):
            if context:
                print(f"Context: {context}")
            
            user_input = input(f"Turn {turn + 1}> ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'reset':
                context = initial_prompt
                print("Context reset.")
                continue
            
            # Add user input to context
            full_prompt = f"{context} {user_input}" if context else user_input
            
            # Generate response
            try:
                generated = self.generate(full_prompt, max_length=50, temperature=0.8)
                print(f"Generated: {generated}")
                
                # Update context
                context = f"{full_prompt} {generated}"
                
                # Trim context if too long
                if len(context) > 500:  # Arbitrary limit
                    context = context[-400:]  # Keep last 400 characters
                    
            except Exception as e:
                print(f"Generation error: {e}")
        
        print("Interactive session ended.")
    
    def compute_perplexity(self, texts: List[str]) -> float:
        """
        Compute perplexity of the model on given texts.
        
        Args:
            texts: List of text strings to evaluate
            
        Returns:
            Average perplexity score
        """
        total_log_likelihood = 0.0
        total_tokens = 0
        
        for text in texts:
            # Tokenize text
            tokenized = self.tokenizer.tokenize([text], padding=True, truncation=True)
            input_ids = tokenized['input_ids'][0]
            
            # Compute log likelihood
            for i in range(1, len(input_ids)):
                if input_ids[i] == self.special_tokens['pad_token_id']:
                    break
                
                # Context up to position i
                context = input_ids[:i]
                if len(context) < self.tokenizer.max_length:
                    padding = [self.special_tokens['pad_token_id']] * (self.tokenizer.max_length - len(context))
                    context = padding + context
                
                # Predict token at position i
                context_tensor = tf.constant([context], dtype=tf.int32)
                predictions = self.model(context_tensor, training=False)
                
                # Handle both 2D and 3D output shapes
                if len(predictions.shape) == 3:
                    logits = predictions[0, -1, :]  # Get logits for the last position (3D case)
                else:
                    logits = predictions[0, :]  # Get logits for the batch (2D case)
                
                # Get log probability of actual token
                log_probs = tf.nn.log_softmax(logits)
                actual_token = input_ids[i]
                token_log_prob = log_probs[actual_token]
                
                total_log_likelihood += float(token_log_prob.numpy())
                total_tokens += 1
        
        # Calculate perplexity
        if total_tokens == 0:
            return float('inf')
        
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = np.exp(-avg_log_likelihood)
        
        return float(perplexity)
