#!/usr/bin/env python3
"""Parquet-based training demo for LSMLite dual CNN pipeline.

This example loads sample conversations from the provided Parquet file,
initializes the embedder, next-token prediction CNN, and response CNN,
trains the dual CNN pipeline briefly, and performs a simple inference.
"""

import logging
import pandas as pd
from pathlib import Path
import sys

# Add project root to Python path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lsm_lite.api import LSMLite
from lsm_lite.utils.config import DualCNNConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_parquet_texts(path: str, max_samples: int = 20):
    """Load conversation texts from a Parquet file."""
    df = pd.read_parquet(path)
    # Combine prompt and text fields if both exist
    if {'prompt', 'text'}.issubset(df.columns):
        texts = (df['prompt'] + ' ' + df['text']).tolist()
    elif 'text' in df.columns:
        texts = df['text'].tolist()
    else:
        # Fallback to joining all string columns
        texts = []
        for _, row in df.iterrows():
            parts = [str(v) for v in row.values if isinstance(v, str)]
            texts.append(' '.join(parts))
    return texts[:max_samples]


def main():
    data_path = Path("train-00000-of-00104.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Sample parquet file not found: {data_path}")

    training_data = load_parquet_texts(str(data_path), max_samples=20)
    print(f"Loaded {len(training_data)} training samples from {data_path}")

    # Initialize API and configuration
    api = LSMLite()
    config = DualCNNConfig(
        embedder_fit_samples=len(training_data),
        embedder_batch_size=4,
        embedder_max_length=64,
        reservoir_size=64,
        attention_heads=2,
        attention_dim=16,
        first_cnn_filters=[8, 16],
        second_cnn_filters=[16, 32],
        wave_window_size=8,
        wave_overlap=2,
        max_wave_storage=32,
        dual_training_epochs=1,
        training_batch_size=2,
    )

    # Setup pipeline and train
    api.setup_dual_cnn_pipeline(training_data=training_data, dual_cnn_config=config)
    api.quick_dual_cnn_train(
        dataset_name=str(data_path),
        max_samples=len(training_data),
        dual_cnn_config=config,
        epochs=1,
        batch_size=2,
    )

    # Demonstrate generation
    prompt = "What is machine learning?"
    generated = api.dual_cnn_generate(prompt, max_length=20)
    print("Prompt:", prompt)
    print("Generated:", generated)


if __name__ == "__main__":
    main()
