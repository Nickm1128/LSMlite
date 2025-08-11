"""Example: Training dual CNN pipeline on a HuggingFace dataset.

This script demonstrates how to:
    1. Download a text dataset from HuggingFace using the ``datasets`` library.
    2. Fit the LSMLite embedder and attentive reservoir.
    3. Train two coordinated CNNs: the first for next-token prediction and the
       second for complete response generation.

The example keeps configuration values intentionally small so it can run on a
CPU in a few seconds.  It also handles environments without internet access
gracefully by falling back to the library's built in sample texts.
"""

import os
import sys

# Allow running this example without installing the package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lsm_lite.api import LSMLite
from lsm_lite.utils.config import DualCNNConfig


def main(dataset_name: str = "ag_news") -> None:
    """Run a tiny end‑to‑end training + generation example."""
    # Configure a lightweight dual CNN pipeline
    config = DualCNNConfig(
        embedder_fit_samples=50,
        embedder_batch_size=8,
        embedder_max_length=32,
        reservoir_size=64,
        attention_heads=2,
        attention_dim=16,
        first_cnn_filters=[16],
        second_cnn_filters=[32],
        wave_window_size=8,
        wave_overlap=2,
        max_wave_storage=20,
        wave_feature_dim=32,
        dual_training_epochs=1,
        training_batch_size=4,
    )

    api = LSMLite()

    try:
        # ``quick_dual_cnn_train`` automatically downloads the dataset, fits the
        # embedder and reservoir, then trains both CNNs.
        results = api.quick_dual_cnn_train(
            dataset_name=dataset_name,
            max_samples=50,
            dual_cnn_config=config,
            epochs=1,
            batch_size=4,
        )
        print("Training complete! Final metrics:", results.get("final_metrics"))

        # Generate a short continuation using both CNNs in coordination
        prompt = "Machine learning"
        generated = api.dual_cnn_generate(prompt=prompt, max_length=20)
        print(f"Prompt: {prompt}\nGenerated: {generated}")

    except Exception as exc:  # pragma: no cover - demonstration only
        # In restricted environments (e.g. no internet), the dataset download may
        # fail.  The DataLoader used internally will then fall back to a small
        # built‑in sample so that the rest of the pipeline can still be
        # exercised.
        print(f"Example could not download dataset: {exc}")


if __name__ == "__main__":  # pragma: no cover - manual execution only
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data.txt")
    main(dataset_name=sample_path)
