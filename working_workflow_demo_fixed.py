#!/usr/bin/env python3
"""Updated LSM Lite workflow demo with recent fixes.

This script mirrors ``working_workflow_example.py`` but adds a final
verification step that the ``DualCNNTrainer`` can be imported and
initialized. The trainer previously failed due to a missing
``AttentiveReservoir`` import; this demo confirms that the issue has
been resolved.
"""

from typing import List

# Configure logging in a minimal way
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core component imports
from lsm_lite.utils.config import DualCNNConfig
from lsm_lite.core.tokenizer import UnifiedTokenizer
from lsm_lite.data.embeddings import SinusoidalEmbedder
from lsm_lite.core.attentive_reservoir import AttentiveReservoir
from lsm_lite.core.rolling_wave_storage import RollingWaveStorage


def create_sample_data() -> List[str]:
    """Return a tiny sample dataset for the demo."""
    return [
        "Hello, how are you today?",
        "Machine learning is fascinating.",
        "Neural networks learn from data.",
        "Reservoir computing enables efficient processing.",
        "Dual CNN architectures coordinate predictions."
    ]


def demonstrate_components() -> bool:
    """Show initialization of core components."""
    print("Testing core components...")

    # 1. Tokenizer
    try:
        tokenizer = UnifiedTokenizer(backend="basic", max_length=32)
        sample = tokenizer.tokenize(["Hello world"])
        print(f"  ✓ Tokenizer ready (vocab: {tokenizer.vocab_size})")
    except Exception as exc:  # pragma: no cover - demo output
        print(f"  ✗ Tokenizer failed: {exc}")
        return False

    # 2. Embedder
    try:
        embedder = SinusoidalEmbedder(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=64,
            max_length=32,
        )
        emb = embedder(sample["input_ids"])  # type: ignore[index]
        print(f"  ✓ Embedder ready (shape: {emb.shape})")
    except Exception as exc:  # pragma: no cover - demo output
        print(f"  ✗ Embedder failed: {exc}")
        return False

    # 3. Attentive reservoir
    try:
        reservoir = AttentiveReservoir(
            input_dim=64,
            reservoir_size=128,
            attention_heads=2,
            attention_dim=32,
        )
        import numpy as np
        dummy = np.random.randn(1, 10, 64)
        res_out = reservoir(dummy)
        if isinstance(res_out, tuple):
            res_states, _ = res_out
            print(f"  ✓ Reservoir ready (states: {res_states.shape})")
        else:
            print(f"  ✓ Reservoir ready (output: {res_out.shape})")
    except Exception as exc:  # pragma: no cover - demo output
        print(f"  ✗ Reservoir failed: {exc}")
        return False

    # 4. Rolling wave storage
    try:
        storage = RollingWaveStorage(
            max_sequence_length=32,
            feature_dim=128,
            window_size=8,
            overlap=2,
            max_memory_mb=50.0,
        )
        import numpy as np
        storage.store_wave(np.random.randn(128), sequence_position=0)
        stats = storage.get_storage_stats()
        print(f"  ✓ Wave storage ready (stored: {stats['stored_count']})")
    except Exception as exc:  # pragma: no cover - demo output
        print(f"  ✗ Wave storage failed: {exc}")
        return False

    # 5. DualCNNTrainer import verification
    try:
        from lsm_lite.training.dual_cnn_trainer import DualCNNTrainer

        class _DummyComponent:
            """Component with minimal attributes used for pipeline stubs."""

            def __init__(self):
                self.trainable_variables = []

        class _DummyWaveStorage:
            def get_storage_stats(self):
                return {
                    "stored_count": 0,
                    "max_capacity": 1,
                    "utilization_percent": 0.0,
                    "memory_used_mb": 0.0,
                    "memory_limit_mb": 1.0,
                }

        class _DummyPipeline:
            """Minimal pipeline mock used only for trainer initialization."""

            def __init__(self, config: DualCNNConfig):
                self.config = config
                self.tokenizer = _DummyComponent()
                self.embedder = _DummyComponent()
                self.reservoir = _DummyComponent()
                self.first_cnn = _DummyComponent()
                self.second_cnn = _DummyComponent()
                self.wave_storage = _DummyWaveStorage()

            def is_initialized(self) -> bool:
                return True

            def get_component_status(self):
                return {
                    "tokenizer": True,
                    "embedder": True,
                    "reservoir": True,
                    "wave_storage": True,
                    "first_cnn": True,
                    "second_cnn": True,
                    "fully_initialized": True,
                }

            def get_fallback_status(self):
                return {"fallback_mode_enabled": False, "single_cnn_fallback": False}

        dummy_pipeline = _DummyPipeline(DualCNNConfig())
        DualCNNTrainer(dummy_pipeline, dummy_pipeline.config)
        print("  ✓ DualCNNTrainer initialized successfully")
    except Exception as exc:  # pragma: no cover - demo output
        print(f"  ✗ DualCNNTrainer failed: {exc}")
        return False

    return True


def main() -> None:
    print("=" * 60)
    print("LSM Lite Workflow Demo (with recent fixes)")
    print("=" * 60)
    create_sample_data()  # Data not directly used but shown for parity

    success = demonstrate_components()
    if success:
        print("\nAll components initialized correctly. Demo complete!")
    else:  # pragma: no cover - demo output
        print("\nDemo encountered issues; see messages above.")


if __name__ == "__main__":
    main()
