# LSM Lite - Lightweight Liquid State Machine for Conversational AI

## Overview

LSM Lite is a streamlined implementation of Liquid State Machine (LSM) for conversational AI applications. The project focuses on simplicity and maintainability while providing core LSM functionality including sparse sine-activated reservoir computing, multi-backend tokenization, and text generation capabilities. This is a rewrite of a more complex LSM system, designed to eliminate over-engineering while preserving essential features like custom tokenizers, 2D/3D CNN processing, and HuggingFace dataset integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## Project Status

**Current Status**: Fully operational LSM Lite implementation completed on August 10, 2025

**Key Achievements**:
- Complete modular architecture with all core components working
- Multi-backend tokenization support (GPT-2, BERT, spaCy)
- Sparse sine-activated reservoir computing implementation
- 2D/3D CNN processing capabilities
- Full model persistence and loading functionality
- Comprehensive CLI interface with training, generation, and evaluation commands
- Interactive example demonstrations with 9 different use cases
- All dependencies successfully installed and type annotation issues resolved

**Recent Changes**:
- Fixed type annotation errors in api.py and persistence.py for None handling
- Resolved import issues after package installation
- Successfully demonstrated working CLI interface and example menu system
- All core LSM functionality verified as operational

## System Architecture

### Core Design Principles
The architecture follows a modular design with clear separation of concerns, minimal dependencies, and straightforward interfaces. The system is built around five main components that work together to provide end-to-end conversational AI functionality.

### Component Architecture

**Core Processing Components (`lsm_lite/core/`)**
- **UnifiedTokenizer**: Provides a consistent interface across multiple tokenizer backends (GPT-2, BERT, spaCy) with automatic vocabulary detection and padding handling
- **SparseReservoir**: Implements sparse sine-activated liquid state machine with parameterized activation functions, spectral radius control, and leak rate dynamics
- **CNNProcessor**: Supports both 2D spatial and 3D spatial-temporal convolutions for processing reservoir outputs

**Data Processing (`lsm_lite/data/`)**
- **DataLoader**: Handles loading from HuggingFace datasets (primarily cosmopedia-v2) and local files with basic text preprocessing and filtering
- **SinusoidalEmbedder**: Combines token embeddings with sinusoidal positional encodings using temperature-controlled frequency scaling

**Training Pipeline (`lsm_lite/training/`)**
- **LSMTrainer**: Orchestrates the complete training process, coordinating all model components and handling the full pipeline from data preparation to model optimization

**Inference System (`lsm_lite/inference/`)**
- **TextGenerator**: Provides autoregressive text generation with temperature sampling, top-k/top-p filtering, and repetition penalty controls

**Utilities (`lsm_lite/utils/`)**
- **LSMConfig**: Dataclass-based configuration management with sensible defaults for all model parameters
- **ModelPersistence**: Complete model serialization including all components (model weights, tokenizer state, embedder configuration)

### Main API Interface
The **LSMLite** class in `api.py` serves as the primary user-facing interface, providing a simplified API that abstracts the complexity of individual components while maintaining full configurability through the config system.

### Architecture Benefits
- **Modularity**: Each component has a single clear responsibility
- **Flexibility**: Multi-backend tokenizer support allows experimentation with different text processing approaches
- **Scalability**: CNN processor supports both 2D and 3D architectures for different spatial-temporal processing needs
- **Maintainability**: Clear interfaces and minimal coupling between components

## External Dependencies

### Core ML Framework
- **TensorFlow/Keras**: Primary deep learning framework for model implementation, training, and inference
- **NumPy**: Numerical computing for array operations and mathematical functions
- **SciPy**: Sparse matrix operations for reservoir connectivity patterns

### NLP and Tokenization
- **Transformers (HuggingFace)**: Provides pre-trained tokenizers (GPT-2, BERT) and model loading utilities
- **spaCy**: Alternative tokenizer backend option for linguistic processing

### Data Handling
- **HuggingFace Datasets**: Primary data source for training datasets, particularly cosmopedia-v2 conversational data

### Development and Logging
- **Python standard library**: Extensive use of logging, json, pickle, pathlib for basic functionality
- **Argparse**: Command-line interface implementation for training and inference scripts

### Model Persistence
The system uses a combination of TensorFlow SavedModel format for neural network components and pickle/JSON for tokenizer states and configuration data, ensuring complete model reproducibility.