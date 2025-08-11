# LSM Lite Complete Workflow Examples

This directory contains comprehensive examples demonstrating the full LSM Lite workflow from initialization to inference using the simple convenience API.

## Files Overview

### Core Workflow Scripts

1. **`complete_workflow_example.py`** - Comprehensive end-to-end workflow demonstration
   - Full pipeline setup with embedder training
   - Dual CNN configuration (next-token prediction + response CNN)
   - Rolling wave storage coordination
   - Training simulation with progress tracking
   - Text generation examples
   - Model persistence workflow
   - **Note**: May encounter tokenizer initialization issues

2. **`working_workflow_example.py`** - **RECOMMENDED** - Working implementation that bypasses complex error handling
   - Direct component testing and initialization
   - Step-by-step component verification
   - Demonstrates actual pipeline flow
   - Includes fallback mechanisms
   - Tests each component individually

3. **`minimal_api_demo.py`** - API usage demonstration without actual execution
   - Shows the intended API usage patterns
   - Explains the workflow steps
   - Architecture overview
   - Production usage examples
   - Perfect for understanding the API design

4. **`simple_workflow_demo.py`** - Streamlined workflow for quick understanding
   - Essential API usage patterns
   - Lightweight configuration
   - Core functionality demonstration
   - Minimal setup for testing

### Configuration

- **`.gitignore`** - Updated to exclude CSV training files and other artifacts
- **`train-00*.csv`** - Training data files (now ignored by git)

## Workflow Steps

### 1. System Initialization
```python
from lsm_lite.api import LSMLite
from lsm_lite.utils.config import DualCNNConfig

# Initialize the API
api = LSMLite()
```

### 2. Embedder Training
The system automatically trains an embedder on your sample data:
- Fits to vocabulary and patterns in training data
- Configures embedding dimensions and sequence length
- Optimizes for downstream CNN processing

### 3. Dual CNN Pipeline Setup
```python
# Configure the dual CNN system
config = DualCNNConfig(
    embedder_fit_samples=1000,
    reservoir_size=128,
    first_cnn_filters=[16, 32, 64],    # Next-token prediction CNN
    second_cnn_filters=[32, 64, 128],  # Response CNN
    wave_window_size=16,
    dual_training_epochs=3
)

# Set up the complete pipeline
pipeline = api.setup_dual_cnn_pipeline(
    training_data=training_data,
    dual_cnn_config=config
)
```

### 4. Next-Token Prediction CNN
- Processes sequences through the attentive reservoir
- Learns to predict the next token in sequences
- Generates rolling wave outputs for coordination

### 5. Response CNN Training
- Uses rolling wave features from the first CNN
- Learns response patterns and generation strategies
- Coordinates with first CNN through wave storage

### 6. Inference and Generation
```python
# Generate text using dual CNN coordination
generated_text = api.dual_cnn_generate(
    prompt="What is machine learning?",
    max_length=50,
    temperature=0.8
)
```

### 7. Model Persistence
```python
# Save trained model
api.save_model("saved_models/my_model")

# Load model later
api.load_model("saved_models/my_model")
```

## Running the Examples

### ✅ **WORKING EXAMPLES** (Recommended)

```bash
# Test basic tokenizer functionality
python test_basic_tokenizer.py

# Complete working workflow - FULLY FUNCTIONAL
python working_workflow_example.py

# API usage overview (no execution)
python minimal_api_demo.py

# Simple workflow demonstration
python simple_workflow_demo.py
```

### ⚠️ **COMPLEX EXAMPLES** (May have issues)

```bash
# Complete workflow with complex error handling (has known issues)
python complete_workflow_example.py

# Existing examples (may need dependencies)
python examples/dual_cnn_api_example.py
python examples/dual_cnn_training_example.py
```

## Current Status

### ✅ **FULLY WORKING**
- **Basic Tokenizer**: No external dependencies, works out of the box
- **`working_workflow_example.py`**: Complete pipeline demonstration
- **All core components**: Embedder, Reservoir, Wave Storage
- **Error message formatting**: Fixed and readable

### ⚠️ **KNOWN ISSUES**
- **`complete_workflow_example.py`**: Complex error recovery system still has formatting bugs
- **External tokenizers**: Require additional libraries (transformers, spacy)

## Tokenizer Information

The system now defaults to a **basic tokenizer** that:
- ✅ **No external dependencies** required
- ✅ **Word-level tokenization** with automatic vocabulary building
- ✅ **Padding and truncation** support
- ✅ **Encode/decode** functionality
- ✅ **Compatible** with all pipeline components

### Using Different Tokenizers

```python
# Basic tokenizer (default, no dependencies)
tokenizer = UnifiedTokenizer(backend='basic')

# GPT-2 tokenizer (requires: pip install transformers)
tokenizer = UnifiedTokenizer(backend='gpt2')

# spaCy tokenizer (requires: pip install spacy)
tokenizer = UnifiedTokenizer(backend='spacy')
```

## Troubleshooting

### If you encounter any issues:

1. **Start with the basic examples**:
   ```bash
   python test_basic_tokenizer.py
   python working_workflow_example.py
   ```

2. **For external tokenizers**, install dependencies:
   ```bash
   pip install transformers torch  # For GPT-2/BERT
   pip install spacy              # For spaCy
   python -m spacy download en_core_web_sm  # English model
   ```

3. **The basic tokenizer works without any external dependencies**

## Key Features Demonstrated

### Convenience API Methods
- `setup_dual_cnn_pipeline()` - One-line pipeline initialization
- `quick_dual_cnn_train()` - Streamlined training workflow
- `dual_cnn_generate()` - Text generation with dual CNN coordination

### Advanced Capabilities
- **Attentive Reservoir**: Enhanced reservoir computing with attention mechanisms
- **Rolling Wave Storage**: Efficient coordination between CNNs
- **Dual CNN Architecture**: Next-token prediction + response generation
- **Error Handling**: Graceful fallbacks and recovery mechanisms
- **Progress Tracking**: Real-time training and setup progress
- **Memory Optimization**: Intelligent memory management

### Configuration Options
- Embedder parameters (vocabulary, dimensions, sequence length)
- Reservoir settings (size, attention heads, sparsity)
- CNN architectures (filters, layers, activation functions)
- Wave storage (window size, overlap, feature dimensions)
- Training parameters (epochs, batch size, learning rates)

## Production Considerations

### Dataset Integration
Replace sample data with real datasets:
```python
# Use actual dataset
results = api.quick_dual_cnn_train(
    dataset_name='cosmopedia-v2',  # or your dataset
    max_samples=10000,
    epochs=10
)
```

### Performance Optimization
- Use larger reservoir sizes for complex tasks
- Tune CNN architectures for your domain
- Adjust wave storage parameters for memory efficiency
- Enable mixed precision training for speed

### Monitoring and Evaluation
- Implement custom progress callbacks
- Add evaluation metrics for your use case
- Monitor wave storage utilization
- Track attention entropy for model health

## Architecture Overview

```
Input Text
    ↓
Tokenizer → Embedder
    ↓
Attentive Reservoir
    ↓
Next-Token CNN ──→ Rolling Wave Storage
    ↓                      ↓
Predictions ←── Response CNN
    ↓
Generated Text
```

## Error Handling

The workflow includes comprehensive error handling:
- Automatic fallback to single CNN if dual CNN fails
- Graceful degradation for attention mechanisms
- Memory management for large datasets
- Configuration validation and intelligent defaults

## Next Steps

1. **Customize Configuration**: Adjust parameters for your specific use case
2. **Integrate Real Data**: Replace sample data with your training corpus
3. **Tune Hyperparameters**: Optimize for your domain and performance requirements
4. **Deploy Models**: Set up serving infrastructure for production use
5. **Monitor Performance**: Implement logging and metrics collection

For more detailed examples, see the `examples/` directory and the comprehensive API documentation.