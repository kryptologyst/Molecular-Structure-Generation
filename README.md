# Molecular Structure Generation

A production-ready framework for generating molecular structures using deep learning. This project implements state-of-the-art generative models for molecular design, including GraphVAE, GraphGAN, and autoregressive approaches.

## Features

- **Multiple Model Architectures**: GraphVAE, GraphGAN, and Autoregressive models
- **Comprehensive Evaluation**: Validity, uniqueness, novelty, QED, SA, LogP metrics
- **Interactive Demo**: Streamlit-based web interface for molecular generation
- **Production Ready**: Proper configuration management, logging, and checkpointing
- **Modern Stack**: PyTorch 2.x, Python 3.10+, with full device support (CUDA/MPS/CPU)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Molecular-Structure-Generation
cd molecular-structure-generation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from src import GraphVAE, MolecularDataset, MolecularSampler

# Load dataset
dataset = MolecularDataset(smiles_list=["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"])

# Create model
model = GraphVAE(atom_types=10, bond_types=4, max_atoms=50)

# Generate samples
sampler = MolecularSampler(model, device="cuda")
samples = sampler.sample_molecules(num_samples=10)
```

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom parameters
python scripts/train.py --epochs 200 --batch_size 128 --lr 0.001
```

### Sampling

```bash
# Generate samples from trained model
python scripts/sample.py --model_path checkpoints/best_model.pt --num_samples 100

# Generate with evaluation
python scripts/sample.py --model_path checkpoints/best_model.pt --num_samples 100 --evaluate
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py
```

## Project Structure

```
molecular-structure-generation/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data handling
│   ├── evaluation/        # Evaluation metrics
│   ├── sampling/          # Sampling utilities
│   ├── training/          # Training logic
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and sampling scripts
├── demo/                  # Interactive demos
├── tests/                 # Unit tests
├── assets/                # Generated samples and visualizations
├── data/                  # Dataset storage
└── checkpoints/           # Model checkpoints
```

## Models

### GraphVAE
Variational Autoencoder for molecular graphs that learns latent representations of molecules. Good for interpolation and controlled generation.

**Key Features:**
- Graph-based molecular representation
- Latent space interpolation
- Beta annealing for training stability
- KL divergence regularization

### GraphGAN
Generative Adversarial Network for molecular graphs that generates realistic molecular structures through adversarial training.

**Key Features:**
- Adversarial training for realistic generation
- Generator-discriminator architecture
- Spectral normalization for stability
- Gradient penalty for training stability

### Autoregressive Generator
Transformer-based autoregressive model that generates SMILES strings directly using attention mechanisms.

**Key Features:**
- Direct SMILES generation
- Transformer architecture
- Attention-based generation
- Temperature and nucleus sampling

## Evaluation Metrics

The framework provides comprehensive evaluation metrics for generated molecules:

- **Validity**: Percentage of chemically valid molecules
- **Uniqueness**: Percentage of unique molecules in generated set
- **Novelty**: Percentage of molecules not in training set
- **QED**: Quantitative Estimate of Drug-likeness
- **SA Score**: Synthetic Accessibility score
- **LogP**: Partition coefficient
- **Lipinski Compliance**: Rule of Five compliance
- **Molecular Weight**: Molecular weight distribution

## Configuration

The project uses YAML-based configuration management:

```yaml
model:
  type: "graph_vae"
  z_dim: 100
  hidden_dim: 128
  max_atoms: 50

training:
  batch_size: 64
  learning_rate: 0.0002
  num_epochs: 100
  beta: 1.0

data:
  dataset: "zinc"
  data_path: "data/"
  train_split: 0.8
```

## Dataset Support

### ZINC Dataset
The framework supports the ZINC dataset for molecular generation. Place your ZINC data in the `data/` directory.

### Toy Dataset
For testing and development, the framework includes a toy dataset generator that creates simple molecular structures.

## Advanced Features

### Molecular Interpolation
GraphVAE models support latent space interpolation between molecules:

```python
# Interpolate between two molecules
interpolated = sampler.interpolate_molecules(mol1, mol2, num_steps=10)
```

### Custom Sampling
Advanced sampling strategies with temperature control, top-k, and nucleus sampling:

```python
samples = sampler.sample_molecules(
    num_samples=100,
    temperature=1.2,
    top_k=50,
    top_p=0.9
)
```

### Evaluation and Visualization
Comprehensive evaluation with automatic visualization:

```python
metrics = MolecularMetrics()
results = metrics.compute_all_metrics(smiles_list)
metrics.plot_metrics_distribution(smiles_list, "metrics.png")
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Code Formatting

```bash
# Format code
black src/ scripts/ tests/
ruff check src/ scripts/ tests/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Adding New Models

To add a new molecular generation model:

1. Implement the model in `src/models/molecular_models.py`
2. Add training logic in `src/training/trainer.py`
3. Add sampling logic in `src/sampling/sampler.py`
4. Update configuration in `configs/default.yaml`
5. Add tests in `tests/test_models.py`

## Performance

The framework is optimized for performance with:

- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Device Support**: Automatic CUDA/MPS/CPU detection and usage
- **Memory Efficiency**: Optimized data loading and batching
- **Parallel Processing**: Multi-worker data loading

## Limitations

- **SMILES Conversion**: Current implementation uses simplified graph-to-SMILES conversion
- **Model Complexity**: Some models may require significant computational resources
- **Dataset Size**: Large datasets may require substantial memory and storage

## Contributing

Contributions are welcome! Please see the contributing guidelines for details on:

- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{molecular_structure_generation,
  title={Molecular Structure Generation: A Modern Deep Learning Framework},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Molecular-Structure-Generation}
}
```

## Acknowledgments

- RDKit for molecular informatics
- PyTorch for deep learning framework
- Streamlit for interactive demos
- The molecular generation research community

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the examples in the `demo/` directory

---

**Note**: This framework is designed for research and educational purposes. Generated molecules should be validated before any real-world applications.
# Molecular-Structure-Generation
