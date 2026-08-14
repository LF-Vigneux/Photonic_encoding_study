# Photonic Encoding Study

A comprehensive research project investigating and comparing different photonic encoding strategies for Quantum Machine Learning (QML) algorithms. This study aims to establish practical guidelines for selecting optimal encoding strategies across various datasets and computational configurations.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Main Arguments](#main-arguments)
  - [Example Commands](#example-commands)
- [Output and Results](#output-and-results)
- [Metric Analysis and Visualization](#metric-analysis-and-visualization)
- [Datasets](#datasets)
- [Encoding Strategies](#encoding-strategies)
- [Complexity Metrics](#complexity-metrics)

## Overview

This project compares classical and quantum-induced complexity metrics across different photonic encoding strategies for quantum machine learning. The study evaluates:

- **Classical Complexity Metrics**: Distributional entropy, correlation order, Kolmogorov complexity, topological complexity, and Wasserstein distance
- **Quantum-Induced Complexity Metrics**: Hilbert space support dimension, quantum Fisher information spread, entanglement entropy, kernel spectrum flatness, locality vs expressibility, and more
- **Visualization**: UMAP embeddings (2D and 3D) showing feature space learned by different encoding layers

The project uses the **Merlin** library for photonic quantum circuits, providing a realistic simulation of photonic quantum computing.

## Repository Structure

```
Photonic_encoding_study/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── cli.json                           # CLI argument schema
├── runner.py                          # Main experiment runner
├── implementation.py                  # Core implementation logic
├── run_dataset_complexity.py          # Dataset complexity experiment
│
├── configs/                           # Configuration files
│   ├── defaults.json                  # Default configurations
│   └── induced_dataset_complexity_exp.json  # Experiment configs
│
├── data/                              # Data loading and preprocessing
│   ├── loader.py                      # Dataset loading functions
│   └── data/
│       ├── hidden_manifold.py         # Synthetic data generation
│       ├── pathmnist.npz              # PathMNIST dataset
│       ├── cifar-10-batches-py/       # CIFAR-10 dataset
│       └── eurosat/                   # EuroSAT satellite dataset
│
├── encodings_merlin/                  # Photonic encoding implementations
│   ├── encoding_layers.py             # Angle, amplitude, Fourier encodings
│   ├── utils.py                       # Utility functions for encodings
│   ├── test_encoding_layers.py        # Unit tests
│
├── dataset_complexity/                # Complexity metric calculations
│   ├── complexity_metrics.py          # Classical and quantum metrics
│   ├── plotter.py                     # Visualization functions
│   ├── umap.py                        # UMAP embedding generation
│   ├── utils.py                       # Helper utilities
│   └── test_utils_datasets_complexity.py  # Tests
│
├── nn_embedding/                      # Neural quantum embedding models
│   ├── lib/
│   │   ├── gate_based_model.py        # Gate-based quantum models
│   │   ├── merlin_based_model.py      # Merlin-based photonic models
│   │   └── training_without_nqe.py    # Training utilities
│   ├── utils/
│   │   ├── gate_based_embedding.py
│   │   ├── merlin_models.py
│   │   ├── merlin_model_utils.py
│   │   └── utils.py                   # Argument parsing and utilities
│   └── tests/                         # Unit tests
│
├── egas/                              # Evolutionary Quantum Algorithm Search
│   ├── photonic_egas.py               # EGAS implementation
│   ├── photonic_circuits.py           # Circuit building utilities
│   ├── photonic_kernel_svm.py
│   ├── gpt.py                         # LLM-based encoding generation
│   ├── photonic_bias.py
│   ├── statevec.py
│   ├── wasserstein.py
│   └── utils/
│       ├── make_tables.py
│       ├── plot_paper_figures.py
│       ├── plot_results.py
│       └── run_fig3_repeats.py
│
├── results/                           # Generated results and visualizations
│   ├── breast8-2-classes/
│   ├── CIFAR8-*/
│   ├── EuroSAT8-*/
│   ├── fashionMNSIT8-*/
│   ├── MNIST8-*/
│   ├── moons2-*/
│   └── wine8-*/
│       ├── complexities/              # Complexity metrics (JSON)
│       └── umaps/                     # UMAP visualizations (HTML/PNG)
│
└── metric_analysis/                   # Analysis and visualization scripts
    ├── classical_metrics_table.py       # LaTeX table of classical metrics
    ├── wasserstein_comparison.py        # Wasserstein distance analysis
    ├── quantum_multi_dataset_panel.py   # Quantum metrics panel across multiple datasets
    ├── normalized_summary.py            # Normalized metrics summary
    ├── nqe_egas_comparison.py           # NQE vs EGAS metrics comparison
    ├── nqe_egas_training_curves.py      # Training curves and performance analysis
    ├── quantum_metrics_panel.py         # Quantum metrics panel visualization
    ├── test_quantum_metrics.py          # Quantum metrics unit tests
    └── results/                         # Generated analysis outputs
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LF-Vigneux/Photonic_encoding_study.git
   cd Photonic_encoding_study
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dependencies Overview**:
   - `merlinquantum==0.4.0`: Photonic quantum circuit simulation (automatically includes PyTorch)
   - `torch==2.10.0`, `torchvision==0.25.0`: Deep learning framework
   - `perceval-quandela==1.2.3`: Photonic quantum computing framework
   - `numpy`, `scipy`, `scikit-learn`: Scientific computing and machine learning
   - `quimb==1.13.0`: Quantum information utilities
   - `matplotlib`, `plotly`: Visualization
   - `pandas`: Data analysis
   - `umap-learn`: Dimensionality reduction for embeddings
   - `ripser`: Topological data analysis (persistent homology)
   - `tqdm`: Progress bars
   - `pytest`: Testing framework
   - `absl-py`: Logging utilities

## Usage

### Running Experiments

The main entry point is `runner.py`, which dispatches different experiments based on configuration.

#### Basic Usage

```bash
python runner.py --config configs/induced_dataset_complexity_exp.json
```

#### Command Line Arguments

```bash
python runner.py \
  --exp_to_run INDUCED_DATASET_COMPLEXITY \
  --dataset_name mnist \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --feature_reduction 8 \
  --computation_space UNBUNCHED \
  --generate_umap_plots true
```

### Main Arguments

#### Experiment Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--exp_to_run` | str | `INDUCED_DATASET_COMPLEXITY` | Experiment name to execute |
| `--dataset_name` | str | `mnist` | Dataset: `mnist`, `fashion`, `kmnist`, `cifar10`, `eurosat`, `breast_cancer`, `wine`, `moons`, `spiral` |
| `--classes` | list[int] | `[0-9]` | Class indices to include (e.g., `0 1 2 3`) |
| `--feature_reduction` | int | 8 | Feature dimension after PCA preprocessing |

#### Quantum Circuit Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n_modes` | int | None | Number of optical modes (auto-computed if None) |
| `--n_photons` | int | None | Number of photons in circuit |
| `--computation_space` | str | `UNBUNCHED` | Photonic space type: `UNBUNCHED` or `BUNCHED` |
| `--num_qubits_per_feature_fourier` | int | 1 | Qubits per feature for Fourier encoding |

#### Classical Complexity Metrics

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hyper_parameters_classical` | list[float] | `[1,1,1,1,1]` | Weights for: entropy, correlation, Kolmogorov, topology, Wasserstein |
| `--max_order_correlation_classical` | int | 4 | Maximum correlation order to compute |
| `--max_dim_topology_classical` | int | 2 | Maximum homology dimension for classical topological complexity |
| `--weights_topology_classical` | list[float] | None | Custom weights for topological features |
| `--max_samples_topology_classical` | int | 1000 | Max samples for topological computation |

#### Quantum-Induced Complexity Metrics

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hyper_parameters_induced` | list[float] | `[1,1,1,1,1,1,1]` | Weights for 7 quantum metrics |
| `--epsilon_hilbert_support_dim_induced` | float | 1e-8 | Threshold for Hilbert space dimension computation |
| `--n_samples_loc_vs_express_induced` | int | 1000 | Samples for locality vs expressibility |
| `--n_bins_loc_vs_express_induced` | int | 50 | Bins for locality vs expressibility histogram |
| `--max_dim_topology_induced` | int | 2 | Maximum homology dimension for quantum topology |
| `--weights_topology_induced` | list[float] | None | Custom weights for quantum topological features |
| `--max_samples_topology_induced` | int | 1000 | Max samples for quantum topological computation |
| `--max_samples_induced` | int | 5000 | Maximum samples for quantum metric computation |

#### Visualization & Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--generate_umap_plots` | bool | `true` | Generate UMAP embedding visualizations |
| `--generate_umap_2d` | bool | `true` | Generate 2D UMAP plots |
| `--generate_umap_3d` | bool | `true` | Generate 3D UMAP plots |
| `--umap_state` | int | 42 | Random seed for UMAP |
| `--umap_num_points_per_class` | int | 50 | Samples per class in UMAP (reduces computation) |
| `--umap_n_neighbors` | int | 15 | Number of neighbors for UMAP algorithm |
| `--umap_n_epochs` | int | 200 | Training epochs for UMAP |

#### Advanced Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--evaluate_evolution` | bool | `false` | Run evolutionary algorithm search for encodings |
| `--randomize_entangling` | bool | `true` | Randomize entangling layer parameters |

### Example Commands

#### Example 1: Basic MNIST Study (2-class)
```bash
python runner.py \
  --dataset_name mnist \
  --classes 0 1 \
  --feature_reduction 8 \
  --generate_umap_plots true
```

#### Example 2: CIFAR-10 Analysis
```bash
python runner.py \
  --dataset_name cifar10 \
  --classes 0 1 2 3 \
  --feature_reduction 8 \
  --n_modes 12 \
  --n_photons 2 \
  --computation_space UNBUNCHED
```

#### Example 3: Custom Complexity Weights
```bash
python runner.py \
  --dataset_name mnist \
  --classes 0 1 2 3 4 5 6 7 8 9 \
  --feature_reduction 16 \
  --hyper_parameters_classical 2.0 1.5 1.0 2.0 1.0 \
  --hyper_parameters_induced 1.5 1.5 1.5 1.5 1.5 1.5 1.5 \
  --max_samples_induced 10000 \
  --umap_num_points_per_class 100
```

#### Example 4: EuroSAT Satellite Classification
```bash
python runner.py \
  --dataset_name eurosat \
  --classes 0 1 2 3 4 5 \
  --feature_reduction 8 \
  --max_order_correlation_classical 5
```

#### Example 5: Boosted Quantum Metrics
```bash
python runner.py \
  --dataset_name mnist \
  --classes 0 1 \
  --feature_reduction 16 \
  --hyper_parameters_induced 2.0 2.0 2.0 2.0 2.0 2.0 2.0 \
  --max_samples_induced 15000 \
  --randomize_entangling false
```

### Running Metric Analysis from implementation.py

Metric analysis scripts can be executed directly via the main `implementation.py` script using predefined config files. This provides a convenient way to run analysis without navigating to the `metric_analysis/` directory.

#### Quick Start - Available Metric Analyses

```bash
# Quantum metrics panel (one PDF per metric showing all encodings)
python implementation.py --config configs/metrics_quantum_panel.json

# Multi-dataset quantum panel (compare metrics across datasets)
python implementation.py --config configs/metrics_multi_dataset_panel.json

# NQE vs EGAS training comparison (with timing and memory instrumentation)
python implementation.py --config configs/metrics_nqe_egas_training.json

# Wasserstein distance comparison across datasets
python implementation.py --config configs/metrics_wasserstein_comparison.json

# Classical metrics LaTeX table
python implementation.py --config configs/metrics_classical_table.json

# NQE vs EGAS metrics comparison
python implementation.py --config configs/metrics_nqe_egas_comparison.json

# Normalized metrics summary
python implementation.py --config configs/metrics_normalized_summary.json
```

#### Creating Custom Metric Configs

You can create custom config files to run metric analyses with different parameters. Example template:

```json
{
    "description": "Custom metric analysis",
    "outdir": "results",
    "exp_to_run": "QUANTUM_METRICS_PANEL"
}
```

Supported `exp_to_run` values:
- `INDUCED_DATASET_COMPLEXITY` - Main complexity experiment (default)
- `QUANTUM_METRICS_PANEL` - Per-encoding quantum metrics visualization
- `QUANTUM_MULTI_DATASET_PANEL` - Multi-dataset quantum metrics panel
- `NQE_EGAS_TRAINING_CURVES` - NQE/EGAS training comparison with instrumentation
- `WASSERSTEIN_COMPARISON` - Wasserstein distance analysis
- `CLASSICAL_METRICS_TABLE` - LaTeX classical metrics table
- `NQE_EGAS_COMPARISON` - Direct NQE vs EGAS comparison
- `NORMALIZED_SUMMARY` - Normalized metrics summary

## Output and Results

### Generated Output Structure

After running an experiment, results are organized in `results/{dataset_name}{num_classes}-{num_features_or_reduced}/`:

```
results/MNIST8-2-classes/
├── complexities/
│   └── dataset_complexity_mnist_results.json
└── umaps/
    ├── u_map_amplitude_mnist_2d.html        # Interactive 3D visualization
    ├── u_map_amplitude_mnist_3d.html
    ├── u_map_angle_mnist_2d.html
    ├── u_map_angle_mnist_3d.html
    ├── u_map_classical_mnist_2d.html        # Classical feature space
    ├── u_map_classical_mnist_3d.html
    ├── u_map_fourier_mnist_2d.html
    └── u_map_fourier_mnist_3d.html
```

### Complexity Metrics Output (JSON)

The `dataset_complexity_{dataset}_results.json` file contains:

```json
{
  "classical": {
    "distributional_entropy": 5.234,
    "correlation_order": 3.102,
    "kolmogorov_complexity": 0.876,
    "topological_complexity": 124.5,
    "wasserstein distance": 2.341,
    "normalized_score": 0.654
  },
  "induced": {
    "angle_encoding": {
      "hilbert_space_support_dim": 512,
      "quantum_fisher_information_spread": 2.134,
      "entanglement_entropy": 1.245,
      "kernel_spectrum_flatness": 0.432,
      "locality_vs_expressibility": 0.678,
      "topological_quantum_complexity": 245.3,
      "multipartite_correlation": 3.456
    },
    "amplitude_encoding": { ... },
    "fourier_basis": { ... }
  }
}
```

### Visualization Examples

#### UMAP Embeddings (2D/3D)

- **Classical UMAP**: Shows the original feature space after preprocessing
- **Angle Encoding UMAP**: Visualizes quantum state embeddings from angle encoding
- **Amplitude Encoding UMAP**: Visualizes quantum state embeddings from amplitude encoding
- **Fourier Basis UMAP**: Visualizes quantum state embeddings from Fourier basis encoding

These interactive HTML plots allow:
- Hover to see individual sample information
- Color-coded by class
- Zoom and pan functionality
- Visualization of encoding effectiveness (good separation = good encoding)

## Metric Analysis and Visualization

The `metric_analysis/` directory contains specialized scripts for generating publication-quality visualizations and comparative analyses of metrics across datasets and encoding strategies.

### Available Analysis Scripts

#### 1. **classical_metrics_table.py**
Generates a LaTeX table summarizing normalized classical complexity metrics across all datasets.

```bash
python metric_analysis/classical_metrics_table.py --output results/classical_metrics_table.tex
```

**Output**:
- LaTeX table with multi-level headers
- Normalized metric values (0-1 scale)
- Dataset-feature-class labeling format

#### 2. **quantum_metrics_panel.py**
Creates PDF panels comparing quantum metrics with one subplot per encoding strategy and bars for each dataset.

```bash
python metric_analysis/quantum_metrics_panel.py --output-dir results
```

**Output**:
- One PDF file per quantum metric (e.g., `hilbert_space_support_dim_per_encoding.pdf`)
- Bar charts with dataset-colored bars
- Legend showing all datasets

**Key Features**:
- Vertical bar labels with percentages
- 2-column layout for optimal readability
- Configurable figure size and font sizes via hyperparameters

#### 3. **quantum_multi_dataset_panel.py**
Generates quantum metrics panel plots comparing multiple datasets with bars for each encoding strategy.

```bash
python metric_analysis/quantum_multi_dataset_panel.py --output results/quantum_multi_dataset_comparison.pdf
```

**Output**:
- Multi-row/multi-column subplot layout for quantum metrics
- One row per dataset, bars for encodings
- Unified legend

#### 4. **nqe_egas_comparison.py**
Compares specific metrics between NQE (Neural Quantum Embedding) and EGAS (Evolutionary Quantum Algorithm Search) methods.

```bash
python metric_analysis/nqe_egas_comparison.py \
  --metrics hilbert_space_support_dim entanglement_entropy \
  --output results/nqe_egas_comparison.pdf
```

**Output**:
- Side-by-side comparison plots (NQE vs EGAS)
- Highlights performance differences

#### 5. **nqe_egas_training_curves.py**
Trains both NQE and EGAS on a chosen dataset and generates training curves, accuracy/loss analysis, and resource usage statistics.

```bash
python metric_analysis/nqe_egas_training_curves.py
```

**Configurable hyperparameters** (at script top):
- `DATASET_NAME`: Dataset to use (default: "wine")
- `FEATURE_REDUCTION`: PCA reduced dimension (default: 8)
- `CLASSES`: Class indices to include (default: (0, 1, 2))
- `NQE_NUM_EPOCHS`: Training epochs for NQE (default: 1000)
- `EGAS_NUM_EPOCHS`: Search iterations for EGAS (default: 4000)
- `EGAS_REFINE_EPOCHS`: Bias refinement epochs (default: 400)

**Output**:
- `nqe_egas_training_curves.pdf`: Training curve plots
- `nqe_egas_training_summary.json`: Detailed summary including:
  - Train/test/total accuracy and loss
  - Training time and time-per-epoch
  - Peak memory usage and memory growth
  - Dataset/split sizes

#### 6. **wasserstein_comparison.py**
Analyzes Wasserstein distance metrics across datasets and encodings.

```bash
python metric_analysis/wasserstein_comparison.py --output results/wasserstein_analysis.pdf
```

#### 7. **normalized_summary.py**
Generates a normalized summary of all metrics with standardized scaling.

```bash
python metric_analysis/normalized_summary.py --output results/summary.pdf
```

#### 8. **test_quantum_metrics.py**
Unit tests for quantum metric computation and visualization functions.

```bash
python metric_analysis/test_quantum_metrics.py
```

### Visualization Conventions

All analysis scripts follow consistent conventions:
- **Dataset labeling**: `{dataset}-{features}-{classes}c` format (e.g., "mnist-8-2c")
- **Encoding colors**: Tab10/Tab20 colormaps for consistent encoding representation
- **Output format**: PDFs at 300 DPI for publication quality
- **LaTeX tables**: `\shortstack` headers for compact multi-line column labels

## Datasets

Supported datasets include:

| Dataset | Classes | Samples | Features | Type |
|---------|---------|---------|----------|------|
| MNIST | 10 | 70,000 | 28×28 | Handwritten digits |
| Fashion-MNIST | 10 | 70,000 | 28×28 | Fashion products |
| Kuzushiji-MNIST | 10 | 70,000 | 28×28 | Japanese characters |
| CIFAR-10 | 10 | 60,000 | 32×32×3 | Natural images |
| EuroSAT | 10 | 27,000 | Variable | Satellite imagery |
| Breast Cancer | 2 | 569 | 30 | Medical data |
| Wine | 3 | 178 | 13 | Chemical analysis |
| Moons | 2 | 1,000 | 2 | Synthetic |
| Spiral | Variable | Variable | 2 | Synthetic |

All image datasets are automatically reduced to specified feature dimensions via PCA.

## Encoding Strategies

The study compares multiple photonic quantum encoding strategies:

### 1. **Angle Encoding**
   - Encodes classical data as rotation angles on photonic modes
   - Produces entangling layers for feature interaction
   - Best for: Small to medium feature dimensions

### 2. **Amplitude Encoding**
   - Encodes data as amplitudes in the photonic state
   - Compact representation but requires state preparation
   - Best for: Higher dimensional feature spaces

### 3. **Fourier Basis Encoding**
   - Uses Fourier features with varying frequencies
   - Provides multi-scale feature representation
   - Best for: Periodic or smooth decision boundaries

### 4. **Dense Angle Encoding**
   - Enhanced angle encoding with additional rotations
   - More expressive but computationally intensive

### 5. **EGAS Encoding** (Optional)
   - Evolutionarily optimized encoding circuits
   - Automatically searches for effective encoding strategies
   - Requires: `--evaluate_evolution true`

## Complexity Metrics

### Classical Metrics

1. **Distributional Entropy**: Information content in data distribution
2. **Correlation Order**: Statistical dependencies between features
3. **Kolmogorov Complexity**: Minimum description length of data
4. **Topological Complexity**: Persistent homology features
5. **Wasserstein Distance**: Class separability in optimal transport sense

### Quantum-Induced Metrics

1. **Hilbert Space Support Dimension**: Effective dimension of quantum feature space
2. **Quantum Fisher Information Spread**: Learnability via quantum gradients
3. **Entanglement Entropy**: Correlation structure in quantum states
4. **Kernel Spectrum Flatness**: Complexity of learned kernel matrix
5. **Locality vs Expressibility**: Trade-off between local and global features
6. **Topological Quantum Complexity**: Quantum topological features
7. **Multipartite Total Correlation**: Global quantum coherence

---

**Author**: Internship research project  
**License**: [Specify your license]  
**Based on**: Neural Quantum Embedding framework and Merlin photonic quantum computing library