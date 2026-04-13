# DualFlow-AMP: De Novo Design of Antimicrobial Peptides via Dual-Guided Flow Matching on Protein Manifold

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥1.9-red.svg)](https://pytorch.org/)

DualFlow-AMP is a generative AI framework for de novo design of antimicrobial peptides (AMPs) using dual-guided flow matching on protein manifold.

Code: https://github.com/007whoami/DualFlow-AMP

---

## Table of Contents

- Introduction
- Key Features
- Installation
- Quick Start
- Usage Guide
- Dataset
- Model Architecture
- Generation
- Citation
- License

---

## Introduction

Antimicrobial resistance (AMR) poses a critical threat to global public health. DualFlow-AMP addresses this challenge by leveraging deep generative models to design novel antimicrobial peptides with enhanced efficacy and reduced resistance potential.

Our approach integrates:
- Flow Matching for stable and efficient trajectory modeling
- Dual Conditioning with evolutionary priors and physicochemical constraints
- Protein Language Models (ESM-2) for evolutionary-aware latent representations
- Manifold-constrained generation for biological plausibility

---

## Key Features

- Dual-Guided Generation: Combines evolutionary homology and physicochemical properties as orthogonal conditioning signals
- Controllable Design: Fine-grained control over peptide properties (net charge, hydrophobicity, length, etc.)
- Efficient Sampling: Flow matching enables faster convergence than traditional diffusion models with deterministic ODE trajectories
- High Performance: Achieves 0.982 CAMPR4 score and 0.931 amPEP score, outperforming state-of-the-art baselines
- Flexible Conditioning: Supports unconditional, physicochemical-conditioned, and homology-guided generation modes
- Validated Designs: AlphaFold3-predicted structures with high confidence (mean pLDDT > 80)

---

## Installation

### Prerequisites

- Python >= 3.13.5
- PyTorch
- CUDA >= 13.0 (recommended for GPU acceleration)
- Linux environment recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/007whoami/DualFlow-AMP.git
cd DualFlow-AMP

# Create and activate conda environment (recommended)
conda create -n dualflow_amp python=3.13.5
conda activate dualflow_amp

# Install Python dependencies
pip install -r requirements.txt
```

### Download Pretrained ESM-2 Model

Before running any pipeline, you must download the frozen ESM-2 (650M) protein language model:

```bash
# Option 1: Download via transformers
python -c "from transformers import EsmModel; EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"

# Option 2: Manual download from Hugging Face
# https://huggingface.co/facebook/esm2_t33_650M_UR50D
```

Note: The ESM-2 encoder weights are frozen throughout training and serve as a fixed feature extractor.

---

## Quick Start

Generate novel antimicrobial peptides with default settings:

```bash
# Run generation script (requires pretrained checkpoint)
bash run_generation.sh
```

Generated sequences will not be saved automated, you should copy the result and save it mannually.

---

## Usage Guide
### Step 1: Generate ESM-2 Embeddings

Before training the main model, generate latent embeddings for your dataset:

```bash
python ./Data_process/extract_embedding_and_phychm.py
```
Note:the sequences data path ,you shuold change in extract_embedding_and_phychm.py

The embeddings are stored as tensors of shape [50, 1280], zero-padded for variable-length sequences.

### Step 2: Pre-train the Sequence Decoder

The position-independent residual MLP decoder (back_to_acid) must be pre-trained to map latent representations back to discrete amino acid sequences:

```bash
cd back_to_acid
./run.sh
```
Note:the parameter should change depends on your files path

### Step 3: Main Model Training

DualFlow-AMP employs a hierarchical two-stage training strategy:

Stage 1: Physicochemical Constraint Learning (enable --pretrain flag)
```bash
bash run.sh
```

Stage 2: Evolutionary Refinement with Cross-Attention (remove --pretrain flag)
```bash
bash run.sh
```

For detailed command-line arguments and advanced options, refer to main.py or run: python main.py --help

### Step 4: Sequence Generation

Generate AMP sequences using the trained model with flexible conditioning modes:

```bash
bash run_generation.sh 
```

Generation Parameters Reference:
- --num_samples : Number of sequences to generate (default: 1000)
- --interval_min : start cfg
- --interval_max : end the cfg 
- --cfg_phychem : control the phychem cfg
- --cfg_family : control the family cfg
- --noise_scale : set the noise scale
- --seed :set the random seed
## Dataset

### Data Sources

AMP sequences are curated from five major public repositories:
- DBAASP v3.0
- dbAMP v3.0
- CAMPR4
- APD6
- DRAMP v3.0

## Model Architecture

### Core Components

1. Protein Manifold Encoder
   - Frozen ESM-2 (650M parameters)
   - Extracts evolutionary-aware embeddings: Z ∈ R^{50×1280}
   - Zero-padding for batch consistency

2. Flow Matching Generator
   - Transformer backbone with cross-attention augmentation
   - x₁-prediction parameterization for optimal transport trajectories
   - Dual classifier-free guidance (CFG) for multi-modal conditioning

3. Sequence Decoder
   - Position-independent residual MLP
   - Maps latent states to discrete amino acid sequences (20-class output)

### Conditioning Strategy

Stage I: Physicochemical Conditioning via Adaptive Layer Normalization (AdaLN)
- 8-dimensional descriptor vector: sequence length, net charge, charge density, GRAVY index, hydrophobic moment, aromatic residue ratio, proline ratio, glycine ratio
- Uniformly modulates all residue embeddings

Stage II: Homology-Guided Cross-Attention
- Cross-attention with reference AMP embeddings
- Evolutionary prior integration via stochastic sampling from homologous triplets
- Captures residue-level correspondences for targeted exploration

---

## Evaluation

### In Silico Predictors Used

- Antimicrobial Activity: CAMPR4, amPEP, AMPscanner V2
- Toxicity: ToxinPred 3.0
- Hemolysis: HemoPI-2 (regression model)
- Structure Prediction: AlphaFold3
---

## Citation

If you use DualFlow-AMP in your research, please cite:

```bibtex
@article{liu2026dualflowamp,
  title={DualFlow-AMP: De Novo Design of Antimicrobial Peptides via Dual-Guided Flow Matching on Protein Manifold},
  author={Liu, Jun and Tuo, Shouheng},
  year={2026},
  url={https://github.com/007whoami/DualFlow-AMP}
}
```
---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## Contributing

We welcome contributions from the community!

1. Fork the repository
2. Create a feature branch: git checkout -b feature/AmazingImprovement
3. Commit your changes: git commit -m 'Add: Amazing improvement'
4. Push to the branch: git push origin feature/AmazingImprovement
5. Open a Pull Request with a clear description of your changes

For major changes, please open an issue first to discuss your proposed modifications.

---

## Contact

For questions, bug reports, or collaboration inquiries:
- Email: liu_jun_root@163.com
---

## Acknowledgments

- ESM-2: Meta AI Research for the pretrained protein language model
- AMP Databases: DBAASP, dbAMP, CAMPR4, APD, DRAMP for curated sequence data
- AlphaFold3: DeepMind for high-accuracy structure prediction
- Evaluation Tools: CAMPR4, amPEP, ToxinPred, HemoPI-2 development teams
- PyTorch & Transformers: Open-source communities for foundational libraries

---