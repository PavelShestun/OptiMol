# 🧬 OptiMol-KEAP1: De Novo Drug Design via GNN-KAN & Transformers

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**OptiMol-KEAP1** is a State-of-the-Art (SOTA) generative Artificial Intelligence framework designed to discover novel, high-affinity molecular inhibitors for the **KEAP1-Nrf2 protein-protein interaction**, a promising therapeutic target for **Alzheimer's disease** and oxidative stress conditions.

By integrating **Kolmogorov-Arnold Networks (KAN)** within Graph Neural Networks (GNN) and utilizing **GPT-style Transformers** for molecular generation, OptiMol achieves superior ligand efficiency compared to traditional methods.

---

## 🚀 Key Innovations

### 1. 🧠 GNN-KAN Predictor (The "Critic")
Unlike standard MLPs, we utilize **Kolmogorov-Arnold Networks (KAN)** as the readout head for our Graph Neural Network.
*   **Advantage:** Captures highly non-linear Structure-Activity Relationships (SAR) from limited small-molecule datasets (ChEMBL).
*   **Metric:** Achieved **RMSE < 1.0** on pIC50 prediction validation.

### 2. 🏗️ Transformer Generator (The "Actor")
We replaced traditional RNNs/LSTMs with a **Decoder-only Transformer (GPT-style)** trained on **SELFIES** representation.
*   **Advantage:** 100% chemical validity, ability to generate complex macrocycles and long-range dependencies.
*   **Pre-training:** Trained on 10k ChEMBL molecules to learn chemical grammar.

### 3. 🎯 Curriculum Reinforcement Learning
To overcome the "sparse reward" problem in de novo design, we implemented a custom RL loop with:
*   **Experience Replay Buffer:** To learn from rare, high-quality molecules.
*   **Curriculum Learning:** Progressively harder tasks (Size -> Rings -> pIC50 -> CNS MPO).
*   **SOTA Reward Function:** Optimizes for pIC50, QED, SA Score (Synthetic Accessibility), and CNS MPO (Blood-Brain Barrier penetration).

---

## 📊 Results

The pipeline successfully identified novel fragment leads and drug-like candidates:

| Metric | OptiMol Best | Benchmark (Fragment) | Note |
|:---|:---|:---|:---|
| **Predicted pIC50** | **~8.5** | > 6.0 | Nanomolar activity potential |
| **Ligand Efficiency (LE)**| **> 0.45** | > 0.30 | High binding energy per atom |
| **CNS MPO** | **> 4.5** | > 4.0 | Suitable for Alzheimer's therapy |
| **Validity** | **100%** | ~90% | Via SELFIES tokenization |

> **Validation:** Top candidates were validated using **GNINA** molecular docking against the KEAP1 Kelch domain (PDB: 4L7B), showing binding affinities < -7.0 kcal/mol.

---

## 🛠️ Installation

### Prerequisites
*   Linux / WSL2 (Recommended)
*   Python 3.10+
*   CUDA (Optional, for GPU acceleration)

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/PavelShestun/OptiMol.git
cd OptiMol

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install GNINA (for Docking validation)
wget https://github.com/gnina/gnina/releases/download/v1.1/gnina -O gnina
chmod +x gnina
```

---

## 🧪 Usage Pipeline

The project is designed as a modular pipeline. Run these scripts in order:

### 1. Data Prep & Pre-training
Download data from ChEMBL and teach the Transformer the "language of chemistry".
```bash
python scripts/pretrain_transformer.py
```

### 2. Train the Predictor (The "Brain")
Train the GNN-KAN model to predict biological activity (pIC50).
```bash
python scripts/train_gnn_predictor.py
```

### 3. Reinforcement Learning (The Optimization)
Fine-tune the Transformer to generate active molecules using Policy Gradient.
```bash
python scripts/train_rl_transformer.py
```
*Note: This step uses Curriculum Learning. Monitor `Avg Reward` in logs.*

### 4. SOTA Mining & Reporting
Generate thousands of candidates, filter for the best properties, and calculate Ligand Efficiency.
```bash
python scripts/generate_sota_report_v2.py
```
*Output: `results/SOTA_MINER_REPORT.csv`*

### 5. Visualization
Create an image grid of the top discovered molecules.
```bash
python scripts/visualize_results.py
```

---

## 📂 Project Structure

```text
OptiMol/
├── data/                   # Raw and Processed Datasets
├── models/                 # Checkpoints (.pt)
├── optimol/                # Core Package
│   ├── models/             
│   │   ├── generator_transformer.py  # GPT Architecture
│   │   └── predictor_gnn.py          # GNN + KAN Architecture
│   └── utils/
│       ├── chemistry.py    # RDKit, QED, CNS MPO calculations
│       └── sota_metrics.py # SA Score, Diversity Penalty
├── scripts/                # Execution Scripts
└── results/                # CSV Reports and Images
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements in the reward function or architecture.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

