# Resource-Adaptive Federated Learning for ECG Arrhythmia Classification

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2Bcpu-EE4C2C.svg)](https://pytorch.org/)
[![WFDB](https://img.shields.io/badge/WFDB-4.3.1-brightgreen.svg)](https://physionet.org/about/software/)
[![Dataset](https://img.shields.io/badge/Dataset-MIT--BIH%20Arrhythmia-orange.svg)](https://physionet.org/content/mitdb/1.0.0/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end framework for biomedical signal preprocessing, patient-level evaluation splits, and Non-IID Dirichlet federated data simulation for ECG heartbeat classification on the **MIT-BIH Arrhythmia Database**, following the **ANSI/AAMI EC57** international standard.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Architecture](#project-architecture)
3. [ECG Preprocessing Pipeline](#ecg-preprocessing-pipeline)
   - [Zero-Phase Butterworth Bandpass Filtering](#1-zero-phase-butterworth-bandpass-filtering)
   - [R-Peak Detection & Segmentation](#2-r-peak-detection--segmentation)
   - [Z-Score Normalization](#3-z-score-normalization)
   - [Linear Interpolation Resampling](#4-linear-interpolation-resampling)
   - [AAMI Heartbeat Categorization](#5-aami-heartbeat-categorization)
   - [Inter-Patient Split Paradigm](#6-inter-patient-split-paradigm)
4. [Federated Non-IID Partitioning](#federated-non-iid-partitioning)
5. [Model Architectures: FastKAN vs. Matched MLP](#model-architectures-fastkan-vs-matched-mlp)
6. [Flower (flwr) Federated Integration](#flower-flwr-federated-integration)
7. [Baseline Experiments & Empirical Results](#baseline-experiments--empirical-results)
   - [Diagnostic Visualizations](#diagnostic-visualizations)
8. [Reinforcement Learning Gymnasium Environment](#reinforcement-learning-gymnasium-environment)
   - [State Space & Telemetry Vector](#1-state-space--telemetry-vector)
   - [Action Space & Dynamic Spline Interpolation](#2-action-space--dynamic-spline-interpolation)
   - [Multi-Objective Reward Function (Equation 7)](#3-multi-objective-reward-function-equation-7)
9. [Verification & Exploratory Data Analysis (EDA)](#verification--exploratory-data-analysis-eda)
10. [Installation & Setup](#installation--setup)
11. [CLI Usage](#cli-usage)
12. [Dataset Summary & Statistics](#dataset-summary--statistics)
13. [Project Roadmap & Next Steps](#project-roadmap--next-steps)
14. [References](#references)

---

## Overview

Heartbeat classification across distributed medical devices (e.g., edge monitors, wearable smartwatches, and clinical bedside telemetry) requires privacy-preserving federated machine learning. However, real-world clinical data exhibits severe **patient heterogeneity** and **class imbalance**. 

This repository provides:
- **Biomedical Signal Processing**: Zero-phase filtering, artifact removal, robust R-peak detection, and dimension downsampling to $n=32$ features.
- **Strict Inter-Patient Paradigm**: Patient-level data splitting (splitting by recording IDs) to prevent patient data leakage between train, validation, and test splits.
- **Federated Simulation**: Class-conditional Dirichlet non-IID partitioning for $N$ clients across varying heterogeneity levels ($\alpha \in [0.1, 0.5, 1.0]$).
- **Fully Deterministic & Configurable**: Controlled via YAML configs and global seeding.

---

## Project Architecture

```
thesis_project/
├── configs/
│   ├── dataset.yaml                # Master dataset, filter, split, & partition configuration
│   ├── fl.yaml                     # Federated learning hyperparameters
│   ├── kan.yaml                    # Kolmogorov-Arnold Network model configuration
│   ├── paper_baseline.yaml         # Multi-algorithm benchmark evaluation config
│   └── ppo.yaml                    # Reinforcement learning controller configuration
├── data/
│   ├── raw/
│   │   └── mit-bih-arrhythmia-database-1.0.0/  # Raw WFDB recordings (.dat, .hea, .atr)
│   ├── processed/
│   │   ├── beats/                  # Extracted .npy heartbeat arrays (X, y, record_ids)
│   │   └── metadata/               # Patient splits, class distributions, partition tables
│   └── partitions/                 # Client index allocations per Dirichlet alpha
├── experiments/
│   └── run_baseline.py             # FL baseline runner across Dirichlet alphas
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   └── 03_partition_analysis.ipynb
├── results/
│   ├── baselines/                  # FL round metrics, summaries, and diagnostic plots
│   │   └── figures/                # Accuracy, F1, and communication efficiency figures
│   └── figures/
│       └── eda/                    # Intermediate verification & diagnostic plots
├── src/
│   ├── preprocessing.py            # Complete ECG filtering, windowing, & split pipeline
│   ├── partition.py                # Dirichlet federated partitioning module
│   ├── dataset.py                  # WFDB record & annotation I/O helpers
│   ├── splits.py                   # Patient-level record splitting
│   ├── visualization.py            # High-resolution clinical ECG plotting routines
│   ├── environment/                # Gymnasium RL environments & telemetry trackers
│   │   ├── f1_env.py               # FederatedKANEnv (Gymnasium.Env) MDP implementation
│   │   ├── fl_env.py               # FederatedKANEnv aliases & exports
│   │   └── telemetry.py            # Client & server 5-dim state telemetry tracker
│   ├── models/                     # Classification architectures (MLP, FastKAN)
│   │   ├── fastkan_model.py        # Modular FastKAN with dynamic (G, k) parameterization
│   │   ├── mlp.py                  # Parameter-matched MLP baseline
│   │   └── kan.py                  # FastKAN aliases & backward compatibility
│   ├── federated/                  # Federated learning clients, servers, and aggregators
│   │   ├── client.py               # Flower NumPyClient with telemetry & local validation
│   │   ├── fedprox.py              # Proximal loss regularization term
│   │   ├── server.py               # Centralized evaluator & FedAvg aggregation
│   │   └── fedavg.py               # FedAvg module aliases
│   └── utils/
│       ├── metrics.py              # AAMI classification metrics (Macro F1, Accuracy)
│       ├── reproducibility.py      # Deterministic seed controls (Python, NumPy, Torch)
│       └── splits.py
├── tests/
│   └── test_environment.py        # Comprehensive test suite for telemetry & Gym environment
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ECG Preprocessing Pipeline

The preprocessing pipeline in [`src/preprocessing.py`](src/preprocessing.py) converts continuous multi-channel ECG telemetry into standardized, feature-resampled heartbeat vectors.

```mermaid
flowchart TD
    A["Raw ECG (.dat, .hea, .atr)"] --> B["Butterworth Bandpass (0.5 - 40 Hz, Order 3)"]
    B --> C["R-Peak Detection & Alignment"]
    C --> D["Fixed-Window Beat Extraction (360 samples = 1.0 s)"]
    D --> E["Z-Score Normalization: z = (x - μ) / σ"]
    E --> F["Linear Interpolation Resampling (n = 32 features)"]
    F --> G["AAMI Class Grouping (N, SVEB, VEB, F, Q)"]
    G --> H["Inter-Patient Split (Train / Val / Test by Record ID)"]
```

### 1. Zero-Phase Butterworth Bandpass Filtering
Eliminates respiration-induced baseline wander ($< 0.5\text{ Hz}$) and high-frequency electromyographic (EMG) or powerline noise ($> 40\text{ Hz}$):
- **Filter order**: $3$ (effective order 6 bidirectional)
- **Cutoffs**: $f_{\text{low}} = 0.5\text{ Hz}$, $f_{\text{high}} = 40.0\text{ Hz}$ ($f_s = 360\text{ Hz}$)
- **Implementation**: Second-order sections (`sosfiltfilt`) for guaranteed numerical stability and zero phase distortion.

### 2. R-Peak Detection & Segmentation
- Extracts a **360-sample window** ($1.0\text{ second}$ duration at $360\text{ Hz}$) centered at the R-peak:
  $$\text{Window} = [r - 180, \; r + 180)$$
- Boundary beats that truncate before $0$ or beyond signal duration are automatically discarded.

### 3. Z-Score Normalization
Each individual beat segment $x$ is normalized to zero mean and unit variance:
$$z = \frac{x - \mu}{\sigma + \epsilon}$$
where $\epsilon = 10^{-8}$ prevents numerical instability for constant or low-amplitude signals.

### 4. Linear Interpolation Resampling
Each 360-sample beat is resampled to a compact representation of $n = 32$ features via continuous linear interpolation:
$$x_{\text{resampled}} = \mathcal{I}_{\text{linear}}(x_{\text{normalized}}, \; n=32)$$
Supports both vectorized `scipy.interpolate.interp1d` and batch-accelerated `torch.nn.functional.interpolate`.

### 5. AAMI Heartbeat Categorization
Annotations from `.atr` files are mapped into the five standard ANSI/AAMI EC57 heartbeat classes. Non-heartbeat annotations (rhythm changes, comments, noise artifacts) are excluded.

| AAMI Class | Description | MIT-BIH Symbols Included | Total Beats |
| :---: | :--- | :--- | :---: |
| **N** | Normal / Bundle Branch Block | `N`, `L`, `R`, `e`, `j`, `B` | 90,582 |
| **SVEB** | Supraventricular Ectopic Beat | `A`, `a`, `J`, `S` | 2,781 |
| **VEB** | Ventricular Ectopic Beat | `V`, `E`, `r` | 7,235 |
| **F** | Fusion of Ventricular & Normal | `F` | 802 |
| **Q** | Paced / Unclassifiable Beat | `/`, `f`, `Q` | 8,038 |
| **Total** | | | **109,438** |

### 6. Inter-Patient Split Paradigm
To avoid intra-patient data leakage (optimistic bias where beats from the same patient appear in both training and test sets), data is partitioned **strictly by recording IDs**:
- **Train Set (70%)**: 33 patients ($75,795$ beats)
- **Validation Set (15%)**: 7 patients ($15,468$ beats)
- **Test Set (15%)**: 8 patients ($18,175$ beats)
- Zero patient overlap is enforced: $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset$, $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$, $\mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$.

---

## Federated Non-IID Partitioning

Federated simulation is implemented in [`src/partition.py`](src/partition.py). To realistically reflect heterogeneity across clinical edge nodes, training sample indices are partitioned among $N = 8$ clients using a class-conditional **Dirichlet distribution**:

$$\mathbf{p}_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_N) \quad \text{for each class } c \in \{N, \text{SVEB}, \text{VEB}, F, Q\}$$

Where $\mathbf{p}_c = [p_{c, 0}, p_{c, 1}, \dots, p_{c, N-1}]$ specifies the fraction of class $c$ samples allocated to client $i$.

- **$\alpha = 0.1$ (Extreme Non-IID)**: High statistical heterogeneity; individual clients specialize heavily in one or two classes while missing others.
- **$\alpha = 0.5$ (Moderate Non-IID)**: Substantial class distribution skew with partial coverage across all classes.
- **$\alpha = 1.0$ (Uniform Dirichlet)**: Smoother class distribution approaching IID characteristics.

Partition indices are serialized to:
- `data/partitions/clients_8_alpha_{alpha}/client_{id}.npy`
- `data/alpha_{alpha}/client_{id}.npy`

---

## Model Architectures: FastKAN vs. Matched MLP

### 1. Fast Kolmogorov-Arnold Network (FastKAN)
Implemented in [`src/models/fastkan_model.py`](src/models/fastkan_model.py), the FastKAN classifier maps 32-sample ECG vectors to 5 AAMI output logits using learnable univariate edge functions:
$$\phi_{j, i}(x_i) = w^b_{j, i} \cdot \text{SiLU}(x_i) + \sum_{m=0}^{G + k - 1} c_{j, i, m} B_m(x_i)$$
- **B-Spline Parameterization**: Evaluates degree-$k$ basis functions across $G$ uniform grid intervals via Cox-de Boor recursion.
- **Dynamic $(G, k)$ Edge Reconfiguration**: Supports dynamically initialized and adaptive $(G, k)$ pairs with $G \in \{3, 5, 7, 10\}$ and $k \in \{2, 3, 4\}$.
- **Active Coefficient Accounting**: `get_parameter_count()` reports exact active parameters ($5,349$ for default $[32, 16, 5]$ with $G=5, k=3$).

### 2. Parameter-Matched MLP Baseline
Implemented in [`src/models/mlp.py`](src/models/mlp.py), the MLP baseline is strictly matched to the parameter budget of FastKAN:
- For FastKAN $[32, 16, 5]$ with $5,349$ parameters, [`build_matched_mlp`](src/models/mlp.py) constructs an equivalent $[32, 57, 55, 5]$ architecture with **$5,351$ parameters** ($\Delta = 2$ parameters) or $[32, 64, 48, 5]$ ($5,477$ parameters), eliminating parameter-scale advantages for a fair empirical comparison.

---

## Flower (flwr) Federated Integration

### 1. NumPyClient with Telemetry
Implemented in [`src/federated/client.py`](src/federated/client.py) (`FlowerECGClient`):
- Local optimization using Adam or SGD with gradient clipping.
- Local validation split evaluation on each client node.
- **Telemetry Extraction**: Calculates and returns:
  - `local_loss`: Mean training loss over epochs.
  - `execution_latency`: Training round wall-clock duration in seconds.
  - `input_variance`: Empirical local data variance $\frac{1}{D}\sum_{d=1}^D \text{Var}(X_{:, d})$.

### 2. FedProx Proximal Regularization
Implemented in [`src/federated/fedprox.py`](src/federated/fedprox.py) (`FedProxCriterion`):
- Penalizes local client model drift under severe non-IID class skew:
  $$\mathcal{L}_{\text{FedProx}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \frac{\mu}{2} \sum_i \|\theta_i - \theta_{t, i}\|_2^2$$
  where $\theta_t$ is the frozen server parameter snapshot at round $t$.

### 3. Global Server Aggregator & Test Evaluator
Implemented in [`src/federated/server.py`](src/federated/server.py):
- Aggregates client updates using weighted FedAvg / FedProx.
- Evaluates the global model on the held-out patient-level test set ($18,175$ beats).
- Calculates communication volume: $\text{Volume} = 2 \times N \times \text{ParamCount} \times 4\text{ bytes}$.

---

## Baseline Experiments & Empirical Results

The full FL simulation across all 9 combinations ($\text{FedAvg-MLP}$, $\text{FedAvg-KAN}$, $\text{FedProx-KAN}$ across $\alpha \in \{0.1, 0.5, 1.0\}$) is executed via [`experiments/run_baseline.py`](experiments/run_baseline.py).

### Benchmark Comparison (MIT-BIH Held-Out Patient-Level Test Set)

| Algorithm | Dirichlet $\alpha$ | Architecture | Active Parameters | Final Test Accuracy | Final Macro F1 | Comm (MB) | Avg Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FedAvg-MLP** | $0.1$ | MLP | $5,477$ | 75.66% | 0.2627 | 1.00 MB | 0.288s |
| **FedAvg-KAN** | $0.1$ | FastKAN | $5,349$ | **86.83%** | **0.3631** | **0.98 MB** | 0.472s |
| **FedProx-KAN** | $0.1$ | FastKAN | $5,349$ | **86.62%** | **0.3618** | **0.98 MB** | 0.588s |
| **FedAvg-MLP** | $0.5$ | MLP | $5,477$ | 75.84% | 0.3288 | 1.00 MB | 0.281s |
| **FedAvg-KAN** | $0.5$ | FastKAN | $5,349$ | **83.69%** | **0.4316** | **0.98 MB** | 0.480s |
| **FedProx-KAN** | $0.5$ | FastKAN | $5,349$ | **82.67%** | **0.4212** | **0.98 MB** | 0.563s |
| **FedAvg-MLP** | $1.0$ | MLP | $5,477$ | 74.73% | 0.3262 | 1.00 MB | 0.276s |
| **FedAvg-KAN** | $1.0$ | FastKAN | $5,349$ | **77.20%** | **0.3848** | **0.98 MB** | 0.503s |
| **FedProx-KAN** | $1.0$ | FastKAN | $5,349$ | 74.53% | 0.3309 | 0.98 MB | 0.596s |

> [!NOTE]
> Under equivalent parameter budgets (~5.3k parameters), FastKAN achieves a **+11.17% accuracy gain** and **+0.1004 Macro F1 increase** over the matched MLP baseline under extreme non-IID skew ($\alpha = 0.1$), demonstrating high expressivity on non-linear morphological ECG features.

### Diagnostic Visualizations

All baseline comparison plots are stored in [`results/baselines/figures/`](results/baselines/figures/):

| Figure | Description |
| :--- | :--- |
| `accuracy_vs_rounds_comparison.png` | Global test accuracy progression over training rounds for FedAvg-MLP, FedAvg-KAN, and FedProx-KAN across $\alpha \in \{0.1, 0.5, 1.0\}$. |
| `macro_f1_vs_rounds_comparison.png` | Macro F1 trajectory comparison showing FastKAN's superior recall on minority AAMI classes under extreme heterogeneity. |
| `communication_efficiency_comparison.png` | Cumulative network footprint (MB) versus achieved Macro F1 trade-off analysis. |

---

## Reinforcement Learning Gymnasium Environment

Implemented in [`src/environment/f1_env.py`](src/environment/f1_env.py) and [`src/environment/telemetry.py`](src/environment/telemetry.py), the `FederatedKANEnv` subclasses `gymnasium.Env` to formulate resource-adaptive federated learning as a multi-client Markov Decision Process (MDP) following **Section 3.2.4 & Section 3.2.5** of the thesis proposal.

```mermaid
flowchart LR
    subgraph Centralized Server
        RL["PPO Meta-Controller"]
        Agg["Weighted FedAvg (G_ref, k_ref)"]
        Eval["Held-Out Test Evaluator"]
    end
    subgraph Edge Clients
        C1["Client 1: (G_1, k_1)"]
        C2["Client 2: (G_2, k_2)"]
        CN["Client N: (G_N, k_N)"]
    end
    RL -- "Actions a_t = (G_i, k_i)" --> Edge Clients
    Edge Clients -- "B-spline Resampled Weights" --> Agg
    Agg -- "Global Model θ_t+1" --> Eval
    Eval -- "ΔAcc_global" --> RL
    Edge Clients -- "Telemetry S_i^t (Loss, Var, Latency, bw, κ)" --> RL
```

### 1. State Space & Telemetry Vector
For each client $i \in \{0, \dots, N-1\}$ at communication round $t$, the client and server track a 5-dimensional telemetry state vector:
$$\mathbf{S}_i^t = \left[ L_{\text{local}}^i, \; \text{Var}(x^i), \; \Delta\tau^i, \; \text{bw}^i, \; \kappa^i \right] \in \mathbb{R}^5$$

- **$L_{\text{local}}^i$**: Most recent training cross-entropy loss on the client's local partition.
- **$\text{Var}(x^i)$**: Empirical input feature variance computed across local ECG beats: $\frac{1}{D}\sum_{d=1}^D \text{Var}(X_{:, d})$.
- **$\Delta\tau^i$**: Wall-clock latency of the local training step in seconds.
- **$\text{bw}^i$**: Simulated edge link bandwidth (Mbps) sampled from uniform $U(1.0, 10.0)$ or log-normal distributions.
- **$\kappa^i$**: Curvature proxy derived from the second-difference statistics of the client's loss trajectory:
  $$\kappa_t^i = \left| L_t^i - 2 L_{t-1}^i + L_{t-2}^i \right|$$

**Observation Space**: Normalized into $[0.0, 1.0]$ via [`TelemetryTracker`](src/environment/telemetry.py):
$$\mathcal{S} = \text{Box}\left(\text{low}=0.0, \; \text{high}=1.0, \; \text{shape}=(N, 5), \; \text{dtype}=\text{np.float32}\right)$$

### 2. Action Space & Dynamic Spline Interpolation
Each client $i$ is dynamically assigned an edge resolution action $a_i^t = (G_i^t, k_i^t)$ from 12 discrete configurations:
$$\mathcal{A}_i = \{3, 5, 7, 10\} \times \{2, 3, 4\} \quad (12 \text{ actions})$$
$$\mathcal{A}_{\text{global}} = \text{MultiDiscrete}([12] \times N)$$

- **Continuous B-Spline Weight Transfer**: Because B-splines represent continuous curves over $[-1, 1]$, server weights on the reference grid $(G_{\text{ref}}=5, k_{\text{ref}}=3)$ are dynamically interpolated to $(G_i^t, k_i^t)$ for local training using PyTorch 1D linear interpolation (`F.interpolate`).
- After local training, updated client weights are interpolated back to $(G_{\text{ref}}, k_{\text{ref}})$, enabling seamless weighted FedAvg parameter aggregation without architectural friction.

### 3. Multi-Objective Reward Function (Equation 7)
The environment computes the scalar step reward:
$$R_t = \Delta\text{Acc}_{\text{global}} - \lambda_1 \sum_{i=1}^N \text{Params}(G_i^t, k_i^t) - \lambda_2 \sum_{i=1}^N \Delta\tau_i^t$$
- $\Delta\text{Acc}_{\text{global}} = \text{Acc}_t - \text{Acc}_{t-1}$: Change in global classification accuracy on held-out test data.
- $\lambda_1$: Active parameter penalty coefficient ($\lambda_1 = 1.0 \times 10^{-6}$ in [`configs/ppo.yaml`](configs/ppo.yaml)).
- $\lambda_2$: Latency penalty coefficient ($\lambda_2 = 0.005$ in [`configs/ppo.yaml`](configs/ppo.yaml)).

---

## Verification & Exploratory Data Analysis (EDA)

All intermediate verification plots are generated deterministically and saved in [`results/figures/eda/`](results/figures/eda/):

| Verification Figure | Description |
| :--- | :--- |
| `raw_vs_filtered_ecg.png` | Demonstrates elimination of baseline wander and high-frequency noise using the 3rd-order Butterworth bandpass filter. |
| `r_peak_detection.png` | Displays detected R-peaks and ground-truth AAMI annotations on the filtered ECG trace. |
| `aami_class_examples.png` | Multi-panel comparison of representative beats for all 5 AAMI classes, comparing 360-sample windows against $n=32$ linearly resampled features. |
| `patient_split_distribution.png` | Log-scale class distribution across Train, Validation, and Test patient splits. |
| `dirichlet_alphas_comparison.png` | Side-by-side heatmaps illustrating client-class proportions across $\alpha \in \{0.1, 0.5, 1.0\}$. |
| `dirichlet_heatmap_alpha_*.png` | Detailed client label distribution heatmaps for each individual $\alpha$. |
| `client_sample_counts_alpha_*.png`| Total number of samples allocated to each federated client per alpha. |

---

## Installation & Setup

### Prerequisites
- Linux / WSL2 (Ubuntu 22.04+)
- Python 3.12+

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/rajukarki467/thesis_project.git
cd thesis_project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## CLI Usage

All pipelines and baseline experiments are executable via command-line interfaces that parse parameters from YAML configs and support deterministic seeding.

### 1. Execute Preprocessing Pipeline
Processes all 48 MIT-BIH recordings, performs filtering, extracts beats, applies normalization and $n=32$ resampling, enforces patient splits, and saves verification figures:
```bash
.venv/bin/python src/preprocessing.py --config configs/dataset.yaml --seed 42
```
*Optional CLI overrides:*
```bash
.venv/bin/python src/preprocessing.py --config configs/dataset.yaml --resample_method torch --seed 123
```

### 2. Execute Federated Partitioning Pipeline
Partitions the training split among simulated federated clients for $\alpha \in [0.1, 0.5, 1.0]$:
```bash
.venv/bin/python src/partition.py --config configs/dataset.yaml --alphas 0.1 0.5 1.0 --num_clients 8 --seed 42
```

### 3. Run Federated Baseline Experiments
Executes complete FL training loops across FedAvg-MLP, FedAvg-KAN, and FedProx-KAN for $\alpha \in \{0.1, 0.5, 1.0\}$, evaluating against the held-out patient test split:
```bash
.venv/bin/python experiments/run_baseline.py --config configs/paper_baseline.yaml --num_rounds 3 --local_epochs 1
```
*Optional CLI overrides:*
```bash
# Run 10 rounds with 3 local epochs and custom FedProx mu:
.venv/bin/python experiments/run_baseline.py --config configs/paper_baseline.yaml --num_rounds 10 --local_epochs 3 --mu 0.01 --seed 42
```

### 4. Execute Gymnasium Environment Rollout Sanity Check
Executes a multi-client 2-step rollout using random action sampling on the `FederatedKANEnv` MDP, verifying telemetry tracking, dynamic B-spline weight interpolation, and multi-objective reward calculation:
```bash
.venv/bin/python src/environment/f1_env.py --config configs/ppo.yaml --steps 2 --seed 42
```

### 5. Run Environment Unit Tests
Runs the complete test suite verifying telemetry normalization, curvature calculations, bandwidth simulation, action bijection, and Gymnasium MDP lifecycle:
```bash
.venv/bin/pytest tests/test_environment.py -v
```

---

## Dataset Summary & Statistics

Following pipeline execution with global seed `42`:

```
Patient-Level Split: 33 Train Records, 7 Validation Records, 8 Test Records
Total Valid Beats Extracted: 109,438 (Resampled to n=32 features)

Split Breakdown:
├── Train Set: 75,795 beats
│   ├── N:    63,349 (83.58%)
│   ├── SVEB:  2,651  (3.50%)
│   ├── VEB:   5,226  (6.89%)
│   ├── F:       417  (0.55%)
│   └── Q:     4,152  (5.48%)
├── Validation Set: 15,468 beats
│   ├── N:    14,653 (94.73%)
│   ├── SVEB:     76  (0.49%)
│   ├── VEB:     733  (4.74%)
│   ├── F:         6  (0.04%)
│   └── Q:         0  (0.00%)
└── Test Set: 18,175 beats
    ├── N:    12,580 (69.22%)
    ├── SVEB:     54  (0.30%)
    ├── VEB:   1,276  (7.02%)
    ├── F:       379  (2.09%)
    └── Q:     3,886 (21.38%)
```

---

## Project Roadmap & Next Steps

### Completed Milestones
- [x] **Phase 1: ECG Signal Preprocessing & AAMI Standardization**
  - Butterworth zero-phase bandpass filter ($0.5-40\text{ Hz}$, order 3).
  - R-peak window segmentation ($360$ samples, $1.0\text{ s}$) and $z$-score normalization.
  - Linear downsampling to $n=32$ features.
  - Patient-level split ($33$ train, $7$ val, $8$ test recordings; zero patient leakage).
- [x] **Phase 2: Federated Non-IID Dirichlet Simulation**
  - Class-conditional Dirichlet partitioning for $N=8$ clients across $\alpha \in \{0.1, 0.5, 1.0\}$.
  - Client metadata serialization and label skew visualization.
- [x] **Phase 3: Deep Learning Architectures**
  - Modular FastKAN classifier with Cox-de Boor B-splines and dynamic $(G, k)$ edge reconfigurability ($G \in \{3, 5, 7, 10\}, k \in \{2, 3, 4\}$).
  - Parameter-matched MLP baseline ($5,351$ vs $5,349$ parameters) for rigorous fair evaluation.
- [x] **Phase 4: Flower (`flwr`) Federated Integration**
  - `FlowerECGClient(NumPyClient)` with client telemetry (`execution_latency`, `local_loss`, `input_variance`) and local validation.
  - FedProx proximal regularization term ($\frac{\mu}{2}\|\theta - \theta_t\|^2$).
  - Centralized test evaluation tracking Accuracy, Macro F1, and communication volume.
- [x] **Phase 5: Baseline Experiments & Benchmarks**
  - Multi-algorithm FL evaluation across all 9 combinations ($\text{FedAvg-MLP}$, $\text{FedAvg-KAN}$, $\text{FedProx-KAN} \times \alpha \in \{0.1, 0.5, 1.0\}$).
  - Serialization of per-round metrics, summary tables, and diagnostic plots in `results/baselines/`.
- [x] **Phase 6: Resource-Adaptive Gymnasium Environment & Telemetry**
  - Implemented 5-dim client-server telemetry tracker (`src/environment/telemetry.py`) with curvature $\kappa^i$, empirical variance, and dynamic bandwidth simulator.
  - Formulated Gymnasium MDP (`src/environment/f1_env.py`) with $\text{Box}(0, 1, (N, 5))$ observation space and $\text{MultiDiscrete}([12] \times N)$ action space.
  - Multi-objective step reward according to Equation (7) ($R_t = \Delta\text{Acc}_{\text{global}} - \lambda_1 \sum \text{Params} - \lambda_2 \sum \Delta\tau$).
  - Continuous B-spline parameter interpolation across heterogeneous edge client configurations.
  - 12 comprehensive unit tests in `tests/test_environment.py`.

### Active & Upcoming Milestones
- [ ] **Phase 7: PPO Reinforcement Learning Policy Optimization**
  - Implement and train PPO actor-critic network (`src/controllers/ppo_controller.py`) interacting with `FederatedKANEnv` to learn Pareto-optimal resource allocation.
- [ ] **Phase 8: Long-Horizon Convergence & Edge Hardware Emulation**
  - Asymptotic convergence analysis (10–50 FL rounds).
  - Profiling memory footprint, peak RAM, and FLOPs on resource-constrained embedded profiles (Raspberry Pi / Jetson emulation).

---

## References

1. **MIT-BIH Database**: Goldberger, A. L., et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation*, 101(23), e215-e220 (2000).
2. **AAMI Standard**: Association for the Advancement of Medical Instrumentation. "Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms." *ANSI/AAMI EC57:1998* (Arlington, VA, 1998).
3. **Inter-Patient Paradigm**: de Chazal, P., O'Dwyer, M., & Reilly, R. B. "A patient-adapting heartbeat classification system using ECG morphology and rhythm features." *IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206 (2004).
4. **Federated Dirichlet Non-IID**: Hsu, T. M. H., Venkataraman, H., & Qi, Z. "Measuring the Effects of Non-Identical Distributions in Federated Learning." *arXiv preprint arXiv:1909.06320* (2019).
