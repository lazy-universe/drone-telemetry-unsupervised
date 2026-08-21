# 🛸 Unsupervised Drone Telemetry Anomaly Detection Framework

A modular, reproducible, and extensible research framework for **Unsupervised Anomaly Detection** in Broadcast Remote ID (ASTM F3411 / OpenDroneID) telemetry. The framework trains solely on genuine flight telemetry (one-class training) to detect adversarial spoofing attacks, flight trajectory manipulation, and low-cost hardware spoofers (ESP32) without requiring attack labels during training.

---

## 📁 Repository Structure

```
drone-telemetry-unsupervised/
├── README.md
├── dataset/
│   ├── genuine_dji_flights/
│   │   └── consistent_dataset/         # 38 authentic DJI flight CSVs (12 distinct airframes)
│   ├── hardware_spoofer/
│   │   └── esp32_source_telemetry.csv  # ESP32 OpenDroneID broadcast telemetry
│   └── spoofed_flights/                # Evaluated spoofed flights (Classes 1-3 & Replay Baseline)
├── implement/
│   ├── initial_setup.py                # Dataset extraction and verification
│   ├── utils/
│   │   ├── classical_ml/               # Pointwise anomaly detection (GMM, Mahalanobis, KNN, iForest)
│   │   ├── deep_learning/              # Sequence Autoencoders (GRU-AE, TCN-AE, CNN-GRU-AE, GRU-VAE)
│   │   ├── dataset_processing/         # Schema alignment, feature engineering, and cleaning
│   │   └── helper/                     # Dynamic feature definitions, paths, and dataset loaders
│   └── workflows/                      # Workflows for feature progression & per-class evaluation
├── notebooks/
│   ├── 01_unsupervised_exploration.ipynb
│   ├── 02_unsupervised_feature_progression.ipynb
│   └── 03_unsupervised_benchmarks_and_eval.ipynb
└── presets/
    ├── run_gpu_pipeline.py             # Feature progression pipeline (Exp 1 - Exp 7)
    ├── run_device_split_benchmark.py   # Cross-device OOD holdout evaluation
    ├── run_per_class_eval.py           # Sub-attack detection accuracy breakdown
    ├── run_ensemble_eval.py            # Multi-model ensemble detector (OR / AND rules)
    ├── run_feature_importance.py       # SHAP & permutation feature importance
    ├── run_ablation.py                 # Systematic feature ablation studies
    └── compute_feature_bounds.py       # Empirical kinematic value ranges (Min, Max, Mean, Std)
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Initial Setup
```bash
python implement/initial_setup.py
```

---

## 🧪 CLI Presets Suite (`presets/`)

All scripts support flexible command-line arguments:

### 1. Feature Progression Pipeline (`run_gpu_pipeline.py`)
Run unsupervised anomaly detection across predefined feature progression sets or custom feature lists:
```bash
# Run baseline 9-feature model across Classical & Deep Autoencoders
python presets/run_gpu_pipeline.py --preset baseline_9_no_yaw --exp-name baseline_eval --model-family all --epochs 15

# Run with custom dynamic features
python presets/run_gpu_pipeline.py --features height,ground_speed,turn_rate,path_curvature --exp-name custom_run
```

Available Feature Progression Presets:
- `raw_coords` (6 features): Raw spatial coordinates + kinematics
- `raw_no_coords` (4 features): Coordinate-free raw signals
- `raw_engineered` (8 features): Raw signals + direct kinematics
- `complete_set` (10 features): Complete kinematic feature set
- `baseline_10` (10 features): Comprehensive 10-feature baseline
- `baseline_9_no_yaw` (9 features): Optimal 9-feature kinematic representation
- `baseline_9_no_pe` (9 features): Feature set omitting prediction error

### 2. Cross-Device OOD Benchmark (`run_device_split_benchmark.py`)
Evaluates model generalization when tested on entirely unseen drone airframes:
```bash
python presets/run_device_split_benchmark.py
```

### 3. Per-Class Sub-Attack Breakdown (`run_per_class_eval.py`)
Evaluates anomaly detection sensitivity against individual spoofing attack categories (Real ESP32, Sim Easy, Sim Medium, Sim Geometry, Sim Baseline):
```bash
python presets/run_per_class_eval.py
```

### 4. Ensemble Detector Evaluation (`run_ensemble_eval.py`)
Evaluates logical OR / AND ensemble combination rules across pointwise and sequential autoencoders:
```bash
python presets/run_ensemble_eval.py
```

### 5. Feature Importance & Bounds (`run_feature_importance.py` / `compute_feature_bounds.py`)
```bash
# Compute SHAP and permutation importance
python presets/run_feature_importance.py

# Calculate exact empirical kinematic bounds
python presets/compute_feature_bounds.py
```

---

## 📓 Interactive Notebooks (`notebooks/`)

1. **[`01_unsupervised_exploration.ipynb`](notebooks/01_unsupervised_exploration.ipynb)**: Exploratory Data Analysis, KDE density distributions, correlation matrices, and flight dynamics.
2. **[`02_unsupervised_feature_progression.ipynb`](notebooks/02_unsupervised_feature_progression.ipynb)**: Step-by-step feature progression experiments (Exp 1 &rarr; Exp 7) comparing Pointwise and Deep Autoencoders.
3. **[`03_unsupervised_benchmarks_and_eval.ipynb`](notebooks/03_unsupervised_benchmarks_and_eval.ipynb)**: Cross-device evaluation, sub-attack breakdown, multi-model ensemble detection, and feature ablations.
