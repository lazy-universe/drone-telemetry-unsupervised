#!/usr/bin/env python3
"""
Preset Library & Standalone Runner: Retraining Experiments
Self-contained within the drone-telemetry-unsupervised repository.

Features:
- Complete isolation across experiments, seeds, and cross-validation folds.
- Automatic ephemeral cache directory clearing before every experiment run.
- Garbage collection and CUDA memory cache clearing between runs to prevent leaks.
- Implements:
  - EXP-8: Multi-Seed Stability Test (5 Seeds) -> Per-Seed Results Table + Mean ± Std
  - EXP-9: Feature Bloat Validation (Multi-Seed Paired Comparison & Statistical Tests)
  - EXP-10: Flight-Wise Cross-Validation (5-Fold CV) -> Per-Fold Results Table + Mean ± Std
  - EXP-13: Device-Wise Split Benchmark (Cross-Hardware Generalization on Ultimate 9)
  - EXP-12: Additional Feature Set Ablations (Exploratory / Optional configs)
  - Automatic Checkpoint Saving (.pth + scalers) and ZIP Bundling
"""

import os
import sys
import gc
import time
import zipfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Path configuration relative to repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implement.utils.helper import get_output_dir
from implement.workflows.per_class_evaluation import get_labeled_datasets, run_per_class_evaluation
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data
from implement.utils.deep_learning.autoencoders import (
    DenseAutoencoder, GRUAutoencoder, TCNAutoencoder, CNNGRUAutoencoder
)
from implement.utils.helper.physics_rules import compute_physics_rule_flags
from presets.run_unsupervised_pipeline import (
    create_sequences, compute_dl_anomaly_scores, run_experiment, clear_ephemeral_cache
)

RETRAINING_OUTPUT_DIR = get_output_dir() / "retraining_experiments"
WINDOW_LEN = 20

ULTIMATE_9 = [
    "motion_smoothness", "heading_speed_consistency", "ground_speed",
    "height", "vertical_speed", "acceleration", "turn_rate",
    "prediction_error", "position_residual_std"
]

ULTIMATE_9_PLUS_YAW = ULTIMATE_9 + ["yaw_acceleration"]

BASELINE_7 = [
    "motion_smoothness", "heading_speed_consistency", "ground_speed",
    "height", "vertical_speed", "acceleration", "turn_rate"
]


def clean_memory():
    """Forces garbage collection and flushes CUDA cache to guarantee total isolation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_tcn_model(X_train_seq, X_val_seq, input_dim=9, seq_len=20, epochs=15, batch_size=64, lr=0.001, patience=7, device="cpu", seed=42):
    """Trains a TCN Autoencoder with early stopping and dynamic threshold calculation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TCNAutoencoder(seq_len=seq_len, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for (bx,) in train_loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            recon_x = model(bx)
            loss = F.mse_loss(recon_x, bx)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(bx)

        val_loss_avg = 0.0
        model.eval()
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                recon_x = model(bx)
                loss = F.mse_loss(recon_x, bx)
                val_loss_avg += loss.item() * len(bx)
        val_loss_avg /= len(val_tensor)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    val_scores = compute_dl_anomaly_scores(model, X_val_seq, device=device)
    return model, val_scores, best_val_loss


def evaluate_predictions(y_pred, test_classes, all_classes):
    """Calculates aggregate and per-class metrics."""
    normal_mask = (test_classes == "Normal DJI")
    spoofed_excl_geo = (test_classes != "Normal DJI") & (test_classes != "Sim Geometry")
    spoofed_incl_geo = (test_classes != "Normal DJI")
    eval_excl_mask = (test_classes != "Sim Geometry")

    normal_acc = np.mean(y_pred[normal_mask] == 0) * 100.0
    far = 100.0 - normal_acc
    spoofed_acc_excl = np.mean(y_pred[spoofed_excl_geo] == 1) * 100.0
    spoofed_acc_incl = np.mean(y_pred[spoofed_incl_geo] == 1) * 100.0

    overall_excl = (
        (np.sum(y_pred[normal_mask] == 0) + np.sum(y_pred[spoofed_excl_geo] == 1))
        / np.sum(eval_excl_mask) * 100.0
    )
    overall_incl = (
        (np.sum(y_pred[normal_mask] == 0) + np.sum(y_pred[spoofed_incl_geo] == 1))
        / len(test_classes) * 100.0
    )

    metrics = {
        "Normal Acc (%)": round(normal_acc, 2),
        "FAR (%)": round(far, 2),
        "Spoofed Acc (Excl Geo) (%)": round(spoofed_acc_excl, 2),
        "Spoofed Acc (Incl Geo) (%)": round(spoofed_acc_incl, 2),
        "Overall Acc (Excl Geo) (%)": round(overall_excl, 2),
        "Overall Acc (Incl Geo) (%)": round(overall_incl, 2),
    }

    for cls in all_classes:
        mask = (test_classes == cls)
        if cls == "Normal DJI":
            metrics[cls] = round(np.mean(y_pred[mask] == 0) * 100.0, 2)
        else:
            metrics[cls] = round(np.mean(y_pred[mask] == 1) * 100.0, 2)

    return metrics


# ==============================================================================
# EXP-8: Multi-Seed Stability Test
# ==============================================================================
def run_exp8_multi_seed(seeds=(17, 42, 137, 521, 2026), epochs=15, patience=7, k_thresh=3.0, save_dir=None, clean_cache=True):
    """
    EXP-8: Multi-Seed Stability Test (5 Seeds) on Ultimate 9.
    Produces a detailed per-seed table and aggregate summary statistics.
    Saves each trained model checkpoint (.pth).
    """
    if clean_cache:
        clear_ephemeral_cache()
    clean_memory()

    if save_dir is None:
        save_dir = RETRAINING_OUTPUT_DIR / "exp8_multi_seed"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    models_dir = save_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("=== EXP-8: MULTI-SEED STABILITY TEST (5 SEEDS) — ULTIMATE 9 ===")
    print(f"=== Seeds: {seeds} | Max Epochs: {epochs} | Patience: {patience} | k: {k_thresh} ===")
    print(f"=== Save Directory: {save_dir} ===")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dji_df, esp32_df = get_labeled_datasets(features=ULTIMATE_9)
    unique_flights = np.sort(dji_df["flight_id"].unique())
    all_classes = ["Normal DJI", "Real ESP32", "Sim Baseline", "Sim Easy", "Sim Geometry", "Sim Hard", "Sim Medium"]

    per_seed_results = []

    for seed in seeds:
        clean_memory()
        print(f"\n>>> Retraining Seed {seed} on Ultimate 9...")
        train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=seed)
        val_fl, test_fl = train_test_split(temp, test_size=0.5, random_state=seed)

        dji_train = dji_df[dji_df["flight_id"].isin(train_fl)].copy()
        dji_val   = dji_df[dji_df["flight_id"].isin(val_fl)].copy()
        dji_test  = dji_df[dji_df["flight_id"].isin(test_fl)].copy()

        test_df = pd.concat([dji_test, esp32_df], ignore_index=True)

        X_train_raw = dji_train[ULTIMATE_9]
        X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
        X_val_scaled   = scaler.transform(imputer.transform(dji_val[ULTIMATE_9]))
        X_test_scaled  = scaler.transform(imputer.transform(test_df[ULTIMATE_9]))

        X_train_seq = create_sequences(X_train_scaled, WINDOW_LEN)
        X_val_seq   = create_sequences(X_val_scaled, WINDOW_LEN)
        X_test_seq  = create_sequences(X_test_scaled, WINDOW_LEN)

        test_classes = test_df["attack_class"].values[WINDOW_LEN - 1:]

        model, val_scores, best_val_loss = train_tcn_model(
            X_train_seq, X_val_seq, input_dim=len(ULTIMATE_9), seq_len=WINDOW_LEN,
            epochs=epochs, patience=patience, device=device, seed=seed
        )

        threshold = float(np.mean(val_scores) + k_thresh * np.std(val_scores))
        test_scores = compute_dl_anomaly_scores(model, X_test_seq, device=device)
        y_pred = (test_scores > threshold).astype(int)

        # Save model checkpoint
        ckpt_path = models_dir / f"tcn_seed_{seed}.pth"
        torch.save({
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "seed": seed,
            "best_val_loss": best_val_loss,
            "features": ULTIMATE_9,
            "window_len": WINDOW_LEN
        }, ckpt_path)

        metrics = evaluate_predictions(y_pred, test_classes, all_classes)
        row = {
            "Seed": seed,
            "Threshold": round(threshold, 6),
            "Best Val Loss": round(best_val_loss, 6),
            **metrics
        }
        per_seed_results.append(row)
        print(f"    Seed {seed} -> Normal Acc: {metrics['Normal Acc (%)']}%, Spoofed Acc (Excl Geo): {metrics['Spoofed Acc (Excl Geo) (%)']}%, Spoofed Acc (Incl Geo): {metrics['Spoofed Acc (Incl Geo) (%)']}% (Saved to {ckpt_path.name})")

    df_per_seed = pd.DataFrame(per_seed_results)

    # Compute Summary Statistics
    numeric_cols = [c for c in df_per_seed.columns if c not in ["Seed"]]
    mean_row = {"Seed": "Mean"}
    std_row = {"Seed": "Std"}
    min_row = {"Seed": "Min"}
    max_row = {"Seed": "Max"}

    for col in numeric_cols:
        mean_row[col] = round(df_per_seed[col].mean(), 2)
        std_row[col] = round(df_per_seed[col].std(), 2)
        min_row[col] = round(df_per_seed[col].min(), 2)
        max_row[col] = round(df_per_seed[col].max(), 2)

    df_summary = pd.DataFrame([mean_row, std_row, min_row, max_row])
    df_full = pd.concat([df_per_seed, df_summary], ignore_index=True)

    print("\n" + "=" * 80)
    print("📊 EXP-8 PER-SEED RESULTS TABLE:")
    print("=" * 80)
    print(df_full.to_string(index=False))

    df_per_seed.to_csv(save_dir / "exp8_per_seed_results.csv", index=False)
    df_full.to_csv(save_dir / "exp8_multi_seed_stability.csv", index=False)

    # Plot Multi-Seed Stability Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    cols_to_plot = ["Normal Acc (%)", "Spoofed Acc (Excl Geo) (%)", "Spoofed Acc (Incl Geo) (%)", "Overall Acc (Excl Geo) (%)"]
    data_to_plot = [df_per_seed[c] for c in cols_to_plot]
    
    bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=cols_to_plot)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, col in enumerate(cols_to_plot):
        y = df_per_seed[col]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'r.', alpha=0.9, markersize=10)

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_title(f"EXP-8: Multi-Seed Stability across 5 Seeds (Ultimate 9)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim([40, 105])
    plt.tight_layout()
    plt.savefig(save_dir / "exp8_multi_seed_stability.png", dpi=300)
    plt.close()

    clean_memory()
    return df_per_seed, df_full


# ==============================================================================
# EXP-9: Feature Bloat Validation (Paired Multi-Seed)
# ==============================================================================
def run_exp9_feature_bloat(seeds=(17, 42, 137, 521, 2026), epochs=15, patience=7, k_thresh=3.0, save_dir=None, clean_cache=True):
    """
    EXP-9: Feature Bloat Validation (Multi-Seed Paired Comparison & Statistical Tests).
    Compares Ultimate 9 vs Ultimate 9 + Yaw Acceleration across identical seeds.
    Saves both model checkpoints for each seed.
    """
    if clean_cache:
        clear_ephemeral_cache()
    clean_memory()

    if save_dir is None:
        save_dir = RETRAINING_OUTPUT_DIR / "exp9_feature_bloat"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    models_dir = save_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("=== EXP-9: FEATURE BLOAT VALIDATION (ULTIMATE 9 vs +YAW ACCELERATION) ===")
    print(f"=== Seeds: {seeds} | Paired Comparison & Statistical Hypothesis Testing ===")
    print(f"=== Save Directory: {save_dir} ===")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dji_df, esp32_df = get_labeled_datasets(features=ULTIMATE_9_PLUS_YAW)
    unique_flights = np.sort(dji_df["flight_id"].unique())
    all_classes = ["Normal DJI", "Real ESP32", "Sim Baseline", "Sim Easy", "Sim Geometry", "Sim Hard", "Sim Medium"]

    paired_rows = []

    for seed in seeds:
        clean_memory()
        print(f"\n>>> Retraining Seed {seed} on (Ultimate 9) vs (Ultimate 9 + Yaw)...")
        train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=seed)
        val_fl, test_fl = train_test_split(temp, test_size=0.5, random_state=seed)

        dji_train = dji_df[dji_df["flight_id"].isin(train_fl)].copy()
        dji_val   = dji_df[dji_df["flight_id"].isin(val_fl)].copy()
        dji_test  = dji_df[dji_df["flight_id"].isin(test_fl)].copy()
        test_df = pd.concat([dji_test, esp32_df], ignore_index=True)
        test_classes = test_df["attack_class"].values[WINDOW_LEN - 1:]

        # 1. Train Ultimate 9
        X_tr_raw_9 = dji_train[ULTIMATE_9]
        X_tr_sc_9, _, sc_9, imp_9 = impute_and_scale_data(X_tr_raw_9, X_tr_raw_9)
        X_val_sc_9 = sc_9.transform(imp_9.transform(dji_val[ULTIMATE_9]))
        X_te_sc_9  = sc_9.transform(imp_9.transform(test_df[ULTIMATE_9]))

        model_9, val_sc_9, loss_9 = train_tcn_model(
            create_sequences(X_tr_sc_9, WINDOW_LEN), create_sequences(X_val_sc_9, WINDOW_LEN),
            input_dim=9, seq_len=WINDOW_LEN, epochs=epochs, patience=patience, device=device, seed=seed
        )
        thresh_9 = float(np.mean(val_sc_9) + k_thresh * np.std(val_sc_9))
        pred_9 = (compute_dl_anomaly_scores(model_9, create_sequences(X_te_sc_9, WINDOW_LEN), device=device) > thresh_9).astype(int)
        m_9 = evaluate_predictions(pred_9, test_classes, all_classes)

        torch.save({"state_dict": model_9.state_dict(), "threshold": thresh_9, "seed": seed, "features": ULTIMATE_9}, models_dir / f"tcn_seed_{seed}.pth")

        # 2. Train Ultimate 9 + Yaw
        X_tr_raw_yaw = dji_train[ULTIMATE_9_PLUS_YAW]
        X_tr_sc_yaw, _, sc_yaw, imp_yaw = impute_and_scale_data(X_tr_raw_yaw, X_tr_raw_yaw)
        X_val_sc_yaw = sc_yaw.transform(imp_yaw.transform(dji_val[ULTIMATE_9_PLUS_YAW]))
        X_te_sc_yaw  = sc_yaw.transform(imp_yaw.transform(test_df[ULTIMATE_9_PLUS_YAW]))

        model_yaw, val_sc_yaw, loss_yaw = train_tcn_model(
            create_sequences(X_tr_sc_yaw, WINDOW_LEN), create_sequences(X_val_sc_yaw, WINDOW_LEN),
            input_dim=10, seq_len=WINDOW_LEN, epochs=epochs, patience=patience, device=device, seed=seed
        )
        thresh_yaw = float(np.mean(val_sc_yaw) + k_thresh * np.std(val_sc_yaw))
        pred_yaw = (compute_dl_anomaly_scores(model_yaw, create_sequences(X_te_sc_yaw, WINDOW_LEN), device=device) > thresh_yaw).astype(int)
        m_yaw = evaluate_predictions(pred_yaw, test_classes, all_classes)

        torch.save({"state_dict": model_yaw.state_dict(), "threshold": thresh_yaw, "seed": seed, "features": ULTIMATE_9_PLUS_YAW}, models_dir / f"tcn_yaw_seed_{seed}.pth")

        paired_rows.append({
            "Seed": seed,
            "Ultimate 9 Normal Acc (%)": m_9["Normal Acc (%)"],
            "+Yaw Normal Acc (%)": m_yaw["Normal Acc (%)"],
            "Normal Delta (%)": round(m_yaw["Normal Acc (%)"] - m_9["Normal Acc (%)"], 2),
            "Ultimate 9 Spoofed Acc (Excl Geo) (%)": m_9["Spoofed Acc (Excl Geo) (%)"],
            "+Yaw Spoofed Acc (Excl Geo) (%)": m_yaw["Spoofed Acc (Excl Geo) (%)"],
            "Spoofed Excl Delta (%)": round(m_yaw["Spoofed Acc (Excl Geo) (%)"] - m_9["Spoofed Acc (Excl Geo) (%)"], 2),
            "Ultimate 9 Spoofed Acc (Incl Geo) (%)": m_9["Spoofed Acc (Incl Geo) (%)"],
            "+Yaw Spoofed Acc (Incl Geo) (%)": m_yaw["Spoofed Acc (Incl Geo) (%)"],
            "Spoofed Incl Delta (%)": round(m_yaw["Spoofed Acc (Incl Geo) (%)"] - m_9["Spoofed Acc (Incl Geo) (%)"], 2),
            "Ultimate 9 Sim Geometry (%)": m_9["Sim Geometry"],
            "+Yaw Sim Geometry (%)": m_yaw["Sim Geometry"],
        })

    df_paired = pd.DataFrame(paired_rows)

    u9_spoof = df_paired["Ultimate 9 Spoofed Acc (Excl Geo) (%)"].values
    yaw_spoof = df_paired["+Yaw Spoofed Acc (Excl Geo) (%)"].values
    t_stat, p_val_t = stats.ttest_rel(yaw_spoof, u9_spoof)
    try:
        w_stat, p_val_w = stats.wilcoxon(yaw_spoof, u9_spoof)
    except Exception:
        p_val_w = np.nan

    print("\n" + "=" * 80)
    print("📊 EXP-9 PAIRED MULTI-SEED BLOAT VALIDATION TABLE:")
    print("=" * 80)
    print(df_paired.to_string(index=False))
    print("\n--- Statistical Hypothesis Testing (Yaw Degradation Effect) ---")
    print(f"Mean Spoofed Delta: {np.mean(df_paired['Spoofed Excl Delta (%)']):.2f}% ± {np.std(df_paired['Spoofed Excl Delta (%)']):.2f}%")
    print(f"Paired t-test: t = {t_stat:.4f}, p-value = {p_val_t:.6f}")
    print(f"Wilcoxon signed-rank: p-value = {p_val_w:.6f}")

    df_paired.to_csv(save_dir / "exp9_feature_bloat_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, row in df_paired.iterrows():
        ax.plot([0, 1], [row["Ultimate 9 Spoofed Acc (Excl Geo) (%)"], row["+Yaw Spoofed Acc (Excl Geo) (%)"]],
                'o-', lw=2, label=f"Seed {row['Seed']}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ultimate 9\n(Clean Residual Manifold)", "Ultimate 9 + Yaw Acceleration\n(Double-Derivative Bloat)"], fontsize=11, fontweight='bold')
    ax.set_ylabel("Spoofed Detection Accuracy (%) [Excl Geo]", fontsize=11, fontweight='bold')
    ax.set_title(f"EXP-9: Feature Bloat Validation across 5 Seeds\n(Paired t-test p = {p_val_t:.4f})", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_dir / "exp9_feature_bloat_validation.png", dpi=300)
    plt.close()

    clean_memory()
    return df_paired


# ==============================================================================
# EXP-10: Flight-Wise Cross-Validation (5-Fold)
# ==============================================================================
def run_exp10_flight_cv(n_splits=5, random_state=42, epochs=15, patience=7, k_thresh=3.0, save_dir=None, clean_cache=True):
    """
    EXP-10: Flight-Wise Cross-Validation (5-Fold CV) on Ultimate 9.
    Produces a per-fold results table and cross-validation mean ± std.
    Saves each fold model checkpoint (.pth).
    """
    if clean_cache:
        clear_ephemeral_cache()
    clean_memory()

    if save_dir is None:
        save_dir = RETRAINING_OUTPUT_DIR / "exp10_flight_cv"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    models_dir = save_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("=== EXP-10: 5-FOLD FLIGHT-WISE CROSS-VALIDATION — ULTIMATE 9 ===")
    print(f"=== Folds: {n_splits} | Stratified by Flight ID | Max Epochs: {epochs} ===")
    print(f"=== Save Directory: {save_dir} ===")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dji_df, esp32_df = get_labeled_datasets(features=ULTIMATE_9)
    unique_flights = np.sort(dji_df["flight_id"].unique())
    all_classes = ["Normal DJI", "Real ESP32", "Sim Baseline", "Sim Easy", "Sim Geometry", "Sim Hard", "Sim Medium"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_flights), start=1):
        clean_memory()
        test_fl = unique_flights[test_idx]
        train_val_fl = unique_flights[train_idx]
        train_fl, val_fl = train_test_split(train_val_fl, test_size=0.2, random_state=random_state)

        print(f"\n>>> Retraining Fold {fold_idx}/{n_splits} — Held-Out Test Flights: {len(test_fl)}...")

        dji_train = dji_df[dji_df["flight_id"].isin(train_fl)].copy()
        dji_val   = dji_df[dji_df["flight_id"].isin(val_fl)].copy()
        dji_test  = dji_df[dji_df["flight_id"].isin(test_fl)].copy()

        test_df = pd.concat([dji_test, esp32_df], ignore_index=True)

        X_train_raw = dji_train[ULTIMATE_9]
        X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
        X_val_scaled   = scaler.transform(imputer.transform(dji_val[ULTIMATE_9]))
        X_test_scaled  = scaler.transform(imputer.transform(test_df[ULTIMATE_9]))

        X_train_seq = create_sequences(X_train_scaled, WINDOW_LEN)
        X_val_seq   = create_sequences(X_val_scaled, WINDOW_LEN)
        X_test_seq  = create_sequences(X_test_scaled, WINDOW_LEN)

        test_classes = test_df["attack_class"].values[WINDOW_LEN - 1:]

        model, val_scores, best_val_loss = train_tcn_model(
            X_train_seq, X_val_seq, input_dim=len(ULTIMATE_9), seq_len=WINDOW_LEN,
            epochs=epochs, patience=patience, device=device, seed=random_state + fold_idx
        )

        threshold = float(np.mean(val_scores) + k_thresh * np.std(val_scores))
        test_scores = compute_dl_anomaly_scores(model, X_test_seq, device=device)
        y_pred = (test_scores > threshold).astype(int)

        # Save fold checkpoint
        torch.save({
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "fold": fold_idx,
            "held_out_flights": list(test_fl),
            "features": ULTIMATE_9
        }, models_dir / f"tcn_fold_{fold_idx}.pth")

        metrics = evaluate_predictions(y_pred, test_classes, all_classes)
        row = {
            "Fold": f"Fold {fold_idx}",
            "Test Flights Count": len(test_fl),
            "Threshold": round(threshold, 6),
            "Best Val Loss": round(best_val_loss, 6),
            **metrics
        }
        fold_results.append(row)
        print(f"    Fold {fold_idx} -> Normal Acc: {metrics['Normal Acc (%)']}%, Spoofed Acc (Excl Geo): {metrics['Spoofed Acc (Excl Geo) (%)']}%")

    df_folds = pd.DataFrame(fold_results)

    numeric_cols = [c for c in df_folds.columns if c not in ["Fold", "Test Flights Count"]]
    mean_row = {"Fold": "CV Mean", "Test Flights Count": "-"}
    std_row = {"Fold": "CV Std", "Test Flights Count": "-"}

    for col in numeric_cols:
        mean_row[col] = round(df_folds[col].mean(), 2)
        std_row[col] = round(df_folds[col].std(), 2)

    df_full_cv = pd.concat([df_folds, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print("\n" + "=" * 80)
    print("📊 EXP-10 5-FOLD CROSS-VALIDATION RESULTS TABLE:")
    print("=" * 80)
    print(df_full_cv.to_string(index=False))

    df_folds.to_csv(save_dir / "exp10_per_fold_results.csv", index=False)
    df_full_cv.to_csv(save_dir / "exp10_flight_cv_results.csv", index=False)

    clean_memory()
    return df_folds, df_full_cv


# ==============================================================================
# EXP-13: Device-Wise Split Benchmark (Cross-Hardware Generalization)
# ==============================================================================
def run_exp13_device_split(save_dir=None, clean_cache=True):
    """
    EXP-13: Device-Wise Split Benchmark on Ultimate 9 (Cross-Hardware Generalization).
    Trains 12 models on 7 drone models and tests on 2 unseen drone models.
    """
    if clean_cache:
        clear_ephemeral_cache()
    clean_memory()

    if save_dir is None:
        save_dir = RETRAINING_OUTPUT_DIR / "exp13_device_split"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("=== EXP-13: DEVICE-WISE SPLIT BENCHMARK (CROSS-HARDWARE GENERALIZATION) ===")
    print(f"=== Save Directory: {save_dir} ===")
    print("=" * 80)

    df_u9 = run_experiment(
        exp_name="all_models_device_split_ultimate9",
        feature_list=ULTIMATE_9,
        model_family="all",
        epochs=15,
        batch_size=64,
        patience=7,
        window_len=20,
        fresh_cache=True,
        use_cache=False,
        split_mode="device",
        exclude_sim_geometry=True
    )

    clean_memory()

    df_b7 = run_experiment(
        exp_name="all_models_device_split_baseline7",
        feature_list=BASELINE_7,
        model_family="all",
        epochs=15,
        batch_size=64,
        patience=7,
        window_len=20,
        fresh_cache=True,
        use_cache=False,
        split_mode="device",
        exclude_sim_geometry=True
    )

    clean_memory()
    return df_u9, df_b7


# ==============================================================================
# EXP-12: Additional Feature Set Ablations (Exploratory)
# ==============================================================================
def run_exp12_additional_feature_ablations(save_dir=None, clean_cache=True):
    """
    EXP-12: Additional Feature Set Ablations (Exploratory / Optional configs).
    Fills all gaps in the ablation story.
    """
    if clean_cache:
        clear_ephemeral_cache()
    clean_memory()

    if save_dir is None:
        save_dir = RETRAINING_OUTPUT_DIR / "exp12_additional_ablations"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("=== EXP-12: ADDITIONAL FEATURE SET ABLATIONS (EXPLORATORY) ===")
    print(f"=== Save Directory: {save_dir} ===")
    print("=" * 80)

    EXTRA_CONFIGS = {
        "baseline_7_plus_pos_only": BASELINE_7 + ["position_residual_std"],
        "pe_only": ["prediction_error"],
        "baseline_7_plus_entropy_only": BASELINE_7 + ["speed_spectral_entropy"],
        "baseline_7_plus_pe_autocorr": BASELINE_7 + ["prediction_error", "prediction_error_autocorrelation"],
    }

    results = []

    for name, feats in EXTRA_CONFIGS.items():
        clean_memory()
        print(f"\n>>> Retraining Ablation: {name} ({len(feats)} features)...")
        df_res = run_experiment(
            exp_name=f"ablation_{name}",
            feature_list=feats,
            model_family="dl",
            epochs=15,
            batch_size=64,
            patience=7,
            window_len=20,
            fresh_cache=True,
            use_cache=False,
            exclude_sim_geometry=True
        )
        if df_res is not None and not df_res.empty:
            tcn_row = df_res[df_res["Model"] == "TCN Autoencoder"].copy()
            if not tcn_row.empty:
                tcn_dict = tcn_row.iloc[0].to_dict()
                tcn_dict["Config"] = name
                results.append(tcn_dict)

    df_extra = pd.DataFrame(results)
    if not df_extra.empty:
        df_extra.to_csv(save_dir / "exp12_additional_ablations.csv", index=False)
        print("\n" + "=" * 80)
        print("📊 EXP-12 ADDITIONAL ABLATIONS SUMMARY:")
        print("=" * 80)
        print(df_extra.to_string(index=False))

    clean_memory()
    return df_extra


def zip_tier2_results(output_zip_path=None):
    """Zips all Retraining outputs (checkpoints, CSVs, PNGs) into a single downloadable bundle."""
    if output_zip_path is None:
        output_zip_path = RETRAINING_OUTPUT_DIR / "retraining_experiments.zip"
    zip_p = Path(output_zip_path)
    zip_p.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Compressing all Retraining results into {zip_p}...")
    with zipfile.ZipFile(zip_p, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(RETRAINING_OUTPUT_DIR):
            for file in files:
                file_path = Path(root) / file
                if file_path == zip_p:
                    continue
                arcname = file_path.relative_to(RETRAINING_OUTPUT_DIR)
                zipf.write(file_path, arcname=arcname)
    print(f"✅ Created Retraining Results Archive: {zip_p} ({zip_p.stat().st_size / (1024*1024):.2f} MB)")
    return zip_p


def main():
    print("=" * 80)
    print("=== STARTING COMPLETE RETRAINING SUITE ===")
    print("=" * 80)

    # 1. EXP-8: Multi-Seed Stability Test (Saves checkpoints tcn_seed_*.pth)
    df_exp8_seed, df_exp8_sum = run_exp8_multi_seed(seeds=[17, 42, 137, 521, 2026])

    # 2. EXP-9: Feature Bloat Validation (Saves checkpoints tcn_yaw_seed_*.pth)
    df_exp9 = run_exp9_feature_bloat(seeds=[17, 42, 137, 521, 2026])

    # 3. EXP-10: Flight-Wise Cross-Validation (Saves fold checkpoints tcn_fold_*.pth)
    df_exp10_folds, df_exp10_sum = run_exp10_flight_cv(n_splits=5)

    # 4. EXP-13: Device-Wise Split Benchmark
    df_exp13_u9, df_exp13_b7 = run_exp13_device_split()

    # 5. EXP-12: Additional Feature Set Ablations
    df_exp12 = run_exp12_additional_feature_ablations()

    # 6. Bundle all checkpoints into zip
    zip_tier2_results()

    print("\n" + "=" * 80)
    print("=== ALL RETRAINING EXPERIMENTS COMPLETED SUCCESSFULLY! ===")
    print("=" * 80)


if __name__ == "__main__":
    main()
