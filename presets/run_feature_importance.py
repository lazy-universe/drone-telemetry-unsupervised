"""
Permutation Feature Importance Analysis for Unsupervised Anomaly Detection
Generates 2x2 Subplot Grids (One subplot per model: Mahalanobis, GMM, Isolation Forest, TCN Autoencoder) for:
  (1) Pure Model Importance (Normal Anomaly Score Delta on Normal DJI data)
  (2) Overall Spoofed Detection Degradation (|Delta TPR|)
  (3) Per-Class Spoofed Detection Degradation (|Delta TPR| for Real ESP32, Sim Baseline, Sim Easy, Sim Medium, Sim Hard, Sim Geometry)
Across the 20-feature all-inclusive set.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.workflows.per_class_evaluation import get_labeled_datasets
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data
from implement.utils.classical_ml.classical_models import GMMWrapper, MahalanobisWrapper
from implement.utils.classical_ml.classical_train_eval import get_anomaly_scores
from implement.utils.deep_learning.dl_models import get_unsupervised_models
from implement.utils.helper.features import ALL_INCLUSIVE_20_FEATURES
from implement.utils.helper import get_output_dir

OUT_DIR = get_output_dir() / "feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_ATTACK_CLASSES = [
    "Real ESP32",
    "Sim Baseline",
    "Sim Easy",
    "Sim Medium",
    "Sim Hard",
    "Sim Geometry"
]

MODEL_PALETTE = {
    "Mahalanobis": "#2980b9",
    "GMM": "#27ae60",
    "Isolation Forest": "#8e44ad",
    "TCN Autoencoder": "#d35400"
}


def clear_ephemeral_cache():
    dataset_dir = PROJECT_ROOT / "implement" / "dataset"
    if dataset_dir.exists():
        for cache_dir in dataset_dir.glob("ephermal_dataset_*"):
            if cache_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(cache_dir)
                    print(f"Cleared ephemeral cache: {cache_dir.name}")
                except Exception as err:
                    print(f"[Warning] Could not clear cache {cache_dir}: {err}")


def create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    if len(data) < seq_len:
        return np.empty((0, seq_len, data.shape[1]))
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i + seq_len])
    return np.array(seqs)


def compute_dl_anomaly_scores(model: nn.Module, data_seq: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    if len(data_seq) == 0:
        return np.array([])
    tensor = torch.tensor(data_seq, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=128, shuffle=False)
    scores = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            recon_x = model(bx)
            mse = F.mse_loss(recon_x, bx, reduction="none").mean(dim=[1, 2])
            scores.extend(mse.cpu().numpy())
    return np.array(scores)


def plot_2x2_model_grid(
    df_data: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    save_filename: str,
    models: list
):
    """Generates a clean 2x2 subplot grid (one subplot per model) showing sorted signed horizontal bars across 20 features."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for i, model_name in enumerate(models):
        ax = axes[i]
        sub_df = df_data[df_data["Model"] == model_name].copy()
        sub_df = sub_df.sort_values(by=value_col, ascending=True)
        
        vals = sub_df[value_col].values
        # Two-tone coloring: Green for positive drops (detection dropped), Red for negative drops (detection increased)
        bar_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in vals]
        
        ax.barh(sub_df["Feature"], vals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.set_title(f"{model_name}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="y", labelsize=9)

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    save_path = OUT_DIR / save_filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved 2x2 plot: {save_path}")


def run_permutation_importance(
    epochs: int = 5,
    batch_size: int = 64,
    window_len: int = 20,
    k_thresh: float = 3.0,
    fresh_cache: bool = True,
    n_repeats: int = 3,
    random_state: int = 42
):
    if fresh_cache:
        clear_ephemeral_cache()

    print("=" * 85)
    print("=== PERMUTATION FEATURE IMPORTANCE EXPERIMENT (20 ALL-INCLUSIVE FEATURES) ===")
    print("=== Models: Mahalanobis, GMM, Isolation Forest, TCN Autoencoder ===")
    print(f"=== Epochs: {epochs} | Window: {window_len} | Repeats: {n_repeats} ===")
    print("=" * 85)

    target_features = ALL_INCLUSIVE_20_FEATURES.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    dji_df, esp32_df = get_labeled_datasets(features=target_features)

    unique_flights = dji_df["flight_id"].unique()
    train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=random_state)
    val_fl, test_fl = train_test_split(temp, test_size=0.5, random_state=random_state)

    dji_train = dji_df[dji_df["flight_id"].isin(train_fl)].copy()
    dji_val   = dji_df[dji_df["flight_id"].isin(val_fl)].copy()
    dji_test  = dji_df[dji_df["flight_id"].isin(test_fl)].copy()

    test_df = pd.concat([dji_test, esp32_df], ignore_index=True)

    # Masks for evaluation
    eval_mask = (test_df["attack_class"] != "Sim Geometry")
    normal_mask = (test_df["attack_class"] == "Normal DJI")
    spoofed_mask = (test_df["attack_class"] != "Normal DJI") & eval_mask

    # Fit Imputer and Scaler on normal training flights only
    X_train_raw = dji_train[target_features]
    X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
    X_val_scaled  = scaler.transform(imputer.transform(dji_val[target_features]))
    X_test_scaled = scaler.transform(imputer.transform(test_df[target_features]))

    # 2. Train Models & Establish Baselines
    models = {
        "Mahalanobis": MahalanobisWrapper(),
        "GMM": GMMWrapper(n_components=3, covariance_type="full", random_state=random_state),
        "Isolation Forest": IsolationForest(n_estimators=100, contamination="auto", random_state=random_state),
    }

    fitted_models = {}
    thresholds = {}
    baseline_normal_scores = {}
    baseline_class_tpr = {}
    baseline_overall_tpr = {}

    all_evaluated_models = ["Mahalanobis", "GMM", "Isolation Forest", "TCN Autoencoder"]

    # Train Classical Pointwise Models
    for name, model in models.items():
        print(f"\nTraining {name} on {len(X_train_scaled)} normal records...")
        model.fit(X_train_scaled)
        fitted_models[name] = model

        val_scores = get_anomaly_scores(model, X_val_scaled)
        thresh = np.mean(val_scores) + k_thresh * np.std(val_scores)
        thresholds[name] = thresh

        # Test scores
        test_scores = get_anomaly_scores(model, X_test_scaled)
        anom_pred = test_scores > thresh

        # Baseline Normal Raw Anomaly Score
        baseline_normal_scores[name] = float(np.mean(test_scores[normal_mask]))

        # Baseline Per-Class TPR
        cls_tpr = {}
        for cls in EVAL_ATTACK_CLASSES:
            c_mask = (test_df["attack_class"] == cls)
            cls_tpr[cls] = float(np.mean(anom_pred[c_mask]) * 100.0) if np.sum(c_mask) > 0 else 0.0
        baseline_class_tpr[name] = cls_tpr

        # Baseline Overall Spoofed TPR
        baseline_overall_tpr[name] = float(np.mean(anom_pred[spoofed_mask]) * 100.0) if np.sum(spoofed_mask) > 0 else 0.0
        print(f"  {name} Baseline -> Normal Mean Score: {baseline_normal_scores[name]:.4f} | Overall Spoofed TPR: {baseline_overall_tpr[name]:.2f}%")

    # Train TCN Autoencoder
    print(f"\nTraining TCN Autoencoder on ({device})...")
    dl_models = get_unsupervised_models(input_dim=len(target_features), seq_len=window_len)
    tcn_net = dl_models["TCN Autoencoder"].to(device)

    X_train_seq = create_sequences(X_train_scaled, window_len)
    X_val_seq   = create_sequences(X_val_scaled, window_len)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(tcn_net.parameters(), lr=0.001)
    best_loss = float("inf")
    best_weights = None

    for ep in range(1, epochs + 1):
        tcn_net.train()
        for (bx,) in train_loader:
            bx = bx.to(device)
            opt.zero_grad()
            recon = tcn_net(bx)
            loss = F.mse_loss(recon, bx)
            loss.backward()
            opt.step()

        tcn_net.eval()
        v_loss = 0.0
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                v_loss += F.mse_loss(tcn_net(bx), bx).item() * len(bx)
        v_loss /= len(X_val_seq)
        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in tcn_net.state_dict().items()}

    tcn_net.load_state_dict(best_weights)
    tcn_net.to(device)

    val_dl_scores = compute_dl_anomaly_scores(tcn_net, X_val_seq, device=device)
    tcn_thresh = np.mean(val_dl_scores) + k_thresh * np.std(val_dl_scores)
    thresholds["TCN Autoencoder"] = tcn_thresh
    fitted_models["TCN Autoencoder"] = tcn_net

    # TCN Baselines
    test_seq_all = create_sequences(X_test_scaled, window_len)
    test_classes_windowed = test_df["attack_class"].values[window_len - 1:]
    eval_mask_dl = (test_classes_windowed != "Sim Geometry")
    norm_mask_dl = (test_classes_windowed == "Normal DJI")
    spoo_mask_dl = (test_classes_windowed != "Normal DJI") & eval_mask_dl

    tcn_test_scores = compute_dl_anomaly_scores(tcn_net, test_seq_all, device=device)
    tcn_anom = tcn_test_scores > tcn_thresh

    baseline_normal_scores["TCN Autoencoder"] = float(np.mean(tcn_test_scores[norm_mask_dl])) if np.sum(norm_mask_dl) > 0 else 0.0
    tcn_cls_tpr = {}
    for cls in EVAL_ATTACK_CLASSES:
        c_mask_dl = (test_classes_windowed == cls)
        tcn_cls_tpr[cls] = float(np.mean(tcn_anom[c_mask_dl]) * 100.0) if np.sum(c_mask_dl) > 0 else 0.0
    baseline_class_tpr["TCN Autoencoder"] = tcn_cls_tpr
    baseline_overall_tpr["TCN Autoencoder"] = float(np.mean(tcn_anom[spoo_mask_dl]) * 100.0) if np.sum(spoo_mask_dl) > 0 else 0.0
    print(f"  TCN Baseline -> Normal Mean MSE: {baseline_normal_scores['TCN Autoencoder']:.4f} | Overall Spoofed TPR: {baseline_overall_tpr['TCN Autoencoder']:.2f}%")

    # 3. Permutation Loop Across 20 Features
    print("\n" + "=" * 85)
    print(">>> Executing Permutation Feature Importance Across 20 Features...")
    print("=" * 85)

    master_records = []
    rng = np.random.RandomState(random_state)

    for feat_idx, feat_name in enumerate(target_features):
        print(f"  [{feat_idx + 1:02d}/20] Permuting feature: {feat_name}...")

        rep_norm_deltas = {m: [] for m in all_evaluated_models}
        rep_class_deltas = {m: {cls: [] for cls in EVAL_ATTACK_CLASSES} for m in all_evaluated_models}
        rep_overall_deltas = {m: [] for m in all_evaluated_models}

        for rep in range(n_repeats):
            shuffled_test_df = test_df.copy()
            shuffled_test_df[feat_name] = rng.permutation(shuffled_test_df[feat_name].values)

            X_shuffled_scaled = scaler.transform(imputer.transform(shuffled_test_df[target_features]))

            # Classical Pointwise Predictions
            for m_name in ["Mahalanobis", "GMM", "Isolation Forest"]:
                m = fitted_models[m_name]
                shuff_scores = get_anomaly_scores(m, X_shuffled_scaled)
                shuff_pred = shuff_scores > thresholds[m_name]

                # (1) Pure Normal Score Delta
                norm_score_delta = float(np.mean(shuff_scores[normal_mask]) - baseline_normal_scores[m_name])
                rep_norm_deltas[m_name].append(norm_score_delta)

                # (2) Per-Class Signed Delta TPR
                for cls in EVAL_ATTACK_CLASSES:
                    c_mask = (test_df["attack_class"] == cls)
                    shuff_cls_tpr = float(np.mean(shuff_pred[c_mask]) * 100.0) if np.sum(c_mask) > 0 else 0.0
                    delta_tpr = baseline_class_tpr[m_name][cls] - shuff_cls_tpr
                    rep_class_deltas[m_name][cls].append(delta_tpr)

                # (3) Overall Signed Delta TPR
                shuff_overall_tpr = float(np.mean(shuff_pred[spoofed_mask]) * 100.0) if np.sum(spoofed_mask) > 0 else 0.0
                overall_delta_tpr = baseline_overall_tpr[m_name] - shuff_overall_tpr
                rep_overall_deltas[m_name].append(overall_delta_tpr)

            # TCN Sequence Predictions
            shuffled_seq = create_sequences(X_shuffled_scaled, window_len)
            tcn_shuff_scores = compute_dl_anomaly_scores(tcn_net, shuffled_seq, device=device)
            tcn_shuff_pred = tcn_shuff_scores > thresholds["TCN Autoencoder"]

            # (1) Pure Normal MSE Delta
            tcn_norm_delta = float(np.mean(tcn_shuff_scores[norm_mask_dl]) - baseline_normal_scores["TCN Autoencoder"]) if np.sum(norm_mask_dl) > 0 else 0.0
            rep_norm_deltas["TCN Autoencoder"].append(tcn_norm_delta)

            # (2) Per-Class Signed Delta TPR
            for cls in EVAL_ATTACK_CLASSES:
                c_mask_dl = (test_classes_windowed == cls)
                shuff_tcn_cls_tpr = float(np.mean(tcn_shuff_pred[c_mask_dl]) * 100.0) if np.sum(c_mask_dl) > 0 else 0.0
                delta_tpr = baseline_class_tpr["TCN Autoencoder"][cls] - shuff_tcn_cls_tpr
                rep_class_deltas["TCN Autoencoder"][cls].append(delta_tpr)

            # (3) Overall Signed Delta TPR
            shuff_tcn_overall_tpr = float(np.mean(tcn_shuff_pred[spoo_mask_dl]) * 100.0) if np.sum(spoo_mask_dl) > 0 else 0.0
            tcn_overall_delta = baseline_overall_tpr["TCN Autoencoder"] - shuff_tcn_overall_tpr
            rep_overall_deltas["TCN Autoencoder"].append(tcn_overall_delta)

        for m_name in all_evaluated_models:
            row = {
                "Feature": feat_name,
                "Model": m_name,
                "Normal Score Delta": float(np.mean(rep_norm_deltas[m_name])),
                "Overall Spoofed Delta TPR (%)": float(np.mean(rep_overall_deltas[m_name])),
            }
            for cls in EVAL_ATTACK_CLASSES:
                row[f"{cls} Delta TPR (%)"] = float(np.mean(rep_class_deltas[m_name][cls]))
            master_records.append(row)

    df_master = pd.DataFrame(master_records)
    master_csv_path = OUT_DIR / "master_feature_importance_per_class.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"\n✅ Saved Master Feature Importance Table to: {master_csv_path}")

    # 4. Generate 2x2 Subplot Grids (One Subplot per Model)
    print("\n>>> Generating 2x2 Subplot Grids (One Subplot per Model)...\n")

    # (1) Pure Normal Score Delta (2x2 Grid)
    plot_2x2_model_grid(
        df_data=df_master,
        value_col="Normal Score Delta",
        title="Pure Model Importance: Normal Anomaly Score Delta (When Feature Permuted)",
        xlabel="Raw Anomaly Score Increase on Clean Normal DJI Data",
        save_filename="importance_pure_normal.png",
        models=all_evaluated_models
    )

    # (2) Overall Spoofed Detection Contribution (2x2 Grid)
    plot_2x2_model_grid(
        df_data=df_master,
        value_col="Overall Spoofed Delta TPR (%)",
        title="Feature Contribution to Spoofed Detection (% TPR Drop When Feature Scrambled)",
        xlabel="Spoofed Detection Contribution (Baseline TPR - Shuffled TPR %)",
        save_filename="importance_overall_spoofed.png",
        models=all_evaluated_models
    )

    # (3) Per-Class 2x2 Grids for each individual attack class
    for cls in EVAL_ATTACK_CLASSES:
        safe_cls_name = cls.lower().replace(" ", "_")
        cls_col = f"{cls} Delta TPR (%)"
        plot_2x2_model_grid(
            df_data=df_master,
            value_col=cls_col,
            title=f"Feature Contribution to Identifying {cls} (% Detection Drop When Scrambled)",
            xlabel=f"{cls} Detection Contribution (Baseline TPR - Shuffled TPR %)",
            save_filename=f"importance_{safe_cls_name}.png",
            models=all_evaluated_models
        )

    print("\n" + "=" * 85)
    print("📊 MASTER FEATURE IMPORTANCE SUMMARY (Preview)")
    print("=" * 85)
    print(df_master.head(12).to_string(index=False))

    return df_master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation Feature Importance")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--fresh-cache", action="store_true", default=True)
    args = parser.parse_args()

    run_permutation_importance(
        epochs=args.epochs,
        n_repeats=args.repeats,
        window_len=args.window_len,
        fresh_cache=args.fresh_cache
    )
