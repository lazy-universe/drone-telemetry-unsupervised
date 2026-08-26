"""
Unsupervised Anomaly Detection Pipeline
Executes end-to-end unsupervised training and multi-class anomaly detection benchmarking across:
- Pointwise Classical Models (Isolation Forest, GMM, Mahalanobis, One-Class SVM, PCA, K-Means, DBSCAN, KNN)
- Deep Sequence Autoencoders (Dense AE, GRU AE, TCN AE, CNN-GRU AE)
"""

import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.workflows.per_class_evaluation import get_labeled_datasets
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data
from implement.utils.classical_ml.classical_models import get_unsupervised_point_models
from implement.utils.classical_ml.classical_train_eval import get_anomaly_scores
from implement.utils.deep_learning.dl_models import get_unsupervised_models
from implement.utils.helper.physics_rules import compute_physics_rule_flags

# Standard Predefined Feature Sets
FEATURE_SETS = {
    "raw_coords": [
        "latitude", "longitude", "course", "ground_speed", "vertical_speed", "height"
    ],
    "raw_no_coords": [
        "course", "ground_speed", "vertical_speed", "height"
    ],
    "raw_engineered": [
        "course", "ground_speed", "vertical_speed", "height",
        "acceleration", "vertical_acceleration", "turn_rate", "path_curvature"
    ],
    "baseline_7": [
        "motion_smoothness", "heading_speed_consistency", "ground_speed", 
        "height", "vertical_speed", "acceleration", "turn_rate"
    ],
    "baseline_7_plus_pos": [
        "motion_smoothness", "heading_speed_consistency", "ground_speed", 
        "height", "vertical_speed", "acceleration", "turn_rate",
        "position_residual_std"
    ],
    "baseline_7_plus_pe": [
        "motion_smoothness", "heading_speed_consistency", "ground_speed", 
        "height", "vertical_speed", "acceleration", "turn_rate",
        "prediction_error"
    ],
    "baseline_7_plus_pe_entropy": [
        "motion_smoothness", "heading_speed_consistency", "ground_speed", 
        "height", "vertical_speed", "acceleration", "turn_rate",
        "prediction_error", "speed_spectral_entropy"
    ],
    "baseline_7_plus_pe_pos": [
        "motion_smoothness", "heading_speed_consistency", "ground_speed", 
        "height", "vertical_speed", "acceleration", "turn_rate",
        "prediction_error", "position_residual_std"
    ],
    "baseline_7_plus_pe_pos_entropy": [
        "motion_smoothness", "heading_speed_consistency", "ground_speed", 
        "height", "vertical_speed", "acceleration", "turn_rate",
        "prediction_error", "position_residual_std", "speed_spectral_entropy"
    ],
    "baseline_7_plus_pe_pos_yaw": [
        "motion_smoothness", "heading_speed_consistency", "ground_speed", 
        "height", "vertical_speed", "acceleration", "turn_rate",
        "prediction_error", "position_residual_std", "yaw_acceleration"
    ],
}


def clear_ephemeral_cache():
    """Clears cached processed ephemeral dataset directories to guarantee fresh feature computation."""
    dataset_dir = PROJECT_ROOT / "implement" / "dataset"
    if dataset_dir.exists():
        for cache_dir in dataset_dir.glob("ephermal_dataset_*"):
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    print(f"Cleared ephemeral cache: {cache_dir.name}")
                except Exception as err:
                    print(f"[Warning] Could not clear cache {cache_dir}: {err}")


def create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Slices continuous time series into sliding window sequence matrices."""
    if len(data) < seq_len:
        return np.empty((0, seq_len, data.shape[1]))
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)


def compute_dl_anomaly_scores(model: nn.Module, data_seq: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Computes MSE reconstruction error anomaly scores for sequence autoencoders."""
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


def run_experiment(
    exp_name: str,
    feature_list: list,
    model_family: str = "all",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    window_len: int = 20,
    k_thresh: float = 3.0,
    patience: int = 7,
    fresh_cache: bool = True,
    use_cache: bool = True,
    enable_physics_rules: bool = False,
    exclude_sim_geometry: bool = True
) -> pd.DataFrame:
    """
    Executes the anomaly detection pipeline for a specified feature set and model family.
    """
    if fresh_cache:
        clear_ephemeral_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print("=" * 80)
    print(f"=== UNSUPERVISED PIPELINE: {exp_name.upper()} ({len(feature_list)} Features) ===")
    print(f"=== Device: {device} | Max Epochs: {epochs} | Patience: {patience} | Window: {window_len} | Physics Rules: {enable_physics_rules} | Exclude Sim Geometry: {exclude_sim_geometry} ===")
    print("=" * 80)

    dji_df, esp32_df = get_labeled_datasets(features=feature_list)

    unique_flights = dji_df["flight_id"].unique()
    train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_fl, test_fl = train_test_split(temp, test_size=0.5, random_state=42)

    dji_train = dji_df[dji_df["flight_id"].isin(train_fl)]
    dji_val   = dji_df[dji_df["flight_id"].isin(val_fl)]
    dji_test  = dji_df[dji_df["flight_id"].isin(test_fl)]

    test_df = pd.concat([dji_test, esp32_df], ignore_index=True)
    all_classes = sorted(list(test_df["attack_class"].unique()))

    # Fit Imputer and Scaler on normal training flights only
    X_train_raw = dji_train[feature_list]
    X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
    X_val_scaled   = scaler.transform(imputer.transform(dji_val[feature_list]))
    X_test_scaled  = scaler.transform(imputer.transform(test_df[feature_list]))

    from implement.utils.helper import get_output_dir
    models_save_dir = get_output_dir() / "pipeline_experiments" / exp_name / "models"
    models_save_dir.mkdir(parents=True, exist_ok=True)

    aggregate_results = []
    per_class_results = []

    # Standalone Physics Rules (if enabled)
    if enable_physics_rules:
        rule_flags_all = compute_physics_rule_flags(test_df)
        eval_mask = (test_df["attack_class"] != "Sim Geometry") if exclude_sim_geometry else np.ones(len(test_df), dtype=bool)
        normal_mask = (test_df["attack_class"] == "Normal DJI") & eval_mask
        spoofed_mask = (test_df["attack_class"] != "Normal DJI") & eval_mask

        norm_acc = np.mean(~rule_flags_all[normal_mask]) * 100.0 if np.sum(normal_mask) > 0 else 0.0
        spoo_acc = np.mean(rule_flags_all[spoofed_mask]) * 100.0 if np.sum(spoofed_mask) > 0 else 0.0
        overall_acc = (
            (np.sum(~rule_flags_all[normal_mask]) + np.sum(rule_flags_all[spoofed_mask])) / np.sum(eval_mask) * 100.0
        ) if np.sum(eval_mask) > 0 else 0.0

        aggregate_results.append({
            "Experiment": exp_name,
            "Model": "Physics Rules Standalone",
            "Type": "Deterministic Rules",
            "Overall Accuracy (%)": round(overall_acc, 2),
            "Normal Accuracy (%)": round(norm_acc, 2),
            "Spoofed Accuracy (%)": round(spoo_acc, 2)
        })

        row_pr = {"Experiment": exp_name, "Model": "Physics Rules Standalone", "Type": "Deterministic Rules"}
        for cls in all_classes:
            mask = (test_df["attack_class"] == cls)
            anom = rule_flags_all[mask]
            acc = np.mean(~anom) * 100.0 if cls == "Normal DJI" else np.mean(anom) * 100.0
            row_pr[cls] = round(acc, 2)
        per_class_results.append(row_pr)

    # 1. Pointwise Classical Models
    if model_family in ["all", "pointwise"]:
        print()
        print(">>> Training / Evaluating Pointwise Classical Models...")
        pw_models = get_unsupervised_point_models()

        for name, model in pw_models.items():
            print(f"  Evaluating {name}...")
            model.fit(X_train_scaled)

            val_scores = get_anomaly_scores(model, X_val_scaled)
            thresh = np.mean(val_scores) + k_thresh * np.std(val_scores)

            model_type_label = "Classical Pointwise + Rules" if enable_physics_rules else "Classical Pointwise"
            
            # Predict on full test set
            all_scores = get_anomaly_scores(model, X_test_scaled)
            anom_pred = all_scores > thresh
            if enable_physics_rules:
                anom_pred = anom_pred | compute_physics_rule_flags(test_df)

            eval_mask = (test_df["attack_class"] != "Sim Geometry") if exclude_sim_geometry else np.ones(len(test_df), dtype=bool)
            normal_mask = (test_df["attack_class"] == "Normal DJI") & eval_mask
            spoofed_mask = (test_df["attack_class"] != "Normal DJI") & eval_mask

            norm_acc = np.mean(~anom_pred[normal_mask]) * 100.0 if np.sum(normal_mask) > 0 else 0.0
            spoo_acc = np.mean(anom_pred[spoofed_mask]) * 100.0 if np.sum(spoofed_mask) > 0 else 0.0
            overall_acc = (
                (np.sum(~anom_pred[normal_mask]) + np.sum(anom_pred[spoofed_mask])) / np.sum(eval_mask) * 100.0
            ) if np.sum(eval_mask) > 0 else 0.0

            aggregate_results.append({
                "Experiment": exp_name,
                "Model": name,
                "Type": model_type_label,
                "Overall Accuracy (%)": round(overall_acc, 2),
                "Normal Accuracy (%)": round(norm_acc, 2),
                "Spoofed Accuracy (%)": round(spoo_acc, 2)
            })

            row_pc = {"Experiment": exp_name, "Model": name, "Type": model_type_label}
            for cls in all_classes:
                mask = (test_df["attack_class"] == cls)
                test_cls_scaled = X_test_scaled[mask]
                if len(test_cls_scaled) > 0:
                    cls_scores = get_anomaly_scores(model, test_cls_scaled)
                    anom = cls_scores > thresh
                    if enable_physics_rules:
                        anom = anom | compute_physics_rule_flags(test_df[mask])
                    acc = np.mean(~anom) * 100.0 if cls == "Normal DJI" else np.mean(anom) * 100.0
                else:
                    acc = 0.0
                row_pc[cls] = round(acc, 2)
            per_class_results.append(row_pc)

    # 2. Deep Learning Sequence Autoencoders
    if model_family in ["all", "dl", "autoencoders"]:
        print()
        print(f">>> Training / Evaluating Deep Learning Autoencoders on ({device})...")
        num_feats = len(feature_list)
        dl_models = get_unsupervised_models(input_dim=num_feats, seq_len=window_len)

        # Prepare Sequences
        X_train_seq = create_sequences(X_train_scaled, window_len)
        X_val_seq   = create_sequences(X_val_scaled, window_len)

        if len(X_train_seq) > 0 and len(X_val_seq) > 0:
            train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
            val_tensor   = torch.tensor(X_val_seq, dtype=torch.float32)
            
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

            for name, net in dl_models.items():
                safe_model_name = name.lower().replace(" ", "_").replace("-", "_")
                ckpt_path = models_save_dir / f"{safe_model_name}.pth"

                loaded_from_cache = False
                thresh = None

                if use_cache and ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device)
                        if isinstance(ckpt, dict) and ckpt.get("features") == feature_list and ckpt.get("window_len") == window_len:
                            net.load_state_dict(ckpt["state_dict"])
                            net.to(device)
                            thresh = ckpt["threshold"]
                            loaded_from_cache = True
                            print(f"  ⚡ Loaded Cached Checkpoint: {ckpt_path}")
                    except Exception as err:
                        print(f"  ⚠️ Retraining {name} due to cache issue: {err}")

                if not loaded_from_cache:
                    print(f"  Training {name} (Patience={patience}, Epochs={epochs})...")
                    net.to(device)
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

                    best_val_loss = float("inf")
                    best_state = None
                    patience_counter = 0

                    for ep in range(1, epochs + 1):
                        net.train()
                        total_train_loss = 0.0
                        for (bx,) in train_loader:
                            bx = bx.to(device)
                            optimizer.zero_grad()
                            recon_x = net(bx)
                            loss = F.mse_loss(recon_x, bx)
                            loss.backward()
                            optimizer.step()
                            total_train_loss += loss.item() * len(bx)

                        train_loss_avg = total_train_loss / len(train_tensor)

                        net.eval()
                        total_val_loss = 0.0
                        with torch.no_grad():
                            for (bx,) in val_loader:
                                bx = bx.to(device)
                                recon_x = net(bx)
                                loss = F.mse_loss(recon_x, bx)
                                total_val_loss += loss.item() * len(bx)

                        val_loss_avg = total_val_loss / len(val_tensor)

                        if val_loss_avg < best_val_loss:
                            best_val_loss = val_loss_avg
                            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"    ⏹️ Early stopping at epoch {ep} (Best Val Loss: {best_val_loss:.6f})")
                                break

                        if ep % 10 == 0 or ep == epochs:
                            print(f"    Epoch {ep}/{epochs} - Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")

                    if best_state is not None:
                        net.load_state_dict(best_state)
                        net.to(device)

                    val_scores = compute_dl_anomaly_scores(net, X_val_seq, device=device)
                    thresh = np.mean(val_scores) + k_thresh * np.std(val_scores)

                    torch.save({
                        "state_dict": net.state_dict(),
                        "threshold": float(thresh),
                        "window_len": window_len,
                        "features": feature_list
                    }, ckpt_path)

                model_type_label = "Deep Learning + Rules" if enable_physics_rules else "Deep Learning"
                
                # Full test sequences
                test_seq_all = create_sequences(X_test_scaled, window_len)
                if len(test_seq_all) > 0:
                    dl_scores_all = compute_dl_anomaly_scores(net, test_seq_all, device=device)
                    dl_anom_all = dl_scores_all > thresh
                    if enable_physics_rules:
                        ph_flags = compute_physics_rule_flags(test_df)
                        if len(ph_flags) >= window_len:
                            dl_anom_all = dl_anom_all | ph_flags[window_len - 1:]

                    test_classes_windowed = test_df["attack_class"].values[window_len - 1:]
                    eval_mask_dl = (test_classes_windowed != "Sim Geometry") if exclude_sim_geometry else np.ones(len(test_classes_windowed), dtype=bool)
                    normal_mask_dl = (test_classes_windowed == "Normal DJI") & eval_mask_dl
                    spoofed_mask_dl = (test_classes_windowed != "Normal DJI") & eval_mask_dl

                    norm_acc = np.mean(~dl_anom_all[normal_mask_dl]) * 100.0 if np.sum(normal_mask_dl) > 0 else 0.0
                    spoo_acc = np.mean(dl_anom_all[spoofed_mask_dl]) * 100.0 if np.sum(spoofed_mask_dl) > 0 else 0.0
                    overall_acc = (
                        (np.sum(~dl_anom_all[normal_mask_dl]) + np.sum(dl_anom_all[spoofed_mask_dl])) / np.sum(eval_mask_dl) * 100.0
                    ) if np.sum(eval_mask_dl) > 0 else 0.0
                else:
                    norm_acc, spoo_acc, overall_acc = 0.0, 0.0, 0.0

                aggregate_results.append({
                    "Experiment": exp_name,
                    "Model": name,
                    "Type": model_type_label,
                    "Overall Accuracy (%)": round(overall_acc, 2),
                    "Normal Accuracy (%)": round(norm_acc, 2),
                    "Spoofed Accuracy (%)": round(spoo_acc, 2)
                })

                row_dl = {"Experiment": exp_name, "Model": name, "Type": model_type_label}
                for cls in all_classes:
                    mask = (test_df["attack_class"] == cls)
                    test_cls_scaled = X_test_scaled[mask]
                    test_cls_seq = create_sequences(test_cls_scaled, window_len)
                    if len(test_cls_seq) > 0:
                        cls_scores = compute_dl_anomaly_scores(net, test_cls_seq, device=device)
                        anom = cls_scores > thresh
                        if enable_physics_rules:
                            physics_flags_cls = compute_physics_rule_flags(test_df[mask])
                            if len(physics_flags_cls) >= window_len:
                                anom = anom | physics_flags_cls[window_len - 1:]
                        acc = np.mean(~anom) * 100.0 if cls == "Normal DJI" else np.mean(anom) * 100.0
                    else:
                        acc = 0.0
                    row_dl[cls] = round(acc, 2)
                per_class_results.append(row_dl)

    df_aggregate = pd.DataFrame(aggregate_results)
    df_per_class = pd.DataFrame(per_class_results)

    out_dir = get_output_dir() / "pipeline_experiments" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = out_dir / f"{exp_name}_aggregate.csv"
    per_class_path = out_dir / f"{exp_name}_results.csv"
    df_aggregate.to_csv(agg_path, index=False)
    df_per_class.to_csv(per_class_path, index=False)

    sim_geo_note = (
        "Note: 'Sim Geometry' is EXCLUDED from aggregate metrics (Overall, Normal, Spoofed Accuracy) for transparency, but retained in the per-class breakdown."
        if exclude_sim_geometry else
        "Note: 'Sim Geometry' is INCLUDED in aggregate metrics (Overall, Normal, Spoofed Accuracy)."
    )

    print()
    print("=" * 80)
    print(f"📊 AGGREGATE PERFORMANCE SUMMARY: {exp_name.upper()}")
    print("=" * 80)
    print(df_aggregate.to_string(index=False))
    print(f"\nℹ️  {sim_geo_note}")

    print()
    print("=" * 80)
    print(f"🎯 PER-CLASS BREAKDOWN RESULTS: {exp_name.upper()}")
    print("=" * 80)
    print(df_per_class.to_string(index=False))

    print(f"\n✅ Saved Aggregate CSV: {agg_path}")
    print(f"✅ Saved Per-Class CSV: {per_class_path}")

    return df_per_class


def main():
    parser = argparse.ArgumentParser(description="Unsupervised Anomaly Detection Pipeline")
    parser.add_argument("--features", nargs="+", default=None, help="Explicit list of features for training/eval")
    parser.add_argument("--preset", choices=list(FEATURE_SETS.keys()), default=None, help="Predefined feature set preset")
    parser.add_argument("--exp-name", type=str, default=None, help="Custom experiment name")
    parser.add_argument("--model-family", choices=["all", "pointwise", "dl", "autoencoders"], default="all")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--no-fresh-cache", action="store_true", help="Do not clear ephemeral dataset cache")
    parser.add_argument("--no-model-cache", action="store_true", help="Do not load cached models; force retrain")
    parser.add_argument("--enable-physics-rules", action="store_true", default=False, help="Enable deterministic physics-based rules")
    parser.add_argument("--include-sim-geometry", action="store_true", default=False, help="Include Sim Geometry in aggregate accuracy metrics (default is to exclude for transparency)")

    args = parser.parse_args()

    if args.features:
        features = []
        for item in args.features:
            for feat in item.split(","):
                feat_clean = feat.strip()
                if feat_clean and feat_clean not in features:
                    features.append(feat_clean)
        exp_name = args.exp_name if args.exp_name else f"custom_{len(features)}_features"
    elif args.preset:
        features = FEATURE_SETS[args.preset]
        exp_name = args.exp_name if args.exp_name else args.preset
    else:
        features = FEATURE_SETS["correlation_16"]
        exp_name = args.exp_name if args.exp_name else "correlation_16"

    run_experiment(
        exp_name=exp_name,
        feature_list=features,
        model_family=args.model_family,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        window_len=args.window_len,
        fresh_cache=not args.no_fresh_cache,
        use_cache=not args.no_model_cache,
        enable_physics_rules=args.enable_physics_rules,
        exclude_sim_geometry=not args.include_sim_geometry
    )


if __name__ == "__main__":
    main()