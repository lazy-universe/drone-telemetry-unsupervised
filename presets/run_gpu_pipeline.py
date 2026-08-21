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

# Predefined Feature Sets
FEATURE_SETS = {
    'raw_coords': ['latitude', 'longitude', 'course', 'ground_speed', 'vertical_speed', 'height'],
    'raw_no_coords': ['course', 'ground_speed', 'vertical_speed', 'height'],
    'raw_engineered': ['course', 'ground_speed', 'vertical_speed', 'height', 'acceleration', 'vertical_acceleration', 'turn_rate', 'path_curvature'],
    'complete_set': ['course', 'ground_speed', 'vertical_speed', 'height', 'acceleration', 'vertical_acceleration', 'turn_rate', 'path_curvature', 'heading_speed_consistency', 'motion_smoothness'],
    'baseline_10': ['height', 'ground_speed', 'vertical_speed', 'acceleration', 'turn_rate', 'path_curvature', 'heading_speed_consistency', 'motion_smoothness', 'prediction_error', 'yaw_acceleration'],
    'baseline_9_no_yaw': ['height', 'ground_speed', 'vertical_speed', 'acceleration', 'turn_rate', 'path_curvature', 'heading_speed_consistency', 'motion_smoothness', 'prediction_error'],
    'baseline_9_no_pe': ['height', 'ground_speed', 'vertical_speed', 'acceleration', 'turn_rate', 'path_curvature', 'heading_speed_consistency', 'motion_smoothness', 'yaw_acceleration'],
}

def clear_ephemeral_cache():
    """Clears all cached processed ephemeral dataset directories to guarantee fresh feature computation for each experiment."""
    dataset_dir = PROJECT_ROOT / "implement" / "dataset"
    if dataset_dir.exists():
        ephemeral_dirs = list(dataset_dir.glob("ephermal_dataset_*"))
        for cache_dir in ephemeral_dirs:
            if cache_dir.exists():
                print(f"🧹 Clearing ephemeral dataset cache: {cache_dir}...")
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✓ Ephemeral cache {cache_dir.name} cleared successfully.")
                except Exception as e:
                    print(f"⚠️ Warning: Could not clear cache {cache_dir}: {e}")

def create_sequences(data, seq_len):
    if len(data) < seq_len:
        return np.empty((0, seq_len, data.shape[1]))
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)

def compute_dl_anomaly_scores(model, data_seq, device='cpu'):
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
            mse = F.mse_loss(recon_x, bx, reduction='none').mean(dim=[1, 2])
            scores.extend(mse.cpu().numpy())
    return np.array(scores)

def run_experiment(exp_name, feature_list, model_family="all", epochs=15, batch_size=64, lr=0.001, window_len=20, k_thresh=3.0, patience=7, fresh_cache=True, use_cache=True):
    if fresh_cache:
        clear_ephemeral_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=========================================================================")
    print(f"=== GPU EXPERIMENT: {exp_name.upper()} ({len(feature_list)} Features) ===")
    print(f"=== Device: {device} | Max Epochs: {epochs} | Patience: {patience} | Window: {window_len} | Use Cache: {use_cache} ===")
    print(f"=========================================================================")

    dji_df, esp32_df = get_labeled_datasets(features=feature_list)

    unique_flights = dji_df['flight_id'].unique()
    train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_fl, test_fl = train_test_split(temp, test_size=0.5, random_state=42)

    dji_train = dji_df[dji_df['flight_id'].isin(train_fl)]
    dji_val   = dji_df[dji_df['flight_id'].isin(val_fl)]
    dji_test  = dji_df[dji_df['flight_id'].isin(test_fl)]

    test_df = pd.concat([dji_test, esp32_df], ignore_index=True)
    all_classes = sorted(list(test_df['attack_class'].unique()))

    # Fit Imputer and Scaler
    X_train_raw = dji_train[feature_list]
    X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
    X_val_scaled   = scaler.transform(imputer.transform(dji_val[feature_list]))
    X_test_scaled  = scaler.transform(imputer.transform(test_df[feature_list]))

    from implement.utils.helper import get_output_dir
    models_save_dir = get_output_dir() / 'gpu_experiments' / exp_name / 'models'
    models_save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Pointwise Classical Models
    if model_family in ['all', 'pointwise']:
        print("\n>>> Training / Evaluating Pointwise Classical Models...")
        pw_models = get_unsupervised_point_models()

        for name, model in pw_models.items():
            print(f"  Evaluating {name}...")
            model.fit(X_train_scaled)

            val_scores = get_anomaly_scores(model, X_val_scaled)
            thresh = np.mean(val_scores) + k_thresh * np.std(val_scores)

            row = {'Experiment': exp_name, 'Model': name, 'Type': 'Classical Pointwise'}

            for cls in all_classes:
                mask = (test_df['attack_class'] == cls)
                test_cls_scaled = X_test_scaled[mask]
                if len(test_cls_scaled) > 0:
                    cls_scores = get_anomaly_scores(model, test_cls_scaled)
                    anom = cls_scores > thresh
                    acc = np.mean(~anom)*100.0 if cls == 'Normal DJI' else np.mean(anom)*100.0
                else:
                    acc = 0.0
                row[cls] = round(acc, 2)
            results.append(row)

    # 2. Deep Learning Autoencoders on GPU with Early Stopping
    if model_family in ['all', 'dl', 'autoencoders']:
        print(f"\n>>> Training / Evaluating Deep Learning Autoencoders on GPU ({device})...")
        
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
                safe_model_name = name.lower().replace(' ', '_').replace('-', '_')
                ckpt_path = models_save_dir / f"{safe_model_name}.pth"

                loaded_from_cache = False
                thresh = None

                if use_cache and ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device)
                        if isinstance(ckpt, dict) and ckpt.get('features') == feature_list and ckpt.get('window_len') == window_len:
                            net.load_state_dict(ckpt['state_dict'])
                            net.to(device)
                            thresh = ckpt['threshold']
                            loaded_from_cache = True
                            print(f"  ⚡ Loaded Cached DL Checkpoint: {ckpt_path}")
                    except Exception as e:
                        print(f"  ⚠️ Could not load cache for {name}: {e}. Retraining...")

                if not loaded_from_cache:
                    print(f"  Training {name} (Patience={patience}, Epochs={epochs})...")
                    net.to(device)
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

                    best_val_loss = float('inf')
                    best_state = None
                    patience_counter = 0

                    for ep in range(1, epochs + 1):
                        # Training Loop
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

                        # Validation Loop for Early Stopping
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

                    # Load best weights
                    if best_state is not None:
                        net.load_state_dict(best_state)
                        net.to(device)

                    # Validation Threshold
                    val_scores = compute_dl_anomaly_scores(net, X_val_seq, device=device)
                    thresh = np.mean(val_scores) + k_thresh * np.std(val_scores)

                    # Save Deep Learning Checkpoint (.pth)
                    torch.save({
                        'state_dict': net.state_dict(),
                        'threshold': float(thresh),
                        'window_len': window_len,
                        'features': feature_list
                    }, ckpt_path)
                    print(f"    💾 Saved DL Checkpoint: {ckpt_path}")

                row = {'Experiment': exp_name, 'Model': name, 'Type': 'Deep Learning (GPU)'}
                for cls in all_classes:
                    mask = (test_df['attack_class'] == cls)
                    test_cls_scaled = X_test_scaled[mask]
                    test_cls_seq = create_sequences(test_cls_scaled, window_len)
                    if len(test_cls_seq) > 0:
                        cls_scores = compute_dl_anomaly_scores(net, test_cls_seq, device=device)
                        anom = cls_scores > thresh
                        acc = np.mean(~anom)*100.0 if cls == 'Normal DJI' else np.mean(anom)*100.0
                    else:
                        acc = 0.0
                    row[cls] = round(acc, 2)
                results.append(row)

    df_res = pd.DataFrame(results)
    out_dir = get_output_dir() / 'gpu_experiments' / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{exp_name}_results.csv"
    df_res.to_csv(csv_path, index=False)

    print(f"\nRESULTS FOR {exp_name.upper()}:")
    print(df_res.to_string(index=False))
    print(f"\n✅ Saved CSV: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="GPU Accelerated Unsupervised Anomaly Detection Pipeline")
    parser.add_argument("--preset", choices=list(FEATURE_SETS.keys()), default=None, help="Predefined feature set preset")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated custom feature list")
    parser.add_argument("--exp-name", type=str, default=None, help="Custom experiment name")
    parser.add_argument("--model-family", choices=["all", "pointwise", "dl", "autoencoders"], default="all")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--no-fresh-cache", action="store_true", help="Do not clear ephemeral dataset cache before running")
    parser.add_argument("--no-model-cache", action="store_true", help="Do not load cached models; force retrain all models")

    args = parser.parse_args()

    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
        exp_name = args.exp_name if args.exp_name else "custom_features"
    elif args.preset:
        features = FEATURE_SETS[args.preset]
        exp_name = args.exp_name if args.exp_name else args.preset
    else:
        features = FEATURE_SETS['complete_set']
        exp_name = args.exp_name if args.exp_name else 'complete_set'

    run_experiment(
        exp_name=exp_name,
        feature_list=features,
        model_family=args.model_family,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        window_len=args.window_len,
        fresh_cache=not args.no_fresh_cache,
        use_cache=not args.no_model_cache
    )

if __name__ == "__main__":
    main()
