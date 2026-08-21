import os
import sys
import numpy as np
import torch
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from implement.initial_setup import get_run_subfolder_name
from implement.utils.helper import (
    get_output_dir,
    get_combined_dataset_dir,
    get_or_preprocess_dji_dataset,
    get_or_preprocess_esp32_dataset,
    UNSUPERVISED_FEATURES,
    WINDOW_LEN,
    log_dataset_statistics
)
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data
from implement.utils.deep_learning.dl_models import get_unsupervised_models
from implement.utils.deep_learning.dl_train_eval_unsupervised import train_and_evaluate_dl_unsupervised


def create_sliding_sequences(df, window_len=WINDOW_LEN, features=UNSUPERVISED_FEATURES):
    """
    Converts telemetry DataFrame into sliding sequence windows.
    Returns X (N, window_len, input_dim) and y (N,).
    """
    X_seq = []
    y_seq = []

    if 'flight_id' in df.columns:
        groups = df.groupby('flight_id')
    else:
        groups = [('all', df)]

    for _, group in groups:
        feat_vals = group[features].values
        labels = group['nature'].values
        if len(group) < window_len:
            continue
        for i in range(len(group) - window_len + 1):
            X_seq.append(feat_vals[i : i + window_len])
            y_seq.append(labels[i + window_len - 1])

    if not X_seq:
        return np.empty((0, window_len, len(features))), np.empty((0,), dtype=int)

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=int)


def run_unsupervised_dl_pipeline(
    mode: str = "multi",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    window_len: int = WINDOW_LEN,
    split_mode: str = "flight",
    patience: int = 7
):
    """
    Executes the unsupervised deep learning autoencoder pipeline (GRU, TCN, CNN-GRU, GRU-VAE).
    """
    # 1. Workspace Paths & initial setup
    output_dir = get_output_dir()
    
    subfolder = f"unsupervised_dl_{split_mode}_w{window_len}_e{epochs}"
    evaluation_dir = output_dir / subfolder
    split_dir = evaluation_dir / 'dataset'
    models_dir = evaluation_dir / 'models'
    plots_dir = evaluation_dir / 'plots'

    split_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("=== WORKFLOW: UNSUPERVISED DEEP LEARNING AUTOENCODERS ===")
    print(f"=== Config: Split={split_mode} | Epochs={epochs} | Batch={batch_size} | Window={window_len} ===")
    print(f"=== Target Output Folder: {evaluation_dir} ===")
    print("=====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load datasets
    print("\n[Step 1/4] Loading DJI (normal) and ESP32 (attack) datasets...")
    dji_df = get_or_preprocess_dji_dataset(filter_length_100=False, features=UNSUPERVISED_FEATURES)
    esp32_df = get_or_preprocess_esp32_dataset()

    print(f"  DJI rows: {len(dji_df)} | ESP32 rows: {len(esp32_df)}")

    # 3. Partition DJI flights into train (70%), val (15%), test (15%)
    dji_clean = dji_df.copy()
    esp32_clean = esp32_df.copy()

    if split_mode == 'flight' and 'flight_id' in dji_clean.columns:
        print("\n[Step 2/4] Partitioning DJI data flight-wise into Train / Val / Test...")
        unique_flights = dji_clean['flight_id'].unique()
        train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
        val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

        dji_train = dji_clean[dji_clean['flight_id'].isin(train_flights)]
        dji_val = dji_clean[dji_clean['flight_id'].isin(val_flights)]
        dji_test = dji_clean[dji_clean['flight_id'].isin(test_flights)]
    else:
        print("\n[Step 2/4] Partitioning DJI data randomly into Train / Val / Test...")
        dji_train, temp_dji = train_test_split(dji_clean, test_size=0.3, random_state=42)
        dji_val, dji_test = train_test_split(temp_dji, test_size=0.5, random_state=42)

    # 4. Scale features using scaler fitted ONLY on dji_train
    print("\nFitting scaler and imputer on normal training data...")
    X_train_raw = dji_train[UNSUPERVISED_FEATURES]
    X_val_raw = dji_val[UNSUPERVISED_FEATURES]
    X_test_dji_raw = dji_test[UNSUPERVISED_FEATURES]
    X_esp32_raw = esp32_clean[UNSUPERVISED_FEATURES]

    X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
    X_val_scaled = scaler.transform(imputer.transform(X_val_raw))
    X_test_dji_scaled = scaler.transform(imputer.transform(X_test_dji_raw))
    X_esp32_scaled = scaler.transform(imputer.transform(X_esp32_raw))

    joblib.dump(scaler, models_dir / 'scaler.joblib')
    joblib.dump(imputer, models_dir / 'imputer.joblib')

    # Reconstruct scaled DataFrames
    dji_train_scaled = dji_train.copy()
    dji_train_scaled[UNSUPERVISED_FEATURES] = X_train_scaled

    dji_val_scaled = dji_val.copy()
    dji_val_scaled[UNSUPERVISED_FEATURES] = X_val_scaled

    dji_test_scaled = dji_test.copy()
    dji_test_scaled[UNSUPERVISED_FEATURES] = X_test_dji_scaled

    esp32_test_scaled = esp32_clean.copy()
    esp32_test_scaled[UNSUPERVISED_FEATURES] = X_esp32_scaled

    # Combine test set
    test_scaled_df = pd.concat([dji_test_scaled, esp32_test_scaled], ignore_index=True)

    # Create sequences
    print(f"\n[Step 3/4] Generating sliding sequences (window_len={window_len})...")
    X_train_seq, y_train_seq = create_sliding_sequences(dji_train_scaled, window_len=window_len)
    X_val_seq, y_val_seq = create_sliding_sequences(dji_val_scaled, window_len=window_len)
    X_test_seq, y_test_seq = create_sliding_sequences(test_scaled_df, window_len=window_len)

    log_dataset_statistics(X_train_seq, y_train_seq, X_val=X_val_seq, y_val=y_val_seq, X_test=X_test_seq, y_test=y_test_seq)

    # 5. Initialize & Train Autoencoders
    print("\n[Step 4/4] Initializing and training 4 Deep Learning Autoencoders...")
    models = get_unsupervised_models(input_dim=len(UNSUPERVISED_FEATURES), seq_len=window_len)

    results = train_and_evaluate_dl_unsupervised(
        models=models,
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
        X_test=X_test_seq,
        y_test=y_test_seq,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        models_dir=models_dir,
        plots_dir=plots_dir,
        patience=patience
    )

    print("\n" + "="*110)
    print("=== FINAL UNSUPERVISED DL AUTOENCODER SUMMARY ===")
    print(f"{'Autoencoder Model':<30}{'Accuracy':<15}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'ROC-AUC':<15}{'PR-AUC':<15}")
    print("-"*110)
    for name, metrics in results.items():
        acc_str = f"{metrics['accuracy']*100:.2f}%"
        prec_str = f"{metrics['precision']*100:.2f}%"
        rec_str = f"{metrics['recall']*100:.2f}%"
        f1_str = f"{metrics['f1_score']*100:.2f}%"
        auc_str = f"{metrics['roc_auc']*100:.2f}%"
        pr_auc_str = f"{metrics['pr_auc']*100:.2f}%"
        print(f"{name:<30}{acc_str:<15}{prec_str:<15}{rec_str:<15}{f1_str:<15}{auc_str:<15}{pr_auc_str:<15}")
    print("========================================================================================================\n")

    print("=== Confusion Matrices ===")
    for name, metrics in results.items():
        print(f"  {name} (Threshold={metrics.get('threshold', 0):.4f}):")
        print(np.array(metrics['confusion_matrix']))
        print()
    print("==========================\n")
    return results
