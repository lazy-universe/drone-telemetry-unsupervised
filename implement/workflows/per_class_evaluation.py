import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from implement.utils.helper import (
    get_output_dir,
    get_or_preprocess_dji_dataset,
    get_or_preprocess_esp32_dataset,
    UNSUPERVISED_FEATURES,
    WINDOW_LEN,
)
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data
from implement.utils.classical_ml.classical_models import get_unsupervised_point_models
from implement.utils.classical_ml.classical_train_eval import get_anomaly_scores
from implement.utils.deep_learning.dl_models import get_unsupervised_models
from implement.utils.deep_learning.dl_train_eval_unsupervised import compute_reconstruction_errors
from torch.utils.data import DataLoader, TensorDataset


def get_labeled_datasets(features: list = None):
    """
    Loads DJI normal flights and ESP32/Simulated attack flights,
    attaching an explicit 'attack_class' column to every sample.
    Excludes simulated normal and hard flights.
    """
    if features is None:
        features = UNSUPERVISED_FEATURES
    dji_df = get_or_preprocess_dji_dataset(filter_length_100=False, features=features)
    dji_df['attack_class'] = 'Normal DJI'

    esp32_df = get_or_preprocess_esp32_dataset()
    
    # Categorize ESP32 & Simulated attack flights
    attack_classes = []
    for fid in esp32_df['flight_id']:
        fid_lower = str(fid).lower()
        if 'esp32' in fid_lower:
            attack_classes.append('Real ESP32')
        elif 'baseline' in fid_lower:
            attack_classes.append('Sim Baseline')
        elif 'easy' in fid_lower:
            attack_classes.append('Sim Easy')
        elif 'medium' in fid_lower:
            attack_classes.append('Sim Medium')
        elif 'hard' in fid_lower:
            attack_classes.append('Sim Hard')
        elif 'geometry' in fid_lower:
            attack_classes.append('Sim Geometry')
        else:
            attack_classes.append('Simulated Attack')

    esp32_df['attack_class'] = attack_classes

    return dji_df, esp32_df


def run_per_class_evaluation(
    model_family: str = "all",
    split_mode: str = "flight",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    window_len: int = WINDOW_LEN,
    k_threshold: float = 3.0,
    features: list = None
):
    """
    Executes per-class anomaly detection evaluation.
    Models train exclusively on Normal DJI flights.
    Evaluation reports Accuracy (%) for each individual flight class.
    """
    if features is None:
        features = UNSUPERVISED_FEATURES
    output_dir = get_output_dir()
    evaluation_dir = output_dir / f"unsupervised_per_class_{model_family}_{split_mode}"
    models_dir = evaluation_dir / 'models'
    tables_dir = evaluation_dir / 'tables'

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("=== EXPERIMENT: PER-CLASS UNSUPERVISED ANOMALY DETECTION ===")
    print(f"=== Model Family: {model_family} | Split Mode: {split_mode} ===")
    print(f"=== Output Dir: {evaluation_dir} ===")
    print("=====================================================================")

    # 1. Load datasets (DJI normal + attack classes)
    dji_df, esp32_df = get_labeled_datasets(features=features)

    # 2. Partition DJI into train/val/test (flight-wise, device-wise, or random)
    if split_mode == 'device' and 'flight_id' in dji_df.columns:
        def extract_device_model(fid):
            fname = str(fid).lower().replace('.csv', '')
            parts = fname.split('_dji_')
            if len(parts) > 1:
                return 'dji_' + parts[1]
            return fname

        dji_df['device_model'] = dji_df['flight_id'].apply(extract_device_model)
        unique_devices = dji_df['device_model'].unique()

        train_devices, temp_devices = train_test_split(unique_devices, test_size=0.3, random_state=42)
        val_devices, test_devices = train_test_split(temp_devices, test_size=0.5, random_state=42)

        dji_train = dji_df[dji_df['device_model'].isin(train_devices)]
        dji_val   = dji_df[dji_df['device_model'].isin(val_devices)]
        dji_test  = dji_df[dji_df['device_model'].isin(test_devices)]
        print(f"Device-level split: {len(train_devices)} Train Models, {len(val_devices)} Val Models, {len(test_devices)} Test Models")

    elif split_mode == 'flight' and 'flight_id' in dji_df.columns:
        unique_flights = dji_df['flight_id'].unique()
        train_flights, temp_flights = train_test_split(unique_flights, test_size=0.3, random_state=42)
        val_flights, test_flights = train_test_split(temp_flights, test_size=0.5, random_state=42)

        dji_train = dji_df[dji_df['flight_id'].isin(train_flights)]
        dji_val   = dji_df[dji_df['flight_id'].isin(val_flights)]
        dji_test  = dji_df[dji_df['flight_id'].isin(test_flights)]
    else:
        dji_train, temp_dji = train_test_split(dji_df, test_size=0.3, random_state=42)
        dji_val, dji_test   = train_test_split(temp_dji, test_size=0.5, random_state=42)

    # 3. Build combined test set: DJI test + attack classes
    test_df = pd.concat([dji_test, esp32_df], ignore_index=True)

    # 4. Fit scaler & imputer on normal train data
    X_train_raw = dji_train[features]
    X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
    joblib.dump(scaler, models_dir / 'scaler.joblib')
    joblib.dump(imputer, models_dir / 'imputer.joblib')

    # Apply scaling to all splits
    X_val_scaled = scaler.transform(imputer.transform(dji_val[features]))
    X_test_scaled = scaler.transform(imputer.transform(test_df[features]))

    all_attack_classes = sorted(list(test_df['attack_class'].unique()))
    print(f"\nUnique classes in evaluation test set: {all_attack_classes}")

    # =========================================================================
    # POINT-WISE MODELS EVALUATION
    # =========================================================================
    if model_family in ['pointwise', 'all']:
        print("\n" + "="*80)
        print("=== EVALUATING POINT-WISE UNSUPERVISED MODELS PER CLASS ===")
        print("="*80)

        pw_models = get_unsupervised_point_models()
        pw_results = []

        for name, model in pw_models.items():
            print(f"Training {name} on {len(X_train_scaled)} clean normal samples...")
            model.fit(X_train_scaled)

            # Compute threshold on normal validation scores
            val_scores = get_anomaly_scores(model, X_val_scaled)
            threshold = val_scores.mean() + k_threshold * val_scores.std()

            row = {'Model': name, 'Threshold': threshold}

            for cls_name in all_attack_classes:
                cls_mask = (test_df['attack_class'] == cls_name)
                cls_X = X_test_scaled[cls_mask]
                scores = get_anomaly_scores(model, cls_X)
                is_anomaly = (scores > threshold)

                if cls_name == 'Normal DJI':
                    # True Negative Rate (Accuracy for normal flights)
                    acc = np.mean(~is_anomaly) * 100.0
                    row[f'{cls_name} (Accuracy %)'] = round(acc, 2)
                else:
                    # True Positive Rate (Accuracy for attacked flights)
                    acc = np.mean(is_anomaly) * 100.0
                    row[f'{cls_name} (Accuracy %)'] = round(acc, 2)

            pw_results.append(row)

        pw_df = pd.DataFrame(pw_results)
        print("\nPoint-wise Per-Class Performance Summary:")
        print(pw_df.to_string(index=False))
        pw_df.to_csv(tables_dir / 'pointwise_per_class_summary.csv', index=False)

    # =========================================================================
    # DEEP LEARNING AUTOENCODERS EVALUATION
    # =========================================================================
    if model_family in ['dl', 'autoencoders', 'all']:
        print("\n" + "="*80)
        print("=== EVALUATING DEEP LEARNING AUTOENCODERS PER CLASS ===")
        print("="*80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def build_seq_df(df):
            X_seq, y_seq, cls_seq = [], [], []
            for _, group in df.groupby('flight_id' if 'flight_id' in df.columns else lambda x: 0):
                feat_vals = scaler.transform(imputer.transform(group[features]))
                labels = group['nature'].values if 'nature' in group.columns else np.zeros(len(group))
                classes = group['attack_class'].values
                if len(group) < window_len:
                    continue
                for i in range(len(group) - window_len + 1):
                    X_seq.append(feat_vals[i : i + window_len])
                    y_seq.append(labels[i + window_len - 1])
                    cls_seq.append(classes[i + window_len - 1])
            return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=int), np.array(cls_seq)

        X_train_seq, _, _ = build_seq_df(dji_train)
        X_val_seq, _, _ = build_seq_df(dji_val)
        X_test_seq, y_test_seq, cls_test_seq = build_seq_df(test_df)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_seq)), batch_size=batch_size, shuffle=False)

        dl_models = get_unsupervised_models(input_dim=len(features), seq_len=window_len)
        dl_results = []

        unique_dl_classes = sorted(list(set(cls_test_seq)))

        for name, model in dl_models.items():
            print(f"\nTraining {name} on {len(X_train_seq)} normal sequences...")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(1, epochs + 1):
                model.train()
                for (batch_x,) in train_loader:
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    recon_x = model(batch_x)
                    loss = torch.nn.functional.mse_loss(recon_x, batch_x)
                    loss.backward()
                    optimizer.step()

            # Dynamic threshold on normal val loader
            val_errors = compute_reconstruction_errors(model, val_loader, device=device)
            threshold = val_errors.mean() + k_threshold * val_errors.std()

            row = {'Model': name, 'Threshold': threshold}

            for cls_name in unique_dl_classes:
                cls_mask = (cls_test_seq == cls_name)
                cls_X = X_test_seq[cls_mask]
                cls_loader = DataLoader(TensorDataset(torch.tensor(cls_X)), batch_size=batch_size, shuffle=False)
                cls_errors = compute_reconstruction_errors(model, cls_loader, device=device)
                is_anomaly = (cls_errors > threshold)

                if cls_name == 'Normal DJI':
                    acc = np.mean(~is_anomaly) * 100.0
                    row[f'{cls_name} (Accuracy %)'] = round(acc, 2)
                else:
                    acc = np.mean(is_anomaly) * 100.0
                    row[f'{cls_name} (Accuracy %)'] = round(acc, 2)

            dl_results.append(row)

        dl_df = pd.DataFrame(dl_results)
        print("\nDeep Learning Per-Class Performance Summary:")
        print(dl_df.to_string(index=False))
        dl_df.to_csv(tables_dir / 'deep_learning_per_class_summary.csv', index=False)

    print("\n=====================================================================")
    print(f"=== PER-CLASS EXPERIMENT COMPLETED. Results saved to {tables_dir} ===")
    print("=====================================================================")
