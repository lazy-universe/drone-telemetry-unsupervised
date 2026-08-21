import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import shap

from implement.utils.helper import UNSUPERVISED_FEATURES
from implement.utils.dataset_processing.dataset_helper import (
    class_balance_and_combine,
    impute_and_scale_data,
    inject_pca_pc1
)


def get_anomaly_scores(model, X):
    """
    Computes anomaly scores for various models where higher score = more anomalous.
    """
    if type(model).__name__ in ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM']:
        return -model.decision_function(X)
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    if hasattr(model, 'score_samples'):
        return -model.score_samples(X)
    return model.predict(X)


def train_and_evaluate_unsupervised_point_models(
    models, X_train, y_train, X_test, y_test, k_threshold=3.0
):
    """
    Fits unsupervised point-wise anomaly detection models on normal training data only (y_train == 0).
    Evaluates dynamic threshold (mean + k*std on normal training scores) on test set.
    """
    X_train_normal = X_train[y_train == 0]
    print(f"Training unsupervised point-wise models on clean normal data only (n_samples={X_train_normal.shape[0]})...")

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_normal)

        # Anomaly scores on normal training data to calculate threshold
        train_scores = get_anomaly_scores(model, X_train_normal)
        threshold = train_scores.mean() + k_threshold * train_scores.std()

        # Anomaly scores on test set
        test_scores = get_anomaly_scores(model, X_test)
        y_pred = (test_scores > threshold).astype(int)

        prec_vals, rec_vals, _ = precision_recall_curve(y_test, test_scores)
        pr_auc = auc(rec_vals, prec_vals)

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, test_scores),
            'pr_auc': pr_auc,
            'threshold': float(threshold),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        print(f"  {name} Results:")
        print(f"    Accuracy : {results[name]['accuracy']*100:.2f}%")
        print(f"    Precision: {results[name]['precision']*100:.2f}%")
        print(f"    Recall   : {results[name]['recall']*100:.2f}%")
        print(f"    F1-score : {results[name]['f1_score']*100:.2f}%")
        print(f"    ROC-AUC  : {results[name]['roc_auc']*100:.2f}%")
        print(f"    PR-AUC   : {results[name]['pr_auc']*100:.2f}%")

    return results


def plot_roc_curves_unsupervised_point(models, X_test, y_test, output_dir):
    """
    Plots Receiver Operating Characteristic (ROC) curves for unsupervised point-wise models.
    """
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        scores = get_anomaly_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_val:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Unsupervised Point-wise Models: ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'roc_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves plot to: {plot_path}")


def explain_unsupervised_point_models(models, X_train_normal, X_test, feature_names, output_dir):
    """
    Generates SHAP beeswarm plots for point-wise unsupervised models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subsample test set for SHAP to make execution fast
    sample_size = min(500, len(X_test))
    indices = np.random.choice(len(X_test), size=sample_size, replace=False)
    X_test_sample = X_test[indices]

    background_size = min(100, len(X_train_normal))
    bg_indices = np.random.choice(len(X_train_normal), size=background_size, replace=False)
    X_bg = X_train_normal[bg_indices]

    for name, model in models.items():
        try:
            print(f"Generating SHAP explanations for {name}...")
            predict_fn = lambda x: get_anomaly_scores(model, x)
            explainer = shap.Explainer(predict_fn, X_bg)
            shap_values = explainer(X_test_sample)
            if feature_names is not None and len(feature_names) == X_test_sample.shape[1]:
                shap_values.feature_names = feature_names

            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, show=False)
            plt.title(f"SHAP Feature Importance: {name}")
            plot_path = output_dir / f"shap_{name.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved SHAP plot for {name} to {plot_path}")
        except Exception as e:
            print(f"  SHAP explanation skipped for {name}: {e}")


def cross_validate_unsupervised_point_models(
    models, dji_df, esp32_df, cv=5, split_mode='random', use_pca=False, random_state=42, features=None, k_threshold=3.0
):
    """
    Performs CV evaluation for unsupervised point-wise models.
    """
    if features is None:
        features = UNSUPERVISED_FEATURES

    print(f"\nRunning {cv}-fold Cross-Validation for Unsupervised Point-wise Models (Split Mode: {split_mode})...")

    cv_metrics = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': [], 'pr_auc': []} for name in models.keys()}

    if split_mode == 'flight' and 'flight_id' in dji_df.columns:
        unique_flights = dji_df['flight_id'].unique()
        kf_dji = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        dji_splits = list(kf_dji.split(unique_flights))

        esp32_clean = esp32_df.drop(columns=['flight_id'], errors='ignore')

        for fold_idx in range(cv):
            print(f"  Processing Fold {fold_idx + 1}/{cv}...")
            dji_train_idx, dji_val_idx = dji_splits[fold_idx]
            train_flights = unique_flights[dji_train_idx]
            val_flights = unique_flights[dji_val_idx]

            dji_train = dji_df[dji_df['flight_id'].isin(train_flights)].drop(columns=['flight_id'])
            dji_val = dji_df[dji_df['flight_id'].isin(val_flights)].drop(columns=['flight_id'])

            # Val combines dji_val (nature=0) and esp32_clean (nature=1)
            val_df = class_balance_and_combine(dji_val, esp32_clean, balance_ratio=None, random_state=random_state)

            X_train = dji_train[features]
            X_val = val_df[features]
            y_val = val_df['nature'].to_numpy()

            X_train_scaled, X_val_scaled, scaler, imputer = impute_and_scale_data(X_train, X_val)

            if use_pca:
                X_train_scaled, X_val_scaled, pca = inject_pca_pc1(X_train_scaled, X_val_scaled)

            for name, model in models.items():
                m = clone(model)
                m.fit(X_train_scaled)
                train_scores = get_anomaly_scores(m, X_train_scaled)
                threshold = train_scores.mean() + k_threshold * train_scores.std()

                val_scores = get_anomaly_scores(m, X_val_scaled)
                y_pred = (val_scores > threshold).astype(int)

                prec_vals, rec_vals, _ = precision_recall_curve(y_val, val_scores)
                pr_auc = auc(rec_vals, prec_vals)

                cv_metrics[name]['accuracy'].append(accuracy_score(y_val, y_pred))
                cv_metrics[name]['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                cv_metrics[name]['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                cv_metrics[name]['f1_score'].append(f1_score(y_val, y_pred, zero_division=0))
                cv_metrics[name]['roc_auc'].append(roc_auc_score(y_val, val_scores))
                cv_metrics[name]['pr_auc'].append(pr_auc)
    else:
        dji_clean = dji_df.drop(columns=['flight_id'], errors='ignore')
        esp32_clean = esp32_df.drop(columns=['flight_id'], errors='ignore')

        kf_dji = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        dji_splits = list(kf_dji.split(dji_clean))

        for fold_idx in range(cv):
            print(f"  Processing Fold {fold_idx + 1}/{cv}...")
            dji_train_idx, dji_val_idx = dji_splits[fold_idx]
            dji_train = dji_clean.iloc[dji_train_idx]
            dji_val = dji_clean.iloc[dji_val_idx]

            val_df = class_balance_and_combine(dji_val, esp32_clean, balance_ratio=None, random_state=random_state)

            X_train = dji_train[features]
            X_val = val_df[features]
            y_val = val_df['nature'].to_numpy()

            X_train_scaled, X_val_scaled, scaler, imputer = impute_and_scale_data(X_train, X_val)

            if use_pca:
                X_train_scaled, X_val_scaled, pca = inject_pca_pc1(X_train_scaled, X_val_scaled)

            for name, model in models.items():
                m = clone(model)
                m.fit(X_train_scaled)
                train_scores = get_anomaly_scores(m, X_train_scaled)
                threshold = train_scores.mean() + k_threshold * train_scores.std()

                val_scores = get_anomaly_scores(m, X_val_scaled)
                y_pred = (val_scores > threshold).astype(int)

                prec_vals, rec_vals, _ = precision_recall_curve(y_val, val_scores)
                pr_auc = auc(rec_vals, prec_vals)

                cv_metrics[name]['accuracy'].append(accuracy_score(y_val, y_pred))
                cv_metrics[name]['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                cv_metrics[name]['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                cv_metrics[name]['f1_score'].append(f1_score(y_val, y_pred, zero_division=0))
                cv_metrics[name]['roc_auc'].append(roc_auc_score(y_val, val_scores))
                cv_metrics[name]['pr_auc'].append(pr_auc)

    results = {}
    for name in models.keys():
        results[name] = {
            'accuracy_mean': np.mean(cv_metrics[name]['accuracy']),
            'accuracy_std': np.std(cv_metrics[name]['accuracy']),
            'precision_mean': np.mean(cv_metrics[name]['precision']),
            'precision_std': np.std(cv_metrics[name]['precision']),
            'recall_mean': np.mean(cv_metrics[name]['recall']),
            'recall_std': np.std(cv_metrics[name]['recall']),
            'f1_score_mean': np.mean(cv_metrics[name]['f1_score']),
            'f1_score_std': np.std(cv_metrics[name]['f1_score']),
            'roc_auc_mean': np.mean(cv_metrics[name]['roc_auc']),
            'roc_auc_std': np.std(cv_metrics[name]['roc_auc']),
            'pr_auc_mean': np.mean(cv_metrics[name]['pr_auc']),
            'pr_auc_std': np.std(cv_metrics[name]['pr_auc'])
        }

    return results
