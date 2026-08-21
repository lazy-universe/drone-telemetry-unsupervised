import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from implement.initial_setup import get_run_subfolder_name
from implement.utils.helper import (
    get_output_dir,
    get_combined_dataset_dir,
    get_or_preprocess_dji_dataset,
    get_or_preprocess_esp32_dataset,
    UNSUPERVISED_FEATURES,
    log_dataset_statistics
)
from implement.utils.dataset_processing.dataset_helper import combine_and_split_unsupervised_data, inject_pca_pc1

from implement.utils.classical_ml.classical_models import get_unsupervised_point_models
from implement.utils.classical_ml.classical_train_eval import (
    train_and_evaluate_unsupervised_point_models,
    cross_validate_unsupervised_point_models,
    plot_roc_curves_unsupervised_point,
    explain_unsupervised_point_models
)


def run_unsupervised_classification_pipeline(
    split_mode: str = 'flight',
    pca: bool = False,
    validate: bool = False,
    cv: int = 5
):
    """
    Executes the unsupervised point-wise anomaly detection telemetry classification pipeline.

    Args:
        split_mode (str): Mode for partitioning DJI flights ('random' or 'flight').
        pca (bool): Inject the first principal component (PC1) as an active training feature.
        validate (bool): Enable cross-validation evaluation. If False, runs normal training & holdout evaluation.
        cv (int): Number of cross-validation folds.
    """
    # 1. Retrieve configured workspace paths
    output_dir = get_output_dir()

    # 2. Compute dynamic subfolder name based on active arguments
    subfolder = "unsupervised_" + (get_run_subfolder_name(pca=pca, validate=validate, cv=cv) or "baseline")

    # Staging paths for this run configuration inside output
    evaluation_dir = output_dir / subfolder
    split_dir = evaluation_dir / 'dataset'
    models_dir = evaluation_dir / 'models'
    plots_dir = evaluation_dir / 'plots'

    # Ensure directories exist
    split_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("=== WORKFLOW: UNSUPERVISED POINT-WISE ANOMALY DETECTION ===")
    print(f"=== Config: PCA={pca} | Split={split_mode} ===")
    if validate:
        print(f"=== Mode: Cross-Validation ({cv}-fold) ===")
    else:
        print("=== Mode: Standard Train-Test Evaluation (80/20 Split) ===")
    print(f"=== Target Output Folder: {evaluation_dir} ===")
    print("=====================================================================")

    # 3. Download, extract, and preprocess DJI dataset
    print("\n[Step 1/5] Loading and preprocessing DJI dataset...")
    dji_df = get_or_preprocess_dji_dataset(filter_length_100=False, features=UNSUPERVISED_FEATURES)
    print(f"  Processed DJI points: {len(dji_df)}")

    # 4. Preprocess ESP32 point logs
    print("\n[Step 2/5] Loading and preprocessing ESP32 dataset...")
    esp32_df = get_or_preprocess_esp32_dataset()
    print(f"  Processed ESP32 points: {len(esp32_df)}")

    # 5. Combine and split datasets
    print(f"\n[Step 3/5] Class balancing, train/test splitting ({split_mode}), and saving combined datasets...")
    intermediate_dir = get_combined_dataset_dir()
    X_train, y_train, X_test, y_test, scaler, imputer = combine_and_split_unsupervised_data(
        dji_df, esp32_df, train_ratio=0.8, split_mode=split_mode, intermediate_dir=intermediate_dir, output_dir=split_dir, features=UNSUPERVISED_FEATURES
    )

    # Save scaler and imputer
    joblib.dump(scaler, models_dir / 'scaler.joblib')
    joblib.dump(imputer, models_dir / 'imputer.joblib')
    print(f"Saved fitted scaler and imputer to: {models_dir}")

    # Log dataset size and dimension details
    log_dataset_statistics(X_train, y_train, X_val=None, y_val=None, X_test=X_test, y_test=y_test)

    # 6. Apply PCA PC1 Injection if specified
    if pca:
        print("\n[Step 3.5/5] Performing PCA PC1 feature injection...")
        X_train, X_test, pca_obj = inject_pca_pc1(X_train, X_test)
        joblib.dump(pca_obj, models_dir / 'pca_model.joblib')
        print(f"Saved fitted PCA model to: {models_dir}")

    # 7. Retrieve point-wise unsupervised models
    print("\n[Step 4/5] Initializing unsupervised anomaly detection models...")
    models = get_unsupervised_point_models()

    feature_names = list(UNSUPERVISED_FEATURES)
    if pca:
        feature_names.append('pca_pc1')

    # 8. Perform evaluation
    if validate:
        print(f"\n[Step 5/5] Evaluating performance using {cv}-fold Cross-Validation...")
        results_cv = cross_validate_unsupervised_point_models(
            models, dji_df, esp32_df, cv=cv, split_mode=split_mode, use_pca=pca, random_state=42
        )

        print("\n" + "="*110)
        print(f"=== FINAL UNSUPERVISED CROSS-VALIDATION PERFORMANCE SUMMARY ({cv}-Fold Mean) ===")
        print(f"{'Anomaly Detection Model':<30}{'Accuracy':<15}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'ROC-AUC':<15}{'PR-AUC':<15}")
        print("-"*110)
        for name, metrics in results_cv.items():
            acc_str = f"{metrics['accuracy_mean']*100:.2f}%"
            prec_str = f"{metrics['precision_mean']*100:.2f}%"
            rec_str = f"{metrics['recall_mean']*100:.2f}%"
            f1_str = f"{metrics['f1_score_mean']*100:.2f}%"
            auc_str = f"{metrics['roc_auc_mean']*100:.2f}%"
            pr_auc_str = f"{metrics['pr_auc_mean']*100:.2f}%"
            print(f"{name:<30}{acc_str:<15}{prec_str:<15}{rec_str:<15}{f1_str:<15}{auc_str:<15}{pr_auc_str:<15}")
        print("========================================================================================================\n")
        return results_cv

    else:
        print("\n[Step 5/5] Evaluating single holdout test set performance...")
        results_holdout = train_and_evaluate_unsupervised_point_models(models, X_train, y_train, X_test, y_test)

        # Save ROC plots
        plot_roc_curves_unsupervised_point(models, X_test, y_test, plots_dir)

        # Generate SHAP explanations for point-wise models (disabled to avoid slow CPU execution)
        # X_train_normal = X_train[y_train == 0]
        # explain_unsupervised_point_models(models, X_train_normal, X_test, feature_names, plots_dir)

        # Save trained models
        for name, model in models.items():
            model_filename = f"{name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, models_dir / model_filename)
        print(f"Saved trained models to: {models_dir}")

        print("\n" + "="*110)
        print("=== FINAL UNSUPERVISED HOLDOUT TEST SET PERFORMANCE SUMMARY ===")
        print(f"{'Anomaly Detection Model':<30}{'Accuracy':<15}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'ROC-AUC':<15}{'PR-AUC':<15}")
        print("-"*110)
        for name, metrics in results_holdout.items():
            acc_str = f"{metrics['accuracy']*100:.2f}%"
            prec_str = f"{metrics['precision']*100:.2f}%"
            rec_str = f"{metrics['recall']*100:.2f}%"
            f1_str = f"{metrics['f1_score']*100:.2f}%"
            auc_str = f"{metrics['roc_auc']*100:.2f}%"
            pr_auc_str = f"{metrics['pr_auc']*100:.2f}%"
            print(f"{name:<30}{acc_str:<15}{prec_str:<15}{rec_str:<15}{f1_str:<15}{auc_str:<15}{pr_auc_str:<15}")
        print("========================================================================================================\n")

        print("=== Confusion Matrices ===")
        for name, metrics in results_holdout.items():
            print(f"  {name}:")
            print(np.array(metrics['confusion_matrix']))
            print()
        print("==========================\n")
        return results_holdout
