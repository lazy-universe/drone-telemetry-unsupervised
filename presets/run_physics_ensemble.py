"""
Preset Script: Physics-Informed Rule Ensemble Evaluation
Evaluates baseline unsupervised ML models standalone vs augmented with deterministic physics rules.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.workflows.per_class_evaluation import get_labeled_datasets
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data
from implement.utils.classical_ml.classical_models import get_unsupervised_point_models
from implement.utils.classical_ml.classical_train_eval import get_anomaly_scores
from implement.utils.helper.physics_rules import (
    compute_physics_rule_flags,
    combine_anomaly_mask_with_physics_rules,
    evaluate_physics_augmented_models,
    PHYSICS_EXTENDED_FEATURES
)
from implement.utils.helper import get_output_dir
from presets.run_unsupervised_pipeline import FEATURE_SETS


def run_physics_ensemble_evaluation(
    feature_preset: str = 'noise_texture_13', 
    threshold_sigma: float = 3.0
) -> pd.DataFrame:
    """
    Evaluates unsupervised models standalone vs combined with deterministic physics-based anomaly rules.
    """
    print("=========================================================================")
    print(f"=== PHYSICS RULE ENSEMBLE BENCHMARK ({feature_preset}) ===")
    print("=========================================================================")

    selected_features = FEATURE_SETS.get(feature_preset, FEATURE_SETS['noise_texture_13'])

    genuine_dji_df, hardware_esp32_df = get_labeled_datasets(features=selected_features)

    unique_flight_identifiers = genuine_dji_df['flight_id'].unique()
    train_flights, remaining_flights = train_test_split(unique_flight_identifiers, test_size=0.3, random_state=42)
    val_flights, test_flights = train_test_split(remaining_flights, test_size=0.5, random_state=42)

    dji_train_set = genuine_dji_df[genuine_dji_df['flight_id'].isin(train_flights)]
    dji_val_set   = genuine_dji_df[genuine_dji_df['flight_id'].isin(val_flights)]
    dji_test_set  = genuine_dji_df[genuine_dji_df['flight_id'].isin(test_flights)]

    composite_test_set = pd.concat([dji_test_set, hardware_esp32_df], ignore_index=True)

    # Fit Imputer and Scaler on normal training flights only
    raw_train_matrix = dji_train_set[selected_features]
    scaled_train_matrix, _, fitted_scaler, fitted_imputer = impute_and_scale_data(raw_train_matrix, raw_train_matrix)
    scaled_val_matrix  = fitted_scaler.transform(fitted_imputer.transform(dji_val_set[selected_features]))
    scaled_test_matrix = fitted_scaler.transform(fitted_imputer.transform(composite_test_set[selected_features]))

    unsupervised_point_models = get_unsupervised_point_models()
    model_predictions_dict = {}

    print(">>> Training Pointwise Unsupervised Models on Normal Flights...")
    for model_name, point_model in unsupervised_point_models.items():
        print(f"  Fitting {model_name}...")
        point_model.fit(scaled_train_matrix)

        validation_anomaly_scores = get_anomaly_scores(point_model, scaled_val_matrix)
        decision_threshold = np.mean(validation_anomaly_scores) + threshold_sigma * np.std(validation_anomaly_scores)

        test_anomaly_scores = get_anomaly_scores(point_model, scaled_test_matrix)
        model_predictions_dict[model_name] = test_anomaly_scores > decision_threshold

    print(">>> Evaluating Standalone Models vs Physics Rules Augmented Ensembles...")
    comparison_dataframe = evaluate_physics_augmented_models(composite_test_set, model_predictions_dict)

    output_directory = get_output_dir() / 'physics_ensemble'
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_output_path = output_directory / f"physics_ensemble_evaluation_{feature_preset}.csv"
    comparison_dataframe.to_csv(csv_output_path, index=False)

    print("\nPHYSICS RULE ENSEMBLE RESULTS:")
    print(comparison_dataframe.to_string(index=False))
    print(f"\n✅ Saved Physics Ensemble results to: {csv_output_path}")

    return comparison_dataframe


def main():
    cli_parser = argparse.ArgumentParser(description="Physics Rule Ensemble Evaluation")
    cli_parser.add_argument(
        "--preset", 
        choices=["noise_texture_13", "physics_extended_16", "baseline_10"], 
        default="noise_texture_13", 
        help="Feature preset to evaluate"
    )
    cli_parser.add_argument(
        "--k-thresh", 
        type=float, 
        default=3.0, 
        help="K-sigma decision threshold factor on validation score distribution"
    )
    parsed_args = cli_parser.parse_args()

    run_physics_ensemble_evaluation(
        feature_preset=parsed_args.preset, 
        threshold_sigma=parsed_args.k_thresh
    )


if __name__ == "__main__":
    main()
