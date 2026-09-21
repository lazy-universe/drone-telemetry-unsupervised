"""
Physics-Informed Anomaly Detection Module
Provides deterministic aerodynamic / kinematic rule-based flags and prediction error (PE) distribution statistics.
These serve as a deterministic guardrail / ensemble layer on top of unsupervised statistical & deep learning models.
"""

from typing import Dict, List, Union
import numpy as np
import pandas as pd

# Prediction error distribution features
PE_DISTRIBUTION_FEATURES: List[str] = [
    'pe_window_mean',
    'pe_window_var',
    'pe_window_skew'
]

# Extended feature set incorporating physics & noise features
PHYSICS_EXTENDED_FEATURES: List[str] = [
    'height', 'ground_speed', 'vertical_speed', 'acceleration', 'turn_rate',
    'path_curvature', 'heading_speed_consistency', 'motion_smoothness',
    'prediction_error', 'yaw_acceleration',
    'prediction_error_autocorrelation', 'position_residual_std', 'speed_spectral_entropy',
    'pe_window_mean', 'pe_window_var', 'pe_window_skew'
]


def compute_physics_rule_flags(telemetry_df: pd.DataFrame) -> np.ndarray:
    """
    Computes deterministic physics-based anomaly flags based on kinematic and aerodynamic thresholds.
    
    Deterministic Rules:
    1. Prediction Error > 10.0m sustained for 3+ consecutive samples (6 seconds at 2Hz).
    2. Heading-Speed Inconsistency > 90 deg at speed > 2.0 m/s sustained for 2+ consecutive samples.
    3. Extreme Yaw Acceleration (instantaneous heading flip): |yaw_acceleration| > 500.0 deg/s^2.
    
    Args:
        telemetry_df (pd.DataFrame): DataFrame containing drone telemetry features.
        
    Returns:
        np.ndarray (bool): Boolean mask where True represents a detected physics violation.
    """
    anomaly_flags = np.zeros(len(telemetry_df), dtype=bool)

    # Rule 1: Sustained high prediction error (trajectory divergence)
    if 'prediction_error' in telemetry_df.columns:
        pe_sustained_violation = (
            (telemetry_df['prediction_error'] > 10.0)
            .rolling(window=3, min_periods=3)
            .sum() >= 3
        )
        anomaly_flags |= pe_sustained_violation.fillna(False).values

    # Rule 2: Sustained severe heading-speed mismatch during forward motion
    if 'heading_speed_consistency' in telemetry_df.columns and 'ground_speed' in telemetry_df.columns:
        hsc_mismatch_at_speed = (
            (telemetry_df['heading_speed_consistency'] > 90.0) & 
            (telemetry_df['ground_speed'] > 2.0)
        )
        hsc_sustained_violation = (
            hsc_mismatch_at_speed
            .rolling(window=2, min_periods=2)
            .sum() >= 2
        )
        anomaly_flags |= hsc_sustained_violation.fillna(False).values

    # Rule 3: Unphysical yaw acceleration spike (instantaneous heading change)
    if 'yaw_acceleration' in telemetry_df.columns:
        extreme_yaw_acc = telemetry_df['yaw_acceleration'].abs() > 500.0
        anomaly_flags |= extreme_yaw_acc.fillna(False).values

    return anomaly_flags


# Backward-compatible alias
physics_rule_flags = compute_physics_rule_flags


def extract_pe_distribution_features(
    prediction_error_series: Union[pd.Series, np.ndarray], 
    window_length: int = 20
) -> Dict[str, np.ndarray]:
    """
    Computes rolling window-level prediction error distribution statistics:
    - pe_window_mean: Rolling mean of prediction error.
    - pe_window_var: Rolling variance of prediction error.
    - pe_window_skew: Rolling skewness of prediction error.
    
    Args:
        prediction_error_series (pd.Series or np.ndarray): Prediction error sequence.
        window_length (int): Rolling window length (default: 20 samples = 10s at 2Hz).
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing the three engineered feature arrays.
    """
    pe_series = pd.Series(prediction_error_series)
    return {
        'pe_window_mean': pe_series.rolling(window=window_length, min_periods=1).mean().fillna(0.0).values,
        'pe_window_var': pe_series.rolling(window=window_length, min_periods=2).var(ddof=0).fillna(0.0).values,
        'pe_window_skew': pe_series.rolling(window=window_length, min_periods=3).skew().fillna(0.0).values,
    }


def combine_anomaly_mask_with_physics_rules(
    model_anomaly_mask: np.ndarray, 
    telemetry_df: pd.DataFrame
) -> np.ndarray:
    """
    Combines statistical/deep learning model anomaly predictions with deterministic physics rules via logical OR.
    
    Args:
        model_anomaly_mask (np.ndarray): Boolean mask of model anomaly predictions (True = anomaly).
        telemetry_df (pd.DataFrame): DataFrame containing raw/engineered telemetry for rule evaluation.
        
    Returns:
        np.ndarray: Combined boolean anomaly mask.
    """
    physics_flags = compute_physics_rule_flags(telemetry_df)
    return model_anomaly_mask | physics_flags


def evaluate_physics_augmented_models(
    labeled_test_df: pd.DataFrame, 
    model_predictions_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Evaluates baseline models vs physics-augmented ensemble variants on labeled multi-class test data.
    
    Args:
        labeled_test_df (pd.DataFrame): Labeled test set containing 'attack_class'.
        model_predictions_dict (Dict[str, np.ndarray]): Mapping from model_name to boolean prediction mask.
        
    Returns:
        pd.DataFrame: Comprehensive accuracy/TNR comparison table.
    """
    attack_classes = sorted(list(labeled_test_df['attack_class'].unique()))
    evaluation_records = []

    # Standalone Deterministic Physics Rules
    standalone_physics_mask = compute_physics_rule_flags(labeled_test_df)
    rules_record = {'Model': 'Physics Rules Standalone', 'Variant': 'Deterministic Rules Only'}
    for cls_name in attack_classes:
        class_mask = (labeled_test_df['attack_class'] == cls_name)
        predicted_anomalies = standalone_physics_mask[class_mask]
        accuracy = (
            np.mean(~predicted_anomalies) * 100.0 if cls_name == 'Normal DJI' 
            else np.mean(predicted_anomalies) * 100.0
        )
        rules_record[cls_name] = round(accuracy, 2)
    evaluation_records.append(rules_record)

    # Models: Baseline vs Physics-Augmented
    for model_name, raw_anomaly_mask in model_predictions_dict.items():
        # Baseline model without rules
        base_record = {'Model': model_name, 'Variant': 'Model Baseline'}
        for cls_name in attack_classes:
            class_mask = (labeled_test_df['attack_class'] == cls_name)
            predicted_anomalies = raw_anomaly_mask[class_mask]
            accuracy = (
                np.mean(~predicted_anomalies) * 100.0 if cls_name == 'Normal DJI' 
                else np.mean(predicted_anomalies) * 100.0
            )
            base_record[cls_name] = round(accuracy, 2)
        evaluation_records.append(base_record)

        # Augmented: Model + Physics Rules (Logical OR)
        augmented_mask = combine_anomaly_mask_with_physics_rules(raw_anomaly_mask, labeled_test_df)
        aug_record = {'Model': model_name, 'Variant': 'Model + Physics Rules (OR)'}
        for cls_name in attack_classes:
            class_mask = (labeled_test_df['attack_class'] == cls_name)
            predicted_anomalies = augmented_mask[class_mask]
            accuracy = (
                np.mean(~predicted_anomalies) * 100.0 if cls_name == 'Normal DJI' 
                else np.mean(predicted_anomalies) * 100.0
            )
            aug_record[cls_name] = round(accuracy, 2)
        evaluation_records.append(aug_record)

    return pd.DataFrame(evaluation_records)
