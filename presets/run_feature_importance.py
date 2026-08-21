"""
Feature Importance Analysis for Unsupervised Anomaly Detection (Accelerated SHAP & Permutation)
Runs: 
 (1) Permutation Importance (sorted)
 (2) Per-class SHAP values (sorted)
 (3) Generates PNG charts directly in implement/output/feature_importance/
"""
import sys, warnings, argparse
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from implement.utils.helper import (
    get_or_preprocess_dji_dataset,
    get_or_preprocess_esp32_dataset,
    
)
from implement.utils.classical_ml.classical_models import GMMWrapper
from implement.utils.classical_ml.classical_train_eval import get_anomaly_scores
from implement.utils.helper.features import INTERSECTING_FEATURES

OUT_DIR = Path('implement/output/feature_importance')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_MAP = {
    'esp32': 'Real ESP32', 'baseline': 'Sim Baseline', 'easy': 'Sim Easy',
    'medium': 'Sim Medium', 'hard': 'Sim Hard', 'geometry': 'Sim Geometry',
}

# All 13 Features Set
ALL_13_FEATURES = INTERSECTING_FEATURES.copy()

def main():
    parser = argparse.ArgumentParser(description="Accelerated Feature Importance Analysis (SHAP & Permutation)")
    parser.add_argument(
        "--include-pred-error",
        action="store_true",
        help="Include prediction_error in feature set"
    )
    args = parser.parse_args()

    target_features = ALL_13_FEATURES.copy()
    if args.include_pred_error and 'prediction_error' not in target_features:
        target_features.append('prediction_error')

    print("=" * 80)
    print("=== ACCELERATED FEATURE IMPORTANCE ANALYSIS (GMM + SHAP + PERMUTATION) ===")
    print(f"=== Features Analyzed ({len(target_features)}): {target_features} ===")
    print("=" * 80)

    # ── 1. LOAD & SPLIT DATA ───────────────────────────────────────────────────────
    print("\nLoading datasets...")
    dji_df = get_or_preprocess_dji_dataset(filter_length_100=False, features=target_features)
    dji_df['attack_class'] = 'Normal DJI'
    esp32_df = get_or_preprocess_esp32_dataset()
    esp32_df['attack_class'] = esp32_df['flight_id'].apply(
        lambda fid: next((v for k, v in ATTACK_MAP.items() if k in str(fid).lower()), 'Other')
    )

    unique_flights = dji_df['flight_id'].unique()
    train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=42)
    val_fl, test_fl = train_test_split(temp, test_size=0.5, random_state=42)
    dji_train = dji_df[dji_df['flight_id'].isin(train_fl)]
    dji_test  = dji_df[dji_df['flight_id'].isin(test_fl)]

    test_df = pd.concat([dji_test, esp32_df], ignore_index=True)
    all_classes = sorted(test_df['attack_class'].unique())

    # ── 2. FIT SCALER + MODEL ─────────────────────────────────────────────────────
    imp = SimpleImputer(strategy='mean')
    scl = StandardScaler()
    X_train = scl.fit_transform(imp.fit_transform(dji_train[target_features]))
    X_test  = scl.transform(imp.transform(test_df[target_features]))

    model = GMMWrapper(n_components=3, covariance_type='full', random_state=42)
    model.fit(X_train)
    train_scores = get_anomaly_scores(model, X_train)
    threshold    = train_scores.mean() + 3.0 * train_scores.std()

    # Baseline per-class detection accuracy
    print(f"\n── BASELINE PERFORMANCE ({len(target_features)} features) ──")
    baseline_rates = {}
    for cls in all_classes:
        mask   = test_df['attack_class'] == cls
        scores = get_anomaly_scores(model, X_test[mask])
        is_anomaly = scores > threshold
        if cls in ['Normal DJI', 'Sim Normal']:
            acc = np.mean(~is_anomaly) * 100.0
        else:
            acc = np.mean(is_anomaly) * 100.0
        baseline_rates[cls] = acc
        print(f"  {cls:30s}: Accuracy = {acc:.1f}%")

    # ── 3. PERMUTATION IMPORTANCE (SORTED) ─────────────────────────────────────────
    print("\n── PERMUTATION IMPORTANCE (shuffling each feature in test set) ──")
    perm_rows = []
    rng = np.random.default_rng(42)

    for feat_idx, feat_name in enumerate(target_features):
        X_test_shuffled = X_test.copy()
        X_test_shuffled[:, feat_idx] = rng.permutation(X_test_shuffled[:, feat_idx])

        row = {'Feature': feat_name}
        for cls in all_classes:
            mask   = test_df['attack_class'] == cls
            scores = get_anomaly_scores(model, X_test_shuffled[mask])
            is_anomaly = scores > threshold
            if cls in ['Normal DJI', 'Sim Normal']:
                acc = np.mean(~is_anomaly) * 100.0
            else:
                acc = np.mean(is_anomaly) * 100.0
            delta = baseline_rates[cls] - acc
            row[f'{cls} delta'] = round(delta, 2)
        perm_rows.append(row)

    perm_df = pd.DataFrame(perm_rows)
    # Sort permutation importance by Normal DJI delta descending
    perm_df['Sort_Metric'] = perm_df['Normal DJI delta'].abs()
    perm_df = perm_df.sort_values(by='Sort_Metric', ascending=False).drop(columns=['Sort_Metric'])
    perm_df.to_csv(OUT_DIR / 'permutation_importance.csv', index=False)
    print(f"Saved sorted: {OUT_DIR / 'permutation_importance.csv'}")

    # ── 4. ACCELERATED SHAP ANALYSIS (SORTED) ──────────────────────────────────────
    print("\n── ACCELERATED SHAP ANALYSIS (subsampled 50 background x 100 test) ──")
    X_background = shap.sample(X_train, 50, random_state=42)
    explainer = shap.KernelExplainer(model.decision_function, X_background)

    shap_results = {}
    for cls in all_classes:
        mask   = test_df['attack_class'] == cls
        X_cls  = X_test[mask]
        n_samples = min(100, len(X_cls))
        idx = rng.choice(len(X_cls), size=n_samples, replace=False)
        X_cls_sub = X_cls[idx]

        print(f"  Computing SHAP for {cls} (n={n_samples})...")
        shap_vals = explainer.shap_values(X_cls_sub, silent=True)
        shap_results[cls] = shap_vals

        safe_name = cls.replace(' ', '_').replace('/', '_')
        shap.summary_plot(
            shap_vals, X_cls_sub,
            feature_names=target_features,
            show=False, plot_type='dot', max_display=len(target_features),
        )
        plt.title(f'SHAP Beeswarm — {cls}', fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'shap_beeswarm_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    shap_mean_rows = []
    for cls, shap_vals in shap_results.items():
        row = {'Class': cls}
        for fi, feat in enumerate(target_features):
            row[feat] = round(float(np.abs(shap_vals[:, fi]).mean()), 4)
        shap_mean_rows.append(row)

    shap_df = pd.DataFrame(shap_mean_rows)
    # Sort SHAP columns by overall mean SHAP score descending
    feat_cols = [c for c in shap_df.columns if c != 'Class']
    mean_shap_scores = shap_df[feat_cols].mean(axis=0).sort_values(ascending=False)
    sorted_cols = ['Class'] + list(mean_shap_scores.index)
    shap_df = shap_df[sorted_cols]
    shap_df.to_csv(OUT_DIR / 'shap_mean_absolute.csv', index=False)
    print(f"Saved sorted: {OUT_DIR / 'shap_mean_absolute.csv'}")

    # ── 5. DIRECT PNG CHARTS GENERATION IN OUT_DIR ────────────────────────────────
    print("\n── GENERATING SORTED VISUALIZATION PNG CHARTS ──")
    
    # Chart A: Permutation Importance Bar Chart (Sorted)
    df_p_plot = perm_df.sort_values(by='Normal DJI delta', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_p_plot['Feature'], df_p_plot['Normal DJI delta'], color='#1f77b4', alpha=0.85)
    ax.set_xlabel('Accuracy Impact Delta (%) when Feature Shuffled', fontsize=11, fontweight='bold')
    ax.set_ylabel('Engineered Telemetry Feature', fontsize=11, fontweight='bold')
    ax.set_title('Permutation Feature Importance (Sorted by Normal DJI Delta)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    for bar in bars:
        width = bar.get_width()
        if abs(width) > 2.0:
            ax.text(width + (0.5 if width >= 0 else -1.5), bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', ha='left', fontsize=9, fontweight='bold')
    plt.savefig(OUT_DIR / 'permutation_importance_bar.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR / 'permutation_importance_bar.png'}")

    # Chart B: SHAP Mean Absolute Values (Sorted, Log Scale)
    df_s_plot = shap_df.set_index('Class').T
    df_s_plot['Mean_SHAP'] = df_s_plot.mean(axis=1)
    df_s_plot = df_s_plot.sort_values(by='Mean_SHAP', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df_s_plot.index, df_s_plot['Mean_SHAP'], color='#2ca02c', alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlabel('Mean Absolute SHAP Value (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Engineered Telemetry Feature', fontsize=11, fontweight='bold')
    ax.set_title('Mean Absolute SHAP Feature Importance (Sorted)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.15, bar.get_y() + bar.get_height()/2, f'{width:.1f}', va='center', ha='left', fontsize=9)
    plt.savefig(OUT_DIR / 'shap_mean_absolute_bar.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR / 'shap_mean_absolute_bar.png'}")

    # Chart C: Permutation Importance Heatmap (Sorted)
    df_h_plot = perm_df.set_index('Feature')
    cols_h = [c for c in df_h_plot.columns if 'delta' in c]
    df_h_clean = df_h_plot[cols_h]
    df_h_clean.columns = [c.replace(' delta', '') for c in cols_h]

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(df_h_clean, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy Delta (%)'}, ax=ax, linewidths=0.5)
    ax.set_title('Permutation Importance Heatmap across Flight Classes (Sorted)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Flight & Attack Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Engineered Feature (Sorted)', fontsize=11, fontweight='bold')
    plt.savefig(OUT_DIR / 'feature_importance_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR / 'feature_importance_heatmap.png'}")

    print(f"\n✅ Pipeline completed! All sorted CSVs and PNG charts generated directly in: {OUT_DIR}")

if __name__ == "__main__":
    main()
