import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.utils.helper import get_output_dir
OUT_DIR = get_output_dir() / 'feature_importance'
ARTIFACT_DIR = OUT_DIR / 'plots'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

perm_csv = OUT_DIR / 'permutation_importance.csv'
shap_csv = OUT_DIR / 'shap_mean_absolute.csv'

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif', 'figure.autolayout': True})

# 1. GENERATE PERMUTATION IMPORTANCE PLOT
if perm_csv.exists():
    df_perm = pd.read_csv(perm_csv)
    # Focus on Sim Hard, Sim Medium, Normal DJI, Real ESP32
    df_perm['Overall Importance (Mean Delta)'] = df_perm.select_dtypes(include=np.number).mean(axis=1)
    df_perm = df_perm.sort_values(by='Overall Importance (Mean Delta)', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_perm['Feature'], df_perm['Normal DJI delta'], color='#1f77b4', alpha=0.85, label='Normal DJI Delta')
    
    # Highlight top features
    for bar in bars:
        width = bar.get_width()
        if width > 10:
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_xlabel('Accuracy Impact Delta (%) when Feature Shuffled', fontsize=11, fontweight='bold')
    ax.set_ylabel('Engineered Telemetry Feature', fontsize=11, fontweight='bold')
    ax.set_title('Permutation Feature Importance (Impact on Normal DJI Boundary)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    img_path = ARTIFACT_DIR / 'permutation_importance_bar.png'
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")

# 2. GENERATE SHAP MEAN ABSOLUTE VALUE PLOT (Log Scale for wide dynamic range)
if shap_csv.exists():
    df_shap = pd.read_csv(shap_csv)
    df_shap_melted = df_shap.set_index('Class').T
    df_shap_melted['Mean SHAP'] = df_shap_melted.mean(axis=1)
    df_shap_melted = df_shap_melted.sort_values(by='Mean SHAP', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot horizontal bar chart for mean SHAP across all classes
    bars = ax.barh(df_shap_melted.index, df_shap_melted['Mean SHAP'], color='#2ca02c', alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlabel('Mean Absolute SHAP Value (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Engineered Telemetry Feature', fontsize=11, fontweight='bold')
    ax.set_title('Mean Absolute SHAP Feature Importance across Flight Classes', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.15, bar.get_y() + bar.get_height()/2, f'{width:.1f}', va='center', ha='left', fontsize=9)

    img_path = ARTIFACT_DIR / 'shap_mean_absolute_bar.png'
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")

# 3. GENERATE HEATMAP OF PERMUTATION DELTA PER CLASS
if perm_csv.exists():
    df_perm = pd.read_csv(perm_csv).set_index('Feature')
    cols = [c for c in df_perm.columns if 'delta' in c]
    clean_cols = [c.replace(' delta', '') for c in cols]
    df_heat = df_perm[cols]
    df_heat.columns = clean_cols

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(df_heat, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy Delta (%)'}, ax=ax, linewidths=0.5)
    ax.set_title('Permutation Importance Heatmap across Flight Classes', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Flight & Attack Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=11, fontweight='bold')

    img_path = ARTIFACT_DIR / 'feature_importance_heatmap.png'
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")

print("All feature importance graphs generated successfully!")
