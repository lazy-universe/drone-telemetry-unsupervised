"""
Script: Compute Feature Value Ranges (Min, Max, Mean, Std, P1, P99)
Calculates exact empirical bounds for all 10 features across:
  1. Normal DJI Flights
  2. Real ESP32 Spoofing Attacks
  3. Simulated Attacks & Sim Normal
"""
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from implement.utils.helper import (
    get_or_preprocess_dji_dataset,
    get_or_preprocess_esp32_dataset,
    get_output_dir,
    UNSUPERVISED_FEATURES
)

def main():
    print("Loading datasets...")
    dji_df = get_or_preprocess_dji_dataset(filter_length_100=False, features=UNSUPERVISED_FEATURES)
    esp32_df = get_or_preprocess_esp32_dataset()

    print("\n" + "=" * 100)
    print("=== FEATURE VALUE RANGES: NORMAL DJI FLIGHTS (Training Base) ===")
    print("=" * 100)

    rows_dji = []
    for feat in UNSUPERVISED_FEATURES:
        if feat in dji_df.columns:
            vals = dji_df[feat].dropna().values
            rows_dji.append({
                'Feature': feat,
                'Min': round(float(np.min(vals)), 4),
                'Max': round(float(np.max(vals)), 4),
                'Mean': round(float(np.mean(vals)), 4),
                'Std': round(float(np.std(vals)), 4),
                'P1 (1%)': round(float(np.percentile(vals, 1)), 4),
                'P99 (99%)': round(float(np.percentile(vals, 99)), 4),
            })
    df_dji_stats = pd.DataFrame(rows_dji)
    print(df_dji_stats.to_string(index=False))

    print("\n" + "=" * 100)
    print("=== FEATURE VALUE RANGES: REAL ESP32 & SPOOFED ATTACKS ===")
    print("=" * 100)

    rows_esp = []
    for feat in UNSUPERVISED_FEATURES:
        if feat in esp32_df.columns:
            vals = esp32_df[feat].dropna().values
            rows_esp.append({
                'Feature': feat,
                'Min': round(float(np.min(vals)), 4),
                'Max': round(float(np.max(vals)), 4),
                'Mean': round(float(np.mean(vals)), 4),
                'Std': round(float(np.std(vals)), 4),
                'P1 (1%)': round(float(np.percentile(vals, 1)), 4),
                'P99 (99%)': round(float(np.percentile(vals, 99)), 4),
            })
    df_esp_stats = pd.DataFrame(rows_esp)
    print(df_esp_stats.to_string(index=False))

    # Save summary tables to CSV
    out_dir = get_output_dir() / 'feature_ranges'
    out_dir.mkdir(parents=True, exist_ok=True)
    df_dji_stats.to_csv(out_dir / 'normal_dji_feature_ranges.csv', index=False)
    df_esp_stats.to_csv(out_dir / 'spoofed_feature_ranges.csv', index=False)
    print(f"\n✅ Saved feature range CSV files to: {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute empirical feature value ranges across normal and spoofed flights")
    parser.parse_args()
    main()
