"""
Informed Ablation Study — tests 8 feature subsets on GMM + Isolation Forest.
ZERO data leakage (Sim Normal is TEST ONLY). Output is Accuracy (%).
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

from implement.utils.helper import (
    get_or_preprocess_dji_dataset,
    get_or_preprocess_esp32_dataset,
    
    UNSUPERVISED_FEATURES,
)
from implement.utils.classical_ml.classical_models import GMMWrapper
from implement.utils.classical_ml.classical_train_eval import get_anomaly_scores

OUT_DIR = Path('implement/output/ablation_study')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_MAP = {
    'esp32': 'Real ESP32', 'baseline': 'Sim Baseline', 'easy': 'Sim Easy',
    'medium': 'Sim Medium', 'hard': 'Sim Hard', 'geometry': 'Sim Geometry',
}

SUBSETS = {
    '① All 9 (baseline)': UNSUPERVISED_FEATURES,
    '② Drop prediction_error': [f for f in UNSUPERVISED_FEATURES if f != 'prediction_error'],
    '③ Drop motion_smoothness': [f for f in UNSUPERVISED_FEATURES if f != 'motion_smoothness'],
    '④ Drop ground_speed': [f for f in UNSUPERVISED_FEATURES if f != 'ground_speed'],
    '⑤ Drop heading_speed_consistency': [f for f in UNSUPERVISED_FEATURES if f != 'heading_speed_consistency'],
    '⑥ Drop pred_err + motion_smooth': [f for f in UNSUPERVISED_FEATURES if f not in ['prediction_error', 'motion_smoothness']],
    '⑦ Drop pred_err + ground_spd + heading_spd_cons': [f for f in UNSUPERVISED_FEATURES if f not in ['prediction_error', 'ground_speed', 'heading_speed_consistency']],
    '⑧ Pure kinematic core': ['acceleration', 'vertical_acceleration', 'vertical_speed', 'turn_rate', 'path_curvature'],
}

# ── LOAD & SPLIT ─────────────────────────────────────────────────────────────
print("Loading datasets...")
dji_df = get_or_preprocess_dji_dataset(filter_length_100=False, features=UNSUPERVISED_FEATURES)
dji_df['attack_class'] = 'Normal DJI'
esp32_df = get_or_preprocess_esp32_dataset()
esp32_df['attack_class'] = esp32_df['flight_id'].apply(
    lambda fid: next((v for k, v in ATTACK_MAP.items() if k in str(fid).lower()), 'Other')
)

unique_flights = dji_df['flight_id'].unique()
train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=42)
_, test_fl = train_test_split(temp, test_size=0.5, random_state=42)
dji_train = dji_df[dji_df['flight_id'].isin(train_fl)]
dji_test  = dji_df[dji_df['flight_id'].isin(test_fl)]

test_df = pd.concat([dji_test, esp32_df], ignore_index=True)
all_classes = sorted(test_df['attack_class'].unique())

# ── RUN ABLATION ───────────────────────────────────────────────────────────────
MODELS_TO_TEST = {'GMM': None, 'Isolation Forest': None}
all_rows = []

for subset_name, feats in SUBSETS.items():
    print(f"\nTesting: {subset_name}  ({len(feats)} features)")
    imp = SimpleImputer(strategy='mean')
    scl = StandardScaler()
    X_tr = scl.fit_transform(imp.fit_transform(dji_train[feats]))
    X_te = scl.transform(imp.transform(test_df[feats]))

    for model_name, _ in MODELS_TO_TEST.items():
        if model_name == 'GMM':
            m = GMMWrapper(n_components=3, covariance_type='full', random_state=42)
        else:
            m = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

        m.fit(X_tr)
        tr_scores = get_anomaly_scores(m, X_tr)
        thresh = tr_scores.mean() + 3.0 * tr_scores.std()

        row = {'Feature Set': subset_name, 'Model': model_name, 'N Features': len(feats)}
        for cls in all_classes:
            mask   = test_df['attack_class'] == cls
            is_anomaly = get_anomaly_scores(m, X_te[mask]) > thresh
            if cls in ['Normal DJI', 'Sim Normal']:
                acc = np.mean(~is_anomaly) * 100.0
            else:
                acc = np.mean(is_anomaly) * 100.0
            row[f'{cls} (Accuracy %)'] = round(acc, 1)
        all_rows.append(row)

results_df = pd.DataFrame(all_rows)
results_df.to_csv(OUT_DIR / 'ablation_results.csv', index=False)
print(f"\n✅ Saved: {OUT_DIR / 'ablation_results.csv'}")
