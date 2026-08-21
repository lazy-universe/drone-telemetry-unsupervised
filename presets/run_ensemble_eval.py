import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.workflows.per_class_evaluation import get_labeled_datasets
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data
from implement.utils.classical_ml.classical_models import get_unsupervised_point_models
from implement.utils.classical_ml.classical_train_eval import get_anomaly_scores

top_10_features = [
    'height',
    'ground_speed',
    'vertical_speed',
    'acceleration',
    'turn_rate',
    'path_curvature',
    'heading_speed_consistency',
    'motion_smoothness',
    'prediction_error',
    'yaw_acceleration'
]

dji_df, esp32_df = get_labeled_datasets(features=top_10_features)

unique_flights = dji_df['flight_id'].unique()
train_fl, temp = train_test_split(unique_flights, test_size=0.3, random_state=42)
val_fl, test_fl = train_test_split(temp, test_size=0.5, random_state=42)

dji_train = dji_df[dji_df['flight_id'].isin(train_fl)]
dji_val   = dji_df[dji_df['flight_id'].isin(val_fl)]
dji_test  = dji_df[dji_df['flight_id'].isin(test_fl)]

test_df = pd.concat([dji_test, esp32_df], ignore_index=True)
all_classes = sorted(test_df['attack_class'].unique())

X_train_raw = dji_train[top_10_features]
X_train_scaled, _, scaler, imputer = impute_and_scale_data(X_train_raw, X_train_raw)
X_val_scaled = scaler.transform(imputer.transform(dji_val[top_10_features]))
X_test_scaled = scaler.transform(imputer.transform(test_df[top_10_features]))

models = get_unsupervised_point_models()

# Fit models and store binary anomaly predictions for validation and test
val_anom_dict = {}
test_anom_dict = {}

for m_name, model in models.items():
    model.fit(X_train_scaled)
    val_scores = get_anomaly_scores(model, X_val_scaled)
    thresh = val_scores.mean() + 3.0 * val_scores.std()

    test_scores = get_anomaly_scores(model, X_test_scaled)
    val_anom_dict[m_name] = val_scores > thresh
    test_anom_dict[m_name] = test_scores > thresh

# Define ensemble combinations
ensembles = {
    'Single: GMM': ['GMM'],
    'Single: Mahalanobis': ['Mahalanobis'],
    'Single: KNN': ['KNN'],
    'Single: Isolation Forest': ['Isolation Forest'],
    'Ensemble: OR (GMM | Mahalanobis)': ['GMM', 'Mahalanobis'],
    'Ensemble: OR (GMM | KNN)': ['GMM', 'KNN'],
    'Ensemble: OR (GMM | Isolation Forest)': ['GMM', 'Isolation Forest'],
    'Ensemble: OR (Mahalanobis | KNN)': ['Mahalanobis', 'KNN'],
    'Ensemble: OR (GMM | Mahalanobis | KNN)': ['GMM', 'Mahalanobis', 'KNN'],
    'Ensemble: OR (ALL Models)': list(models.keys())
}

print("=========================================================================")
print("=== EXPERIMENT: LOGICAL 'OR' ENSEMBLE EVALUATION (10-FEATURE SET) ===")
print("=========================================================================")

results = []
for ens_name, model_list in ensembles.items():
    # Compute logical OR of test anomaly masks
    test_or_mask = np.zeros(len(test_df), dtype=bool)
    for m_name in model_list:
        if m_name in test_anom_dict:
            test_or_mask = test_or_mask | test_anom_dict[m_name]

    row = {'Configuration': ens_name}
    for cls in all_classes:
        mask = (test_df['attack_class'] == cls)
        cls_is_anom = test_or_mask[mask]
        
        if cls == 'Normal DJI':
            acc = np.mean(~cls_is_anom) * 100.0 # True Negative Rate
        else:
            acc = np.mean(cls_is_anom) * 100.0  # True Positive Rate / Detection Accuracy
        row[cls] = round(acc, 2)
    results.append(row)

df_res = pd.DataFrame(results)

from implement.utils.helper import get_output_dir
out_dir = get_output_dir() / 'ensemble_eval'
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "ensemble_or_results.csv"
df_res.to_csv(csv_path, index=False)

print("\n" + df_res.to_string(index=False))
print(f"\n✅ Ensemble evaluation completed! CSV saved to {csv_path}")

# Generate Bar Chart for Ensembles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif', 'figure.autolayout': True})

fig, ax = plt.subplots(figsize=(14, 7))
plot_classes = [c for c in ['Normal DJI', 'Real ESP32', 'Sim Easy', 'Sim Medium', 'Sim Geometry', 'Sim Baseline'] if c in all_classes]
x = np.arange(len(plot_classes))

selected_ens = [
    'Single: GMM',
    'Single: KNN',
    'Ensemble: OR (GMM | Mahalanobis)',
    'Ensemble: OR (GMM | KNN)',
    'Ensemble: OR (GMM | Mahalanobis | KNN)'
]

plot_df = df_res[df_res['Configuration'].isin(selected_ens)].copy()
num_bars = len(plot_df)
width = 0.8 / max(num_bars, 1)
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

for idx, (_, row) in enumerate(plot_df.iterrows()):
    c_label = row['Configuration']
    vals = [row[c] for c in plot_classes]
    offset = (idx - num_bars/2 + 0.5) * width
    rects = ax.bar(x + offset, vals, width, label=c_label, color=colors[idx % len(colors)], alpha=0.9)
    for rect in rects:
        h = rect.get_height()
        if h > 5:
            ax.text(rect.get_x() + rect.get_width()/2., h + 1.0, f'{h:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Accuracy / TNR (%)', fontsize=11, fontweight='bold')
ax.set_title('Pointwise Model Ensembling: Single Models vs. Logical OR Ensembles (10 Features)', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(plot_classes, fontsize=10.5, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(title='Model / Ensemble', frameon=True, facecolor='white', loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

img_path = out_dir / 'barchart_ensemble_comparison.png'
plt.savefig(img_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved bar chart: {img_path}")
