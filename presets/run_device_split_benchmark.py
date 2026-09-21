"""
Preset Script: Device-Level Split Benchmark Evaluation (split_mode='device')
Evaluates pointwise models and deep learning autoencoders using random_state=42 device-level splits.
Trains exclusively on genuine DJI flights from a subset of drone models, and tests on held-out unseen DJI drone models + all attack classes.
"""
import sys
import warnings
import argparse
from pathlib import Path

# Add project root directory to Python path for importing implement modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from implement.workflows.per_class_evaluation import run_per_class_evaluation

ULTIMATE_9 = [
    "motion_smoothness", "heading_speed_consistency", "ground_speed",
    "height", "vertical_speed", "acceleration", "turn_rate",
    "prediction_error", "position_residual_std"
]

BASELINE_7 = [
    "motion_smoothness", "heading_speed_consistency", "ground_speed",
    "height", "vertical_speed", "acceleration", "turn_rate"
]

FEATURE_SETS = {
    "ultimate_9": ULTIMATE_9,
    "baseline_7_plus_pe_pos": ULTIMATE_9,
    "baseline_7": BASELINE_7,
    "baseline_7_plus_pe": BASELINE_7 + ["prediction_error"],
    "baseline_7_plus_pos": BASELINE_7 + ["position_residual_std"],
    "baseline_7_plus_pe_pos_yaw": ULTIMATE_9 + ["yaw_acceleration"],
}


def main():
    parser = argparse.ArgumentParser(description="Preset: Device-Level Split Benchmark Evaluation")
    parser.add_argument(
        "--feature-set",
        choices=list(FEATURE_SETS.keys()) + ["all"],
        default="ultimate_9",
        help="Feature set to evaluate (default: ultimate_9)"
    )
    parser.add_argument(
        "--model-family",
        choices=["all", "pointwise", "dl", "autoencoders"],
        default="all",
        help="Model family to evaluate (default: all)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs for deep learning models (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--k-threshold",
        type=float,
        default=3.0,
        help="k threshold multiplier (default: 3.0)"
    )
    parser.add_argument(
        "--features",
        nargs='+',
        default=None,
        help="Explicit list of features to evaluate (space-separated or comma-separated)"
    )
    args = parser.parse_args()

    if args.features:
        custom_feats = []
        for item in args.features:
            for feat in item.split(','):
                feat_clean = feat.strip()
                if feat_clean and feat_clean not in custom_feats:
                    custom_feats.append(feat_clean)
        sets_to_run = [(f"custom_{len(custom_feats)}_features", custom_feats)]
    else:
        if args.feature_set == "all":
            # Run the primary unique sets
            sets_to_run = [
                ("ultimate_9", ULTIMATE_9),
                ("baseline_7", BASELINE_7),
                ("baseline_7_plus_pe", BASELINE_7 + ["prediction_error"]),
                ("baseline_7_plus_pos", BASELINE_7 + ["position_residual_std"]),
                ("baseline_7_plus_pe_pos_yaw", ULTIMATE_9 + ["yaw_acceleration"]),
            ]
        else:
            sets_to_run = [(args.feature_set, FEATURE_SETS[args.feature_set])]

    print("=" * 80)
    print("=== RUNNING UNSUPERVISED EVALUATION: DEVICE-LEVEL SPLIT (random_state=42) ===")
    print(f"=== Model Family: {args.model_family} | Epochs: {args.epochs} | k_thresh: {args.k_threshold} ===")
    print("=" * 80)

    for name, f_list in sets_to_run:
        print("\n" + "=" * 80)
        print(f"=== DEVICE SPLIT EXPERIMENT: {name} ({len(f_list)} features) ===")
        print(f"=== Features: {f_list} ===")
        print("=" * 80)

        run_per_class_evaluation(
            model_family=args.model_family,
            split_mode="device",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=0.001,
            k_threshold=args.k_threshold,
            features=f_list
        )


if __name__ == "__main__":
    main()
