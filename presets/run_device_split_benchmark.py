"""
Preset Script: Device-Level Split Benchmark Evaluation (split_mode='device')
Evaluates pointwise models and deep learning autoencoders using random_state=42 device-level splits for:
  1. Refined 8 Features
  2. Candidate A (9 Features)
  3. Candidate B (9 Features)
"""
import sys, warnings, argparse
from pathlib import Path

# Add project root directory to Python path for importing implement modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from implement.workflows.per_class_evaluation import run_per_class_evaluation

REFINED_8 = [
    'height',
    'ground_speed',
    'vertical_speed',
    'acceleration',
    'turn_rate',
    'path_curvature',
    'heading_speed_consistency',
    'motion_smoothness'
]

CANDIDATE_A = REFINED_8 + ['yaw_acceleration']
CANDIDATE_B = REFINED_8 + ['prediction_error']

FEATURE_SETS = {
    '8_features': REFINED_8,
    'candidate_a': CANDIDATE_A,
    'candidate_b': CANDIDATE_B
}

def main():
    parser = argparse.ArgumentParser(description="Preset: Device-Level Split Benchmark Evaluation")
    parser.add_argument(
        "--feature-set",
        choices=["all", "8_features", "candidate_a", "candidate_b"],
        default="all",
        help="Feature set to evaluate (default: all)"
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
        default=5,
        help="Number of epochs for deep learning autoencoders (default: 5)"
    )
    args = parser.parse_args()

    sets_to_run = FEATURE_SETS.items() if args.feature_set == "all" else [(args.feature_set, FEATURE_SETS[args.feature_set])]

    print("=" * 80)
    print("=== RUNNING UNSUPERVISED EVALUATION: DEVICE-LEVEL SPLIT (random_state=42) ===")
    print(f"=== Model Family: {args.model_family} | Epochs: {args.epochs} ===")
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
            batch_size=64,
            lr=0.001,
            k_threshold=3.0,
            features=f_list
        )

if __name__ == "__main__":
    main()
