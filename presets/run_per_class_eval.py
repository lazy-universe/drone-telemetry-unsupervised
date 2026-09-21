import os
import sys
import argparse
from pathlib import Path

# Add project root directory to Python path for importing implement modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.workflows.per_class_evaluation import run_per_class_evaluation
from implement.utils.helper import WINDOW_LEN

def main():
    parser = argparse.ArgumentParser(description="Workflow Preset: Per-Class Unsupervised Anomaly Detection Evaluation")
    parser.add_argument(
        "--model-family",
        choices=["all", "pointwise", "dl", "autoencoders"],
        default="all",
        help="Model family to evaluate: 'all' (both), 'pointwise', or 'dl'/'autoencoders' (default: all)"
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "flight"],
        default="flight",
        help="Partitioning mode for DJI flights ('random' or 'flight') (default: flight)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train deep learning autoencoders (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for optimization (default: 0.001)"
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=WINDOW_LEN,
        help=f"Custom window length for sequence data (default: {WINDOW_LEN})"
    )
    parser.add_argument(
        "--k-threshold",
        type=float,
        default=3.0,
        help="Threshold multiplier k (threshold = mean + k*std) (default: 3.0)"
    )
    parser.add_argument(
        "--features",
        nargs='+',
        default=None,
        help="Explicit list of features to use for evaluation (space-separated or comma-separated)"
    )
    args = parser.parse_args()

    features_list = None
    if args.features:
        features_list = []
        for item in args.features:
            for feat in item.split(','):
                feat_clean = feat.strip()
                if feat_clean and feat_clean not in features_list:
                    features_list.append(feat_clean)

    run_per_class_evaluation(
        model_family=args.model_family,
        split_mode=args.split_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window_len=args.window_len,
        k_threshold=args.k_threshold,
        features=features_list
    )

if __name__ == "__main__":
    main()
