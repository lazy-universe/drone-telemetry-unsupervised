"""
Legacy forwarder for run_unsupervised_pipeline.py.
Forwards all calls and exports to presets/run_unsupervised_pipeline.py.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from presets.run_unsupervised_pipeline import (
    FEATURE_SETS,
    run_experiment,
    main,
    clear_ephemeral_cache,
    create_sequences,
    compute_dl_anomaly_scores
)

if __name__ == '__main__':
    main()
