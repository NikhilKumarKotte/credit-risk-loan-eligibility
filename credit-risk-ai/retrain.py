"""
retrain.py
----------
Model retraining script — regenerates dataset (optional) and retrains
all candidate models, saving the new best model bundle.

Usage:
    python retrain.py                    # retrain on existing data
    python retrain.py --regenerate-data  # regenerate dataset first
    python retrain.py --samples 20000    # custom dataset size
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retrain the credit risk model.")
    parser.add_argument(
        "--regenerate-data", action="store_true",
        help="Regenerate the synthetic dataset before retraining."
    )
    parser.add_argument(
        "--samples", type=int, default=12000,
        help="Number of samples to generate (default: 12000)."
    )
    parser.add_argument(
        "--data-path", type=str, default=str(ROOT / "data" / "loan_data.csv"),
        help="Path to the CSV dataset."
    )
    args = parser.parse_args()

    # ── Optional: regenerate data ─────────────────────────────────────────────
    if args.regenerate_data:
        logger.info(f"Regenerating dataset with {args.samples:,} samples …")
        from generate_dataset import generate_loan_dataset
        df = generate_loan_dataset(n_samples=args.samples)
        data_path = Path(args.data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"New dataset saved → {data_path}")
    else:
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Dataset not found at {data_path}. Use --regenerate-data flag.")
            sys.exit(1)

    # ── Retrain ───────────────────────────────────────────────────────────────
    logger.info("Starting retraining pipeline …")
    from train_model import train_and_select
    result = train_and_select(data_path=data_path)

    best = result["best_model"]
    m = result["metrics"][best]
    logger.info("Retraining complete.")
    print(f"\n{'='*50}")
    print(f"  Best Model  : {best}")
    print(f"  ROC-AUC     : {m['roc_auc']}")
    print(f"  Accuracy    : {m['accuracy']}")
    print(f"  Precision   : {m['precision']}")
    print(f"  Recall      : {m['recall']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
