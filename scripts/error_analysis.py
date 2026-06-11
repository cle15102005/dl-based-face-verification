"""
CLI entry point: Visual error analysis (False Accept / False Reject cases).

Outputs side-by-side error visualizations to outputs/error_cases/.

Usage (from project root):
    python scripts/error_analysis.py
    python scripts/error_analysis.py --max-cases 20
    python scripts/error_analysis.py --checkpoint outputs/checkpoints/arcface_s.pt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import run_error_analysis

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize false accept and false reject cases.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override.")
    parser.add_argument("--max-cases", type=int, default=10, help="Max cases per error type to visualize.")
    args = parser.parse_args()
    run_error_analysis(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        max_cases=args.max_cases,
    )
