"""
CLI entry point: Evaluate all 6 models on LFW 6,000-pair protocol.

This script runs the full evaluation
pipeline and exports outputs/metrics/comparison_table.csv.

Usage (from project root):
    python scripts/evaluate_lfw.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_lfw import run_evaluation

if __name__ == "__main__":
    run_evaluation()
