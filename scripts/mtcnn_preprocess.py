"""
CLI entry point: MTCNN face preprocessing.

Runs src/data/mtcnn_preprocess.py from the project root so imports resolve correctly.

Usage (from project root):
    python scripts/mtcnn_preprocess.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mtcnn_preprocess import preprocess_and_split

if __name__ == "__main__":
    preprocess_and_split()
