"""Root launcher for training from the workspace root."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

PROJECT_DIR = Path(__file__).resolve().parent / "email_classifier"
sys.path.insert(0, str(PROJECT_DIR))
runpy.run_module("src.train", run_name="__main__")
