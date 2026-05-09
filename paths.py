"""Backward-compatible path helpers.

This module re-exports everything from ``src.paths`` so that existing
scripts (e.g. ``Raspberry/eval_tflite_multidataset.py``) that import
``paths.project_root`` continue to work without changes.
"""

from src.paths import *  # noqa: F401,F403