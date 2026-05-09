"""Repository path helpers for notebooks and scripts.

Provides a single source of truth for resolving the project root,
replacing the duplicated inline definitions previously found in every notebook.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Resolve the project root from any working directory.

    Search strategy (in order):
      1. Walk upward looking for a directory that contains both ``.git``
         and ``results/`` — the canonical repo marker.
      2. Check the current directory or its parent for ``datasets/`` or
         ``results/`` as a fallback (e.g. fresh clone before training).
      3. Fall back to ``cwd``.
    """
    cwd = Path.cwd().resolve()
    for d in [cwd, *cwd.parents]:
        if (d / ".git").is_dir():
            return d
    if (cwd / "datasets").is_dir() or (cwd / "results").is_dir():
        return cwd
    if (cwd.parent / "datasets").is_dir() or (cwd.parent / "results").is_dir():
        return cwd.parent
    return cwd


ROOT = project_root()

DATASETS_DIR = ROOT / "datasets"
KERAS_MODELS_DIR = ROOT / "keras_models"
TFLITE_MODELS_DIR = ROOT / "tflite_models"
SAVED_MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
EXPORTS_DIR = ROOT / "exports"
CHECKPOINT_DIR = ROOT / "checkpoints"
MCUNET_OFFICIAL_PATH = ROOT / "third_party" / "mcunet-official"