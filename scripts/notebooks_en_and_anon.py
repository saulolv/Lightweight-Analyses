"""
Load each notebook, apply English translations + path anonymization to all string values, write UTF-8 JSON.
Run: .venv/Scripts/python scripts/notebooks_en_and_anon.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from cifar_phrases import PHRASES_CIFAR  # noqa: E402
from phrases_pass2 import PHRASES_PASS2  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"

# Generic anonymized project root in outputs (no real usernames)
# Use a lambda in re.sub: replacement strings must not use "\\U" (regex template escape).
RE_WIN = re.compile(r"[A-Za-z]:\\Users\\[^/\\\n\r\"']+", re.IGNORECASE)
RE_UNIX = re.compile(r"/Users/[^/\\\n\r\"']+")
RE_PIBIC = re.compile(r"PIBIC", re.IGNORECASE)
RE_SAUO = re.compile(r"\\\\saulo\\\\", re.IGNORECASE)

ANON_ROOT_WIN = "C:/Users/anonymous/project/Lightweight Analyses"
ANON_ROOT_UNIX = "/home/anonymous/project/Lightweight Analyses"


def anonymize_paths(s: str) -> str:
    """Replace host-specific paths; no real usernames in printed paths."""
    s = RE_SAUO.sub(r"\\anonymous\\", s)
    s = RE_PIBIC.sub("project", s)
    s = RE_WIN.sub(lambda _m: ANON_ROOT_WIN, s)
    s = s.replace(ANON_ROOT_WIN + "/.venv", ANON_ROOT_WIN + "/.venv")
    s = RE_UNIX.sub(ANON_ROOT_UNIX, s)
    return s


# Longest-first: multi-line and phrases before single words
PHRASES: list[tuple[str, str]] = [
    # --- wake_vision_robustness intro (split across lines in JSON) ---
    (
        "## Robustez com Corrupções — Wake Vision\n",
        "## Robustness to corruptions — Wake Vision\n",
    ),
    (
        "Este notebook avalia models treinados no **Wake Vision** (pessoa vs não-pessoa) usando as **19 corrupções** de Hendrycks & Dietterich (2019), aplicadas **on-the-fly** via `imagecorruptions`.\n",
        "This notebook evaluates models trained on **Wake Vision** (person vs non-person) using the **19 corruptions** from Hendrycks & Dietterich (2019), applied **on-the-fly** via `imagecorruptions`.\n",
    ),
    (
        "- **15 common** (usadas no cálculo do mCE oficial):",
        "- **15 common** (used in the official mCE):",
    ),
    (
        "- **4 extra/validation** (não entram no mCE oficial):",
        "- **4 extra/validation** (not part of the official mCE):",
    ),
    (
        "### Por que on-the-fly?\n",
        "### Why on-the-fly?\n",
    ),
    # Continuation of wake_vision text - need full paragraph from file; use substrings
    (
        "No CIFAR-10/100, os datasets de corrupção (CIFAR-10-C, CIFAR-100-C) já vêm pré-gerados como `.npy` porque as imagens são 32×32. \n",
        "On CIFAR-10/100, corruption datasets (CIFAR-10-C, CIFAR-100-C) are shipped as pre-generated `.npy` because images are 32×32. \n",
    ),
    (
        "No Wake Vision, o test set tem ~55K imagens de tamanhos variados (redimensionadas para 224×224). Pré-gerar e salvar todas as corrupções em disco",
        "On Wake Vision, the test set has ~55K images at varied sizes (resized to 224×224). Pre-generating and saving all corruptions to disk",
    ),
    # --- mobile export ---
    (
        "Pipeline para exportar o dataset de test **Wake Vision** em formatos otimizados para dispositivos móveis:\n",
        "Pipeline to export the **Wake Vision** test set in mobile-friendly formats:\n",
    ),
    (
        "1. **NPZ**: Compactado, fácil para carregar em Python\n",
        "1. **NPZ**: compressed, easy to load in Python\n",
    ),
    (
        "2. **PNG + CSV**: Imagens individuais com metadados (classes, tamanhos originais)\n",
        "2. **PNG + CSV**: individual images with metadata (class, original size)\n",
    ),
    (
        "3. **TFLite**: Models quantizados (opcionalmente) para Android/Raspberry Pi\n",
        "3. **TFLite**: optionally quantized models for Android / Raspberry Pi\n",
    ),
    (
        "O dataset tem **~55K imagens de tamanhos variados** do test de pessoa vs não-pessoa no Wake Vision.\n",
        "The dataset has **~55K varied-size images** in the person vs non-person test split of Wake Vision.\n",
    ),
    (
        "Uma amostra estratificada de **500 imagens** é exportada para avaliar em dispositivos.",
        "A stratified sample of **500 images** is exported for on-device evaluation.",
    ),
    # --- wake_vision_analyses header ---
    (
        "(Harvard Edge Computing Lab)\n",
        "(paper authors, see citation)\n",
    ),
    (
        "— classificação binária: **pessoa** vs **não-pessoa**\n",
        "— binary classification: **person** vs **non-person**\n",
    ),
    (
        "MobileNetV3-Small (pré-treinado no ImageNet, fine-tuned)\n",
        "MobileNetV3-Small (ImageNet pre-trained, fine-tuned)\n",
    ),
    (
        "download via `tensorflow_datasets` → preprocess → treino em 2 fases (head → fine-tune) → avaliação",
        "download via `tensorflow_datasets` → preprocess → two-phase training (head → fine-tune) → evaluation",
    ),
    # --- cifar-10-c / cifar10c ---
    (
        "## Robustez com CIFAR-10-C (Common Corruptions)\n",
        "## Robustness on CIFAR-10-C (common corruptions)\n",
    ),
    (
        "Este notebook avalia models treinados no CIFAR-10 no **CIFAR-10-C** (corrupções que simulam degradações comuns em imagens naturais) e reporta **accuracy por severidade (1–5)** e médias.\n",
        "This notebook evaluates models trained on CIFAR-10 on **CIFAR-10-C** (corruptions that mimic common degradations) and reports **per-severity accuracy (1–5)** and means.\n",
    ),
    ("## Carregando o CIFAR-10", "## Loading CIFAR-10"),
    ("## Carregando o CIFAR-100", "## Loading CIFAR-100"),
    ("Não achei", "Could not find"),
    (
        "Download concluído:",
        "Download complete:",
    ),
    (
        "OK (já extraído):",
        "OK (already extracted):",
    ),
    (
        "Clean accuracy carregada do cache:",
        "Clean accuracy loaded from cache:",
    ),
    (
        "clean_acc carregada do cache",
        "clean_acc loaded from cache",
    ),
    # analyses_results mojibake + PT comments (fix to proper UTF-8 English)
    (
        "Comparao de Tamanho",
        "File size comparison",
    ),
    (
        "Comparação de Tamanho",
        "File size comparison",
    ),
    # Baseline prints
    ("Baseline (pior no clean):", "Baseline (worst on clean):"),
    ("# pior no clean", "# worst on clean"),
    (")  # pior no clean", ")  # worst on clean"),
    (
        "print(\"\\nBaseline (pior no clean):\",",
        "print(\"\\nBaseline (worst on clean):\",",
    ),
    (
        "baseline = max(clean, key=lambda d: d[\"clean_err\"])  # pior no clean",
        "baseline = max(clean, key=lambda d: d[\"clean_err\"])  # worst on clean",
    ),
    # cifar10c comments (remaining after partial)
    (
        "raise FileNotFoundError(f\"Não achei labels.npy em {base_dir.resolve()}\")",
        "raise FileNotFoundError(f\"Could not find labels.npy in {base_dir.resolve()}\")",
    ),
    (
        "raise FileNotFoundError(f\"Não achei {path}\")",
        "raise FileNotFoundError(f\"Could not find {path}\")",
    ),
    (
        "raise RuntimeError(f\"Não achei baseline model: {BASELINE_NAME}\")",
        "raise RuntimeError(f\"Could not find baseline model: {BASELINE_NAME}\")",
    ),
    # cifar-10-c_robustness (many cells)
    (
        "from __future__ import annotations\n",
        "from __future__ import annotations\n",  # no-op
    ),
    (
        "\n# Pasta onde o CIFAR-10-C ficará cacheado\n",
        "\n# Directory to cache CIFAR-10-C\n",
    ),
    (
        "\n# Localiza a raiz do CIFAR-10-C (onde ficam labels.npy e os arquivos *.npy de corrupção)\n",
        "\n# Locate CIFAR-10-C root (labels.npy and corruption *.npy files)\n",
    ),
    (
        "\n# Define o diretório base mesmo se você já tiver instalado (evita NameError fora de ordem)\n",
        "\n# Set base directory even if already set (avoids out-of-order NameError)\n",
    ),
    (
        "\n# (dica) Se der NameError por variáveis, rode esta célula antes",
        "\n# (tip) If variables raise NameError, run this cell first",
    ),
    (
        "Se der NameError por variáveis, rode esta célula antes",
        "If you get NameError for variables, run this cell first",
    ),
    (
        "variáveis",
        "variables",
    ),
    (
        "corrupção",
        "corruption",
    ),
    (
        "corrupções",
        "corruptions",
    ),
    (
        "degradações",
        "degradations",
    ),
    (
        "Avaliação",
        "Evaluation",
    ),
    (
        "persistência",
        "persistence",
    ),
    (
        "Célula",
        "Cell",
    ),
    (
        "célula",
        "cell",
    ),
    (
        "células",
        "cells",
    ),
    (
        "autônoma",
        "standalone",
    ),
    (
        "baixado",
        "downloaded",
    ),
    (
        "Diretório",
        "Directory",
    ),
    (
        "diretório",
        "directory",
    ),
    (
        "ficará cacheado",
        "will be cached",
    ),
    (
        "CIFAR-10 limpo",
        "clean CIFAR-10",
    ),
    (
        "CIFAR-10 limpo",
        "clean CIFAR-10",
    ),
]

# Add explicit multi-line comment blocks (cifar-10-c)
MORE: list[tuple[str, str]] = [
    (
        "# Arquivos de corrupção disponíveis (cada um com shape (50000, 32, 32, 3))\n",
        "# Corruption files (each with shape (50000, 32, 32, 3))\n",
    ),
    (
        "# Convenção do CIFAR-10-C: 5 severidades, 10k imagens cada\n",
        "# CIFAR-10-C convention: 5 severities, 10k images each\n",
    ),
    (
        "# Baseline = model com pior accuracy no clean (calculado na célula anterior).\n",
        "# Baseline = model with worst clean accuracy (from previous cell).\n",
    ),
    (
        "# mostra o que já tinha salvo\n",
        "# show previously stored values\n",
    ),
    (
        "# Carrega baseline_acc salvo (se existir) e só calcula o que faltar\n",
        "# Load saved baseline_acc if any; only compute what is missing\n",
    ),
    (
        "# CE por model/corrupção (resume via JSONL)\n",
        "# CE by model/corruption (resumed from JSONL)\n",
    ),
    (
        "        # Mantém a métrica antiga (média das razões) só para referência/debug\n",
        "        # Keep legacy metric (mean of ratios) for reference/debug\n",
    ),
    (
        "        # Métrica principal (bate com a fórmula do paper): razão das somas\n",
        "        # Main metric (matches the paper): ratio of sums\n",
    ),
    (
        "\n# Avalia no CIFAR-10 limpo para TODOS os models e escolhe baseline = pior clean acc\n",
        "\n# Evaluate on clean CIFAR-10 for all models; baseline = worst clean acc\n",
    ),
    # MCUNet
    (
        "## MCUNet Oficial — Avaliação de robustez no CIFAR-10-C\n",
        "## MCUNet (official) — CIFAR-10-C robustness\n",
    ),
    (
        "Célula **autônoma**: só precisa que o CIFAR-10-C esteja baixado (células 1–4) e que o model",
        "Standalone: only needs CIFAR-10-C downloaded (cells 1–4) and the model",
    ),
    (
        "# MCUNet Oficial — avaliação de robustez no CIFAR-10-C\n",
        "# MCUNet (official) — CIFAR-10-C robustness\n",
    ),
    (
        "\n# (célula autônoma: só precisa que o CIFAR-10-C esteja baixado)\n",
        "\n# (standalone: only needs CIFAR-10-C downloaded)\n",
    ),
]

PHRASES = PHRASES + MORE
# Sort once at start of apply_phrases with PASS2

# After phrase replace, mojibake fix for old broken encoding in analyses_results
MOJIBAKE: list[tuple[str, str]] = [
    ("Robustness AnalysismCE and Relative mCE", "Robustness Analysis — mCE and Relative mCE"),
    (" CIFAR-10C ", " CIFAR-10-C "),
    ("# CIFAR-10C ", "# CIFAR-10-C "),
    (" CIFAR-100C ", " CIFAR-100-C "),
    (", mCE e Relative mCE", ", mCE and Relative mCE"),
    (", mCE e Relative mCE", ", mCE and Relative mCE"),
]


_PHRASE_ORDERED: list[tuple[str, str]] | None = None


def _all_phrases() -> list[tuple[str, str]]:
    global _PHRASE_ORDERED
    if _PHRASE_ORDERED is None:
        merged = list(PHRASES) + list(PHRASES_PASS2) + list(PHRASES_CIFAR)
        merged.sort(key=lambda x: len(x[0]), reverse=True)
        _PHRASE_ORDERED = merged
    return _PHRASE_ORDERED


def apply_phrases(s: str) -> str:
    for a, b in _all_phrases():
        s = s.replace(a, b)
    for a, b in MOJIBAKE:
        s = s.replace(a, b)
    return s


def walk(obj: Any) -> Any:
    if isinstance(obj, str):
        return anonymize_paths(apply_phrases(obj))
    if isinstance(obj, list):
        return [walk(x) for x in obj]
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    return obj


def process(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    new = walk(data)
    text = json.dumps(new, ensure_ascii=False, indent=1) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print("OK", path.name)


def main() -> None:
    for p in sorted(NOTEBOOKS.glob("*.ipynb")):
        process(p)


if __name__ == "__main__":
    main()
