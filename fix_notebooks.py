#!/usr/bin/env python3
"""
fix_notebooks.py
----------------
Scans every .ipynb in the notebooks/ folder and:
  1. Translates Portuguese text → English in:
       - code comments (lines starting with #)
       - print() string arguments (source code)
       - cell outputs (stdout text saved inside the notebook)
       - markdown cells
  2. Anonymises any Windows/Linux paths that contain a real username.

Run from the repo root:
    python fix_notebooks.py

A backup of each file is written as <name>.ipynb.bak
"""

import json, re, copy
from pathlib import Path

TRANSLATIONS = [
    # ── source-code comments ──────────────────────────────────────────────
    ("# Permite unificar nomes de models vindos de arquivos diferentes",
     "# Allows unifying model names from different files"),
    ("# Medições",            "# Measurements"),
    ("# Load/Processar Dados: LATÊNCIA",
     "# Load/Process Data: LATENCY"),
    ("# Com 100 runs por dataset e 3 datasets (CIFAR10, CIFAR100 e WakeVision), temos até 300 blocos.",
     "# With 100 runs per dataset and 3 datasets (CIFAR10, CIFAR100 and WakeVision), we have up to 300 blocks."),
    ("# Testes para Tamanho (Size)",  "# Tests for Size"),
    ("# O tamanho do model é determinístico e não possui variância num mesmo pipeline.",
     "# Model size is deterministic and has no variance within the same pipeline."),
    ("# Aqui usamos os datasets CIFAR-10, CIFAR-100 e Wake Vision para uma comparação descritiva.",
     "# Here we use the CIFAR-10, CIFAR-100, and Wake Vision datasets for a descriptive comparison."),
    ("# Menor tamanho é melhor",      "# Smaller size is better"),
    ("# Ranqueamento",   "# Ranking"),
    ("# Dados de",       "# Data for"),
    ("# Carregando",     "# Loading"),
    ("# Calculando",     "# Computing"),
    ("# Gerando",        "# Generating"),
    ("# Plotando",       "# Plotting"),
    ("# Configurações",  "# Settings"),
    ("# Configuração",   "# Configuration"),
    ("# Resultados",     "# Results"),
    ("# Parâmetros",     "# Parameters"),
    ("# Modelos",        "# Models"),
    ("# Importações",    "# Imports"),
    ("# Importando",     "# Importing"),
    ("# Funções",        "# Functions"),
    ("# Função",         "# Function"),
    ("# Leitura",        "# Reading"),
    ("# Análise",        "# Analysis"),
    ("# Visualização",   "# Visualisation"),
    ("# Comparação",     "# Comparison"),
    ("# Testes de",      "# Tests for"),
    ("# Teste de",       "# Test for"),
    ("# Testes Friedman","# Friedman Tests"),
    ("# Teste Friedman", "# Friedman Test"),
    # ── print() arguments / stdout ───────────────────────────────────────
    ("Amostra dos Dados de Robustez (CE):",  "Robustness Data Sample (CE):"),
    ("--- TESTE FRIEDMAN PARA ROBUSTNESS (CE) ---",
     "--- FRIEDMAN TEST FOR ROBUSTNESS (CE) ---"),
    ("--- TESTE FRIEDMAN PARA ROBUSTEZ (CE) ---",
     "--- FRIEDMAN TEST FOR ROBUSTNESS (CE) ---"),
    ("Amostra dos Dados de Latency:",        "Latency Data Sample:"),
    ("Amostra dos Dados de Latência:",       "Latency Data Sample:"),
    ("--- TESTE FRIEDMAN PARA LATÊNCIA ---",
     "--- FRIEDMAN TEST FOR LATENCY ---"),
    ("--- RANQUEAMENTO PARA TAMANHO DO MODELO (TFLite MB) ---",
     "--- RANKING FOR MODEL SIZE (TFLite MB) ---"),
    ("Tamanho (MB):",                        "Size (MB):"),
    ("Ranques (1 = Menor/Melhor):",          "Ranks (1 = Smallest/Best):"),
    ("--- RANQUEAMENTO PARA LATÊNCIA P90 (MOBILE) ---",
     "--- RANKING FOR P90 LATENCY (MOBILE) ---"),
    ("--- RANQUEAMENTO PARA",   "--- RANKING FOR"),
    ("Amostra dos Dados",       "Data Sample"),
    ("--- TESTE FRIEDMAN",      "--- FRIEDMAN TEST"),
    ("Gerando diagrama de diferença crítica",
     "Generating critical-difference diagram"),
    ("Gerando o diagrama",      "Generating the diagram"),
    ("Carregando modelo",       "Loading model"),
    ("Carregando dados",        "Loading data"),
    ("Resultados para",         "Results for"),
    ("Processando",             "Processing"),
    ("Calculando",              "Computing"),
    ("Tamanho do modelo",       "Model size"),
    ("Número de parâmetros",    "Number of parameters"),
    ("Acurácia",    "Accuracy"),
    ("Precisão",    "Precision"),
    ("Revocação",   "Recall"),
    ("Erro de Corrupção", "Corruption Error"),
    ("Latência",    "Latency"),
    ("Robustez",    "Robustness"),
    # ── markdown ─────────────────────────────────────────────────────────
    ("## Análise",      "## Analysis"),
    ("## Resultados",   "## Results"),
    ("## Comparação",   "## Comparison"),
    ("## Configuração", "## Configuration"),
    ("## Introdução",   "## Introduction"),
    ("## Conclusão",    "## Conclusion"),
    ("### Análise",     "### Analysis"),
    ("### Resultados",  "### Results"),
]

PATH_RE = re.compile(
    r'(?<=[/\\])(?!anonymous(?:[/\\]|$))'
    r'([A-Za-z][A-Za-z0-9_.-]{1,30})'
    r'(?=[/\\])',
)

def anonymise_path(text):
    if not re.search(r'(?:[A-Za-z]:[/\\]|/home/|/Users/|C:\\)', text):
        return text
    return PATH_RE.sub("anonymous", text)

def fix_string(s):
    for pt, en in TRANSLATIONS:
        s = s.replace(pt, en)
    return anonymise_path(s)

def fix_lines(lines):
    return [fix_string(l) for l in lines]

def fix_cell(cell):
    cell = copy.deepcopy(cell)
    src = cell.get("source")
    if isinstance(src, list):
        cell["source"] = fix_lines(src)
    elif isinstance(src, str):
        cell["source"] = fix_string(src)
    for out in cell.get("outputs", []):
        for key in ("text", "traceback"):
            if isinstance(out.get(key), list):
                out[key] = fix_lines(out[key])
            elif isinstance(out.get(key), str):
                out[key] = fix_string(out[key])
        for mime, val in out.get("data", {}).items():
            if mime.startswith("text/"):
                if isinstance(val, list):
                    out["data"][mime] = fix_lines(val)
                elif isinstance(val, str):
                    out["data"][mime] = fix_string(val)
    return cell

def fix_notebook(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    path.with_suffix(".ipynb.bak").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    nb["cells"] = [fix_cell(c) for c in nb.get("cells", [])]
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ✓  {path.name}")

def main():
    root = Path("notebooks")
    if not root.is_dir():
        print("ERROR: run from the repo root (notebooks/ not found).")
        return
    nbs = sorted(root.rglob("*.ipynb"))
    print(f"Found {len(nbs)} notebook(s).\n")
    for p in nbs:
        fix_notebook(p)
    print("\nDone. Backups saved as *.ipynb.bak")
    print("Remove them after review:  rm notebooks/*.bak")

if __name__ == "__main__":
    main()