"""Second-pass phrase replacements (Portuguese/PT-EN code comments → English). Merged in notebooks_en_and_anon.py."""

# Longest keys first; applied after first phrase pass, same walk.
PHRASES_PASS2: list[tuple[str, str]] = [
    # --- wake_vision_robustness c0 (remaining PT) ---
    (
        "Pre-generating and saving all corruptions to disk ocuparia **~780 GB** (19 corruptions × 5 severidades × 55K × 224×224×3).",
        "Pre-generating and saving all corruptions to disk would use **~780 GB** (19 corruptions × 5 severities × 55K × 224×224×3).",
    ),
    (
        "A solução é usar a biblioteca `imagecorruptions` para aplicar cada corruption imagem a imagem, avaliar, e descartar — sem salvar nada em disco.\n",
        "The solution is to use `imagecorruptions` to apply each corruption per image, evaluate, and discard—without writing anything to disk.\n",
    ),
    (
        "### Métricas\n",
        "### Metrics\n",
    ),
    (
        "(severidades 1–5)\n",
        "(severities 1–5)\n",
    ),
    (
        "- **mCE:** média do CE sobre as **15 corruptions comuns** (métrica oficial, Hendrycks 2019)\n",
        "- **mCE:** mean CE over the **15 common corruptions** (official metric, Hendrycks 2019)\n",
    ),
    (
        "- **mCE-19:** média do CE sobre todas as 19 corruptions (informativo)\n",
        "- **mCE-19:** mean CE over all 19 corruptions (informational)\n",
    ),
    (
        "- **Relative mCE:** ajusta pela clean accuracy de cada model",
        "- **Relative mCE:** adjusts for each model’s clean accuracy",
    ),
    # device anonymization
    ("(Samsung Galaxy S24+)", "(reference mobile device)"),
    ("Samsung Galaxy S24+", "reference mobile device"),
    # analyses_results
    (
        "# --- Registrar camada customizada (necessário para carregar MCUNet_Official.keras) ---",
        "# --- Register custom layer (required to load MCUNet_Official.keras) ---",
    ),
    (
        '"""Normalização ImageNet: (x/255 - mean) / std"""',
        '"""ImageNet normalization: (x/255 - mean) / std"""',
    ),
    (
        "# Models a comparar (apenas MCUNet Official, sem variantes)\n",
        "# Models to compare (MCUNet Official only, no variants)\n",
    ),
    (
        "# Mapeamento: nome do model → arquivo .keras por dataset\n",
        "# Map: model name → .keras file per dataset\n",
    ),
    (
        "# ===================== Funções auxiliares =====================\n",
        "# ===================== Helper functions =====================\n",
    ),
    (
        '"""Carrega models keras e calcula clean accuracy no test set.\n    Results são cacheados em JSON para evitar re-avaliação."""',
        '"""Load Keras models and compute test-set clean accuracy.\n    Results are cached in JSON to avoid re-evaluation."""',
    ),
    (
        'print(f"  Carregando {name} de {model_path}...")',
        'print(f"  Loading {name} from {model_path}...")',
    ),
    (
        'print(f"Calculando clean accuracy para {dataset_name}...")',
        'print(f"Computing clean accuracy for {dataset_name}...")',
    ),
    (
        "print(f\"  Cache salvo em: {cache_path}\")",
        "print(f\"  Cache written to: {cache_path}\")",
    ),
    (
        "    # Cachear results\n",
        "    # Write cache file\n",
    ),
    (
        '"""Calcula mCE e Relative mCE a partir dos results de CE."""',
        '"""Compute mCE and Relative mCE from CE results."""',
    ),
    (
        '    """Plota plot scatter: Accuracy vs mCE / Relative mCE com anotações."""',
        '    """Scatter: accuracy vs mCE / Relative mCE with annotations."""',
    ),
    (
        "    # Plotando com marcadores\n",
        "    # Markers and lines\n",
    ),
    (
        "    # --- Offsets automáticos (baseados na posição relativa de cada ponto) ---\n",
        "    # --- Automatic label offsets (from relative point positions) ---\n",
    ),
    (
        "    # Ordena por accuracy para decidir offsets de forma robusta\n",
        "    # Sort by accuracy for robust label placement\n",
    ),
    (
        "        rank = list(sorted_idx).index(i)  # posição por accuracy\n",
        "        rank = list(sorted_idx).index(i)  # rank by accuracy\n",
    ),
    (
        "        # Offsets padrão: model com menor accuracy → direita,\n        # maior accuracy → esquerda, meio → acima\n",
        "        # Default offsets: lowest accuracy → right,\n        # highest → left, middle → above\n",
    ),
    (
        "        if rank == 0:  # menor accuracy\n",
        "        if rank == 0:  # lowest accuracy\n",
    ),
    (
        "        elif rank == len(model_names) - 1:  # maior accuracy\n",
        "        elif rank == len(model_names) - 1:  # highest accuracy\n",
    ),
    (
        "        else:  # meio\n",
        "        else:  # middle\n",
    ),
    (
        "        # Anotação mCE (azul)\n",
        "        # mCE annotation (blue)\n",
    ),
    (
        "        # Anotação Relative mCE (laranja)\n",
        "        # Relative mCE annotation (orange)\n",
    ),
    (
        "        print(f\"Plot salvo em: {pdf_path}\")",
        "        print(f\"Plot saved to: {pdf_path}\")",
    ),
    (
        'print("✓ Funções auxiliares carregadas.")',
        'print("✓ Helper functions loaded.")',
    ),
    (
        "# CIFAR-10-C — mCE e Relative mCE (baseline: MCUNet_Official)\n",
        "# CIFAR-10-C — mCE and Relative mCE (baseline: MCUNet_Official)\n",
    ),
    (
        "# 1) Clean accuracy (com cache automático)\n",
        "# 1) Clean accuracy (with automatic cache)\n",
    ),
    (
        "# 2) Calcular mCE e Relative mCE\n",
        "# 2) Compute mCE and Relative mCE\n",
    ),
    (
        "# 3) Tabela resumo\n",
        "# 3) Summary table\n",
    ),
    (
        "    flag = \" ← baseline\" if m == BASELINE_NAME else \"\"\n",
        '    flag = " <- baseline" if m == BASELINE_NAME else ""\n',
    ),
    (
        "  # pode não existir",
        "  # may be missing",
    ),
    (
        "    Retorna tamanho do arquivo em MB (0.0 se não existir).",
        "    Return file size in MB (0.0 if missing).",
    ),
    (
        "    Comprime o arquivo com ZIP (deflate) e retorna o tamanho comprimido em MB.",
        "    Deflate-zip the file and return compressed size in MB.",
    ),
    (
        "# --- Coletar tamanhos ---\n",
        "# --- Collect file sizes ---\n",
    ),
    (
        "# --- Tabela ---\n",
        "# --- Table ---\n",
    ),
    (
        'print(f"CIFAR-10 — File size comparison dos Models")',
        'print(f"CIFAR-10 — model file size comparison")',
    ),
    (
        "ax.set_ylabel(\"Tamanho (MB)\")\n",
        'ax.set_ylabel("Size (MB)")\n',
    ),
    (
        'ax.set_title("CIFAR-10: File size comparison dos Models")',
        'ax.set_title("CIFAR-10: model file size comparison")',
    ),
    (
        "print(\"Plot salvo em: cifar10_model_size_comparison.pdf\")",
        "print(\"Plot saved to: cifar10_model_size_comparison.pdf\")",
    ),
    (
        "# CIFAR-100-C — mCE e Relative mCE (baseline: MCUNet_Official)\n",
        "# CIFAR-100-C — mCE and Relative mCE (baseline: MCUNet_Official)\n",
    ),
    (
        'print(f"CIFAR-100 — File size comparison dos Models")',
        'print(f"CIFAR-100 — model file size comparison")',
    ),
    (
        'ax.set_title("CIFAR-100: File size comparison dos Models")',
        'ax.set_title("CIFAR-100: model file size comparison")',
    ),
    (
        "print(\"Plot salvo em: cifar100_model_size_comparison.pdf\")",
        "print(\"Plot saved to: cifar100_model_size_comparison.pdf\")",
    ),
    # latency
    (
        "print(f\"Aviso: arquivo Android não encontrado (ignorado): {file}\")",
        "print(f\"Note: Android file not found (skipped): {file}\")",
    ),
    (
        "print(f\"Aviso: arquivo Pi não encontrado (ignorado): {pi_path}\")",
        "print(f\"Note: Pi file not found (skipped): {pi_path}\")",
    ),
    # Statistical (partial) — more in extra batch
    (
        "print(\"Amostra dos Dados de Latência:\")",
        'print("Latency data sample:")',
    ),
    (
        "    print(\"aeon indisponível. Executando `pip install aeon` no terminal se necessário.\")",
        '    print("aeon unavailable. Run `pip install aeon` in a terminal if needed.")',
    ),
    # wake common
    (
        "print(\"GPUs disponíveis:\", gpus)",
        'print("Available GPUs:", gpus)',
    ),
    (
        "GPUs disponíveis:",
        "Available GPUs:",
    ),
    (
        '"Carregar models .keras"',
        '"Load .keras models"',
    ),
    (
        "# Registrar camada customizada (necessário para carregar MCUNet)\n",
        "# Register custom layer (required to load MCUNet)\n",
    ),
    (
        "print(f\"AVISO: {path} não encontrado, pulando...\")",
        "print(f\"WARNING: {path} not found, skipping...\")",
    ),
    (
        "print(f\"\\n{len(models)} models disponíveis\")",
        "print(f\"\\n{len(models)} models available\")",
    ),
    (
        "Materializando test set em memória (pode demorar um pouco)...",
        "Materializing test set in memory (may take a while)...",
    ),
    (
        "print(f\"Distribuição: no_person=",
        "print(f\"Distribution: no_person=",
    ),
    (
        "fragmentacao de VRAM",
        "VRAM fragmentation",
    ),
    # wake plot
    (
        'ax.set_ylabel("Corrupção")',
        'ax.set_ylabel("Corruption")',
    ),
    (
        "# Marcar quais são extra\n",
        "# Mark which are extra/validation\n",
    ),
    (
        "mCE    = 15 common corruptions (métrica oficial, Hendrycks 2019)\n",
        "mCE    = 15 common corruptions (official metric, Hendrycks 2019)\n",
    ),
    (
        "print(\"mCE    = 15 common corruptions (métrica oficial, Hendrycks 2019)\")",
        "print(\"mCE    = 15 common corruptions (official metric, Hendrycks 2019)\")",
    ),
    (
        "## 1 — Carregar models .keras",
        "## 1 — Load .keras models",
    ),
    (
        "## 2 — Carregar Wake Vision test set",
        "## 2 — Load Wake Vision test set",
    ),
    (
        "# Materializa o test set em memória (necessário para aplicar corruptions por imagem)\n",
        "# Materialize the test set in memory (needed for per-image corruptions)\n",
    ),
    (
        "plt.xlabel('mCE (%) $\\leftarrow$ Robustez Absoluta', fontsize=12)\n",
        "plt.xlabel('mCE (%) $\\leftarrow$ Absolute robustness', fontsize=12)\n",
    ),
    (
        "plt.ylabel('Relative mCE (%) $\\leftarrow$ Robustez Relativa', fontsize=12)\n",
        "plt.ylabel('Relative mCE (%) $\\leftarrow$ Relative robustness', fontsize=12)\n",
    ),
    (
        "Corrupção                Tipo    MobileNetV3Small  EfficientNetB0          MCUNet\n",
        "Corruption               Type   MobileNetV3Small  EfficientNetB0          MCUNet\n",
    ),
    # Ending must be a real \\n, not the two-char sequence backslash + n
    (
        "print(f\"\\n{'Corrupção':<24s} {'Tipo':<8s}\", end=\"\")\n",
        "print(f\"\\n{'Corruption':<24s} {'Type':<8s}\", end=\"\")\n",
    ),
    (
        "print(f\"\\nBaseline para CE/mCE: {BASELINE_NAME} (acc_clean={clean_acc_baseline:.4f})\")",
        "print(f\"\\nBaseline for CE/mCE: {BASELINE_NAME} (acc_clean={clean_acc_baseline:.4f})\")",
    ),
    (
        "Baseline para CE/mCE:",
        "Baseline for CE/mCE:",
    ),
    (
        "Cada resultado é salvo linha a linha. Se o kernel reiniciar, as corruptions já avaliadas são puladas.\n",
        "Each result is written line by line. If the kernel restarts, already-finished corruptions are skipped.\n",
    ),
    (
        "Cada resultado é salvo linha a linha. Se o kernel reiniciar, as corruptions já avaliadas são puladas.",
        "Each result is written line by line. If the kernel restarts, already-finished corruptions are skipped.",
    ),
    (
        "Corrupções comuns (",
        "Common corruptions (",
    ),
    (
        "Corrupções extra/validation (",
        "Extra/validation corruptions (",
    ),
    (
        "    \"\"\"Resize + cast para float32. Não normaliza — cada model faz internamente.\"\"\"",
        '    """Resize + cast to float32. No normalization here—each model does it internally."""',
    ),
    (
        "    # gaussian usa channel_axis=-1 diretamente para evitar depender do patch 1\n",
        "    # gaussian: channel_axis=-1 to avoid depending on patch #1\n",
    ),
    (
        "  AVISO: corruption_dict não encontrado — corrupt() pode usar versão lenta",
        "  WARNING: corruption_dict not found — corrupt() may use a slow version",
    ),
    (
        "return  # versão antiga, sem necessidade de patch\n",
        "return  # old version, no patch needed\n",
    ),
    (
        "# A implementação original usa dois for-loops Python de H×W iters por imagem.\n",
        "# The original uses two H×W Python for-loops per image.\n",
    ),
    (
        "Para 224×224 × 55 K imagens isso são ~5 bilhões de iters Python — horas de execução.\n",
        "For 224×224 × 55K images that is ~5 billion Python iters—hours of runtime.\n",
    ),
    (
        "Esta versão substitui os loops por operações numpy equivalentes (~100–1000x mais rápida).\n",
        "This version replaces the loops with equivalent numpy operations (~100–1000x faster).\n",
    ),
    (
        "    # CRÍTICO: corrupt() usa corruption_dict construído na importação com\n    # referências diretas às funções — precisa ser atualizado também.\n",
        "    # CRITICAL: corrupt() uses corruption_dict built at import with\n    # direct function refs—it must be updated too.\n",
    ),
    (
        "# Compatibilidade NumPy 2.x para bibliotecas legadas (imagecorruptions)\n",
        "# NumPy 2.x compatibility for legacy libs (imagecorruptions)\n",
    ),
    (
        "# Usamos TODAS as 19 corruptions na avaliação\n",
        "# We use all 19 corruptions in evaluation\n",
    ),
    (
        "# Paralelização CPU para TODAS as corruptions por batch.\n",
        "# CPU parallelization for all corruptions per batch.\n",
    ),
    (
        "    # Adicionar constraints para rótulos para que não cruzem as linhas",
        "    # Label constraints to avoid line crossings",
    ),
    (
        "plt.xlabel('mCE (%) $\\\\leftarrow$ Robustez Absoluta', fontsize=12)\n",
        "plt.xlabel('mCE (%) $\\\\leftarrow$ Absolute robustness', fontsize=12)\n",
    ),
    (
        "plt.ylabel('Relative mCE (%) $\\\\leftarrow$ Robustez Relativa', fontsize=12)\n",
        "plt.ylabel('Relative mCE (%) $\\\\leftarrow$ Relative robustness', fontsize=12)\n",
    ),
    (
        'print("Cada model é carregado separadamente para evitar acumular VRAM\\n")',
        'print("Each model is loaded separately to limit VRAM use\\n")',
    ),
    (
        "gc.collect()  # coleta manual entre corruptions para ajudar a liberar memória",
        "gc.collect()  # manual collect between corruptions to free memory",
    ),
    (
        "# Persistência em JSONL\n",
        "# JSONL persistence\n",
    ),
    # mobile export sections
    (
        "## Seção 0 — Setup e Imports",
        "## Section 0 — Setup and imports",
    ),
    (
        "  # Use None para nao forcar uma GPU especifica\n",
        "  # Use None to not pin a specific GPU\n",
    ),
    (
        "  # Numero de imagens para amostrar\n",
        "  # Number of images to sample\n",
    ),
    (
        "## Seção 1 — Carregado Wake Vision Test Set",
        "## Section 1 — Load Wake Vision test set",
    ),
    (
        'print("Carregando Wake Vision test set...")',
        'print("Loading Wake Vision test set...")',
    ),
    (
        "Carregando Wake Vision test set...",
        "Loading Wake Vision test set...",
    ),
    (
        "# Carrega dataset completo\n",
        "# Load full dataset\n",
    ),
    (
        "# Converte para listas em memoria\n",
        "# Build in-memory lists\n",
    ),
    (
        "print(f\"✓ Dataset carregado:",
        "print(f\"✓ Dataset loaded:",
    ),
    (
        "## Seção 2 — Estratified Sampling (500 imagens)\n",
        "## Section 2 — Stratified sampling (500 images)\n",
    ),
    (
        "  Proporção:",
        "  Proportion:",
    ),
    (
        "print(f\"  Distribuição:",
        "print(f\"  Distribution:",
    ),
    (
        "Distribuição: no_person=",
        "Distribution: no_person=",
    ),
    (
        "print(f\"  Proporção:",
        "print(f\"  Proportion:",
    ),
    (
        "# Amostragem estratificada para balancear classes\n",
        "# Stratified sampling to balance classes\n",
    ),
    (
        "## Seção 3 — Exportar NPZ (uint8 compactado)\n",
        "## Section 3 — Export NPZ (packed uint8)\n",
    ),
    (
        "Preparando imagens para NPZ",
        "Preparing images for NPZ",
    ),
    (
        "## Seção 4 — Exportar PNG + CSV\n",
        "## Section 4 — Export PNG + CSV\n",
    ),
    (
        "## Seção 5 — Carregar Models .keras\n",
        "## Section 5 — Load .keras models\n",
    ),
    (
        "        print(f\"⚠ Não encontrado: {name}\")",
        "        print(f\"⚠ Not found: {name}\")",
    ),
    (
        'print("Convertendo models para TFLite...\\n")',
        'print("Converting models to TFLite...\\n")',
    ),
    (
        "## Seção 6 — Converter para TFLite (sem quantização)\n",
        "## Section 6 — Convert to TFLite (no quantization)\n",
    ),
    (
        "## Seção 7 — Gerar README_MOBILE.md\n",
        "## Section 7 — Generate README_MOBILE.md\n",
    ),
    (
        "Dataset otimizado para avaliação em dispositivos móveis.\n",
        "Dataset prepared for on-device evaluation.\n",
    ),
    (
        "img = images[0]                  # uint8, tamanho variável",
        "img = images[0]                  # uint8, variable size",
    ),
    (
        "### 2. PNG + CSV (para aplicativos móveis)\n",
        "### 2. PNG + CSV (for mobile apps)\n",
    ),
    (
        "As imagens do Wake Vision têm **tamanhos originais variables**",
        "Wake Vision images have **variable original sizes**",
    ),
    (
        "✓ Exportação concluída com sucesso!",
        "✓ Export finished successfully!",
    ),
    (
        "📍 Directory de exportação",
        "📍 Export directory",
    ),
    (
        "## Seção 8 — Resumo e Estatísticas",
        "## Section 8 — Summary and statistics",
    ),
    (
        "NUM_CLASSES = 2         # pessoa / não-pessoa",
        "NUM_CLASSES = 2         # person / non-person",
    ),
    (
        "print(\"GPUs disponíveis:\", gpus)\n",
        "print(\"Available GPUs:\", gpus)\n",
    ),
    (
        "Nenhuma GPU detectada",
        "No GPU detected",
    ),
    # mobile export: headers without trailing newline in last line of source list
    (
        "## Seção 2 — Estratified Sampling (500 imagens)",
        "## Section 2 — Stratified sampling (500 images)",
    ),
    (
        "print(f\"  Distribuicao: no_person=",
        "print(f\"  Distribution: no_person=",
    ),
    (
        "print(f\"Classe: {['no_person', 'person'][class_id]}",
        "print(f\"Class: {['no_person', 'person'][class_id]}",
    ),
    (
        ", confiança: {confidence:.4f}",
        ", confidence: {confidence:.4f}",
    ),
    (
        "print(f\"Latência: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms\")",
        "print(f\"Latency: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms\")",
    ),
    (
        "print(\"RESUMO DA EXPORTAÇÃO\")",
        "print(\"EXPORT SUMMARY\")",
    ),
    # Global PT verbs in comments/prints (safe in this repo)
    (
        "Carregando",
        "Loading",
    ),
    (
        "Carregar",
        "Load",
    ),
    (
        "Carrega ",
        "Load ",
    ),
    (
        "# Carrega",
        "# Load",
    ),
    (
        "  Latência",
        "  Latency",
    ),
    (
        "Latência",
        "Latency",
    ),
    (
        "média",
        "mean",
    ),
    (
        "mínima",
        "min",
    ),
    (
        "máxima",
        "max",
    ),
    (
        "utilitários",
        "utilities",
    ),
    (
        "Hiperparâmetros",
        "Hyperparameters",
    ),
    (
        "## 3 — Funções de avaliação com corruptions on-the-fly",
        "## 3 — Evaluation with on-the-fly corruptions",
    ),
    (
        "## 5 — Evaluation de corruptions (com persistence JSONL)",
        "## 5 — Corruption evaluation (with JSONL persistence)",
    ),
    (
        "models disponíveis",
        "models available",
    ),
    (
        "# Carrega results (funciona mesmo com kernel reiniciado)\n",
        "# Load results (works after kernel restarts too)\n",
    ),
    (
        "BASELINE_NAME = \"MCUNet\" # Fallback/Hardcode automático",
        "BASELINE_NAME = \"MCUNet\"  # auto fallback (hardcoded)",
    ),
    (
        "# 2. Recalcular CE para todos os models\n",
        "# 2. Recompute CE for all models\n",
    ),
    (
        "CRÍTICO: corrupt() usa corruption_dict construído na importação com\n",
        "CRITICAL: corrupt() uses corruption_dict built at import with\n",
    ),
    (
        "  # 15 para mCE padrão",
        "  # 15 for default mCE",
    ),
    (
        "    # referências diretas às funções — precisa ser atualizado também.\n",
        "    # direct function references — those must be updated as well.\n",
    ),
    (
        "    # Patch no atributo do módulo (usado por chamadas diretas)\n",
        "    # Patch module attribute (for direct call paths)\n",
    ),
    (
        "## Como Carregar em Python\n",
        "## How to load in Python\n",
    ),
    (
        "## Informações Importantes\n",
        "## Important information\n",
    ),
    (
        "Para inferência em model, sempre redimensione para 224×224.\n",
        "For inference, always resize to 224×224.\n",
    ),
    (
        "Cada model é carregado separadamente para evitar acumular VRAM\n",
        "Each model is loaded separately to limit VRAM use\n",
    ),
    (
        "Cada model é carregado separadamente para evitar acumular VRAM",
        "Each model is loaded separately to limit VRAM use",
    ),
    (
        "## Seção 3 — Exportar NPZ (uint8 compactado)",
        "## Section 3 — Export NPZ (packed uint8)",
    ),
    (
        "## Seção 4 — Exportar PNG + CSV",
        "## Section 4 — Export PNG + CSV",
    ),
    (
        "## Seção 5 — Load Models .keras",
        "## Section 5 — Load .keras models",
    ),
    (
        "## Seção 6 — Converter para TFLite (sem quantização)",
        "## Section 6 — Convert to TFLite (no quantization)",
    ),
    (
        "## Seção 7 — Gerar README_MOBILE.md",
        "## Section 7 — Generate README_MOBILE.md",
    ),
    (
        "print(f\"Exportando {SAMPLE_SIZE} imagens PNG...\")\n",
        "print(f\"Exporting {SAMPLE_SIZE} PNG images...\")\n",
    ),
    (
        "    # Converte para uint8 se necessário\n",
        "    # Convert to uint8 if needed\n",
    ),
    (
        "    # Converte para TFLite (sem quantização por padrão)\n",
        "    # Convert to TFLite (not quantized by default)\n",
    ),
    (
        "    # Libera memória\n",
        "    # Free memory\n",
    ),
    (
        "# Obtém tensor info\n",
        "# Get tensor info\n",
    ),
    (
        "# Obtém predição\n",
        "# Get prediction\n",
    ),
    (
        "// Obtém resultado\n",
        "// Get result\n",
    ),
    (
        "### Tamanhos de Imagem Variáveis\n",
        "### Variable image sizes\n",
    ),
    (
        "Para medir latência e accuracy em dispositivos:\n",
        "To measure latency and accuracy on devices:\n",
    ),
    (
        "treino será na CPU (muito mais lento)",
        "training will run on CPU (much slower)",
    ),
    (
        "versão curada com menor taxa de erro",
        "curated version with lower labeling error",
    ),
    (
        "devem ser filtradas para avaliação padrão.",
        "should be filtered out for standard evaluation.",
    ),
    (
        "augmentação opcional",
        "optional augmentation",
    ),
    (
        "Augmentação simples",
        "Simple augmentation",
    ),
    (
        "## 5 — Hyperparameters e utilities de treino",
        "## 5 — Hyperparameters and training utilities",
    ),
    (
        "# ---------- Hyperparameters de treino ----------\n",
        "# ---------- Training hyperparameters ----------\n",
    ),
    (
        "# ---------- Funções utilitárias ----------\n",
        "# ---------- Utility functions ----------\n",
    ),
    (
        "    \"\"\"Input + camada de pré-processamento (layer puro → serializável).\"\"\"",
        "    \"\"\"Input + preprocess layer (plain layer, serializable).\"\"\"",
    ),
    (
        "    \"\"\"Label smoothing compatível com labels sparse (int).\"\"\"",
        "    \"\"\"Label smoothing for sparse (int) labels.\"\"\"",
    ),
    (
        "    \"\"\"AdamW quando disponível; fallback para Adam.\"\"\"",
        "    \"\"\"AdamW if available; otherwise Adam.\"\"\"",
    ),
    (
        "        include_preprocessing=False,  # já fazemos o Rescaling acima",
        "        include_preprocessing=False,  # we already rescale above",
    ),
    (
        "    \"\"\"Fase 1: treina cabeça (backbone congelado). Fase 2: fine-tune parcial.\"\"\"",
        "    \"\"\"Phase 1: train head (frozen backbone). Phase 2: partial fine-tune.\"\"\"",
    ),
    (
        "Fase 1 — Treino da cabeça",
        "Phase 1 — head training",
    ),
    (
        "## 8 — Evaluation e métricas",
        "## 8 — Evaluation and metrics",
    ),
    (
        "Métricas clássicas a partir da matriz de confusão",
        "Classic metrics from the confusion matrix",
    ),
    (
        "Avalia model e plota heatmap + matriz de confusão",
        "Evaluate model, plot heatmap and confusion matrix",
    ),
    (
        " matriz de confusão",
        " confusion matrix",
    ),
    (
        "não tem 'best_weights'. Rode o treino antes",
        "does not have 'best_weights'. Run training first",
    ),
    (
        "Mesmo protocolo de treino (head → fine-tune) aplicado ao",
        "Same training protocol (head → fine-tune) applied to",
    ),
    (
        "O EfficientNetB0 já possui normalização interna",
        "EfficientNetB0 already has internal normalization",
    ),
    (
        "que contém models pré-treinados no ImageNet",
        "with ImageNet pre-trained models",
    ),
    (
        "1. Load o model PyTorch pré-treinado",
        "1. Load the pre-trained PyTorch model",
    ),
    (
        "O workflow é:\n",
        "Workflow:\n",
    ),
    (
        "O MCUNet usa normalização ImageNet",
        "MCUNet uses ImageNet normalization",
    ),
    (
        "MCUNet Oficial — Models disponíveis",
        "MCUNet (official) — available models",
    ),
    (
        "Parâmetros totais (PyTorch):",
        "Total parameters (PyTorch):",
    ),
    (
        "Resolução original:",
        "Native resolution:",
    ),
    (
        "Descrição:",
        "Description:",
    ),
    (
        "1. Load o model PyTorch",
        "1. Load the PyTorch model",
    ),
    # Statistical_Tests, cifar, etc.
    (
        "Gerando CD Diagram para Latência",
        "Generating critical-difference diagram for latency",
    ),
    (
        "# 1. Carregar/Processar Dados: ROBUSTEZ (CE por Corrupção)\n",
        "# 1. Load data: robustness (CE by corruption)\n",
    ),
    (
        "Amostra dos Dados de Latência",
        "Latency data sample",
    ),
    (
        "# 2. Carregar/Processar Dados: LATÊNCIA\n",
        "# 2. Load data: latency\n",
    ),
    (
        "ROBUSTEZ",
        "ROBUSTNESS",
    ),
    (
        "    \"\"\"Avalia um model em uma corruption para todas as severidades.\"\"\"",
        "    \"\"\"Evaluate one model on one corruption across all severities.\"\"\"",
    ),
    (
        "# Determina automaticamente o model com a menor accuracy para ser o baseline\n",
        "# Pick the lowest-accuracy model as the baseline automatically\n",
    ),
    (
        "# Desduplicar (último registro por chave)\n",
        "# De-duplicate (keep last record per key)\n",
    ),
    (
        "    # Sempre usa o wrapper corrupt() para manter compatibilidade com corruptions\n",
        "    # Always use corrupt() wrapper for compatibility with corruptions\n",
    ),
    (
        "# Converte para uint8 e salva como NPZ\n",
        "# Convert to uint8 and save as NPZ\n",
    ),
    (
        "# Registra camada customizada para MCUNet\n",
        "# Register custom layer for MCUNet\n",
    ),
    (
        "### 1. NPZ (recomendado para prototipagem)\n",
        "### 1. NPZ (recommended for prototyping)\n",
    ),
    (
        "# Prepara input\n",
        "# Prepare input\n",
    ),
    (
        "// Prepara input (float32, shape 1x224x224x3)\n",
        "// Prepare input (float32, shape 1x224x224x3)\n",
    ),
    (
        "// Prepara output (float32, shape 1x2)\n",
        "// Prepare output (float32, shape 1x2)\n",
    ),
    (
        "        # Simple augmentation para imagens do mundo real\n",
        "        # Simple augmentation for real-world images\n",
    ),
    (
        "    # Resize para o tamanho do backbone\n",
        "    # Resize to backbone input size\n",
    ),
    (
        "IMG_SIZE = 224          # MobileNetV3 foi projetado para 224×224\n",
        "IMG_SIZE = 224          # MobileNetV3 is designed for 224×224\n",
    ),
    (
        "    \"\"\"Recria o model (não compilado), carrega os melhores pesos e salva em .keras.\"\"\"",
        "    \"\"\"Rebuild the model (uncompiled), load best weights, save to .keras.\"\"\"",
    ),
    (
        "    \"\"\"MobileNetV3-Small com cabeça customizada para classificação binária.\"\"\"",
        "    \"\"\"MobileNetV3-Small with a custom head for binary classification.\"\"\"",
    ),
    (
        "    \"\"\"EfficientNetB0 com cabeça customizada para Wake Vision.\"\"\"",
        "    \"\"\"EfficientNetB0 with a custom head for Wake Vision.\"\"\"",
    ),
    (
        "    # EfficientNet já normaliza internamente — stem usa Identity\n",
        "    # EfficientNet already normalizes internally — stem uses Identity\n",
    ),
    (
        "    # Evita dupla normalização se a versão tiver preprocessing embutido\n",
        "    # Avoid double normalization if preprocessing is built in\n",
    ),
    (
        "2. Converter os pesos para Keras (PyTorch → Keras)\n",
        "2. Convert weights to Keras (PyTorch → Keras)\n",
    ),
    (
        "# Camada customizada para normalização ImageNet (registrada para serialização)\n",
        "# Custom ImageNet normalization layer (registered for serialization)\n",
    ),
    (
        "Constrói um model Keras baseado na arquitetura MCUNet oficial.",
        "Build a Keras model from the official MCUNet architecture.",
    ),
    (
        "Load pesos pré-treinados do ImageNet (PyTorch) e converte para Keras.",
        "Load ImageNet pre-trained PyTorch weights and convert to Keras.",
    ),
    (
        "    Retorna (model_completo, backbone) para compatibilidade com train_head_and_finetune.\n",
        "    Returns (full_model, backbone) for train_head_and_finetune.\n",
    ),
    (
        "    # Load model PyTorch com pesos pré-treinados\n",
        "    # Load PyTorch model with pre-trained weights\n",
    ),
    (
        "    # Extrai configuração da rede\n",
        "    # Read network config\n",
    ),
    (
        "    # Parâmetros de BatchNorm\n",
        "    # BatchNorm parameters\n",
    ),
    (
        "    # Helper para ativação\n",
        "    # Activation helper\n",
    ),
    (
        "            pass  # Ignora erros de camadas não mapeadas\n",
        "            pass  # ignore unmapped layer errors\n",
    ),
    (
        "    # Congela backbone para treinar só a cabeça\n",
        "    # Freeze backbone to train the head only\n",
    ),
    (
        "Gerando CD Diagram para Robustez",
        "Generating critical-difference diagram for robustness",
    ),
    (
        "Gerando CD Diagram para Latency",
        "Generating critical-difference diagram for latency",
    ),
    (
        "# 1. Load/Processar Dados: ROBUSTNESS (CE por Corrupção)\n",
        "# 1. Load data: robustness (CE by corruption)\n",
    ),
    (
        "Visualização de diferenças críticas do aeon",
        "Critical-difference visualization (aeon)",
    ),
    (
        "# ===================== Configurações Iniciais =====================\n",
        "# ===================== Initial settings =====================\n",
    ),
    (
        "# 3. Funções Estatísticas (Friedman / Diferença Crítica)\n",
        "# 3. Statistical functions (Friedman / critical difference)\n",
    ),
    (
        "        print(\"Aviso: Existem NaNs nos ranqueamentos. Removendo linhas incompletas para o CD plot.\")",
        "        print(\"Warning: NaNs in ranks. Dropping incomplete rows for the CD plot.\")",
    ),
    (
        "Extraindo para CIFAR-10-C, CIFAR-100-C e Wake Vision\n",
        "Extracting for CIFAR-10-C, CIFAR-100-C, and Wake Vision\n",
    ),
    (
        "# 4. Testes para Robustez (Corruption Error)\n",
        "# 4. Tests for robustness (corruption error)\n",
    ),
    (
        "# 5. Testes para Latency\n",
        "# 5. Tests for latency\n",
    ),
]

# Deduplicate by first element (keep longer run order by rebuilding from sorted merge in main)
PHRASES_PASS2.sort(key=lambda t: len(t[0]), reverse=True)
