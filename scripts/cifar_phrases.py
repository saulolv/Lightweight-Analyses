"""PT→EN for cifar-10_analyses.ipynb and cifar-100.analyses.ipynb (key visible strings + comments)."""
from __future__ import annotations

PHRASES_CIFAR: list[tuple[str, str]] = [
    (
        "# Load CIFAR-10 (download automático na 1ª vez)\n",
        "# Load CIFAR-10 (downloaded automatically on the first run)\n",
    ),
    (
        "# Carrega CIFAR-10 (download automático na 1ª vez)\n",
        "# Load CIFAR-10 (downloaded automatically on the first run)\n",
    ),
    (
        "# Mantém imagens em [0,255] (uint8) — o preprocess_input dos models cuida da escala certa\n",
        "# Keep images in [0, 255] (uint8) — each model’s preprocess_input handles scaling\n",
    ),
    (
        "# Ajusta rótulos para shape (N,) (int)\n",
        "# Squeeze labels to shape (N,) with dtype int\n",
    ),
    (
        'print("Exemplo de rótulo:", int(y_train[0]), "->", class_names[int(y_train[0])])',
        'print("Example label:", int(y_train[0]), "->", class_names[int(y_train[0])])',
    ),
    (
        "## Preparação do dataset (train/val/test)\n",
        "## Dataset preparation (train/val/test)\n",
    ),
    (
        "# Split train -> train/val (segurando os últimos VAL_SIZE exemplos como validação)\n",
        "# Split train into train/val (last VAL_SIZE examples = validation)\n",
    ),
    (
        "        # Augmentação clássica do CIFAR: pad + random crop + flip\n",
        "        # Classic CIFAR augmentation: pad + random crop + flip\n",
    ),
    (
        "    # Resize para o tamanho do backbone (mantém o model \"limpo\" p/ TFLite)\n",
        "    # Resize to backbone size (keeps a clean graph for TFLite)\n",
    ),
    (
        "## Configuração comum (para todos os models)\n",
        "## Common settings (all models)\n",
    ),
    (
        "# Como o dataset já entrega imagens (IMG_SIZE, IMG_SIZE, 3), o input do model fica fixo.\n",
        "# The dataset already yields (IMG_SIZE, IMG_SIZE, 3) so the model input is fixed.\n",
    ),
    (
        "Esta cell concentra tudo que é compartilhado entre os models (augmentação, callbacks, hiperparâmetros e funções auxiliares). Para testar novos models (ex.: MCU-Net), basta criar um `build_*()` e chamar `train_head_and_finetune(...)`.\n",
        "This cell holds everything shared across models (augmentation, callbacks, hyperparameters, helpers). To try a new model (e.g. MCU‑Net), add a `build_*()` and call `train_head_and_finetune(...)`.\n",
    ),
    (
        "    \"\"\"Cria Input + camada de pré-processamento do model.\n",
        "    \"\"\"Build Input + a preprocess layer for the model.\n",
    ),
    (
        "    Importante: usar **layers** (e não `Lambda` com funções Python) torna o model\n",
        "    Prefer using **Keras layers** (not a Python `Lambda`) so the model is\n",
    ),
    (
        "    salvável/carregável como `.keras` sem precisar de `custom_objects`.\n",
        "    easy to save/load as `.keras` without extra `custom_objects`.\n",
    ),
    (
        "    \"\"\"Label smoothing compatível com labels sparse.\n",
        "    \"\"\"Label smoothing for sparse (int) labels.\n",
    ),
    (
        "    Alguns ambientes não suportam `label_smoothing` em SparseCategoricalCrossentropy.\n",
        "    Some versions don’t support `label_smoothing` in SparseCategoricalCrossentropy.\n",
    ),
    (
        '    """AdamW quando disponível; fallback para Adam se não existir."""',
        '    """AdamW when available; otherwise Adam."""',
    ),
    (
        "    \"\"\"Recria o model (não compilado), carrega os melhores pesos e salva em `.keras`.\n",
        "    \"\"\"Reload the uncompiled model, load best weights, and save a `.keras` file.\n",
    ),
    (
        "    Isso evita problemas de serialização do treino (loss custom, etc.).\n",
        "    This avoids common training serialization issues (custom loss, etc.).\n",
    ),
    (
        "    # Fine-tune (ou 2ª fase de treino quando base=None)\n",
        "    # Fine-tune (2nd training phase when base=None)\n",
    ),
    (
        "        print(\"Arquivo não encontrado (pulei):\", keras_path)",
        "        print(\"File not found (skipping):\", keras_path)",
    ),
    (
        "    raise ValueError(f\"Activation não suportada: {act_name}\")",
        "    raise ValueError(f\"Unsupported activation: {act_name}\")",
    ),
    (
        "    \"\"\"Mede latência de inferência do model Keras.\"\"\"",
        "    \"\"\"Measure Keras model inference latency.\"\"\"",
    ),
    (
        "    # Medir latência",
        "    # Measure latency",
    ),
    (
        "    \"\"\"Conta parâmetros total e treináveis.\"\"\"",
        "    \"\"\"Count total and trainable parameters.\"\"\"",
    ),
    (
        "Dica: para Android, normalmente você vai querer exportar só um **subconjunto** (ex.: 500–2000 imagens) pra não inflar o app.\n",
        "Tip: on Android, export only a **subset** (e.g. 500–2000 images) to keep the app size small.\n",
    ),
    (
        "Depois de reiniciar o kernel, variables como `mnv3_results` podem não existir — então aqui a exportação funciona **por caminho de arquivo**.\n",
        "After a kernel restart, variables like `mnv3_results` may be gone — this export path uses **file paths** only.\n",
    ),
    (
        "Você pode exportar de duas formas:\n",
        "You can export in two ways:\n",
    ),
    (
        "# IMPORTANTE: Registra a camada customizada caso não tenha sido definida ainda\n",
        "# Register the custom layer if it is not in scope yet\n",
    ),
    (
        "        print(f\"⚠ Model não encontrado: {keras_path}\")",
        "        print(f\"⚠ Model not found: {keras_path}\")",
    ),
    (
        "Dica: para Android, normalmente você vai querer exportar",
        "Tip: for Android, you usually want to export",
    ),
    (
        "Também incluir TFLite para comparação de tamanho",
        "Also include TFLite for a size comparison",
    ),
    (
        "    model_tag=\"MCUNet_TL\",  # Nome diferente para não sobrescrever",
        "    model_tag=\"MCUNet_TL\",  # distinct tag to avoid overwrites",
    ),
    (
        "COMPARAÇÃO COMPLETA",
        "FULL COMPARISON",
    ),
    (
        "Comparação: Todas as versões do MCUNet (Accuracy, Tamanho, Latency)",
        "Comparison: all MCUNet variants (accuracy, size, latency)",
    ),
    (
        "print(\"MCUNet Oficial - Models disponíveis:\")",
        'print("MCUNet (official) — available models:")',
    ),
    (
        "    # Entrada do dataset está em float32 no range [0,255]\n",
        "    # Dataset output is float32 in [0, 255]\n",
    ),
    (
        "        include_preprocessing=False,  # já fazemos o Rescaling(-1..1) acima",
        "        include_preprocessing=False,  # we already rescale to (-1, 1) above",
    ),
    # Statistical — leftover
    (
        "print(\"\\nGerando CD Diagram para Robustez...\")",
        "print(\"\\nGenerating critical-difference diagram for robustness...\")",
    ),
    (
        "Neste test, os Tratamentos são os \"Models\" e os Blocos são \"Dataset\" + \"Corruption\".",
        "Here treatments are the models, and blocks are (dataset, corruption) pairs.",
    ),
    (
        "Neste test, os Tratamentos são os \"Models\" e os Blocos são as \"Runs\" (agrupadas por dataset).",
        "Here treatments are models, and blocks are run groups per dataset.",
    ),
    (
        "plot_per_class_accuracy_heatmap(m[\"per_class_accuracy\"], class_names, title=f\"{title} - accuracy por classe\")",
        "plot_per_class_accuracy_heatmap(m[\"per_class_accuracy\"], class_names, title=f\"{title} - per-class accuracy\")",
    ),
    (
        "plot_confusion_matrix_heatmap(cm, class_names, title=f\"{title} - confusion matrix (normalizada)\", normalize=True)",
        "plot_confusion_matrix_heatmap(cm, class_names, title=f\"{title} - confusion matrix (normalized)\", normalize=True)",
    ),
    (
        "    \"\"\"Treina cabeça (base congelada) e depois faz fine-tuning parcial (se base != None).",
        "    \"\"\"Train the head (frozen base) then fine-tune partially (if base is not None).",
    ),
    (
        "            # Boa prática: manter BatchNorm congelado no fine-tune",
        "            # Keep BatchNorm in inference mode during fine-tuning (common practice)",
    ),
    (
        "# Salva o melhor model como .keras (carregável) e avalia (accuracy/F1/heatmap)\n",
        "# Save the best model as .keras and evaluate (accuracy, F1, heatmaps)\n",
    ),
    (
        "Aqui usamos o **repositório oficial do MCUNet** (`third_party/mcunet-official`) que contém:\n",
        "We use the **official MCUNet** repo (`third_party/mcunet-official`), which provides:\n",
    ),
    (
        "print(f\"Parâmetros totais: {total_params:,}\")",
        "print(f\"Total parameters: {total_params:,}\")",
    ),
    (
        "print(f\"Parâmetros treináveis: {trainable_params:,}\")",
        "print(f\"Trainable parameters: {trainable_params:,}\")",
    ),
    (
        "A ideia aqui é salvar o **mesmo `x_test/y_test`** em formatos fáceis de levar para outros dispositivos:\n",
        "Here we store the same **`x_test`/`y_test`** in portable formats for other devices:\n",
    ),
    (
        "latência de inferência",
        "inference latency",
    ),
    (
        "# Treino (comum)\n",
        "# Training (shared)\n",
    ),
    (
        "    - Salva o melhor checkpoint por val_accuracy (útil para exportar/convert.\n",
        "    - Keeps the best val_accuracy checkpoint (handy to export/convert later).\n",
    ),
    (
        "# O mcunet-in4 (512KB SRAM, 2MB Flash) - maior model, melhor para comparação\n",
        "# mcunet-in4 (512KB SRAM, 2MB Flash) is the largest variant—best for apples-to-apples comparison\n",
    ),
    (
        "   Medindo latência (",
        "   Measuring latency (",
    ),
    (
        "⚠ Model não encontrado:",
        "⚠ Model not found:",
    ),
    (
        "        # Parâmetros\n",
        "        # Parameters\n",
    ),
    (
        "        print(f\"   Medindo latência ({NUM_RUNS} runs)...\")",
        "        print(f\"   Measuring latency ({NUM_RUNS} runs)...\")",
    ),
    (
        "Parâmetros totais:",
        "Total parameters:",
    ),
    (
        "Parâmetros treináveis:",
        "Trainable parameters:",
    ),
    (
        "print(\"Comparação dos tamanhos dos models em .keras:\")",
        'print("Keras model size comparison:")',
    ),
    (
        "print(\"Comparação dos tamanhos dos models em .tflite:\")",
        'print("TFLite model size comparison:")',
    ),
    (
        "Comparação dos tamanhos dos models em .keras:\n",
        "Keras model size comparison:\n",
    ),
    (
        "Comparação dos tamanhos dos models em .tflite:\n",
        "TFLite model size comparison:\n",
    ),
    (
        "print(\"COMPARAÇÃO: MCUNet (do zero) vs MCUNet (Transfer Learning)\")",
        'print("COMPARISON: MCUNet (scratch) vs MCUNet (transfer learning)")',
    ),
    (
        "COMPARAÇÃO: MCUNet (do zero) vs MCUNet (Transfer Learning)\n",
        "COMPARISON: MCUNet (scratch) vs MCUNet (transfer learning)\n",
    ),
    (
        "ANÁLISE DE LATÊNCIA - MODELOS TFLITE CIFAR-10\n",
        "TFLITE MODEL LATENCY — CIFAR-10\n",
    ),
    (
        "print(\"ANÁLISE DE LATÊNCIA - MODELOS TFLITE CIFAR-10\")",
        "print(\"TFLITE MODEL LATENCY — CIFAR-10\")",
    ),
    (
        "NUM_INFERENCE_RUNS = 100  # Número de inferências para medir latência\n",
        "NUM_INFERENCE_RUNS = 100  # inferences to measure latency\n",
    ),
    (
        "ANÁLISE DE LATÊNCIA - MODELOS TFLITE CIFAR-10",
        "TFLITE MODEL LATENCY — CIFAR-10",
    ),
    # cifar-100 (subset)
    (
        "        print(\"PyTorch não está instalado. Pulando carregamento de pesos pré-treinados.\")",
        "        print(\"PyTorch is not installed. Skipping pre-trained weight loading.\")",
    ),
    (
        "        print(f\"Arquivo de pesos não encontrado: {pytorch_weights_path}\")",
        "        print(f\"Weight file not found: {pytorch_weights_path}\")",
    ),
    (
        "    print(f\"Carregados {loaded_count} camadas com pesos pré-treinados do ImageNet.\")",
        "    print(f\"Loaded {loaded_count} ImageNet pre-trained layer groups.\")",
    ),
    (
        "Tip: for Android, you usually want to export só um **subconjunto** (ex.: 500–2000 imagens) pra não inflar o app.",
        "Tip: for Android, you usually want to export only a **subset** (e.g. 500–2000 images) to keep the app small.",
    ),
    (
        "## Comparação completa (Accuracy, Tamanho, Latency, FPS)\n",
        "## Full comparison (accuracy, size, latency, FPS)\n",
    ),
    (
        "            # Se não encontrar, usa valores padrão",
        "            # if missing, fall back to defaults",
    ),
    (
        "Pesos carregados",
        "Weights loaded",
    ),
]

PHRASES_CIFAR.sort(key=lambda t: len(t[0]), reverse=True)
