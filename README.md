# Reproducibility Package

> **Anonymous submission** — This repository is provided for peer review purposes only. Author identities have been removed in compliance with the double-blind review policy.

This repository contains the complete reproducibility package for the paper:

> *"Lightweight CNNs for Edge Vision: A Cross-Benchmark Study of Robustness and Efficiency"*

The package covers all stages of the experimental pipeline: model training and export, corruption-robustness evaluation, on-device benchmarking on Android smartphones and Raspberry Pi Zero 2 W, statistical testing, and figure generation.

---

## Repository Structure

```
.
├── notebooks/                   # Jupyter notebooks — training, robustness, latency, figures
│   ├── cifar-10_analyses.ipynb
│   ├── cifar-100_analyses.ipynb
│   ├── wake_vision_analyses.ipynb
│   ├── wake_vision_mobile_export.ipynb
│   ├── cifar10c_corruption_error.ipynb
│   ├── cifar100c_corruption_error.ipynb
│   ├── wake_vision_robustness.ipynb
│   ├── robustness_results.ipynb
│   ├── latency_analyses.ipynb
│   ├── Statistical_Tests.ipynb
│   └── output/                  # Intermediate notebook outputs
├── Raspberry/                   # Raspberry Pi benchmark scripts and Docker environment
│   ├── eval_tflite_multidataset.py
│   └── Dockerfile
├── App/                         # Android benchmark application (Kotlin / Jetpack Compose)
├── exports/                     # Exported TFLite model files (.tflite)
├── results/                     # Raw benchmark JSON results (Android + Raspberry Pi)
├── figures/                     # Generated figures used in the paper
└── paths.py                     # Centralized path configuration
```

---

## Experimental Setup

### Architectures

Three lightweight CNN architectures are compared:

| Model | Backbone | Input size | TFLite size |
|---|---|---|---|
| MobileNetV3-Small | ImageNet pretrained | 160×160 (CIFAR), 224×224 (Wake Vision) | ~3.6 MB |
| EfficientNet-B0 | ImageNet pretrained | 160×160 (CIFAR), 224×224 (Wake Vision) | ~15.3 MB |
| MCUNet | ImageNet pretrained (converted from PyTorch) | 160×160 (CIFAR), 224×224 (Wake Vision) | ~5.4 MB |

### Datasets

- **CIFAR-10** — 10-class natural image classification (input resized to 160×160)
- **CIFAR-100** — 100-class natural image classification (input resized to 160×160)
- **Wake Vision** — binary person detection, TinyML-oriented (`train_quality` / `validation` / `test` splits via `tensorflow_datasets`)

### Corruption Benchmarks

Robustness is evaluated using the CIFAR-C protocol: 15 corruption types × 5 severity levels = 75 conditions per dataset. The same corruption taxonomy is applied to Wake Vision to produce a matched benchmark.

### Deployment Protocol

All models are deployed as **float32 TensorFlow Lite** with **CPU-only** inference and **no quantization or hardware delegate**.

| Device | Threads | Notes |
|---|---|---|
| Samsung Galaxy S24+ | 4 | Kotlin benchmark app |
| Samsung Galaxy A14 | 4 | Kotlin benchmark app |
| Raspberry Pi Zero 2 W | 1 | Python script, executed inside Docker (`--threads 1`) |

Each benchmark run uses 10 warm-up inferences followed by 400 images per dataset (balanced subset). Models are executed independently per dataset on all devices.

> **Note on thread count (Pi Zero 2 W):** All reported results were obtained with `--threads 1`. Multi-threaded TFLite inference on low-power ARM boards (Cortex-A53) incurs synchronization overhead that can negate parallelism gains; restricting to a single thread also places the Pi Zero 2 W in the same operating regime as single-core embedded processors (e.g., Cortex-M4/M7 class), making it a conservative proxy for the most resource-constrained tier of edge deployment.

---

## Reproducing the Results

### 1. Training and Export (Notebooks)

Open and run the following notebooks in order for each dataset:

```
notebooks/cifar-10_analyses.ipynb        # CIFAR-10 training + TFLite export
notebooks/cifar-100_analyses.ipynb      # CIFAR-100 training + TFLite export
notebooks/wake_vision_analyses.ipynb    # Wake Vision training
notebooks/wake_vision_mobile_export.ipynb  # Wake Vision TFLite export
```

Exported `.tflite` files are saved to `exports/`.

### 2. Corruption Robustness

Run the following notebooks to compute per-corruption error and aggregate metrics (mCE, relative mCE):

```
notebooks/cifar10c_corruption_error.ipynb
notebooks/cifar100c_corruption_error.ipynb
notebooks/wake_vision_robustness.ipynb
notebooks/robustness_results.ipynb
```

### 3. Android Benchmark

The Android benchmark application is located in `App/`. It is implemented in **Kotlin with Jetpack Compose** and uses `tflite` interpreter with CPU inference (no delegate, up to 4 threads).

To reproduce:
1. Open the `App/` folder in Android Studio.
2. Build and install on a physical Android device.
3. Place the exported `.tflite` models and image subsets in the expected paths.
4. Run the benchmark — results are saved as JSON to `results/`.

### 4. Raspberry Pi Benchmark (Docker)

The Raspberry Pi evaluation uses a Python script executed inside Docker for environment reproducibility.

**Build the Docker image** (run from the `Raspberry/` folder):

```bash
docker build -t lightweight-pi .
```

**Run the benchmark** (mounting models and results from the host):

```bash
docker run --rm -it \
  -v /path/to/exports:/app/exports \
  -v /path/to/results:/app/results \
  lightweight-pi python eval_tflite_multidataset.py --threads 1
```

> **Important:** `--threads 1` is now the script default and must be preserved to reproduce the paper's reported results. Do not omit or increase this value unless you intentionally want to measure multi-threaded behaviour.

The script runs each model independently per dataset with a single thread, performs 10 warm-up inferences, then evaluates 400 images. Results are written to `results/raspberry/results/results_multidataset.json`.

### 5. Statistical Tests and Figures

```
notebooks/Statistical_Tests.ipynb     # Friedman + Nemenyi analyses
notebooks/latency_analyses.ipynb      # Latency and throughput figures
```

All generated figures are saved to `figures/` in PDF format.

---

## Dependencies

### Python Notebooks

The notebooks were developed with Python 3.10. The main dependencies are:

```
tensorflow >= 2.12
tensorflow-datasets
numpy < 2
pandas
matplotlib
seaborn
scipy
scikit-posthocs
pillow
```

Install with:

```bash
pip install tensorflow tensorflow-datasets "numpy<2" pandas matplotlib seaborn scipy scikit-posthocs pillow
```

### Raspberry Pi (Docker)

The `Raspberry/Dockerfile` installs all required dependencies automatically:

```
tflite-runtime
numpy < 2
pillow
opencv-python-headless
```

See `Raspberry/Dockerfile` for the complete environment definition.

---

## Results Files

Raw benchmark results are stored in `results/` as JSON files.

### Android
| File | Source |
|---|---|
| `s24p_metrics_*.json` | Samsung Galaxy S24+ benchmark app |
| `a14_metrics_*.json` | Samsung Galaxy A14 benchmark app |

### Raspberry Pi
Individual Raspberry Pi benchmark files are located in `results/raspberry/results/`:

| File | Description |
|---|---|
| `cifar10.json` | CIFAR-10 benchmark results |
| `cifar100.json` | CIFAR-100 benchmark results |
| `wakevision.json` | Wake Vision benchmark results |
| `cifar10_efficientnet.json` | Additional EfficientNet CIFAR-10 run |
| `log.txt` | Execution log |
| `results/raspberry/results/results_merged.json` | Merged Pi Zero 2 W results (Docker, 1 thread) |

Older legacy files (`metrics_*_cpu_500imgs_*.json`) are preserved for reference but are not used in the current analyses.

---

## Notes on MCUNet

MCUNet was originally released as a PyTorch model. It is imported from `third_party/mcunet-official/` and the conversion to Keras is handled automatically by the relevant notebooks.

---

## License

This repository is made available for peer review purposes. License information will be provided upon paper acceptance.
