"""Evaluate TFLite models on CIFAR-10, CIFAR-100 and Wake Vision NPZ test sets."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import project_root

# Import TFLite interpreter from tflite_runtime or tensorflow
tflite: Any | None = None
TFLITE_IMPORT_ERROR: str | None = None
try:
    import tflite_runtime.interpreter as tflite  # type: ignore[assignment]
except Exception:
    try:
        import tensorflow as tf

        tflite = SimpleNamespace(Interpreter=tf.lite.Interpreter)
        print("Warning: using TensorFlow Lite (tf.lite.Interpreter) as fallback.")
    except Exception as import_err:
        TFLITE_IMPORT_ERROR = (
            "Failed to import tflite_runtime or tensorflow. "
            "Install one of them (recommended: python3-tflite-runtime on Raspberry Pi). "
            f"Original error: {import_err}"
        )


ROOT = project_root()

DATASET_SPECS = {
    "cifar10": {
        "npz_file": ROOT / "exports" / "cifar10" / "cifar10_test_uint8.npz",
        "models": [
            {"name": "mobilenetv3_small", "file": ROOT / "tflite_models" / "MobileNetV3Small_CIFAR10.tflite"},
            {"name": "efficientnet_b0", "file": ROOT / "tflite_models" / "EfficientNetB0_CIFAR10.tflite"},
            {"name": "lcnn", "file": ROOT / "tflite_models" / "MCUNet_CIFAR10.tflite"},
        ],
        "class_names": None,  # Prefer class_names from NPZ when available
    },
    "cifar100": {
        "npz_file": ROOT / "exports" / "cifar100" / "cifar100_test_uint8.npz",
        "models": [
            {"name": "mobilenetv3_small", "file": ROOT / "tflite_models" / "MobileNetV3Small_CIFAR100.tflite"},
            {"name": "efficientnet_b0", "file": ROOT / "tflite_models" / "EfficientNetB0_CIFAR100.tflite"},
            {"name": "lcnn", "file": ROOT / "tflite_models" / "MCUNet_CIFAR100.tflite"},
        ],
        "class_names": None,  # Prefer class_names from NPZ when available
    },
    "wakevision": {
        "npz_file": ROOT / "exports" / "wakevision" / "wakevision_test_uint8.npz",
        "models": [
            {"name": "mobilenetv3_small", "file": ROOT / "tflite_models" / "MobileNetV3Small_wake_vision.tflite"},
            {"name": "efficientnet_b0", "file": ROOT / "tflite_models" / "EfficientNetB0_wake_vision.tflite"},
            {"name": "lcnn", "file": ROOT / "tflite_models" / "MCUNet_wake_vision.tflite"},
        ],
        "class_names": ["no_person", "person"],
    },
}

DEFAULT_DATASETS = ["cifar10", "cifar100", "wakevision"]
DEFAULT_MAX_IMAGES = 400
DEFAULT_WARMUP_RUNS = 10
# Single-thread default matches the paper's CPU-only deployment claim.
# Multi-threaded TFLite on low-power ARM boards (Cortex-A53) incurs
# synchronization overhead that can negate parallelism gains; using 1 thread
# also places the Pi Zero 2 W in the same operating regime as single-core
# embedded processors (e.g., Cortex-M4/M7), making it a conservative proxy
# for the most resource-constrained tier of edge deployment.
DEFAULT_NUM_THREADS = 1


def _read_proc_rss_bytes() -> int | None:
    """Read current RSS from /proc/self/status (Linux)."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None

    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                # Format: VmRSS:    123456 kB
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except Exception:
        return None

    return None


def _read_peak_rss_bytes() -> int | None:
    """Read peak RSS from resource.getrusage when available."""
    try:
        import resource

        ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports ru_maxrss in KiB; macOS reports bytes.
        if platform.system().lower() == "darwin":
            return ru_maxrss
        return ru_maxrss * 1024
    except Exception:
        return None


def _current_rss_bytes() -> int | None:
    """Best-effort current RSS in bytes."""
    rss = _read_proc_rss_bytes()
    if rss is not None:
        return rss

    # Fallback: derive from ru_maxrss (may overestimate current usage).
    return _read_peak_rss_bytes()


def _decode_label_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _as_label_vector(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    return labels.astype(np.int64)


def load_dataset(dataset_name: str, dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images, labels and class names from a supported NPZ file or directory."""
    if dataset_path.is_dir():
        import glob
        files = sorted(glob.glob(str(dataset_path / "*.png")))
        images = []
        labels = []
        for f in files:
            images.append(str(f))
            lbl = int(Path(f).stem.split("_")[-1])
            labels.append(lbl)
        default_names = DATASET_SPECS.get(dataset_name, {}).get("class_names") or ["0", "1"]
        return np.array(images, dtype=object), np.array(labels, dtype=np.int64), [str(x) for x in default_names]

    with np.load(dataset_path, allow_pickle=True) as data:
        keys = set(data.files)

        # CIFAR exports: x_test, y_test, class_names
        if {"x_test", "y_test"}.issubset(keys):
            images = np.asarray(data["x_test"])
            labels = _as_label_vector(data["y_test"])
            if "class_names" in keys:
                class_names = [_decode_label_name(v) for v in np.asarray(data["class_names"]).tolist()]
            else:
                n_classes = int(labels.max()) + 1 if labels.size else 0
                class_names = [str(i) for i in range(n_classes)]
            return images, labels, class_names

        # Wake Vision export: images, labels (+ optional heights, widths)
        if {"images", "labels"}.issubset(keys):
            images = np.asarray(data["images"])
            labels = _as_label_vector(data["labels"])
            default_names = DATASET_SPECS.get(dataset_name, {}).get("class_names") or ["0", "1"]
            return images, labels, [str(x) for x in default_names]

    raise ValueError(f"Unsupported NPZ structure in '{dataset_path}'.")


def balanced_sample_indices(labels: np.ndarray, max_images: int) -> np.ndarray:
    """Sample indices trying to keep classes balanced (deterministic by input order)."""
    labels = _as_label_vector(labels)
    n = labels.shape[0]
    if max_images <= 0 or n == 0:
        return np.array([], dtype=np.int64)
    if max_images >= n:
        return np.arange(n, dtype=np.int64)

    unique_labels = sorted(np.unique(labels).tolist())
    by_class = {label: np.where(labels == label)[0].tolist() for label in unique_labels}

    per_class_target = max(1, max_images // max(1, len(unique_labels)))
    selected: list[int] = []

    for label in unique_labels:
        selected.extend(by_class[label][:per_class_target])

    if len(selected) < max_images:
        remaining_needed = max_images - len(selected)
        leftovers: list[int] = []
        for label in unique_labels:
            leftovers.extend(by_class[label][per_class_target:])
        selected.extend(leftovers[:remaining_needed])

    return np.asarray(selected[:max_images], dtype=np.int64)


def _resize_image(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image with PIL while preserving channels."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims [H, W, C], got shape {image.shape}.")

    # PIL supports uint8 robustly; convert to uint8 for resize then keep semantic 0..255.
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((target_w, target_h))
    return np.asarray(pil_img)


def _quantize_if_needed(image: np.ndarray, input_dtype: np.dtype, input_details: dict[str, Any]) -> np.ndarray:
    """Convert image to model dtype; use quantization params for integer inputs."""
    image = np.asarray(image)
    quant_scale, quant_zero_point = input_details.get("quantization", (0.0, 0))

    if input_dtype in (np.float32, np.float64):
        return image.astype(input_dtype)

    if np.issubdtype(input_dtype, np.integer):
        if quant_scale and quant_scale > 0:
            q = np.round(image.astype(np.float32) / quant_scale + quant_zero_point)
            dtype_info = np.iinfo(input_dtype)
            q = np.clip(q, dtype_info.min, dtype_info.max)
            return q.astype(input_dtype)
        return image.astype(input_dtype)

    return image.astype(input_dtype)


def preprocess_image(image_hwc: np.ndarray | str | Path, input_details: dict[str, Any]) -> np.ndarray:
    """Resize and cast one image to match TFLite input tensor details."""
    if isinstance(image_hwc, (str, Path)):
        import PIL.Image
        image_hwc = np.asarray(PIL.Image.open(image_hwc).convert("RGB"))

    input_shape = input_details["shape"]
    input_dtype = input_details["dtype"]

    if len(input_shape) != 4:
        raise ValueError(f"Unsupported input shape {input_shape}; expected 4D.")

    # Handle NHWC and NCHW.
    if input_shape[-1] in (1, 3):
        target_h = int(input_shape[1])
        target_w = int(input_shape[2])
        channel_order = "nhwc"
    else:
        target_h = int(input_shape[2])
        target_w = int(input_shape[3])
        channel_order = "nchw"

    image_hwc = np.asarray(image_hwc)
    if image_hwc.ndim == 2:
        image_hwc = np.stack([image_hwc] * 3, axis=-1)
    if image_hwc.ndim != 3:
        raise ValueError(f"Expected image [H, W, C], got shape {image_hwc.shape}.")

    if image_hwc.shape[0] != target_h or image_hwc.shape[1] != target_w:
        image_hwc = _resize_image(image_hwc, target_w=target_w, target_h=target_h)

    if image_hwc.shape[-1] == 1 and (input_shape[-1] == 3 or input_shape[1] == 3):
        image_hwc = np.repeat(image_hwc, 3, axis=-1)

    prepared = _quantize_if_needed(image_hwc, input_dtype=input_dtype, input_details=input_details)

    if channel_order == "nhwc":
        prepared = np.expand_dims(prepared, axis=0)
    else:
        prepared = np.transpose(prepared, (2, 0, 1))
        prepared = np.expand_dims(prepared, axis=0)

    return prepared


def create_interpreter(model_path: Path, num_threads: int | None) -> Any:
    if tflite is None:
        raise RuntimeError(TFLITE_IMPORT_ERROR or "TFLite runtime is not available.")

    if num_threads is None:
        interpreter = tflite.Interpreter(model_path=str(model_path))
    else:
        try:
            interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=int(num_threads))
        except TypeError:
            interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def evaluate_model(
    dataset_name: str,
    model_spec: dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    warmup_runs: int,
    num_threads: int | None,
) -> dict[str, Any]:
    print(f"\n--- Evaluating {dataset_name} / {model_spec['name']} ---")

    model_path = Path(model_spec["file"])
    if not model_path.exists():
        return {
            "dataset": dataset_name,
            "model": model_spec["name"],
            "model_file": str(model_path),
            "error": "model not found",
        }

    interpreter = create_interpreter(model_path, num_threads=num_threads)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_info = input_details[0]
    output_info = output_details[0]
    memory_before_bytes = _current_rss_bytes()
    peak_before_bytes = _read_peak_rss_bytes()

    if images.shape[0] == 0:
        return {
            "dataset": dataset_name,
            "model": model_spec["name"],
            "model_file": str(model_path),
            "error": "empty input dataset",
        }

    print(f"Running {warmup_runs} warmup inferences...")
    warmup_input = preprocess_image(images[0], input_info)
    import gc
    for _ in range(max(0, warmup_runs)):
        interpreter.set_tensor(input_info["index"], warmup_input)
        interpreter.invoke()
    gc.collect()

    latencies_ms: list[float] = []
    correct_predictions = 0
    total_time_start = time.perf_counter()

    for idx, (image, true_idx) in enumerate(zip(images, labels), start=1):
        if idx % 50 == 0 or idx == images.shape[0]:
            print(f"  Processing {idx}/{images.shape[0]}...")

        input_data = preprocess_image(image, input_info)
        interpreter.set_tensor(input_info["index"], input_data)

        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        latencies_ms.append((end_time - start_time) * 1000.0)

        output = interpreter.get_tensor(output_info["index"])
        predicted_idx = int(np.argmax(output))
        if predicted_idx == int(true_idx):
            correct_predictions += 1

    total_time_ms = (time.perf_counter() - total_time_start) * 1000.0
    memory_after_bytes = _current_rss_bytes()
    peak_after_bytes = _read_peak_rss_bytes()

    images_processed = int(images.shape[0])
    accuracy = correct_predictions / images_processed if images_processed > 0 else 0.0

    lat_np = np.asarray(latencies_ms, dtype=np.float64)
    lat_avg = float(np.mean(lat_np))
    lat_min = float(np.min(lat_np))
    lat_max = float(np.max(lat_np))
    lat_median = float(np.median(lat_np))
    lat_p90 = float(np.percentile(lat_np, 90))
    lat_std = float(np.std(lat_np))
    throughput = float(images_processed / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    print(
        f"  Accuracy: {accuracy:.4f} ({correct_predictions}/{images_processed}) | "
        f"Latency avg/med/p90: {lat_avg:.2f}/{lat_median:.2f}/{lat_p90:.2f} ms | "
        f"Throughput: {throughput:.2f} img/s"
    )

    return {
        "dataset": dataset_name,
        "model": model_spec["name"],
        "model_file": str(model_path),
        "images_processed": images_processed,
        "num_classes": len(class_names),
        "correct": int(correct_predictions),
        "accuracy": float(accuracy),
        "latency_ms": {
            "avg": lat_avg,
            "min": lat_min,
            "median": lat_median,
            "p90": lat_p90,
            "max": lat_max,
            "std": lat_std,
        },
        "throughput_img_s": throughput,
        "total_time_ms": float(total_time_ms),
        # Keep Android-style field names for downstream compatibility.
        "memoryUsageBeforeBytes": memory_before_bytes,
        "memoryUsageAfterBytes": memory_after_bytes,
        "memoryUsageDeltaBytes": (
            int(memory_after_bytes - memory_before_bytes)
            if memory_before_bytes is not None and memory_after_bytes is not None
            else None
        ),
        "peakMemoryUsageBeforeBytes": peak_before_bytes,
        "peakMemoryUsageAfterBytes": peak_after_bytes,
        "peakMemoryUsageDeltaBytes": (
            int(peak_after_bytes - peak_before_bytes)
            if peak_before_bytes is not None and peak_after_bytes is not None
            else None
        ),
    }


def parse_dataset_overrides(entries: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    if not entries:
        return overrides
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid dataset override '{entry}'. Use format: dataset=path/to/file.npz")
        dataset, path_str = entry.split("=", 1)
        dataset = dataset.strip().lower()
        if dataset not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset in override '{dataset}'.")
        overrides[dataset] = Path(path_str).expanduser().resolve()
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TFLite models for CIFAR-10, CIFAR-100 and Wake Vision using NPZ test sets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help=f"Datasets to evaluate. Options: {', '.join(DATASET_SPECS.keys())}. Default: all.",
    )
    parser.add_argument(
        "--dataset-file",
        nargs="*",
        default=None,
        help="Optional NPZ override(s) in format dataset=/path/file.npz. Example: cifar10=exports/cifar10/cifar10_test_uint8.npz",
    )
    parser.add_argument(
        "--only-models",
        nargs="*",
        default=None,
        help="Evaluate only models with names containing these terms (case-insensitive).",
    )
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES, help="Max images per dataset (balanced).")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUP_RUNS, help="Warmup runs per model.")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=(
            "Number of TFLite interpreter threads (default: 1). "
            "The reported results in the paper were obtained with --threads 1, "
            "which places the Pi Zero 2 W in the same single-core regime as "
            "Cortex-M class embedded processors."
        ),
    )
    parser.add_argument("--output", default="results_multidataset.json", help="Output report JSON path.")
    return parser.parse_args()


def _normalize_datasets(requested: list[str]) -> list[str]:
    normalized: list[str] = []
    seen = set()
    for name in requested:
        key = name.strip().lower()
        if key not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset '{name}'. Use one of: {', '.join(DATASET_SPECS.keys())}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    return normalized


def _filter_models(models: list[dict[str, Any]], only_terms: list[str] | None) -> list[dict[str, Any]]:
    if not only_terms:
        return models
    terms = [x.lower() for x in only_terms]
    return [m for m in models if any(t in str(m["name"]).lower() for t in terms)]


def main() -> int:
    args = parse_args()

    try:
        selected_datasets = _normalize_datasets(args.datasets)
        dataset_overrides = parse_dataset_overrides(args.dataset_file)
    except ValueError as err:
        print(f"Error: {err}")
        return 2

    report: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(ROOT),
        "selected_datasets": selected_datasets,
        "max_images": int(args.max_images),
        "warmups": int(args.warmups),
        "threads": args.threads,
        "results": [],
    }

    for dataset_name in selected_datasets:
        spec = DATASET_SPECS[dataset_name]
        npz_file = dataset_overrides.get(dataset_name, Path(spec["npz_file"]))
        model_specs = _filter_models(spec["models"], args.only_models)

        dataset_entry: dict[str, Any] = {
            "dataset": dataset_name,
            "npz_file": str(npz_file),
            "models": [],
        }

        if not npz_file.exists():
            msg = "npz dataset file not found"
            print(f"Warning: {dataset_name}: {msg}: {npz_file}")
            for model_spec in model_specs:
                dataset_entry["models"].append(
                    {
                        "dataset": dataset_name,
                        "model": model_spec["name"],
                        "model_file": str(model_spec["file"]),
                        "error": msg,
                    }
                )
            report["results"].append(dataset_entry)
            continue

        try:
            images, labels, class_names = load_dataset(dataset_name, npz_file)
        except Exception as err:
            msg = f"failed to load npz: {err}"
            print(f"Warning: {dataset_name}: {msg}")
            for model_spec in model_specs:
                dataset_entry["models"].append(
                    {
                        "dataset": dataset_name,
                        "model": model_spec["name"],
                        "model_file": str(model_spec["file"]),
                        "error": msg,
                    }
                )
            report["results"].append(dataset_entry)
            continue

        if images.shape[0] != labels.shape[0]:
            msg = f"images/labels size mismatch: {images.shape[0]} vs {labels.shape[0]}"
            print(f"Warning: {dataset_name}: {msg}")
            for model_spec in model_specs:
                dataset_entry["models"].append(
                    {
                        "dataset": dataset_name,
                        "model": model_spec["name"],
                        "model_file": str(model_spec["file"]),
                        "error": msg,
                    }
                )
            report["results"].append(dataset_entry)
            continue

        chosen_indices = balanced_sample_indices(labels, max_images=int(args.max_images))
        images_eval = images[chosen_indices]
        labels_eval = labels[chosen_indices]

        dataset_entry["num_images_total"] = int(images.shape[0])
        dataset_entry["num_images_eval"] = int(images_eval.shape[0])
        dataset_entry["num_classes"] = int(len(class_names))

        print(
            f"\nDataset: {dataset_name} | NPZ: {npz_file} | "
            f"Images total/eval: {images.shape[0]}/{images_eval.shape[0]}"
        )

        for model_spec in model_specs:
            model_file = Path(model_spec["file"])
            if not model_file.exists():
                print(f"Warning: model not found, skipping: {model_file}")
                dataset_entry["models"].append(
                    {
                        "dataset": dataset_name,
                        "model": model_spec["name"],
                        "model_file": str(model_file),
                        "error": "model not found",
                    }
                )
                continue

            try:
                metrics = evaluate_model(
                    dataset_name=dataset_name,
                    model_spec=model_spec,
                    images=images_eval,
                    labels=labels_eval,
                    class_names=class_names,
                    warmup_runs=int(args.warmups),
                    num_threads=args.threads,
                )
                dataset_entry["models"].append(metrics)
            except Exception as err:
                print(f"Error evaluating {dataset_name}/{model_spec['name']}: {err}")
                dataset_entry["models"].append(
                    {
                        "dataset": dataset_name,
                        "model": model_spec["name"],
                        "model_file": str(model_file),
                        "error": str(err),
                    }
                )

        report["results"].append(dataset_entry)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
