import re

file_path = 'tools/eval_tflite_multidataset.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update load_dataset_from_npz -> load_dataset
new_load_dataset = '''def load_dataset(dataset_name: str, dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    \"\"\"Load images, labels and class names from a supported NPZ file or directory.\"\"\"
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

    with np.load(dataset_path, allow_pickle=True) as data:'''

content = content.replace('def load_dataset_from_npz(dataset_name: str, npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:\n    \"\"\"Load images, labels and class names from a supported NPZ file.\"\"\"\n    with np.load(npz_path, allow_pickle=True) as data:', new_load_dataset)
content = content.replace('raise ValueError(f"Unsupported NPZ structure in \'{npz_path}\'.")', 'raise ValueError(f"Unsupported NPZ structure in \'{dataset_path}\'.")')
content = content.replace('images, labels, class_names = load_dataset_from_npz(dataset_name, npz_file)', 'images, labels, class_names = load_dataset(dataset_name, npz_file)')

# 2. Update preprocess_image to accept file paths
new_preprocess = '''def preprocess_image(image_hwc: np.ndarray | str | Path, input_details: dict[str, Any]) -> np.ndarray:
    \"\"\"Resize and cast one image to match TFLite input tensor details.\"\"\"
    if isinstance(image_hwc, (str, Path)):
        import PIL.Image
        image_hwc = np.asarray(PIL.Image.open(image_hwc).convert("RGB"))
'''
content = content.replace('def preprocess_image(image_hwc: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:\n    \"\"\"Resize and cast one image to match TFLite input tensor details.\"\"\"', new_preprocess)

# 3. Explicit memory cleanup during eval:
new_evaluate = '''
    import gc
    for _ in range(max(0, warmup_runs)):
        interpreter.set_tensor(input_info["index"], warmup_input)
        interpreter.invoke()
    gc.collect()

    latencies_ms: list[float] = []'''
content = content.replace('''
    for _ in range(max(0, warmup_runs)):
        interpreter.set_tensor(input_info["index"], warmup_input)
        interpreter.invoke()

    latencies_ms: list[float] = []''', new_evaluate)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched successfully")
