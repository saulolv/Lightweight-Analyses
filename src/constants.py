"""Centralized constants shared across all notebooks and scripts.

Import with:  from src.constants import *
"""

from __future__ import annotations

SEED = 42

IMG_SIZE_CIFAR = 160
IMG_SIZE_WAKEVISION = 224
INPUT_SHAPE_CIFAR = (IMG_SIZE_CIFAR, IMG_SIZE_CIFAR, 3)
INPUT_SHAPE_WAKEVISION = (IMG_SIZE_WAKEVISION, IMG_SIZE_WAKEVISION, 3)

BATCH_SIZE = 64
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "wakevision": 2}

EPOCHS_HEAD = 40
EPOCHS_FINE = 20
HEAD_LR = 1e-3
FINE_LR = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
FINE_TUNE_RATIO = 0.8

MCUNET_NET_ID = "mcunet-in4"

WARMUP_INFERENCES = 10
EVAL_IMAGES = 400

SEVERITIES = [1, 2, 3, 4, 5]

CORRUPTION_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 128

VAL_SIZE = 5_000

MODEL_NAMES = ["MobileNetV3Small", "EfficientNetB0", "MCUNet"]

DISPLAY_NAMES = {
    "MobileNetV3Small": "MobileNetV3-Small",
    "EfficientNetB0": "EfficientNet-B0",
    "MCUNet_Official": "MCUNet",
    "MCUNet": "MCUNet",
}

DATASET_DISPLAY = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "wakevision": "Wake Vision",
    "CIFAR-10": "CIFAR-10",
    "CIFAR-100": "CIFAR-100",
    "Wake Vision": "Wake Vision",
}

MODEL_DISPLAY = {
    "efficientnetb0": "EfficientNet-B0",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet-b0": "EfficientNet-B0",
    "efficientnet": "EfficientNet-B0",
    "lcnn": "MCUNet",
    "mcunet": "MCUNet",
    "mobilenetv3small": "MobileNetV3-Small",
    "mobilenetv3_small": "MobileNetV3-Small",
    "mobilenetv3-small": "MobileNetV3-Small",
    "mobilenet": "MobileNetV3-Small",
}