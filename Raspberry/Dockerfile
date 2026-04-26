FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Basic dependencies required by TFLite, numpy, and OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install TFLite runtime and dependencies
# numpy<2 is required for compatibility with tflite-runtime on ARM
RUN pip install --no-cache-dir \
    tflite-runtime \
    "numpy<2" \
    pillow \
    opencv-python-headless

# Working directory
WORKDIR /app

# Copy project into the container
COPY . /app

# Default shell — run your benchmark scripts manually
CMD ["bash"]
