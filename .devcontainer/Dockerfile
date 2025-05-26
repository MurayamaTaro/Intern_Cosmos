# ------------------------------------------------------------
# ベースは NVIDIA NGC: PyTorch 24.10 (CUDA 12.6, torch-2.5.0a0, TE-1.11)
# ------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:24.10-py3

# 1) 追加で必要になる system ライブラリ
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libgl1-mesa-glx git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Python パッケージ（torch/vision/audio/transformer-engine を除外済み）
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# 3) 作業ディレクトリ
WORKDIR /workspace
