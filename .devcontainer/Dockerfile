# ベース:  NVIDIA NGC PyTorch 24.10 (CUDA 12.6, torch-2.5.0a0, TE-1.11)
FROM nvcr.io/nvidia/pytorch:24.10-py3

# システムライブラリ
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libgl1-mesa-glx git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ（torch/vision/audio/transformer-engineを除外済み）
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    # wheel競合をなくす
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir --force-reinstall \
        numpy==1.26.4 opencv-contrib-python==4.8.0.74

# gitユーザー情報
RUN git config --global user.name  "MurayamaTaro" && \
    git config --global user.email "muramura10taro@gmail.com"

# 警告防止
ENV PYTHONWARNINGS="ignore"
ENV TORCH_CPP_LOG_LEVEL=ERROR

# ワークスペース
WORKDIR /workspace
COPY bashrc /root/.bashrc
CMD ["bash"]
