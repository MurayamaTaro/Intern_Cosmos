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
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    # wheel 競合を潰す
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir --force-reinstall \
        numpy==1.26.4 opencv-contrib-python==4.8.0.74

# git のユーザー情報
RUN git config --global user.name  "MurayamaTaro" && \
    git config --global user.email "muramura10taro@gmail.com"

# 3) 警告防止
ENV PYTHONWARNINGS="ignore"
# ENV PYTHONWARNINGS="ignore::UserWarning"
# ENV PYTHONWARNINGS="ignore::FutureWarning"
# NCCL や C++ 側のログを ERROR 以上に
ENV TORCH_CPP_LOG_LEVEL=ERROR

# 4) ワークスペース
WORKDIR /workspace
COPY bashrc /root/.bashrc
CMD ["bash"]
