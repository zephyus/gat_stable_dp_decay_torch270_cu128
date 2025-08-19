# ============================================================
# MARL (TSC base): Ubuntu 22.04 + CUDA 12.8 (runtime+cudnn)
# Python 3.11
# Env A: PyTorch 2.7.0 (cu128) + torchvision/torchaudio
# Env B: TensorFlow 2.16 GPU
# Sci stack + seaborn + tensorboard + psutil
# SUMO wheels + SUMO_HOME
# GRF (google-research/football) 系統依賴
# 重點：CUPTI/NCCL 路徑優先序用 activate.d 控制（不污染全域）
# ============================================================

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PIP_NO_CACHE_DIR=1

# --- 基本系統工具（含 tmux 與 GRF 常見依賴工具鏈） ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential wget curl git ca-certificates \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
      libffi-dev libssl-dev cmake swig tmux \
    && rm -rf /var/lib/apt/lists/*

# 有些基底 image 會多一份舊的 CUDA source，先移掉避免 apt 告警
RUN if [ -f /etc/apt/sources.list.d/cuda.list ]; then rm /etc/apt/sources.list.d/cuda.list; fi

# ------------------------------------------------------------
# Miniconda
# ------------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-lc"]

# 嚴格 channel priority + mamba
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --yes \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r --yes \
 && conda config --system --set channel_priority strict \
 && conda config --system --add channels conda-forge \
 && conda config --system --remove channels defaults || true \
 && conda install -n base -c conda-forge --override-channels mamba -y

# ------------------------------------------------------------
# 建兩個乾淨環境
#   A) torch311：專供 PyTorch + RL + SUMO
#   B) tf311   ：專供 TensorFlow
# ------------------------------------------------------------
RUN mamba create -y -n torch311 python=3.11 \
 && mamba create -y -n tf311    python=3.11

# 直接使用 torch311 環境內 python/pip，避免在 build 階段多次啟動 shell activate
ENV TORCH_PY=/opt/conda/envs/torch311/bin/python \
    TORCH_PIP=/opt/conda/envs/torch311/bin/pip \
    TF_PY=/opt/conda/envs/tf311/bin/python \
    TF_PIP=/opt/conda/envs/tf311/bin/pip

# 讓登入時預設啟用 torch311（可手動切 tf311）
RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate torch311' >> /root/.bashrc

# ------------------------------------------------------------
# 科學計算/視覺化 + tensorboard + psutil（裝到 torch311）
# ------------------------------------------------------------
RUN $TORCH_PY -m pip install --upgrade pip \
 && $TORCH_PIP install \
     numpy==2.1.* scipy==1.14.* pandas==2.2.* \
     matplotlib==3.9.* seaborn==0.13.* tensorboard psutil

# ------------------------------------------------------------
# PyTorch 2.7.0 (cu128) + torchvision/torchaudio（torch311）
# ------------------------------------------------------------
RUN $TORCH_PIP install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# ------------------------------------------------------------
# 安裝 CUDA 12.8 的 CUPTI（APT），並讓 linker 看得到 targets 路徑
# 不碰 pip 版 CUPTI/NCCL，避免版本打架
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends cuda-cupti-12-8 \
 && echo "$(realpath /usr/local/cuda)/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/cuda-targets.conf \
 && ldconfig

# 在 torch311 加入 activate/deactivate 腳本：優先使用 torch 自帶 .so，其次系統 CUDA targets
RUN mkdir -p /opt/conda/envs/torch311/etc/conda/activate.d /opt/conda/envs/torch311/etc/conda/deactivate.d \
 && printf '%s\n' \
'#!/usr/bin/env bash' \
'export TORCH_LIB=/opt/conda/envs/torch311/lib/python3.11/site-packages/torch/lib' \
'export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"' \
'export LD_LIBRARY_PATH="${TORCH_LIB}:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"' \
> /opt/conda/envs/torch311/etc/conda/activate.d/10-torch-libs.sh \
 && printf '%s\n' \
'#!/usr/bin/env bash' \
'export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH}"' \
'unset _OLD_LD_LIBRARY_PATH' \
'unset TORCH_LIB' \
> /opt/conda/envs/torch311/etc/conda/deactivate.d/10-torch-libs.sh \
 && chmod +x /opt/conda/envs/torch311/etc/conda/activate.d/10-torch-libs.sh \
 && chmod +x /opt/conda/envs/torch311/etc/conda/deactivate.d/10-torch-libs.sh

# ------------------------------------------------------------
# TensorFlow（tf311）
# 2.16 的 pip 版已內含 CUDA/cuDNN 對應輪（無需額外裝 toolkit）
# ------------------------------------------------------------
RUN $TF_PY -m pip install --upgrade pip \
 && $TF_PIP install 'tensorflow[and-cuda]==2.16.1' \
 && $TF_PY - <<'PY'
import pathlib, tensorflow as tf
root = pathlib.Path(tf.__file__).resolve().parent
nvdir = root.parent / 'nvidia'
# 把 TF 附帶的 nvidia/*.so 連回 tensorflow 資料夾，避免部分動態連結找不到
if nvdir.exists():
    for so in nvdir.glob('**/lib/*.so*'):
        dst = root / so.name
        try:
            if not dst.exists():
                dst.symlink_to(so)
        except Exception:
            pass
print("TF GPU libs linked into:", root)
PY

# ------------------------------------------------------------
# GRF（google-research/football）需要的系統庫
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1-mesa-dev \
      libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
      libboost-all-dev libdirectfb-dev \
      mesa-utils xvfb x11vnc \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# SUMO（wheel 版）+ 設定 SUMO_HOME（安裝到 torch311）
# ------------------------------------------------------------
RUN $TORCH_PIP install eclipse-sumo==1.24.0 traci==1.24.0 sumolib==1.24.0 \
 && /opt/conda/envs/torch311/bin/sumo -V || true
ENV SUMO_HOME=/opt/conda/envs/torch311/share/sumo

# ------------------------------------------------------------
# 工作目錄
# ------------------------------------------------------------
RUN mkdir -p /workspace/my_deeprl_network
WORKDIR /workspace

CMD ["/bin/bash"]
