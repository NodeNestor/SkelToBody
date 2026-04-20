FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# 8.9 = RTX 4060. torch 2.6.0 does not know sm_120 (Blackwell/5060 Ti); for that
# you'd need torch 2.7+ which in turn forces cu128/cu130 and a different base.
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Python 3.10 to match TRELLIS.2 setup.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git wget build-essential ninja-build \
    libgl1 libglib2.0-0 libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 \
    libjpeg-dev libgoogle-perftools4 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch (pinned to TRELLIS.2's tested combo) ---
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# --- Constraint file to STOP downstream installs from upgrading torch/triton ---
RUN echo "torch==2.6.0" > /tmp/constraints.txt && \
    echo "torchvision==0.21.0" >> /tmp/constraints.txt && \
    echo "triton==3.2.0" >> /tmp/constraints.txt

# --- Basic deps (from setup.sh --basic, minus gradio/lpips we don't need) ---
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh pymeshlab transformers tensorboard pandas zstandard \
    numpy==1.26.4 scipy scikit-image tifffile \
    SimpleITK nibabel pydicom \
    pyarrow safetensors huggingface_hub einops \
    kornia timm \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
        git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- QLoRA stack (separate step, constrained) ---
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    peft bitsandbytes accelerate

# --- Flash Attention (version from setup.sh) ---
RUN pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt \
    flash-attn==2.7.3

# --- TRELLIS.2 custom packages (o_voxel + FlexGEMM) ---
WORKDIR /workspace
COPY TRELLIS.2 /workspace/TRELLIS.2
# Clone FlexGEMM FRESH (with --recursive for its submodules); o_voxel already has its submodules
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install --no-build-isolation -c /tmp/constraints.txt /tmp/extensions/nvdiffrast && \
    cp -r /workspace/TRELLIS.2/o-voxel /tmp/extensions/o-voxel && \
    pip install --no-build-isolation -c /tmp/constraints.txt /tmp/extensions/o-voxel && \
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM && \
    pip install --no-build-isolation -c /tmp/constraints.txt /tmp/extensions/FlexGEMM

ENV PYTHONPATH=/workspace/TRELLIS.2:/workspace:$PYTHONPATH
ENV ATTN_BACKEND=flash_attn
ENV SPARSE_CONV_BACKEND=flex_gemm

CMD ["/bin/bash"]
