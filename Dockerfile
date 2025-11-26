FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 

WORKDIR /app

# 1. Install system tools and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies and PyTorch (latest nightly with CUDA 12.4 for sm_120 support)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

# 3. Install Depth Anything V2
RUN pip3 install --no-cache-dir depth-anything-v2 2>/dev/null || echo "depth-anything-v2 not available in PyPI, will use fallback"

# 4. Copy project code (The .dockerignore file will stop this from copying 74GB)
COPY . . 

CMD ["tail", "-f", "/dev/null"]
