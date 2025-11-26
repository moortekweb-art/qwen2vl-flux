# Comprehensive Deployment Guide for Qwen2VL-FLUX

This guide covers complete setup, configuration, deployment, and integration strategies for your production Qwen2VL-FLUX pipeline.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Docker Setup and Production Image](#docker-setup)
3. [Secure Configuration Management](#secure-configuration)
4. [Environment Deployment Strategies](#deployment-strategies)
5. [Frontend Integration](#frontend-integration)
6. [Monitoring and Optimization](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

- **GPUs**: 2x NVIDIA GPUs (tested with RTX A6000, L40S, or equivalent)
  - GPU 0 (cuda:0): Qwen2VL encoder, text encoders (≈20-25GB)
  - GPU 1 (cuda:1): FLUX transformer, VAE (≈40-50GB)
- **VRAM Total**: 80GB minimum (recommended 160GB+ for batch processing)
- **CPU**: 16+ cores recommended for preprocessing
- **RAM**: 64GB minimum
- **Storage**: 200GB minimum for models + outputs

### Software

- **NVIDIA Driver**: 555+ (supports CUDA 12.4)
- **CUDA**: 12.4 (nightly PyTorch required for sm_120 support)
- **Python**: 3.10 or 3.11
- **Docker**: 20.10+ with nvidia-docker

---

## Docker Setup and Production Image

### Understanding Your Current Setup

Your Dockerfile uses:
- Base image: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- Python dependencies: Locked to specific versions in requirements.txt
- PyTorch: Nightly build from CUDA 12.4 index
- .dockerignore: Prevents copying 74GB of checkpoints

### Step 1: Build the Development Docker Image

```bash
cd /path/to/qwen2vl-flux
docker build -t qwen2vl-flux:dev .
```

Expected build time: 15-20 minutes
Image size: ~12GB (without checkpoints)

### Step 2: Run Development Container

```bash
docker run -it \
  --gpus all \
  --runtime=nvidia \
  -v /path/to/qwen2vl-flux:/app \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CHECKPOINT_DIR=/app/checkpoints \
  qwen2vl-flux:dev bash
```

### Step 3: Save Production Image

Once you've validated the development image works:

```bash
# Run container and keep it active
docker run -d \
  --gpus all \
  --name qwen-flux-prod \
  --runtime=nvidia \
  -v /path/to/qwen2vl-flux:/app \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CHECKPOINT_DIR=/app/checkpoints \
  qwen2vl-flux:dev tail -f /dev/null

# Commit the running container to a new image
docker commit qwen2vl-flux-prod qwen2vl-flux:production

# Tag with version
docker tag qwen2vl-flux:production qwen2vl-flux:v1.0-production

# Save to tar for backup (optional, but recommended)
docker save qwen2vl-flux:production | gzip > qwen2vl-flux-production.tar.gz

# Clean up temporary container
docker stop qwen-flux-prod
docker rm qwen-flux-prod
```

Image size after production build: ~12GB
Checkpoint data (stored separately): ~74GB

### Step 4: Verify Production Image

```bash
# List available images
docker images | grep qwen2vl

# Test the production image
docker run -it \
  --gpus all \
  --runtime=nvidia \
  -v /mnt/models:/app/checkpoints \
  qwen2vl-flux:production python -c "from model import FluxModel; print('✓ Model imports successful')"
```

### Optimized Docker Compose for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  qwen-flux-inference:
    image: qwen2vl-flux:production
    container_name: qwen2vl-flux-prod
    runtime: nvidia
    working_dir: /app

    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CHECKPOINT_DIR=/models/checkpoints
      - CUDA_LAUNCH_BLOCKING=1
      - TORCH_CUDA_ARCH_LIST=sm_80,sm_90
      - PYTHONUNBUFFERED=1

    volumes:
      # Read-only source code
      - ./src:/app/src:ro
      # Shared model checkpoints
      - /mnt/fast-nvme/models:/models:ro
      # Writable output directory
      - ./outputs:/app/outputs
      # Cache directory (optional)
      - inference-cache:/app/cache

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; print('GPU 0:', torch.cuda.get_device_name(0))"]
      interval: 60s
      timeout: 10s
      retries: 3

    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

    stdin_open: true
    tty: true

  # Optional: ComfyUI Frontend
  comfyui-server:
    image: qwen2vl-flux:production
    container_name: comfyui-prod
    runtime: nvidia
    working_dir: /app/ComfyUI

    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - CHECKPOINT_DIR=/models/checkpoints
      - QWEN2VL_FLUX_PATH=/app

    ports:
      - "8188:8188"

    volumes:
      - ./ComfyUI:/app/ComfyUI
      - /mnt/fast-nvme/models:/models:ro
      - ./comfyui-outputs:/app/ComfyUI/output

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

    depends_on:
      - qwen-flux-inference

    command: python main.py --listen 0.0.0.0 --port 8188

volumes:
  inference-cache:
    driver: local
```

---

## Secure Configuration Management

### Step 1: Create Secrets Management File

Create `config/.env.production` (ADD TO .gitignore):

```bash
# Hugging Face API
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_HOME=/models/hf-cache

# Inference Settings
INFERENCE_DEVICE=cuda:0
TRANSFORMER_DEVICE=cuda:1
BATCH_SIZE=2
NUM_INFERENCE_STEPS=28

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/inference.log

# Security
ENABLE_AUTH=true
API_KEY=your-secure-api-key-here
```

### Step 2: Secure Token Storage

```bash
# Option 1: Docker Secrets (Recommended for Swarm)
echo "hf_xxxxxxxxxxxx" | docker secret create hf_token -

# Option 2: Environment Variables (Less secure, but simpler)
export HF_TOKEN="your-token"
docker run -e HF_TOKEN=$HF_TOKEN ...

# Option 3: Secret File Mount
mkdir -p /mnt/secrets
echo "hf_xxxxxxxxxxxx" > /mnt/secrets/hf_token
chmod 600 /mnt/secrets/hf_token
docker run -v /mnt/secrets:/run/secrets:ro ...
```

### Step 3: Create Secure Config Loader

Create `config/config.py`:

```python
import os
from pathlib import Path
from typing import Optional

class Config:
    """Secure configuration loader"""

    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or '.env.production'
        self.load_env()

    def load_env(self):
        """Load environment variables from file or system"""
        if Path(self.env_file).exists():
            with open(self.env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')

    @property
    def hf_token(self) -> str:
        """Get HF token (from file or env)"""
        # Try from file first
        token_file = Path('/run/secrets/hf_token')
        if token_file.exists():
            return token_file.read_text().strip()
        # Fall back to environment
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("HF_TOKEN not found in secrets or environment")
        return token

    @property
    def checkpoint_dir(self) -> Path:
        return Path(os.getenv('CHECKPOINT_DIR', '/app/checkpoints'))

    @property
    def inference_device(self) -> str:
        return os.getenv('INFERENCE_DEVICE', 'cuda:0')

    @property
    def transformer_device(self) -> str:
        return os.getenv('TRANSFORMER_DEVICE', 'cuda:1')

    @property
    def batch_size(self) -> int:
        return int(os.getenv('BATCH_SIZE', '1'))

    @property
    def num_inference_steps(self) -> int:
        return int(os.getenv('NUM_INFERENCE_STEPS', '28'))

# Global config instance
config = Config()
```

### Step 4: Update model.py to Use Config

```python
from config.config import config

class FluxModel:
    def __init__(self, is_turbo=False, device=None, required_features=None):
        # Use config instead of hardcoded paths
        self.device = torch.device(device or config.inference_device)
        self.checkpoint_dir = config.checkpoint_dir
        # ... rest of initialization
```

---

## Deployment Strategies

### Strategy 1: Local Dual-GPU Setup

```bash
# Set environment
export NVIDIA_VISIBLE_DEVICES=0,1
export CHECKPOINT_DIR=/mnt/models/checkpoints

# Run with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Monitor
docker-compose -f docker-compose.prod.yml logs -f qwen-flux-inference
```

### Strategy 2: Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen2vl-flux
  namespace: inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qwen2vl-flux
  template:
    metadata:
      labels:
        app: qwen2vl-flux
    spec:
      containers:
      - name: qwen-flux
        image: qwen2vl-flux:production
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "0"
        - name: CHECKPOINT_DIR
          value: /models
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secrets
              key: token
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "30Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: 1
            memory: "50Gi"
            cpu: "16"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: outputs
          mountPath: /app/outputs
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - import torch; assert torch.cuda.is_available()
          initialDelaySeconds: 30
          periodSeconds: 60
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: outputs
        persistentVolumeClaim:
          claimName: outputs-pvc
      nodeSelector:
        accelerator: nvidia-gpu
```

Deploy with:
```bash
kubectl apply -f k8s/deployment.yaml
```

### Strategy 3: API Server Deployment

Create `api/server.py`:

```python
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
import asyncio
from typing import Optional
from model import FluxModel
from config.config import config

app = FastAPI(title="Qwen2VL-FLUX API")

# Global model instance (lazy load)
model = None

def get_model():
    global model
    if model is None:
        model = FluxModel(
            is_turbo=False,
            device=config.inference_device,
            required_features=['controlnet', 'depth', 'line']
        )
    return model

@app.on_event("startup")
async def startup():
    """Preload model on startup"""
    print("Preloading model...")
    get_model()
    print("Model ready!")

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = "",
    mode: str = "variation",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    aspect_ratio: str = "1:1",
):
    """Generate images using Qwen2VL-FLUX"""
    from PIL import Image
    import io

    try:
        # Load image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Generate
        model = get_model()
        images = model.generate(
            input_image_a=pil_image,
            prompt=prompt,
            mode=mode,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            aspect_ratio=aspect_ratio,
        )

        # Save and return
        output_path = f"/app/outputs/generated_{int(time.time())}.png"
        images[0].save(output_path)

        return JSONResponse({
            "status": "success",
            "output": output_path
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run with:
```bash
docker run -p 8000:8000 qwen2vl-flux:production python api/server.py
```

---

## Frontend Integration

### Integration with OpenWebUI

OpenWebUI uses ComfyUI as a backend. See `COMFYUI_INTEGRATION.md` for full setup.

Quick summary:
1. Set up ComfyUI with custom Qwen2VL-FLUX nodes
2. Create a workflow JSON
3. Configure OpenWebUI to use ComfyUI as image generation backend
4. Export workflow to OpenWebUI

### Integration with Gradio UI

Create `ui/gradio_app.py`:

```python
import gradio as gr
from PIL import Image
import io
from model import FluxModel
from config.config import config

model = FluxModel(
    device=config.inference_device,
    required_features=['controlnet', 'depth', 'line']
)

def generate_image(input_image, prompt, mode, steps, guidance):
    """Gradio generation function"""
    images = model.generate(
        input_image_a=input_image,
        prompt=prompt,
        mode=mode,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    )
    return images

# Create UI
with gr.Blocks() as demo:
    gr.Markdown("# Qwen2VL-FLUX Generator")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image", type="pil")
            prompt = gr.Textbox(label="Prompt", lines=3)
            mode = gr.Radio(
                ["variation", "img2img", "controlnet"],
                label="Generation Mode"
            )

        with gr.Column():
            steps = gr.Slider(1, 100, value=28, label="Inference Steps")
            guidance = gr.Slider(1.0, 15.0, value=3.5, label="Guidance Scale")
            generate_btn = gr.Button("Generate", scale=2)

    output = gr.Gallery(label="Generated Images")

    generate_btn.click(
        fn=generate_image,
        inputs=[input_img, prompt, mode, steps, guidance],
        outputs=output
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
```

Run with:
```bash
docker run -p 7860:7860 qwen2vl-flux:production python ui/gradio_app.py
```

---

## Monitoring and Optimization

### GPU Memory Monitoring

Create `monitoring/gpu_monitor.py`:

```python
import torch
import psutil
import time
from datetime import datetime

def monitor_gpu():
    """Monitor GPU memory usage"""
    for i in range(2):
        props = torch.cuda.get_device_properties(i)
        util = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100

        print(f"\nGPU {i}: {props.name}")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB")
        print(f"  Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f}GB")
        print(f"  Total: {props.total_memory / 1e9:.2f}GB")
        print(f"  Utilization: {util:.1f}%")

def optimize_memory():
    """Clear unnecessary memory"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Run monitoring
while True:
    monitor_gpu()
    time.sleep(5)
```

### Performance Benchmarking

```bash
# Benchmark generation speed
time python -c "
from model import FluxModel
from PIL import Image

model = FluxModel(device='cuda')
img = Image.new('RGB', (512, 512))

for i in range(5):
    images = model.generate(img, prompt='test', num_inference_steps=28)
    print(f'Iteration {i}: Complete')
"
```

Expected performance:
- Variation: 45-60 seconds per image
- With turbo mode: 15-20 seconds per image
- Batch processing: Near-linear scaling

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**:
```bash
# Reduce batch size
export BATCH_SIZE=1

# Enable memory-efficient mode
export TORCH_CUDA_ARCH_LIST=sm_80,sm_90

# Clear cache between generations
python -c "import torch; torch.cuda.empty_cache()"
```

### Issue: "Model not found"

**Solution**:
```bash
# Verify checkpoint directory
ls -la /path/to/checkpoints/

# Set correct path
export CHECKPOINT_DIR=/correct/path/to/checkpoints
```

### Issue: "GPU not detected in Docker"

**Solution**:
```bash
# Verify nvidia-docker installation
docker run --rm --runtime=nvidia nvidia/cuda:12.4.1-base nvidia-smi

# Check docker daemon config
cat /etc/docker/daemon.json | grep nvidia
```

### Issue: "Slow generation speed"

**Solution**:
```bash
# Check GPU utilization
nvidia-smi

# Enable turbo mode in model initialization
model = FluxModel(is_turbo=True)

# Reduce inference steps (20-28 is usually sufficient)
num_inference_steps=20
```

---

## Backup and Recovery

### Automated Backup of Production Image

```bash
#!/bin/bash
# backup.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/mnt/backups/docker-images"

mkdir -p $BACKUP_DIR

# Save image
docker save qwen2vl-flux:production | gzip > $BACKUP_DIR/qwen2vl-flux-$TIMESTAMP.tar.gz

# Keep only last 3 backups
ls -t $BACKUP_DIR/qwen2vl-flux-*.tar.gz | tail -n +4 | xargs rm -f

echo "Backup completed: $BACKUP_DIR/qwen2vl-flux-$TIMESTAMP.tar.gz"
```

Schedule with cron:
```bash
# Weekly backup
0 2 * * 0 /path/to/backup.sh
```

---

## Production Checklist

- [ ] Docker image built and tested
- [ ] Production image saved and backed up
- [ ] Environment variables configured securely
- [ ] Model checkpoints downloaded
- [ ] GPU memory allocation verified
- [ ] Monitoring and logging enabled
- [ ] API/Frontend deployed and tested
- [ ] Health checks configured
- [ ] Backup strategy implemented
- [ ] Documentation complete and accessible
- [ ] Team trained on deployment procedures

---

## Additional Resources

- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch CUDA Support](https://pytorch.org/get-started/locally/)
- [ComfyUI Integration Guide](./COMFYUI_INTEGRATION.md)
- Your repository's README for model details
