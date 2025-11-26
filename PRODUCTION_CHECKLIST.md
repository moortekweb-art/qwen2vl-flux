# Production Deployment Checklist

Use this checklist to ensure your Qwen2VL-FLUX deployment is production-ready.

## Pre-Deployment (Week Before)

### Infrastructure & Hardware
- [ ] Verify 2 GPUs available with 24GB+ VRAM each
- [ ] Confirm NVIDIA drivers installed (555+)
- [ ] Verify CUDA 12.4 toolkit installed
- [ ] Check disk space: 200GB available for models
- [ ] Confirm network connectivity for model downloads
- [ ] Set up monitoring infrastructure (Prometheus/Grafana if applicable)

### Model Acquisition
- [ ] Download Qwen2VL model checkpoints (~12GB)
- [ ] Download FLUX model (~55GB)
- [ ] Download ControlNet model (~3GB)
- [ ] Verify model integrity (checksums if available)
- [ ] Place in `checkpoints/` directory
- [ ] Test model loading locally

### Software Prerequisites
- [ ] Install Docker and nvidia-docker
- [ ] Install ComfyUI (if using web interface)
- [ ] Install Python 3.10 or 3.11
- [ ] Install git for version control
- [ ] Set up backup storage location
- [ ] Configure log rotation

### Credentials & Security
- [ ] Obtain Hugging Face API token
- [ ] Store token in secure vault (not plaintext)
- [ ] Generate API keys for external access
- [ ] Set up SSL/TLS certificates (if public endpoint)
- [ ] Configure firewall rules
- [ ] Enable audit logging

---

## Deployment Phase (Day Of)

### 1. Code Preparation (30 min)

```bash
# [ ] Clone repository
git clone https://github.com/your-repo/qwen2vl-flux.git
cd qwen2vl-flux

# [ ] Verify git status is clean
git status  # Should show no uncommitted changes

# [ ] Check recent commits
git log --oneline -5

# [ ] Verify all documentation present
ls -la QUICKSTART.md COMFYUI_INTEGRATION.md DEPLOYMENT_GUIDE.md

# [ ] Verify custom nodes present
ls -la comfyui_nodes/
```

### 2. Environment Configuration (20 min)

```bash
# [ ] Create .env.production file
cat > .env.production << 'EOF'
QWEN2VL_FLUX_PATH=/opt/models/qwen2vl-flux
CHECKPOINT_DIR=/opt/models/qwen2vl-flux/checkpoints
NVIDIA_VISIBLE_DEVICES=0,1
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
EOF

# [ ] Verify permissions are restricted
chmod 600 .env.production

# [ ] Test environment variables load
source .env.production
echo "Checkpoint dir: $CHECKPOINT_DIR"

# [ ] Verify paths exist
ls -la "$CHECKPOINT_DIR"
```

### 3. Docker Build (15-30 min)

```bash
# [ ] Build development image
docker build -t qwen2vl-flux:dev .

# [ ] Verify build completed successfully
docker images | grep qwen2vl-flux

# [ ] Test development image
docker run --rm --gpus all qwen2vl-flux:dev python -c \
  "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# [ ] Build production image
docker run -d --gpus all --name qwen-flux-prod \
  --runtime=nvidia \
  -v /opt/models:/app/checkpoints \
  qwen2vl-flux:dev tail -f /dev/null

# [ ] Save production image
docker commit qwen-flux-prod qwen2vl-flux:production

# [ ] Tag version
docker tag qwen2vl-flux:production qwen2vl-flux:v1.0

# [ ] Clean up temporary container
docker stop qwen-flux-prod && docker rm qwen-flux-prod

# [ ] Verify production image
docker images | grep "qwen2vl-flux:production"

# [ ] Back up production image
docker save qwen2vl-flux:production | gzip > \
  /backup/qwen2vl-flux-production-$(date +%Y%m%d).tar.gz

# [ ] Verify backup file
ls -lh /backup/qwen2vl-flux-production-*.tar.gz
```

### 4. Docker Compose Validation (10 min)

```bash
# [ ] Verify docker-compose.prod.yml exists
ls -la docker-compose.prod.yml

# [ ] Validate YAML syntax
docker-compose -f docker-compose.prod.yml config > /dev/null && \
  echo "✓ docker-compose.yml is valid"

# [ ] Test pulling/building images
docker-compose -f docker-compose.prod.yml pull 2>/dev/null || \
  echo "Note: Pull not needed for local images"

# [ ] Dry-run deployment
docker-compose -f docker-compose.prod.yml config | head -20
```

### 5. ComfyUI Setup (if using) (20 min)

```bash
# [ ] Copy custom nodes
cp -r comfyui_nodes /path/to/ComfyUI/custom_nodes/qwen2vl_flux

# [ ] Verify custom node files
ls -la /path/to/ComfyUI/custom_nodes/qwen2vl_flux/

# [ ] Check node registration
grep -r "NODE_CLASS_MAPPINGS" \
  /path/to/ComfyUI/custom_nodes/qwen2vl_flux/

# [ ] Create workflows directory if needed
mkdir -p /path/to/ComfyUI/workflows

# [ ] Copy example workflows
cp workflows/*.json /path/to/ComfyUI/workflows/
```

### 6. GPU Configuration (10 min)

```bash
# [ ] Verify NVIDIA drivers
nvidia-smi

# [ ] Check CUDA support
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

# [ ] Test GPU Docker support
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi

# [ ] Verify GPU device ordering
nvidia-smi -i 0 && nvidia-smi -i 1

# [ ] Check memory per GPU
nvidia-smi --query-gpu=index,memory.total --format=csv
```

### 7. Database/Storage Setup (if applicable)

```bash
# [ ] Create output directory with proper permissions
mkdir -p /data/outputs
chmod 755 /data/outputs

# [ ] Set up volume mounts for Docker
mkdir -p /data/models
mkdir -p /data/outputs
mkdir -p /data/logs

# [ ] Verify disk space
df -h /data/
```

---

## Testing Phase (Day Of, After Deployment)

### 1. System Startup Tests (15 min)

```bash
# [ ] Start container
docker-compose -f docker-compose.prod.yml up -d

# [ ] Wait for startup
sleep 10

# [ ] Check container status
docker-compose -f docker-compose.prod.yml ps

# [ ] Check logs for errors
docker-compose -f docker-compose.prod.yml logs qwen-flux-inference | head -50

# [ ] Verify GPU detection in container
docker-compose -f docker-compose.prod.yml exec qwen-flux-inference \
  python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2. Model Loading Tests (30 min)

```bash
# [ ] Test Qwen2VL model loads
docker-compose -f docker-compose.prod.yml exec qwen-flux-inference \
  python -c "from model import FluxModel; m = FluxModel(device='cuda:0'); \
    print('✓ Qwen2VL loaded')"

# [ ] Test FLUX model loads
docker-compose -f docker-compose.prod.yml exec qwen-flux-inference \
  python -c "from model import FluxModel; m = FluxModel(device='cuda:1'); \
    print('✓ FLUX loaded')"

# [ ] Test full pipeline initialization
docker-compose -f docker-compose.prod.yml exec qwen-flux-inference \
  python -c "from model import FluxModel; \
    m = FluxModel(required_features=['controlnet', 'depth', 'line']); \
    print('✓ Full pipeline initialized')"
```

### 3. Generation Tests (30-45 min)

```bash
# [ ] Create test image
python << 'EOF'
from PIL import Image
img = Image.new('RGB', (512, 512), color='blue')
img.save('test_image.png')
print("✓ Test image created")
EOF

# [ ] Test variation generation
docker-compose -f docker-compose.prod.yml exec -T qwen-flux-inference \
  python /app/test_generation.py

# [ ] Check output files
ls -la ComfyUI/output/ 2>/dev/null || echo "No output directory yet"

# [ ] Verify generated images
if [ -f "output/generated_*.png" ]; then
  echo "✓ Generation successful"
else
  echo "✗ Generation failed - check logs"
fi
```

### 4. Performance Tests (20 min)

```bash
# [ ] Monitor GPU usage during generation
nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader -l 1

# [ ] Check generation speed
# Expected: 45-60 seconds per image (28 steps, single GPU)

# [ ] Verify memory usage
docker stats --no-stream qwen-flux-inference

# [ ] Record baseline metrics
echo "GPU Memory:" >> metrics.txt
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> metrics.txt
```

### 5. API/Interface Tests (if applicable) (15 min)

```bash
# [ ] Test ComfyUI web interface
curl -s http://localhost:8188/system_stats | python -m json.tool | head -20

# [ ] Load example workflow in ComfyUI
# [ ] Queue generation through web UI
# [ ] Verify output appears

# [ ] Test API endpoint (if implemented)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test"}' 2>/dev/null | head -20
```

### 6. Error Handling Tests (15 min)

```bash
# [ ] Test graceful handling of invalid inputs
# [ ] Test OOM handling with large batch size
# [ ] Test invalid generation parameters
# [ ] Test missing model handling
# [ ] Verify error messages are logged

# [ ] Check error logs
docker-compose -f docker-compose.prod.yml logs | grep -i error
```

---

## Post-Deployment (First Week)

### Monitoring Setup

- [ ] Configure log aggregation (ELK, Datadog, etc.)
- [ ] Set up GPU monitoring (Prometheus metrics)
- [ ] Create alerting rules for:
  - [ ] GPU temperature > 80°C
  - [ ] GPU memory > 90%
  - [ ] API response time > 60s
  - [ ] Generation failures
- [ ] Set up performance dashboards
- [ ] Configure log retention policies

### Performance Baseline

- [ ] Record startup time
  ```bash
  time docker-compose -f docker-compose.prod.yml up -d
  ```
- [ ] Record generation time for standard prompt
  ```bash
  time python -c "from model import FluxModel; ..."
  ```
- [ ] Record memory usage per generation
- [ ] Document baseline metrics

### Backup Verification

- [ ] Test Docker image restoration from backup
  ```bash
  docker load < qwen2vl-flux-production.tar.gz
  ```
- [ ] Test model checkpoint backup/restore
- [ ] Verify database backups (if applicable)
- [ ] Document backup procedures

### Documentation Updates

- [ ] Update runbook with your specific setup
- [ ] Document custom environment variables
- [ ] Record GPU/hardware configuration
- [ ] Document any deviations from standard setup
- [ ] Create on-call guides

### Team Training

- [ ] Train ops team on startup/shutdown procedures
- [ ] Train on-call staff on troubleshooting
- [ ] Create decision trees for common issues
- [ ] Schedule knowledge transfer sessions

---

## Security Checklist

### Access Control
- [ ] Verify firewall rules configured
- [ ] Limit container port access
- [ ] Require authentication for API
- [ ] Use private Docker registry (if applicable)
- [ ] Encrypt credentials in transit

### Secrets Management
- [ ] HF_TOKEN not in plaintext files
- [ ] API keys stored in secure vault
- [ ] Docker secrets configured (if Swarm)
- [ ] Kubernetes secrets configured (if K8s)
- [ ] Audit log access to secrets

### Container Security
- [ ] Run containers as non-root user
- [ ] Use read-only filesystems where possible
- [ ] Scan images for vulnerabilities
- [ ] Keep base images updated
- [ ] Implement resource limits

### Monitoring & Logging
- [ ] Enable audit logging
- [ ] Monitor for anomalous behavior
- [ ] Set up intrusion detection (if on public network)
- [ ] Log all API access
- [ ] Archive logs securely

---

## Troubleshooting Quick Reference

### If Models Don't Load
```bash
# 1. Check checkpoint directory
ls -la $CHECKPOINT_DIR

# 2. Verify CHECKPOINT_DIR environment variable
echo $CHECKPOINT_DIR

# 3. Check file permissions
chmod -R 755 $CHECKPOINT_DIR

# 4. Verify GPU access
nvidia-smi

# 5. Check container logs
docker logs qwen-flux-inference
```

### If GPU Not Detected
```bash
# 1. Verify nvidia-docker installed
docker run --rm --runtime=nvidia nvidia/cuda:12.4.1-base nvidia-smi

# 2. Check NVIDIA_VISIBLE_DEVICES
echo $NVIDIA_VISIBLE_DEVICES

# 3. Verify docker daemon config
cat /etc/docker/daemon.json | grep nvidia

# 4. Restart docker daemon
sudo systemctl restart docker
```

### If Generation Fails
```bash
# 1. Check available VRAM
nvidia-smi

# 2. Reduce batch_size to 1

# 3. Reduce num_inference_steps to 20

# 4. Check container logs
docker logs -f qwen-flux-inference | tail -100

# 5. Verify model files are not corrupted
```

### If Slow Generation
```bash
# 1. Monitor GPU utilization
watch nvidia-smi

# 2. Check CPU usage
top

# 3. Check disk I/O
iostat -x 1

# 4. Reduce inference_steps

# 5. Check if other processes using GPU
lsof /dev/nvidia*
```

---

## Sign-Off

**System Administrator**: _________________ Date: _______

**QA/Testing**: _________________ Date: _______

**Operations Lead**: _________________ Date: _______

**Security Review**: _________________ Date: _______

---

## Post-Deployment Notes

Use this section to document:
- Any deviations from standard setup
- Custom configurations implemented
- Known issues and workarounds
- Performance observations
- Team feedback and improvements

```
[Document your notes here]
```

---

## Rollback Plan

If deployment fails, follow this rollback procedure:

1. **Stop current deployment**
   ```bash
   docker-compose -f docker-compose.prod.yml down
   ```

2. **Restore from backup**
   ```bash
   docker load < /backup/qwen2vl-flux-production-previous.tar.gz
   ```

3. **Verify backup image**
   ```bash
   docker images | grep qwen2vl-flux
   ```

4. **Restart with previous version**
   ```bash
   docker tag qwen2vl-flux:previous qwen2vl-flux:production
   docker-compose -f docker-compose.prod.yml up -d
   ```

5. **Run verification tests** (see Testing Phase above)

6. **Root cause analysis** - Determine what went wrong

---

**Deployment Complete!** 🎉

Your Qwen2VL-FLUX pipeline is now production-ready. Continue monitoring and follow the post-deployment procedures above.

For support, refer to:
- QUICKSTART.md - Quick reference
- COMFYUI_INTEGRATION.md - ComfyUI setup
- DEPLOYMENT_GUIDE.md - Detailed deployment info
- PROJECT_SUMMARY.md - Complete overview
