
### Minimum
- OS: Linux is preferred for production, but macOS can run for testing.
- Python: `3.11` (repo uses `python3.11`)
- Disk: `~100 MB` for code + models
  - `efficientnet_b1_crop_mini.pt` ≈ 25 MB
  - `efficientnet_b0_disease_mini.pt` ≈ 16 MB
  - plus JSON mapping files and Python packages
- RAM: `4 GB`
- CPU: `2 vCPU`
- Dependencies: from requirements.txt
  - `fastapi`
  - `uvicorn[standard]`
  - `torch>=2.0.0`
  - `torchvision>=0.15.0`
  - `timm`
  - `albumentations`
  - `opencv-python-headless`
  - `ultralytics`
  - `numpy`
  - `pillow`
  - `python-multipart`

### Recommended
- RAM: `8 GB`
- CPU: `4 vCPU`
- Disk: `200 MB` to allow logs, model updates, and package cache
- Python virtual environment for isolation

### GPU / Accelerator
If you want faster inference or higher throughput:
- NVIDIA GPU with CUDA support
  - GPU memory: `4 GB` minimum, `8 GB` recommended
- Or Apple Silicon with `torch.backends.mps` support
- If using GPU, install appropriate PyTorch build for CUDA/MPS

### Production considerations
- Run with `uvicorn main:app --host 0.0.0.0 --port 8000`
- For higher traffic, use a process manager or ASGI server with workers
  - e.g. `gunicorn -k uvicorn.workers.UvicornWorker`
- Keep only one model load per process to avoid duplicated memory use
- If you use CPU-only inference, expect each model and request pipeline to consume additional RAM while processing
