# =====================================================
# VILA-M3 Medical Imaging Demo
# =====================================================

# NVIDIA CUDA setup (original)
demo_m3:
	pip install -U setuptools && cd thirdparty/VILA && ./environment_setup.sh
	pip install -U python-dotenv gradio monai[nibabel,pynrrd,skimage,fire,ignite] torchxrayvision huggingface_hub colored
	mkdir -p $(HOME)/.cache/torch/hub/bundle
	python -m monai.bundle download vista3d --version 0.5.4 --bundle_dir $(HOME)/.cache/torch/hub/bundle
	python -m monai.bundle download brats_mri_segmentation --version 0.5.2 --bundle_dir $(HOME)/.cache/torch/hub/bundle
	unzip $(HOME)/.cache/torch/hub/bundle/vista3d_v0.5.4.zip -d $(HOME)/.cache/torch/hub/bundle/vista3d_v0.5.4

# AMD ROCm setup (MI300X/MI300A)
setup_amd:
	@bash ./setup_amd_inference.sh

# Run the demo
run_demo:
	cd m3/demo && python3 gradio_m3.py

# Check GPU status
check_gpu:
	@rocm-smi --showproductname 2>/dev/null || nvidia-smi 2>/dev/null || echo "No GPU detected"
	@python3 -c "import torch; print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')"

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: demo_m3 setup_amd run_demo check_gpu clean
