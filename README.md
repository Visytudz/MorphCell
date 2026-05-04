# MorphCell

## Installation

This repository uses `uv` for Python environment management. Dependencies,
including the `pointnet2-ops` CUDA extension, are declared in `pyproject.toml`
and locked in `uv.lock`.

Run the following from the repository root:

```bash
# Install C++ build tools required by the CUDA extension build.
sudo apt update
sudo apt install -y build-essential ninja-build

# Install CUDA Toolkit 11.8 from NVIDIA, then verify nvcc.
# Download: https://developer.nvidia.com/cuda-11-8-0-download-archive
# CUDA 11.8 is recommended because the locked PyTorch build uses
# `torch==2.7.1+cu118`.
nvcc --version

# Install uv if needed.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install MorphCell.
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Set the dataset root when data is not stored under the repository's
`datasets/` directory:

```bash
export MORPHCELL_DATA_ROOT=/path/to/datasets
```

Verify the installation:

```bash
python -c "import torch; import pointnet2_ops; import morphcell; print(torch.__version__)"
```

Run a training entry point:

```bash
python morphcell/main.py --config-name=pretrain_dfn
```
