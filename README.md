# MorphCell

## Installation

This repository uses `uv` for Python environment management. Dependencies,
including the `pointnet2-ops` CUDA extension, are declared in `pyproject.toml`
and locked in `uv.lock`.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, create the environment and install MorphCell:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Verify the installation:

```bash
python -c "import torch; import pointnet2_ops; import morphcell; print(torch.__version__)"
```

Run a training entry point:

```bash
python morphcell/main.py --config-name=pretrain_dfn
```
